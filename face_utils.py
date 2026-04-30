import base64
import os
import pickle
import zlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA

from db import (
    clear_legacy_user_data,
    get_all_users,
    get_face_samples_for_user,
    replace_face_samples,
    update_user_average_eye_distance,
    update_user_feature_summary,
)


try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pca_model.pkl")
FACE_SIZE = (120, 120)
MIN_CAPTURE_SAMPLES = 10
MAX_CAPTURE_SAMPLES = 20
DEFAULT_DISTANCE_THRESHOLD = 520.0
DEFAULT_EYE_THRESHOLD = 0.08
DEFAULT_CORRELATION_THRESHOLD = 0.08
MIN_CORRELATION_WITH_EYE = 0.1


if mp is not None:  # pragma: no branch - simple initialization
    FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
else:
    FACE_MESH = None


def get_face_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))


def get_eye_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"))


def decode_base64_image(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    binary = base64.b64decode(image_data)
    image_array = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode the submitted image.")
    return frame


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast_boosted = clahe.apply(gray)
    equalized = cv2.equalizeHist(contrast_boosted)
    normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(normalized, FACE_SIZE)


def detect_largest_face(frame_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width == 0 or height == 0:
        return None

    resize_scale = min(1.0, 960.0 / max(width, height))
    search_gray = gray
    if resize_scale < 1.0:
        search_gray = cv2.resize(
            gray,
            (max(1, int(width * resize_scale)), max(1, int(height * resize_scale))),
            interpolation=cv2.INTER_LINEAR,
        )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    detection_variants = [
        search_gray,
        cv2.equalizeHist(search_gray),
        clahe.apply(search_gray),
    ]
    min_dimension = min(search_gray.shape[:2])
    min_face_size = max(56, int(min_dimension * 0.12))
    cascade = get_face_cascade()

    faces: List[Tuple[int, int, int, int]] = []
    detection_settings = [
        (1.15, 5, min_face_size),
        (1.1, 4, max(48, int(min_face_size * 0.9))),
        (1.05, 3, 40),
    ]
    for variant in detection_variants:
        for scale_factor, min_neighbors, min_size in detection_settings:
            detected_faces = cascade.detectMultiScale(
                variant,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_size, min_size),
            )
            if len(detected_faces) == 0:
                continue
            for x, y, w, h in detected_faces.tolist():
                if resize_scale < 1.0:
                    faces.append(
                        (
                            int(x / resize_scale),
                            int(y / resize_scale),
                            int(w / resize_scale),
                            int(h / resize_scale),
                        )
                    )
                else:
                    faces.append((int(x), int(y), int(w), int(h)))

    if not faces:
        return None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.12)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    face = frame_bgr[y1:y2, x1:x2]
    return face, (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def _eye_center_from_landmarks(landmarks: Any, indices: List[int], width: int, height: int) -> np.ndarray:
    points = []
    for index in indices:
        landmark = landmarks.landmark[index]
        points.append([landmark.x * width, landmark.y * height])
    return np.mean(np.array(points, dtype=np.float32), axis=0)


def _normalized_eye_distance_from_boxes(eyes: np.ndarray, face_width: int) -> float:
    if len(eyes) < 2 or face_width <= 0:
        return 0.0

    candidates = sorted(eyes.tolist(), key=lambda item: item[2] * item[3], reverse=True)[:6]
    centers = [
        np.array([ex + ew / 2.0, ey + eh / 2.0], dtype=np.float32)
        for ex, ey, ew, eh in candidates
    ]
    best_pair_distance = 0.0
    for left_index in range(len(centers)):
        for right_index in range(left_index + 1, len(centers)):
            horizontal_gap = abs(centers[left_index][0] - centers[right_index][0])
            if horizontal_gap > best_pair_distance:
                best_pair_distance = horizontal_gap
    return float(best_pair_distance / face_width) if best_pair_distance > 0 else 0.0


def calculate_normalized_eye_distance(face_bgr: np.ndarray) -> float:
    face_height, face_width = face_bgr.shape[:2]
    if face_width == 0:
        return 0.0

    if FACE_MESH is not None:
        rgb_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        results = FACE_MESH.process(rgb_face)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            left_eye = _eye_center_from_landmarks(landmarks, [33, 133, 159, 145], face_width, face_height)
            right_eye = _eye_center_from_landmarks(landmarks, [362, 263, 386, 374], face_width, face_height)
            return float(np.linalg.norm(left_eye - right_eye) / face_width)

    # Haar eye detection is less stable, so we focus on the upper-face ROI and boost contrast first.
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    upper_height = max(1, int(face_height * 0.62))
    upper_face = gray[:upper_height, :]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    enhanced_upper = cv2.equalizeHist(clahe.apply(upper_face))

    min_eye_width = max(12, int(face_width * 0.08))
    min_eye_height = max(12, int(face_height * 0.08))
    eye_cascade = get_eye_cascade()

    eyes = eye_cascade.detectMultiScale(
        enhanced_upper,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(min_eye_width, min_eye_height),
    )
    normalized_distance = _normalized_eye_distance_from_boxes(eyes, face_width)
    if normalized_distance > 0:
        return normalized_distance

    # Final fallback on the whole preprocessed face with slightly stricter pairing.
    enhanced_full = cv2.equalizeHist(clahe.apply(gray))
    eyes = eye_cascade.detectMultiScale(
        enhanced_full,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_eye_width, min_eye_height),
    )
    return _normalized_eye_distance_from_boxes(eyes, face_width)


def flatten_face(face_gray: np.ndarray) -> np.ndarray:
    return face_gray.astype(np.float32).flatten()


def encode_feature_vector(vector: np.ndarray) -> bytes:
    vector_bytes = vector.astype(np.float32).tobytes()
    return zlib.compress(vector_bytes)


def decode_feature_vector(feature_blob: bytes) -> np.ndarray:
    vector_bytes = zlib.decompress(feature_blob)
    return np.frombuffer(vector_bytes, dtype=np.float32)


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    correlation = np.corrcoef(a, b)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def is_valid_eye_distance(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return 0.05 <= numeric_value <= 0.8


def valid_eye_distances(values: List[float]) -> List[float]:
    return [float(value) for value in values if is_valid_eye_distance(value)]


def _nearest_sample_stats(sample_projections: np.ndarray) -> Tuple[List[float], List[float]]:
    if len(sample_projections) <= 1:
        return [], []

    nearest_distances: List[float] = []
    nearest_correlations: List[float] = []
    for index, projection in enumerate(sample_projections):
        best_distance = float("inf")
        best_correlation = 0.0
        for other_index, other_projection in enumerate(sample_projections):
            if index == other_index:
                continue

            candidate_distance = float(np.linalg.norm(projection - other_projection))
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_correlation = safe_correlation(projection, other_projection)

        if best_distance != float("inf"):
            nearest_distances.append(best_distance)
            nearest_correlations.append(best_correlation)

    return nearest_distances, nearest_correlations


def _build_user_candidates(
    projection: np.ndarray,
    eye_distance: float,
    user_profiles: Dict[int, Dict[str, Any]],
    sample_profiles: List[Dict[str, Any]],
    ignore_sample_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    matches_by_user: Dict[int, List[Dict[str, Any]]] = {}

    for sample in sample_profiles:
        sample_id = sample.get("sample_id")
        if ignore_sample_id is not None and sample_id == ignore_sample_id:
            continue

        user_id = int(sample["user_id"])
        sample_projection = np.asarray(sample["projection"], dtype=np.float32)
        sample_eye_distance = float(sample.get("eye_distance", 0.0))
        matches_by_user.setdefault(user_id, []).append(
            {
                "distance": float(np.linalg.norm(projection - sample_projection)),
                "correlation": safe_correlation(projection, sample_projection),
                "eye_difference": (
                    abs(eye_distance - sample_eye_distance)
                    if is_valid_eye_distance(eye_distance) and is_valid_eye_distance(sample_eye_distance)
                    else None
                ),
            }
        )

    candidates: List[Dict[str, Any]] = []
    for user_id, sample_matches in matches_by_user.items():
        profile = user_profiles.get(user_id)
        if not profile:
            continue

        sample_matches.sort(key=lambda match: match["distance"])
        top_matches = sample_matches[: min(3, len(sample_matches))]
        profile_projection = np.asarray(profile["projection"], dtype=np.float32)
        profile_eye_distance = float(profile["average_eye_distance"])
        profile_eye_difference = (
            abs(eye_distance - profile_eye_distance)
            if is_valid_eye_distance(eye_distance) and is_valid_eye_distance(profile_eye_distance)
            else sample_matches[0]["eye_difference"]
        )

        candidates.append(
            {
                "user_id": user_id,
                "nearest_sample_distance": float(sample_matches[0]["distance"]),
                "top_sample_distance": float(np.mean([match["distance"] for match in top_matches])),
                "correlation": float(max(match["correlation"] for match in top_matches)),
                "eye_difference": profile_eye_difference,
                "profile_distance": float(np.linalg.norm(projection - profile_projection)),
            }
        )

    candidates.sort(
        key=lambda candidate: (
            candidate["nearest_sample_distance"],
            candidate["top_sample_distance"],
            candidate["profile_distance"],
        )
    )
    return candidates


def passes_match_rules(
    best_distance: float,
    second_best_distance: float,
    eye_difference: Optional[float],
    correlation: float,
    profile: Dict[str, Any],
    distance_threshold: float,
    eye_threshold: float,
    distance_margin: float,
    correlation_threshold: float,
    profile_distance: Optional[float] = None,
    support_distance: Optional[float] = None,
) -> bool:
    required_margin = max(distance_margin, best_distance * 0.1)
    margin_condition = second_best_distance == float("inf") or (second_best_distance - best_distance) >= required_margin
    ratio_condition = second_best_distance == float("inf") or best_distance <= second_best_distance * 0.92
    effective_distance_limit = max(
        distance_threshold,
        float(profile.get("sample_distance_limit", profile.get("distance_limit", distance_threshold))),
    )
    profile_correlation_floor = float(
        profile.get("sample_correlation_floor", profile.get("correlation_floor", correlation_threshold))
    )
    correlation_floor = max(correlation_threshold, MIN_CORRELATION_WITH_EYE, profile_correlation_floor)
    eye_condition = eye_difference is None or eye_difference <= eye_threshold

    # Eye detection can fail or drift on webcam frames. A very strong PCA/correlation
    # match should still identify an already-enrolled user instead of false rejecting.
    strong_appearance_condition = (
        correlation >= max(correlation_floor + 0.05, 0.5)
        or best_distance <= effective_distance_limit * 0.82
    )
    support_condition = True
    if support_distance is not None:
        support_condition = support_distance <= max(
            float(profile.get("distance_limit", effective_distance_limit * 1.25)),
            effective_distance_limit * 1.2,
        )

    profile_condition = True
    if profile_distance is not None:
        profile_condition = profile_distance <= max(
            float(profile.get("distance_limit", effective_distance_limit * 1.35)),
            effective_distance_limit * 1.3,
        )

    return (
        margin_condition
        and ratio_condition
        and best_distance <= effective_distance_limit
        and correlation >= correlation_floor
        and support_condition
        and profile_condition
        and (eye_condition or strong_appearance_condition)
    )


def extract_samples_from_images(images: List[str], min_required: int = MIN_CAPTURE_SAMPLES) -> List[Dict[str, Any]]:
    if len(images) < min_required:
        raise ValueError(f"Capture at least {min_required} samples.")

    processed_samples: List[Dict[str, Any]] = []
    for image_data in images[:MAX_CAPTURE_SAMPLES]:
        frame = decode_base64_image(image_data)
        detected = detect_largest_face(frame)
        if not detected:
            continue

        face_bgr, _ = detected
        face_gray = preprocess_face(face_bgr)
        feature_vector = flatten_face(face_gray)
        processed_samples.append(
            {
                "feature_blob": encode_feature_vector(feature_vector),
                "eye_distance": calculate_normalized_eye_distance(face_bgr),
            }
        )

    if len(processed_samples) < min_required:
        raise ValueError(
            f"Only {len(processed_samples)} valid faces were detected. Capture clearer images and try again."
        )

    return processed_samples


def migrate_legacy_samples_if_needed() -> None:
    users = get_all_users()
    for user in users:
        if user["sample_count"] > 0 or not user["legacy_image_paths"]:
            continue

        migrated_samples: List[Dict[str, Any]] = []
        default_eye_distance = float(user["legacy_features"].get("eye_distance", user["average_eye_distance"]))
        for sample_index, image_path in enumerate(user["legacy_image_paths"][:MAX_CAPTURE_SAMPLES], start=1):
            if not os.path.exists(image_path):
                continue

            face_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if face_gray is None:
                continue

            resized = cv2.resize(face_gray, FACE_SIZE)
            feature_vector = flatten_face(resized)
            migrated_samples.append(
                {
                    "feature_blob": encode_feature_vector(feature_vector),
                    "eye_distance": default_eye_distance,
                    "sample_index": sample_index,
                }
            )

        if migrated_samples:
            replace_face_samples(user["id"], migrated_samples)
            clear_legacy_user_data(user["id"])


def _load_training_dataset() -> Tuple[np.ndarray, List[int], List[float], Dict[int, Dict[str, Any]]]:
    migrate_legacy_samples_if_needed()
    users = get_all_users()

    sample_vectors: List[np.ndarray] = []
    labels: List[int] = []
    eye_distances: List[float] = []
    user_lookup: Dict[int, Dict[str, Any]] = {}

    for user in users:
        samples = get_face_samples_for_user(user["id"])
        if not samples:
            continue

        user_lookup[user["id"]] = user
        for sample in samples:
            sample_vectors.append(decode_feature_vector(sample["feature_blob"]))
            labels.append(user["id"])
            eye_distances.append(float(sample["eye_distance"]))

    if not sample_vectors:
        return np.array([], dtype=np.float32), labels, eye_distances, user_lookup

    return np.array(sample_vectors, dtype=np.float32), labels, eye_distances, user_lookup


def _calculate_thresholds(
    projections: np.ndarray,
    labels: List[int],
    eye_distances: List[float],
    user_profiles: Dict[int, Dict[str, Any]],
    sample_profiles: List[Dict[str, Any]],
) -> Tuple[float, float, float, float]:
    genuine_distances: List[float] = []
    genuine_eye_differences: List[float] = []
    genuine_correlations: List[float] = []
    impostor_best_distances: List[float] = []
    impostor_best_eye_differences: List[float] = []
    impostor_best_correlations: List[float] = []

    for index, user_id in enumerate(labels):
        true_profile = user_profiles[user_id]
        same_user_candidates = [
            sample
            for sample in sample_profiles
            if sample["user_id"] == user_id and sample["sample_id"] != index
        ]
        if same_user_candidates:
            genuine_candidate = min(
                same_user_candidates,
                key=lambda sample: float(np.linalg.norm(projections[index] - sample["projection"])),
            )
            genuine_distances.append(float(np.linalg.norm(projections[index] - genuine_candidate["projection"])))
            genuine_correlations.append(safe_correlation(projections[index], genuine_candidate["projection"]))
        else:
            genuine_distances.append(float(np.linalg.norm(projections[index] - true_profile["projection"])))
            genuine_correlations.append(safe_correlation(projections[index], true_profile["projection"]))

        sample_eye_distance = eye_distances[index]
        profile_eye_distance = float(true_profile["average_eye_distance"])
        if is_valid_eye_distance(sample_eye_distance) and is_valid_eye_distance(profile_eye_distance):
            genuine_eye_differences.append(abs(sample_eye_distance - profile_eye_distance))

        impostor_candidates = []
        for sample in sample_profiles:
            other_user_id = int(sample["user_id"])
            if other_user_id == user_id:
                continue

            profile = user_profiles[other_user_id]
            sample_eye_available = is_valid_eye_distance(sample_eye_distance)
            profile_eye_available = is_valid_eye_distance(float(profile["average_eye_distance"]))
            eye_difference = (
                abs(sample_eye_distance - float(profile["average_eye_distance"]))
                if sample_eye_available and profile_eye_available
                else None
            )
            impostor_candidates.append(
                (
                    float(np.linalg.norm(projections[index] - sample["projection"])),
                    eye_difference,
                    safe_correlation(projections[index], sample["projection"]),
                )
            )

        if impostor_candidates:
            best_impostor_distance, best_impostor_eye_difference, best_impostor_correlation = min(
                impostor_candidates,
                key=lambda item: item[0],
            )
            impostor_best_distances.append(best_impostor_distance)
            if best_impostor_eye_difference is not None:
                impostor_best_eye_differences.append(best_impostor_eye_difference)
            impostor_best_correlations.append(best_impostor_correlation)

    if not genuine_distances:
        return DEFAULT_DISTANCE_THRESHOLD, DEFAULT_EYE_THRESHOLD, 0.25, DEFAULT_CORRELATION_THRESHOLD

    if impostor_best_distances:
        distance_threshold = float((max(genuine_distances) + min(impostor_best_distances)) / 2.0)
        distance_margin = float(np.clip(min(impostor_best_distances) - max(genuine_distances), 40.0, 5000.0))
    else:
        distance_threshold = float(np.percentile(genuine_distances, 95) * 1.08)
        distance_margin = max(distance_threshold * 0.02, 40.0)
    distance_threshold = max(distance_threshold, 500.0)

    if genuine_eye_differences and impostor_best_eye_differences:
        eye_threshold = float(
            np.clip((max(genuine_eye_differences) + min(impostor_best_eye_differences)) / 2.0, 0.015, 0.18)
        )
    elif genuine_eye_differences:
        eye_threshold = float(np.clip(np.percentile(genuine_eye_differences, 95) * 1.15, 0.015, 0.18))
    else:
        eye_threshold = DEFAULT_EYE_THRESHOLD

    if genuine_correlations and impostor_best_correlations:
        correlation_threshold = float(
            np.clip(
                (min(genuine_correlations) + max(impostor_best_correlations)) / 2.0,
                0.02,
                0.35,
            )
        )
    elif genuine_correlations:
        correlation_threshold = float(np.clip(min(genuine_correlations) * 0.95, 0.02, 0.35))
    else:
        correlation_threshold = DEFAULT_CORRELATION_THRESHOLD

    return distance_threshold, eye_threshold, distance_margin, correlation_threshold


def _evaluate_model(
    projections: np.ndarray,
    labels: List[int],
    eye_distances: List[float],
    user_profiles: Dict[int, Dict[str, Any]],
    sample_profiles: List[Dict[str, Any]],
    distance_threshold: float,
    eye_threshold: float,
    distance_margin: float,
    correlation_threshold: float,
) -> Dict[str, float]:
    correct = 0
    false_accepts = 0
    false_rejects = 0

    for index, true_user_id in enumerate(labels):
        candidates = _build_user_candidates(
            projection=projections[index],
            eye_distance=eye_distances[index],
            user_profiles=user_profiles,
            sample_profiles=sample_profiles,
            ignore_sample_id=index,
        )
        best_candidate = candidates[0] if candidates else None
        second_best_distance = candidates[1]["nearest_sample_distance"] if len(candidates) > 1 else float("inf")
        candidate_profile = user_profiles.get(best_candidate["user_id"]) if best_candidate else None
        matched = bool(candidate_profile) and passes_match_rules(
            best_distance=best_candidate["nearest_sample_distance"],
            second_best_distance=second_best_distance,
            eye_difference=best_candidate["eye_difference"],
            correlation=best_candidate["correlation"],
            profile=candidate_profile,
            distance_threshold=distance_threshold,
            eye_threshold=eye_threshold,
            distance_margin=distance_margin,
            correlation_threshold=correlation_threshold,
            profile_distance=best_candidate["profile_distance"],
            support_distance=best_candidate["top_sample_distance"],
        )
        if matched and best_candidate["user_id"] == true_user_id:
            correct += 1
        elif matched and best_candidate["user_id"] != true_user_id:
            false_accepts += 1
        else:
            false_rejects += 1

    total = len(labels) or 1
    accuracy = (correct / total) * 100.0
    far = (false_accepts / total) * 100.0
    frr = (false_rejects / total) * 100.0
    return {
        "samples": float(len(labels)),
        "accuracy": round(accuracy, 2),
        "false_accept_rate": round(far, 2),
        "false_reject_rate": round(frr, 2),
    }


def train_model_from_database() -> Dict[str, Any]:
    vectors, labels, eye_distances, user_lookup = _load_training_dataset()
    if len(vectors) < 3 or len(set(labels)) < 1:
        raise ValueError("At least 3 processed face samples are required before training.")

    sample_count, feature_count = vectors.shape
    n_components = min(sample_count - 1, feature_count, 50)
    pca = PCA(n_components=n_components, whiten=False, svd_solver="randomized", random_state=42)
    projections = pca.fit_transform(vectors)
    sample_profiles = [
        {
            "sample_id": index,
            "user_id": int(labels[index]),
            "projection": projections[index],
            "eye_distance": float(eye_distances[index]),
        }
        for index in range(len(labels))
    ]

    user_profiles: Dict[int, Dict[str, Any]] = {}
    for user_id in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == user_id]
        user_samples = projections[indices]
        user_eye_distances = [eye_distances[index] for index in indices]
        user_details = user_lookup[user_id]
        mean_projection = np.mean(user_samples, axis=0)
        valid_user_eye_distances = valid_eye_distances(user_eye_distances)
        average_eye_distance = float(np.mean(valid_user_eye_distances)) if valid_user_eye_distances else 0.0
        user_sample_distances = [float(np.linalg.norm(sample - mean_projection)) for sample in user_samples]
        user_sample_correlations = [safe_correlation(sample, mean_projection) for sample in user_samples]
        nearest_sample_distances, nearest_sample_correlations = _nearest_sample_stats(user_samples)
        distance_limit = float(max(user_sample_distances) * 1.1) if user_sample_distances else float("inf")
        correlation_floor = (
            float(max(min(user_sample_correlations) * 0.95, 0.02))
            if user_sample_correlations
            else DEFAULT_CORRELATION_THRESHOLD
        )
        sample_distance_limit = (
            float(max(nearest_sample_distances) * 1.08)
            if nearest_sample_distances
            else distance_limit
        )
        sample_correlation_floor = (
            float(max(min(nearest_sample_correlations) * 0.95, 0.05))
            if nearest_sample_correlations
            else correlation_floor
        )
        user_profiles[user_id] = {
            "user_id": user_id,
            "name": user_details["name"],
            "person_identifier": user_details["person_identifier"],
            "email": user_details["email"],
            "projection": mean_projection,
            "average_eye_distance": average_eye_distance,
            "sample_count": len(indices),
            "max_training_distance": max(user_sample_distances) if user_sample_distances else 0.0,
            "min_training_correlation": min(user_sample_correlations) if user_sample_correlations else 0.0,
            "distance_limit": distance_limit,
            "correlation_floor": correlation_floor,
            "sample_distance_limit": sample_distance_limit,
            "sample_correlation_floor": sample_correlation_floor,
        }
        update_user_average_eye_distance(user_id, average_eye_distance)
        update_user_feature_summary(
            user_id,
            {
                "pca_projection": mean_projection.tolist(),
                "average_eye_distance": average_eye_distance,
                "sample_count": len(indices),
                "max_training_distance": user_profiles[user_id]["max_training_distance"],
                "min_training_correlation": user_profiles[user_id]["min_training_correlation"],
            },
        )

    distance_threshold, eye_threshold, distance_margin, correlation_threshold = _calculate_thresholds(
        projections,
        labels,
        eye_distances,
        user_profiles,
        sample_profiles,
    )
    metrics = _evaluate_model(
        projections,
        labels,
        eye_distances,
        user_profiles,
        sample_profiles,
        distance_threshold,
        eye_threshold,
        distance_margin,
        correlation_threshold,
    )

    payload = {
        "pca": pca,
        "user_profiles": user_profiles,
        "sample_profiles": sample_profiles,
        "projections": projections,
        "labels": labels,
        "eye_distances": eye_distances,
        "distance_threshold": round(distance_threshold, 4),
        "eye_threshold": round(eye_threshold, 4),
        "distance_margin": round(distance_margin, 4),
        "correlation_threshold": round(correlation_threshold, 4),
        "matching_mode": "sample_profiles",
        "face_size": FACE_SIZE,
        "metrics": metrics,
        "trained_users": len(user_profiles),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(payload, model_file)

    return payload


def remove_trained_model() -> None:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)


def _ensure_sample_profiles(model: Dict[str, Any]) -> Dict[str, Any]:
    projections = model.get("projections")
    labels = model.get("labels")
    eye_distances = model.get("eye_distances")
    if projections is None:
        projections = []
    if labels is None:
        labels = []
    if eye_distances is None:
        eye_distances = []
    user_profiles = model.get("user_profiles") or {}
    needs_refresh = model.get("matching_mode") != "sample_profiles"
    if len(projections) != len(labels):
        return model

    if not model.get("sample_profiles"):
        model["sample_profiles"] = [
            {
                "sample_id": index,
                "user_id": int(labels[index]),
                "projection": projections[index],
                "eye_distance": float(eye_distances[index]) if index < len(eye_distances) else 0.0,
            }
            for index in range(len(labels))
        ]
        needs_refresh = True

    for user_id, profile in user_profiles.items():
        if profile.get("sample_distance_limit") is not None and profile.get("sample_correlation_floor") is not None:
            continue

        user_sample_projections = np.array(
            [sample["projection"] for sample in model["sample_profiles"] if int(sample["user_id"]) == int(user_id)],
            dtype=np.float32,
        )
        nearest_distances, nearest_correlations = _nearest_sample_stats(user_sample_projections)
        distance_limit = float(profile.get("distance_limit", model.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD)))
        correlation_floor = float(
            profile.get("correlation_floor", model.get("correlation_threshold", DEFAULT_CORRELATION_THRESHOLD))
        )
        profile["sample_distance_limit"] = (
            float(max(nearest_distances) * 1.08)
            if nearest_distances
            else distance_limit
        )
        profile["sample_correlation_floor"] = (
            float(max(min(nearest_correlations) * 0.95, 0.05))
            if nearest_correlations
            else correlation_floor
        )
        needs_refresh = True

    projections_array = np.asarray(projections, dtype=np.float32)
    labels_list = [int(label) for label in labels]
    eye_distances_list = [float(value) for value in eye_distances]
    if needs_refresh and len(projections_array) and user_profiles:
        distance_threshold, eye_threshold, distance_margin, correlation_threshold = _calculate_thresholds(
            projections_array,
            labels_list,
            eye_distances_list,
            user_profiles,
            model["sample_profiles"],
        )
        model["distance_threshold"] = round(distance_threshold, 4)
        model["eye_threshold"] = round(eye_threshold, 4)
        model["distance_margin"] = round(distance_margin, 4)
        model["correlation_threshold"] = round(correlation_threshold, 4)
        model["metrics"] = _evaluate_model(
            projections_array,
            labels_list,
            eye_distances_list,
            user_profiles,
            model["sample_profiles"],
            distance_threshold,
            eye_threshold,
            distance_margin,
            correlation_threshold,
        )
        model["matching_mode"] = "sample_profiles"

    return model


def load_model() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as model_file:
        return _ensure_sample_profiles(pickle.load(model_file))


def get_model_summary() -> Dict[str, Any]:
    model = load_model()
    if not model:
        return {
            "available": False,
            "trained_users": 0,
            "distance_threshold": 0,
            "eye_threshold": 0,
            "distance_margin": 0,
            "correlation_threshold": 0,
            "metrics": {},
        }

    return {
        "available": True,
        "trained_users": int(model.get("trained_users", 0)),
        "distance_threshold": float(model.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD)),
        "eye_threshold": float(model.get("eye_threshold", DEFAULT_EYE_THRESHOLD)),
        "distance_margin": float(model.get("distance_margin", 0.25)),
        "correlation_threshold": float(model.get("correlation_threshold", DEFAULT_CORRELATION_THRESHOLD)),
        "metrics": model.get("metrics", {}),
    }


def recognize_face(image_data: str) -> Dict[str, Any]:
    model = load_model()
    if not model:
        raise ValueError("The model is not trained yet. Train the system from the admin portal first.")

    frame = decode_base64_image(image_data)
    detected = detect_largest_face(frame)
    if not detected:
        return {
            "matched": False,
            "status": "no_face",
            "message": "No Face Detected",
            "bounding_box": None,
        }

    face_bgr, bounding_box = detected
    processed_face = preprocess_face(face_bgr)
    eye_distance = calculate_normalized_eye_distance(face_bgr)
    feature_vector = flatten_face(processed_face).reshape(1, -1)
    projection = model["pca"].transform(feature_vector)[0]
    sample_profiles = model.get("sample_profiles") or []
    user_profiles = model["user_profiles"]
    candidates = _build_user_candidates(
        projection=projection,
        eye_distance=eye_distance,
        user_profiles=user_profiles,
        sample_profiles=sample_profiles,
    )
    best_match = candidates[0] if candidates else None
    second_best_distance = candidates[1]["nearest_sample_distance"] if len(candidates) > 1 else float("inf")

    if best_match is None:
        return {
            "matched": False,
            "status": "no_face",
            "message": "No Face Detected",
            "bounding_box": None,
        }

    distance_threshold = float(model.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD))
    eye_threshold = float(model.get("eye_threshold", DEFAULT_EYE_THRESHOLD))
    distance_margin = float(model.get("distance_margin", 0.25))
    correlation_threshold = float(model.get("correlation_threshold", DEFAULT_CORRELATION_THRESHOLD))
    profile = user_profiles[best_match["user_id"]]
    matched = passes_match_rules(
        best_distance=best_match["nearest_sample_distance"],
        second_best_distance=second_best_distance,
        eye_difference=best_match["eye_difference"],
        correlation=best_match["correlation"],
        profile=profile,
        distance_threshold=distance_threshold,
        eye_threshold=eye_threshold,
        distance_margin=distance_margin,
        correlation_threshold=correlation_threshold,
        profile_distance=best_match["profile_distance"],
        support_distance=best_match["top_sample_distance"],
    )

    if matched:
        return {
            "matched": True,
            "status": "granted",
            "message": "Face Detected - Access Granted",
            "user": {
                "id": best_match["user_id"],
                "name": profile["name"],
                "person_identifier": profile["person_identifier"],
                "email": profile["email"],
            },
            "pca_distance": round(best_match["nearest_sample_distance"], 4),
            "eye_difference": round(best_match["eye_difference"], 4) if best_match["eye_difference"] is not None else None,
            "correlation": round(best_match["correlation"], 4),
            "bounding_box": {
                "x": bounding_box[0],
                "y": bounding_box[1],
                "w": bounding_box[2],
                "h": bounding_box[3],
            },
        }

    return {
        "matched": False,
        "status": "denied",
        "message": "Face Detected - Access Denied (Unknown User)",
        "user": None,
        "pca_distance": round(best_match["nearest_sample_distance"], 4),
        "eye_difference": round(best_match["eye_difference"], 4) if best_match["eye_difference"] is not None else None,
        "correlation": round(best_match["correlation"], 4),
        "bounding_box": {
            "x": bounding_box[0],
            "y": bounding_box[1],
            "w": bounding_box[2],
            "h": bounding_box[3],
        },
    }
