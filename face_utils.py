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
MIN_CORRELATION_WITHOUT_EYE = 0.25
STRICT_EYE_THRESHOLD_CAP = 0.12


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
    faces = get_face_cascade().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face = frame_bgr[y : y + h, x : x + w]
    return face, (int(x), int(y), int(w), int(h))


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
) -> Tuple[float, float, float, float]:
    genuine_distances: List[float] = []
    genuine_eye_differences: List[float] = []
    genuine_correlations: List[float] = []
    impostor_best_distances: List[float] = []
    impostor_best_eye_differences: List[float] = []
    impostor_best_correlations: List[float] = []

    for index, user_id in enumerate(labels):
        true_profile = user_profiles[user_id]
        genuine_distance = float(np.linalg.norm(projections[index] - true_profile["projection"]))
        genuine_distances.append(genuine_distance)
        genuine_correlations.append(safe_correlation(projections[index], true_profile["projection"]))
        sample_eye_distance = eye_distances[index]
        profile_eye_distance = float(true_profile["average_eye_distance"])
        if is_valid_eye_distance(sample_eye_distance) and is_valid_eye_distance(profile_eye_distance):
            genuine_eye_differences.append(abs(sample_eye_distance - profile_eye_distance))

        impostor_candidates = []
        for other_user_id, profile in user_profiles.items():
            if other_user_id == user_id:
                continue
            sample_eye_available = is_valid_eye_distance(sample_eye_distance)
            profile_eye_available = is_valid_eye_distance(float(profile["average_eye_distance"]))
            eye_difference = (
                abs(sample_eye_distance - float(profile["average_eye_distance"]))
                if sample_eye_available and profile_eye_available
                else None
            )
            impostor_candidates.append(
                (
                    float(np.linalg.norm(projections[index] - profile["projection"])),
                    eye_difference,
                    safe_correlation(projections[index], profile["projection"]),
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
    distance_threshold: float,
    eye_threshold: float,
    distance_margin: float,
    correlation_threshold: float,
) -> Dict[str, float]:
    correct = 0
    false_accepts = 0
    false_rejects = 0

    for index, true_user_id in enumerate(labels):
        best_user_id = None
        best_distance = float("inf")
        best_eye_difference: Optional[float] = None
        best_correlation = -1.0
        second_best_distance = float("inf")

        for candidate_user_id, profile in user_profiles.items():
            candidate_distance = float(np.linalg.norm(projections[index] - profile["projection"]))
            sample_eye_available = is_valid_eye_distance(eye_distances[index])
            profile_eye_distance = float(profile["average_eye_distance"])
            profile_eye_available = is_valid_eye_distance(profile_eye_distance)
            candidate_eye_difference = (
                abs(eye_distances[index] - profile_eye_distance)
                if sample_eye_available and profile_eye_available
                else None
            )
            candidate_correlation = safe_correlation(projections[index], profile["projection"])
            if candidate_distance < best_distance:
                second_best_distance = best_distance
                best_distance = candidate_distance
                best_eye_difference = candidate_eye_difference
                best_correlation = candidate_correlation
                best_user_id = candidate_user_id
            elif candidate_distance < second_best_distance:
                second_best_distance = candidate_distance

        eye_available = best_eye_difference is not None
        strict_eye_threshold = min(eye_threshold, STRICT_EYE_THRESHOLD_CAP)
        eye_condition = eye_available and best_eye_difference <= strict_eye_threshold
        margin_condition = second_best_distance == float("inf") or (second_best_distance - best_distance) >= distance_margin
        candidate_profile = user_profiles.get(best_user_id) if best_user_id is not None else None
        profile_distance_limit = candidate_profile["distance_limit"] if candidate_profile else distance_threshold
        profile_correlation_floor = candidate_profile["correlation_floor"] if candidate_profile else correlation_threshold
        profile_requires_eye = (
            bool(candidate_profile)
            and is_valid_eye_distance(float(candidate_profile["average_eye_distance"]))
        )
        matched = margin_condition and (
            (
                eye_condition
                and best_distance <= min(distance_threshold, profile_distance_limit)
                and best_correlation >= max(correlation_threshold, MIN_CORRELATION_WITH_EYE, profile_correlation_floor)
            )
            or (
                (not profile_requires_eye)
                and
                (not eye_available)
                and best_distance <= min(distance_threshold, profile_distance_limit)
                and best_correlation >= max(correlation_threshold + 0.02, MIN_CORRELATION_WITHOUT_EYE, profile_correlation_floor)
            )
        )
        if matched and best_user_id == true_user_id:
            correct += 1
        elif matched and best_user_id != true_user_id:
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

    user_profiles: Dict[int, Dict[str, Any]] = {}
    for user_id in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == user_id]
        user_samples = projections[indices]
        user_eye_distances = [eye_distances[index] for index in indices]
        user_details = user_lookup[user_id]
        mean_projection = np.mean(user_samples, axis=0)
        average_eye_distance = float(np.mean(user_eye_distances))
        user_sample_distances = [float(np.linalg.norm(sample - mean_projection)) for sample in user_samples]
        user_sample_correlations = [safe_correlation(sample, mean_projection) for sample in user_samples]
        distance_limit = float(max(user_sample_distances) * 1.1) if user_sample_distances else float("inf")
        correlation_floor = (
            float(max(min(user_sample_correlations) * 0.95, 0.02))
            if user_sample_correlations
            else DEFAULT_CORRELATION_THRESHOLD
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
        }
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
    )
    metrics = _evaluate_model(
        projections,
        labels,
        eye_distances,
        user_profiles,
        distance_threshold,
        eye_threshold,
        distance_margin,
        correlation_threshold,
    )

    payload = {
        "pca": pca,
        "user_profiles": user_profiles,
        "projections": projections,
        "labels": labels,
        "eye_distances": eye_distances,
        "distance_threshold": round(distance_threshold, 4),
        "eye_threshold": round(eye_threshold, 4),
        "distance_margin": round(distance_margin, 4),
        "correlation_threshold": round(correlation_threshold, 4),
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


def load_model() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


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

    best_match: Optional[Dict[str, Any]] = None
    second_best_distance = float("inf")
    for user_id, profile in model["user_profiles"].items():
        pca_distance = float(np.linalg.norm(projection - profile["projection"]))
        profile_eye_distance = float(profile["average_eye_distance"])
        eye_difference = (
            abs(eye_distance - profile_eye_distance)
            if is_valid_eye_distance(eye_distance) and is_valid_eye_distance(profile_eye_distance)
            else None
        )
        correlation = safe_correlation(projection, profile["projection"])
        candidate = {
            "user_id": int(user_id),
            "name": profile["name"],
            "person_identifier": profile["person_identifier"],
            "email": profile["email"],
            "pca_distance": pca_distance,
            "eye_difference": eye_difference,
            "correlation": correlation,
            "distance_limit": float(profile.get("distance_limit", distance_threshold if "distance_threshold" in locals() else DEFAULT_DISTANCE_THRESHOLD)),
            "correlation_floor": float(profile.get("correlation_floor", DEFAULT_CORRELATION_THRESHOLD)),
        }

        if best_match is None or candidate["pca_distance"] < best_match["pca_distance"]:
            if best_match is not None:
                second_best_distance = best_match["pca_distance"]
            best_match = candidate
        elif candidate["pca_distance"] < second_best_distance:
            second_best_distance = candidate["pca_distance"]

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
    eye_available = best_match["eye_difference"] is not None
    strict_eye_threshold = min(eye_threshold, STRICT_EYE_THRESHOLD_CAP)
    eye_condition = eye_available and best_match["eye_difference"] <= strict_eye_threshold
    margin_condition = second_best_distance == float("inf") or (second_best_distance - best_match["pca_distance"]) >= distance_margin
    profile_distance_limit = float(best_match.get("distance_limit", distance_threshold))
    profile_correlation_floor = float(best_match.get("correlation_floor", correlation_threshold))
    profile_requires_eye = is_valid_eye_distance(
        float(model["user_profiles"][best_match["user_id"]]["average_eye_distance"])
    )

    matched = margin_condition and (
        (
            eye_condition
            and best_match["pca_distance"] <= min(distance_threshold, profile_distance_limit)
            and best_match["correlation"] >= max(correlation_threshold, MIN_CORRELATION_WITH_EYE, profile_correlation_floor)
        )
        or (
            (not profile_requires_eye)
            and
            (not eye_available)
            and best_match["pca_distance"] <= min(distance_threshold, profile_distance_limit)
            and best_match["correlation"] >= max(correlation_threshold + 0.02, MIN_CORRELATION_WITHOUT_EYE, profile_correlation_floor)
        )
    )

    if matched:
        return {
            "matched": True,
            "status": "granted",
            "message": "Face Detected - Access Granted",
            "user": {
                "id": best_match["user_id"],
                "name": best_match["name"],
                "person_identifier": best_match["person_identifier"],
                "email": best_match["email"],
            },
            "pca_distance": round(best_match["pca_distance"], 4),
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
        "pca_distance": round(best_match["pca_distance"], 4),
        "eye_difference": round(best_match["eye_difference"], 4) if best_match["eye_difference"] is not None else None,
        "correlation": round(best_match["correlation"], 4),
        "bounding_box": {
            "x": bounding_box[0],
            "y": bounding_box[1],
            "w": bounding_box[2],
            "h": bounding_box[3],
        },
    }
