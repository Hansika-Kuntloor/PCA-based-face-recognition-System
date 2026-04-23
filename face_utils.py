import base64
import os
import pickle
import uuid

import cv2
import numpy as np
from sklearn.decomposition import PCA

from db import get_all_users, update_user_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "captures")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pca_model.pkl")
FACE_SIZE = (120, 120)
DEFAULT_THRESHOLD = 5200.0



def get_face_cascade():
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))


def get_eye_cascade():
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"))


def decode_base64_image(image_data):
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    binary = base64.b64decode(image_data)
    image_array = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode the submitted image.")
    return frame


def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast_boosted = clahe.apply(gray)
    equalized = cv2.equalizeHist(contrast_boosted)
    normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(normalized, FACE_SIZE)


def detect_largest_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face = frame_bgr[y:y + h, x:x + w]
    return face, (int(x), int(y), int(w), int(h))


def calculate_interocular_distance(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eyes = get_eye_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
    if len(eyes) < 2:
        return 0.0

    eyes = sorted(eyes, key=lambda item: item[0])[:2]
    centers = []
    for ex, ey, ew, eh in eyes:
        centers.append((ex + ew / 2.0, ey + eh / 2.0))

    left_eye, right_eye = centers[0], centers[1]
    return float(np.linalg.norm(np.array(left_eye) - np.array(right_eye)))


def flatten_face(face_gray):
    return face_gray.astype(np.float32).flatten()


def save_registration_samples(name, images):
    if not name.strip():
        raise ValueError("Name is required.")
    if len(images) < 3:
        raise ValueError("Capture at least 3 face samples for reliable enrollment.")

    safe_name = "_".join(name.strip().split()).lower()
    user_dir = os.path.join(DATASET_DIR, safe_name)
    os.makedirs(user_dir, exist_ok=True)

    stored_paths = []
    processed_faces = []
    eye_distances = []

    for image_data in images:
        frame = decode_base64_image(image_data)
        detected = detect_largest_face(frame)
        if not detected:
            continue

        face_bgr, _ = detected
        face_gray = preprocess_face(face_bgr)
        processed_faces.append(face_gray)
        eye_distances.append(calculate_interocular_distance(face_bgr))

        filename = f"{uuid.uuid4().hex}.png"
        output_path = os.path.join(user_dir, filename)
        cv2.imwrite(output_path, face_gray)
        stored_paths.append(output_path)

    if len(processed_faces) < 3:
        raise ValueError("Could not detect clear faces in enough samples. Try better framing and lighting.")

    mean_vector = np.mean([flatten_face(face) for face in processed_faces], axis=0).tolist()
    mean_eye_distance = float(np.mean(eye_distances)) if eye_distances else 0.0

    return {
        "image_paths": stored_paths,
        "mean_eye_distance": mean_eye_distance,
        "features": {
            "face_vector": mean_vector,
            "eye_distance": mean_eye_distance,
        },
        "sample_count": len(processed_faces),
    }


def collect_training_data():
    users = get_all_users()
    face_vectors = []
    labels = []
    eye_distances = []

    for user in users:
        user_vectors = []
        feature_eye_distance = 0.0
        if isinstance(user.get("features"), dict):
            feature_eye_distance = float(user["features"].get("eye_distance", 0.0))

        for image_path in user["image_paths"]:
            if not os.path.exists(image_path):
                continue
            face_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if face_gray is None:
                continue

            vector = flatten_face(face_gray)
            user_vectors.append(vector)
            face_vectors.append(vector)
            labels.append(user["name"])
            eye_distances.append(feature_eye_distance)

        if user_vectors:
            update_user_features(
                user["name"],
                {
                    "face_vector": np.mean(user_vectors, axis=0).tolist(),
                    "eye_distance": feature_eye_distance,
                },
            )

    return face_vectors, labels, eye_distances


def safe_correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    correlation = np.corrcoef(a, b)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def evaluate_training_accuracy(projections, labels, eye_distances, user_profiles):
    correct = 0

    for idx, sample_projection in enumerate(projections):
        best_label = "Unknown"
        best_score = float("inf")

        for label, profile in user_profiles.items():
            profile_projection = np.array(profile["projection"])
            euclidean = float(np.linalg.norm(sample_projection - profile_projection))
            correlation_penalty = 1.0 - safe_correlation(sample_projection, profile_projection)
            eye_penalty = abs(eye_distances[idx] - float(profile["eye_distance"])) / 15.0
            composite = euclidean + (250.0 * correlation_penalty) + (eye_penalty / 3.0)


            if composite < best_score:
                best_score = composite
                best_label = label

        if best_label == labels[idx]:
            correct += 1

    accuracy = (correct / len(labels)) * 100.0 if labels else 0.0
    return {"samples": float(len(labels)), "accuracy": round(accuracy, 2)}


def train_pca_model():
    vectors, labels, eye_distances = collect_training_data()
    if len(vectors) < 3:
        raise ValueError("At least 3 valid face samples are required before training.")

    x = np.array(vectors, dtype=np.float32)
    sample_count, feature_count = x.shape
    n_components = min(sample_count, feature_count, max(2, sample_count - 1))

    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=42)
    projections = pca.fit_transform(x)

    label_profiles = {}
    for idx, label in enumerate(labels):
        label_profiles.setdefault(label, {"projections": [], "eye_distances": []})
        label_profiles[label]["projections"].append(projections[idx])
        label_profiles[label]["eye_distances"].append(eye_distances[idx])

    user_profiles = {}
    for label, values in label_profiles.items():
        user_profiles[label] = {
            "projection": np.mean(np.array(values["projections"]), axis=0),
            "eye_distance": float(np.mean(np.array(values["eye_distances"]))) if values["eye_distances"] else 0.0,
        }

    metrics = evaluate_training_accuracy(projections, labels, eye_distances, user_profiles)

    payload = {
        "pca": pca,
        "user_profiles": user_profiles,
        "threshold": DEFAULT_THRESHOLD,
        "face_size": FACE_SIZE,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(payload, model_file)

    return payload


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


def recognize_face(image_data):
    model = load_model()
    if not model:
        raise ValueError("Model not trained yet. Train the system before recognition.")

    frame = decode_base64_image(image_data)
    detected = detect_largest_face(frame)
    if not detected:
        return {"matched": False, "name": "Unknown", "message": "No face detected."}

    face_bgr, bbox = detected
    face_gray = preprocess_face(face_bgr)
    eye_distance = calculate_interocular_distance(face_bgr)
    vector = flatten_face(face_gray).reshape(1, -1)
    projection = model["pca"].transform(vector)[0]

    best_label = "Unknown"
    best_distance = float("inf")
    best_correlation = 0.0
    best_eye_gap = 0.0

    for label, profile in model["user_profiles"].items():
        profile_projection = np.array(profile["projection"])
        euclidean = float(np.linalg.norm(projection - profile_projection))
        correlation = safe_correlation(projection, profile_projection)
        eye_gap = abs(eye_distance - float(profile["eye_distance"]))
        composite = euclidean + (250.0 * (1.0 - correlation)) + (eye_gap / 40.0)

        if composite < best_distance:
            best_distance = composite
            best_label = label
            best_correlation = correlation
            best_eye_gap = eye_gap

    threshold = float(model.get("threshold", DEFAULT_THRESHOLD))
    matched = best_distance <= threshold


    return {
        "matched": matched,
        "name": best_label if matched else "Unknown",
        "distance": round(best_distance, 2),
        "correlation": round(best_correlation, 4),
        "eye_gap": round(best_eye_gap, 2),
        "bounding_box": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
        "message": "Face recognized successfully." if matched else "No confident match found.",
        "metrics": model.get("metrics", {}),
    }



