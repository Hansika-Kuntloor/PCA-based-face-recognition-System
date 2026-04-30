from collections import Counter
import math

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session, url_for

from db import create_auth_log
from face_utils import get_model_summary, recognize_face


recognition_bp = Blueprint("recognition", __name__)

AUTH_SAMPLE_LIMIT = 10
MIN_AUTH_SAMPLE_MATCHES = 4
AUTH_MATCH_RATIO = 0.6


def _best_metric_result(results):
    face_results = [result for result in results if result.get("status") != "no_face"]
    if not face_results:
        return None
    return min(face_results, key=lambda result: result.get("pca_distance", float("inf")))


def _required_sample_matches(total_samples: int) -> int:
    limited_samples = max(1, min(total_samples, AUTH_SAMPLE_LIMIT))
    return max(MIN_AUTH_SAMPLE_MATCHES, math.ceil(limited_samples * AUTH_MATCH_RATIO))


def recognize_from_samples(images):
    sample_results = [recognize_face(image) for image in images[:AUTH_SAMPLE_LIMIT]]
    face_results = [result for result in sample_results if result.get("status") != "no_face"]
    granted_results = [result for result in face_results if result.get("matched") and result.get("user")]

    if not face_results:
        return {
            "matched": False,
            "status": "no_face",
            "message": "No Face Detected in captured samples",
            "bounding_box": None,
            "sample_count": len(sample_results),
            "valid_samples": 0,
            "matched_samples": 0,
        }

    user_counts = Counter(result["user"]["id"] for result in granted_results)
    if user_counts:
        best_user_id, matched_samples = user_counts.most_common(1)[0]
        if matched_samples >= _required_sample_matches(len(sample_results)):
            best_user_results = [result for result in granted_results if result["user"]["id"] == best_user_id]
            best_result = min(best_user_results, key=lambda result: result.get("pca_distance", float("inf")))
            return {
                **best_result,
                "message": "Face Samples Matched - Access Granted",
                "sample_count": len(sample_results),
                "valid_samples": len(face_results),
                "matched_samples": matched_samples,
            }

    best_result = _best_metric_result(face_results) or face_results[0]
    return {
        **best_result,
        "matched": False,
        "status": "denied",
        "message": "Face Samples Captured - Access Denied",
        "user": None,
        "sample_count": len(sample_results),
        "valid_samples": len(face_results),
        "matched_samples": len(granted_results),
    }


@recognition_bp.route("/authenticate")
def authenticate():
    if not session.get("admin_id"):
        flash("Admin login is required to scan and detect faces.", "error")
        return redirect(url_for("admin.login"))
    return render_template("authenticate.html", model_summary=get_model_summary(), scan_context="admin")


@recognition_bp.route("/recognize", methods=["POST"])
def recognize():
    if not session.get("admin_id"):
        return jsonify({"success": False, "message": "Admin login is required to scan and detect faces."}), 401

    payload = request.get_json(silent=True) or {}
    images = payload.get("images")
    image = payload.get("image")
    if not images and not image:
        return jsonify({"success": False, "message": "At least one face sample is required."}), 400
    if images and not isinstance(images, list):
        return jsonify({"success": False, "message": "Face samples must be submitted as a list."}), 400

    try:
        result = recognize_from_samples(images) if images else recognize_face(image)
        if result.get("status") == "granted":
            result = {**result, "message": "Face Matched - User Data Found"}
        elif result.get("status") == "denied":
            result = {**result, "message": "Face Scanned - No Registered User Found"}

        if result["status"] == "granted":
            create_auth_log(
                status="GRANTED",
                message=result["message"],
                matched_user_id=result["user"].get("id") if result.get("user") else None,
                matched_name=result["user"].get("name") if result.get("user") else None,
                pca_distance=result.get("pca_distance"),
                eye_difference=result.get("eye_difference"),
                correlation=result.get("correlation"),
            )
        elif result["status"] == "denied":
            create_auth_log(
                status="DENIED",
                message=result["message"],
                matched_name="Unknown User",
                pca_distance=result.get("pca_distance"),
                eye_difference=result.get("eye_difference"),
                correlation=result.get("correlation"),
            )
        else:
            create_auth_log(status="NO_FACE", message=result["message"])

        return jsonify({"success": True, **result})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
