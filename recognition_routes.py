from flask import Blueprint, flash, jsonify, redirect, render_template, request, session, url_for

from db import create_auth_log
from face_utils import get_model_summary, recognize_face


recognition_bp = Blueprint("recognition", __name__)


@recognition_bp.route("/authenticate")
def authenticate():
    if not session.get("admin_id") and not session.get("portal_user_id"):
        flash("Please login before using face authentication.", "error")
        return redirect(url_for("user_portal.user_login"))
    return render_template("authenticate.html", model_summary=get_model_summary())


@recognition_bp.route("/recognize", methods=["POST"])
def recognize():
    if not session.get("admin_id") and not session.get("portal_user_id"):
        return jsonify({"success": False, "message": "Please login before using face authentication."}), 401

    payload = request.get_json(silent=True) or {}
    image = payload.get("image")
    if not image:
        return jsonify({"success": False, "message": "Image is required."}), 400

    try:
        result = recognize_face(image)
        portal_user_id = session.get("portal_user_id")
        if (
            portal_user_id
            and not session.get("admin_id")
            and result.get("matched")
            and result.get("user", {}).get("id") != int(portal_user_id)
        ):
            result = {
                **result,
                "matched": False,
                "status": "denied",
                "message": "Face Detected - Access Denied",
                "user": None,
            }

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
