import json

from flask import Blueprint, flash, redirect, render_template, request, url_for

from auth_utils import admin_required
from db import delete_user_by_id, get_all_users, get_user_by_id, replace_face_samples, update_user_details, create_user
from face_utils import (
    MAX_CAPTURE_SAMPLES,
    MIN_CAPTURE_SAMPLES,
    extract_samples_from_images,
    get_model_summary,
    remove_trained_model,
    train_model_from_database,
)


user_bp = Blueprint("user_admin", __name__, url_prefix="/admin/users")


def parse_samples_from_form() -> list[str]:
    raw_samples = request.form.get("samples_json", "[]").strip()
    if not raw_samples:
        return []
    try:
        parsed = json.loads(raw_samples)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def train_if_possible() -> None:
    users = get_all_users()
    total_samples = sum(user["sample_count"] for user in users)
    if total_samples >= 3:
        train_model_from_database()
    else:
        remove_trained_model()


@user_bp.route("/")
@admin_required
def list_users():
    return render_template("admin_users.html", users=get_all_users(), model_summary=get_model_summary())


@user_bp.route("/new", methods=["GET", "POST"])
@admin_required
def create():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        person_identifier = request.form.get("person_identifier", "").strip()
        email = request.form.get("email", "").strip()
        captured_samples = parse_samples_from_form()

        if not name:
            flash("Name is required.", "error")
            return render_template("admin_user_form.html", mode="create", min_samples=MIN_CAPTURE_SAMPLES, max_samples=MAX_CAPTURE_SAMPLES)
        if len(captured_samples) < MIN_CAPTURE_SAMPLES:
            flash(f"Capture between {MIN_CAPTURE_SAMPLES} and {MAX_CAPTURE_SAMPLES} samples before saving.", "error")
            return render_template("admin_user_form.html", mode="create", min_samples=MIN_CAPTURE_SAMPLES, max_samples=MAX_CAPTURE_SAMPLES)

        try:
            processed_samples = extract_samples_from_images(captured_samples, min_required=MIN_CAPTURE_SAMPLES)
            user_id = create_user(name, person_identifier, email)
            replace_face_samples(user_id, processed_samples)
            train_if_possible()
            flash("User added and model updated successfully.", "success")
            return redirect(url_for("user_admin.list_users"))
        except Exception as exc:
            flash(str(exc), "error")

    return render_template("admin_user_form.html", mode="create", min_samples=MIN_CAPTURE_SAMPLES, max_samples=MAX_CAPTURE_SAMPLES)


@user_bp.route("/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def edit(user_id: int):
    user = get_user_by_id(user_id)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("user_admin.list_users"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        person_identifier = request.form.get("person_identifier", "").strip()
        email = request.form.get("email", "").strip()
        captured_samples = parse_samples_from_form()

        if not name:
            flash("Name is required.", "error")
            return render_template(
                "admin_user_form.html",
                mode="edit",
                user=user,
                min_samples=MIN_CAPTURE_SAMPLES,
                max_samples=MAX_CAPTURE_SAMPLES,
            )

        try:
            update_user_details(user_id, name, person_identifier, email)
            if captured_samples:
                processed_samples = extract_samples_from_images(captured_samples, min_required=MIN_CAPTURE_SAMPLES)
                replace_face_samples(user_id, processed_samples)
            train_if_possible()
            flash("User updated successfully.", "success")
            return redirect(url_for("user_admin.list_users"))
        except Exception as exc:
            flash(str(exc), "error")

    return render_template(
        "admin_user_form.html",
        mode="edit",
        user=user,
        min_samples=MIN_CAPTURE_SAMPLES,
        max_samples=MAX_CAPTURE_SAMPLES,
    )


@user_bp.route("/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete(user_id: int):
    user = get_user_by_id(user_id)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("user_admin.list_users"))

    delete_user_by_id(user_id)
    train_if_possible()
    flash(f"{user['name']} deleted successfully.", "success")
    return redirect(url_for("user_admin.list_users"))
