import json

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from auth_utils import admin_required
from db import (
    create_auth_log,
    create_user,
    delete_user_by_id,
    find_user_by_login_details,
    get_all_users,
    get_user_by_id,
    replace_face_samples,
    update_user_details,
)
from face_utils import (
    MAX_CAPTURE_SAMPLES,
    MIN_CAPTURE_SAMPLES,
    extract_samples_from_images,
    get_model_summary,
    remove_trained_model,
    train_model_from_database,
)


user_bp = Blueprint("user_admin", __name__, url_prefix="/admin/users")
public_user_bp = Blueprint("user_portal", __name__, url_prefix="/user")


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


def render_user_login(form_data: dict | None = None):
    return render_template(
        "user_login.html",
        form_data=form_data or {},
    )


@public_user_bp.route("/login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        person_identifier = request.form.get("person_identifier", "").strip()
        email = request.form.get("email", "").strip()
        form_data = {
            "name": name,
            "person_identifier": person_identifier,
            "email": email,
        }

        if not name:
            flash("Name is required.", "error")
            return render_user_login(form_data)
        if not person_identifier and not email:
            flash("Enter a User ID or email so the admin can identify this registration.", "error")
            return render_user_login(form_data)

        try:
            user = find_user_by_login_details(name, person_identifier, email)
            if user:
                user_id = int(user["id"])
                update_user_details(user_id, name, person_identifier, email)
                message = "User details updated successfully."
            else:
                user_id = create_user(name, person_identifier, email)
                message = "User registered successfully."

            session.pop("admin_id", None)
            session.pop("admin_username", None)
            session["portal_user_id"] = user_id
            create_auth_log(
                status="USER_LOGIN",
                message=message,
                matched_user_id=user_id,
                matched_name=name,
            )
            flash(message, "success")
            return redirect(url_for("user_portal.profile"))
        except Exception as exc:
            flash(str(exc), "error")
            return render_user_login(form_data)

    return render_user_login()


@public_user_bp.route("/profile")
def profile():
    user_id = session.get("portal_user_id")
    if not user_id:
        flash("Please register first.", "error")
        return redirect(url_for("user_portal.user_login"))

    user = get_user_by_id(int(user_id))
    if not user:
        session.pop("portal_user_id", None)
        flash("User record was not found. Please register again.", "error")
        return redirect(url_for("user_portal.user_login"))

    return render_template("user_profile.html", user=user, model_summary=get_model_summary())


@public_user_bp.route("/logout")
def user_logout():
    session.pop("portal_user_id", None)
    flash("User logged out successfully.", "success")
    return redirect(url_for("user_portal.user_login"))


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
