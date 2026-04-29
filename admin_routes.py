from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from auth_utils import admin_required
from db import count_admins, create_admin, create_auth_log, get_admin_by_username, get_all_users, list_auth_logs
from face_utils import get_model_summary, train_model_from_database


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/")
def admin_root():
    if count_admins() == 0:
        return redirect(url_for("admin.setup"))
    if request.method == "GET" and session.get("admin_id"):
        return redirect(url_for("admin.dashboard"))
    return redirect(url_for("admin.login"))


@admin_bp.route("/setup", methods=["GET", "POST"])
def setup():
    if count_admins() > 0:
        return redirect(url_for("admin.login"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if len(username) < 3:
            flash("Username must be at least 3 characters long.", "error")
            return render_template("admin_setup.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("admin_setup.html")
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("admin_setup.html")

        create_admin(username, generate_password_hash(password))
        flash("Admin account created successfully. Please log in.", "success")
        return redirect(url_for("admin.login"))

    return render_template("admin_setup.html")


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if count_admins() == 0:
        return redirect(url_for("admin.setup"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        admin = get_admin_by_username(username)

        if admin and check_password_hash(admin["password_hash"], password):
            session.pop("portal_user_id", None)
            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]
            create_auth_log(
                status="ADMIN_LOGIN_SUCCESS",
                message="Admin login successful.",
                matched_name=admin["username"],
            )
            flash("Logged in successfully.", "success")
            return redirect(url_for("admin.dashboard"))

        if username:
            create_auth_log(
                status="ADMIN_LOGIN_FAILED",
                message="Invalid admin username or password.",
                matched_name=username,
            )
        flash("Invalid admin username or password.", "error")

    return render_template("admin_login.html")


@admin_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("admin.login"))


@admin_bp.route("/dashboard")
@admin_required
def dashboard():
    model_summary = get_model_summary()
    users = get_all_users()
    logs = list_auth_logs(limit=10)
    return render_template(
        "admin_dashboard.html",
        users=users,
        logs=logs,
        model_summary=model_summary,
    )


@admin_bp.route("/scan")
@admin_required
def scan():
    return render_template("authenticate.html", model_summary=get_model_summary(), scan_context="admin")


@admin_bp.route("/train", methods=["POST"])
@admin_required
def train():
    try:
        model = train_model_from_database()
        metrics = model.get("metrics", {})
        flash(
            (
                f"Training completed. Accuracy: {metrics.get('accuracy', 0)}%, "
                f"FAR: {metrics.get('false_accept_rate', 0)}%, "
                f"FRR: {metrics.get('false_reject_rate', 0)}%."
            ),
            "success",
        )
    except Exception as exc:
        flash(str(exc), "error")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/logs")
@admin_required
def logs():
    return render_template("admin_logs.html", logs=list_auth_logs(limit=200))
