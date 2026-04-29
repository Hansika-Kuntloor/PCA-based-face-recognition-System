import os

from flask import Flask, flash, redirect, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from admin_routes import admin_bp
from db import count_admins, get_admin_by_id, get_user_by_id, init_db
from recognition_routes import recognition_bp
from user_routes import public_user_bp, user_bp


SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "secure-face-recognition-secret")


app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 24 * 1024 * 1024


@app.before_request
def startup() -> None:
    init_db()


@app.context_processor
def inject_global_template_data():
    admin = None
    portal_user = None
    if session.get("admin_id"):
        admin = get_admin_by_id(int(session["admin_id"]))
    if session.get("portal_user_id"):
        portal_user = get_user_by_id(int(session["portal_user_id"]))
    return {
        "current_admin": admin,
        "current_portal_user": portal_user,
        "admin_logged_in": admin is not None,
        "user_logged_in": portal_user is not None,
        "admin_exists": count_admins() > 0,
    }


@app.route("/")
def root():
    if count_admins() == 0:
        return redirect(url_for("admin.setup"))
    if session.get("admin_id"):
        return redirect(url_for("admin.dashboard"))
    return redirect(url_for("user_portal.user_login"))


@app.route("/login")
def legacy_login():
    return redirect(url_for("admin.login"))


@app.route("/dashboard")
def legacy_dashboard():
    if session.get("admin_id"):
        return redirect(url_for("admin.dashboard"))
    return redirect(url_for("recognition.authenticate"))


@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(_error):
    if request.path.startswith("/admin/users") or request.path.startswith("/user/login"):
        flash(
            "Captured samples were too large to upload. The app now compresses them, but please try again with a fresh capture.",
            "error",
        )
        fallback = "user_portal.user_login" if request.path.startswith("/user/login") else "user_admin.list_users"
        return redirect(request.referrer or url_for(fallback))
    return "Request entity too large.", 413


app.register_blueprint(admin_bp)
app.register_blueprint(user_bp)
app.register_blueprint(public_user_bp)
app.register_blueprint(recognition_bp)


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
