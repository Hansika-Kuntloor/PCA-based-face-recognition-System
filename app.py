import os
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from db import count_auth_users, create_auth_user, get_all_users, get_auth_user_by_username, init_db, upsert_user
from face_utils import load_model, recognize_face, save_registration_samples, train_pca_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "pca-face-recognition-secret")


app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


@app.before_request
def startup() -> None:
    init_db()


@app.route("/")
def root():
    if session.get("logged_in"):
        return redirect(url_for("index"))
    return redirect(url_for("login"))


def render_login_page(login_error=None, success_message=None):
    return render_template(
        "login.html",
        login_error=login_error,
        success_message=success_message,
        has_auth_users=count_auth_users() > 0,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        auth_user = get_auth_user_by_username(username)
        if auth_user and check_password_hash(auth_user["password_hash"], password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))

        return render_login_page(login_error="Invalid username or password.")

    success_message = None
    if request.args.get("registered") == "1":
        success_message = "Account created successfully. Please log in."
    return render_login_page(success_message=success_message)


def render_register_page(signup_error=None):
    return render_template(
        "register.html",
        signup_error=signup_error,
        has_auth_users=count_auth_users() > 0,
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_register_page()

    username = request.form.get("register_username", "").strip()
    password = request.form.get("register_password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()

    if len(username) < 3:
        return render_register_page(signup_error="Username must be at least 3 characters long.")
    if len(password) < 6:
        return render_register_page(signup_error="Password must be at least 6 characters long.")
    if password != confirm_password:
        return render_register_page(signup_error="Passwords do not match.")
    if get_auth_user_by_username(username):
        return render_register_page(signup_error="That username is already registered.")

    create_auth_user(username, generate_password_hash(password))
    return redirect(url_for("login", registered="1"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def index():
    model = load_model()
    metrics = model.get("metrics", {}) if model else {}
    return render_template("index.html", users=get_all_users(), metrics=metrics, model_available=bool(model))


@app.route("/users-page")
@login_required
def users_page():
    return render_template("users.html", users=get_all_users())


@app.route("/register", methods=["POST"])
@login_required
def register():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name", "").strip()
    images = payload.get("images", [])

    try:
        result = save_registration_samples(name, images)
        user_id = upsert_user(name, result["image_paths"], result["features"])
        return jsonify(
            {
                "success": True,
                "user_id": user_id,
                "name": name,
                "sample_count": result["sample_count"],
                "mean_eye_distance": round(result["mean_eye_distance"], 2),
                "message": "User registered. Run training to update the PCA model.",
            }
        )
    except Exception as exc:
        app.logger.exception("Registration failed")
        return jsonify({"success": False, "message": str(exc)}), 400


@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    try:
        model = train_pca_model()
        return jsonify(
            {
                "success": True,
                "message": "PCA model trained successfully.",
                "metrics": model.get("metrics", {}),
                "registered_users": len(model.get("user_profiles", {})),
            }
        )
    except Exception as exc:
        app.logger.exception("Training failed")
        return jsonify({"success": False, "message": str(exc)}), 400


@app.route("/recognize", methods=["POST"])
@login_required
def recognize():
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")

    if not image:
        return jsonify({"success": False, "message": "Image is required."}), 400

    try:
        result = recognize_face(image)
        return jsonify({"success": True, **result})
    except Exception as exc:
        app.logger.exception("Recognition failed")
        return jsonify({"success": False, "message": str(exc)}), 400


@app.route("/users", methods=["GET"])
@login_required
def users():
    return jsonify({"success": True, "users": get_all_users()})


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
