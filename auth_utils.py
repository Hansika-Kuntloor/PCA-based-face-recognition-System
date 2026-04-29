from functools import wraps

from flask import flash, redirect, session, url_for


def admin_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("admin_id"):
            flash("Please log in to continue.", "error")
            return redirect(url_for("admin.login"))
        return view_func(*args, **kwargs)

    return wrapped
