import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "face_recognition.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                image_path TEXT NOT NULL,
                features TEXT
            )
            """
        )
    conn.close()


def count_auth_users() -> int:
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) AS count FROM auth_users").fetchone()["count"]
    conn.close()
    return int(count)


def create_auth_user(username: str, password_hash: str) -> int:
    conn = get_connection()
    with conn:
        cursor = conn.execute(
            "INSERT INTO auth_users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        user_id = cursor.lastrowid
    conn.close()
    return int(user_id)


def get_auth_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, password_hash FROM auth_users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
    }


def upsert_user(name: str, image_paths: List[str], features: Optional[Any]) -> int:
    conn = get_connection()
    features_json = json.dumps(features) if features is not None else None
    image_paths_json = json.dumps(image_paths)

    with conn:
        existing = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE users SET image_path = ?, features = ? WHERE id = ?",
                (image_paths_json, features_json, existing["id"]),
            )
            user_id = existing["id"]
        else:
            cursor = conn.execute(
                "INSERT INTO users (name, image_path, features) VALUES (?, ?, ?)",
                (name, image_paths_json, features_json),
            )
            user_id = cursor.lastrowid

    conn.close()
    return user_id


def update_user_features(name: str, features: Optional[Any]) -> None:
    conn = get_connection()
    features_json = json.dumps(features) if features is not None else None
    with conn:
        conn.execute("UPDATE users SET features = ? WHERE name = ?", (features_json, name))
    conn.close()


def get_all_users() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT id, name, image_path, features FROM users ORDER BY id ASC").fetchall()
    conn.close()

    users: List[Dict[str, Any]] = []
    for row in rows:
        users.append(
            {
                "id": row["id"],
                "name": row["name"],
                "image_paths": json.loads(row["image_path"]) if row["image_path"] else [],
                "features": json.loads(row["features"]) if row["features"] else [],
            }
        )
    return users


def get_user_by_name(name: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, name, image_path, features FROM users WHERE name = ?",
        (name,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "name": row["name"],
        "image_paths": json.loads(row["image_path"]) if row["image_path"] else [],
        "features": json.loads(row["features"]) if row["features"] else [],
    }
