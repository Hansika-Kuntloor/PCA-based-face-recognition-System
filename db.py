import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "face_recognition.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(column["name"] == column_name for column in columns)


def ensure_column(conn: sqlite3.Connection, table_name: str, column_definition: str) -> None:
    column_name = column_definition.split()[0]
    if not column_exists(conn, table_name, column_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                person_identifier TEXT,
                email TEXT,
                average_eye_distance REAL NOT NULL DEFAULT 0,
                image_path TEXT,
                features TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        ensure_column(conn, "users", "person_identifier TEXT")
        ensure_column(conn, "users", "email TEXT")
        ensure_column(conn, "users", "average_eye_distance REAL NOT NULL DEFAULT 0")
        # SQLite cannot add a column with CURRENT_TIMESTAMP via ALTER TABLE on an existing table,
        # so legacy databases receive nullable text columns first and are backfilled below.
        ensure_column(conn, "users", "created_at TEXT")
        ensure_column(conn, "users", "updated_at TEXT")

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_person_identifier
            ON users(person_identifier)
            WHERE person_identifier IS NOT NULL AND person_identifier != ''
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                sample_index INTEGER NOT NULL,
                feature_blob BLOB NOT NULL,
                eye_distance REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                matched_user_id INTEGER,
                matched_name TEXT,
                status TEXT NOT NULL,
                pca_distance REAL,
                eye_difference REAL,
                correlation REAL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (matched_user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """
        )

        if table_exists(conn, "auth_users") and count_admins(conn) == 0:
            legacy_admins = conn.execute("SELECT username, password_hash FROM auth_users").fetchall()
            for admin in legacy_admins:
                conn.execute(
                    "INSERT OR IGNORE INTO admins (username, password_hash) VALUES (?, ?)",
                    (admin["username"], admin["password_hash"]),
                )

        conn.execute(
            """
            UPDATE users
            SET updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP),
                created_at = COALESCE(created_at, CURRENT_TIMESTAMP)
            """
        )

    conn.close()


def count_admins(conn: Optional[sqlite3.Connection] = None) -> int:
    should_close = conn is None
    conn = conn or get_connection()
    count = conn.execute("SELECT COUNT(*) AS count FROM admins").fetchone()["count"]
    if should_close:
        conn.close()
    return int(count)


def create_admin(username: str, password_hash: str) -> int:
    conn = get_connection()
    with conn:
        cursor = conn.execute(
            "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        admin_id = cursor.lastrowid
    conn.close()
    return int(admin_id)


def get_admin_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, password_hash, created_at FROM admins WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def get_admin_by_id(admin_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, created_at FROM admins WHERE id = ?",
        (admin_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def _user_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    legacy_image_paths = json.loads(row["image_path"]) if row["image_path"] else []
    stored_features = json.loads(row["features"]) if row["features"] else {}
    return {
        "id": row["id"],
        "name": row["name"],
        "person_identifier": row["person_identifier"] or "",
        "email": row["email"] or "",
        "average_eye_distance": float(row["average_eye_distance"] or 0),
        "sample_count": int(row["sample_count"] or 0),
        "legacy_image_paths": legacy_image_paths,
        "legacy_features": stored_features,
        "feature_summary": stored_features,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_all_users() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            u.*,
            COUNT(fs.id) AS sample_count
        FROM users u
        LEFT JOIN face_samples fs ON fs.user_id = u.id
        GROUP BY u.id
        ORDER BY u.id ASC
        """
    ).fetchall()
    conn.close()
    return [_user_from_row(row) for row in rows]


def count_users() -> int:
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
    conn.close()
    return int(count)


def count_face_samples() -> int:
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) AS count FROM face_samples").fetchone()["count"]
    conn.close()
    return int(count)


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            u.*,
            COUNT(fs.id) AS sample_count
        FROM users u
        LEFT JOIN face_samples fs ON fs.user_id = u.id
        WHERE u.id = ?
        GROUP BY u.id
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return _user_from_row(row)


def get_user_by_name(name: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            u.*,
            COUNT(fs.id) AS sample_count
        FROM users u
        LEFT JOIN face_samples fs ON fs.user_id = u.id
        WHERE lower(u.name) = lower(?)
        GROUP BY u.id
        """,
        (name,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return _user_from_row(row)


def find_user_by_login_details(name: str, person_identifier: str, email: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = None

    if person_identifier:
        row = conn.execute(
            """
            SELECT
                u.*,
                COUNT(fs.id) AS sample_count
            FROM users u
            LEFT JOIN face_samples fs ON fs.user_id = u.id
            WHERE lower(u.person_identifier) = lower(?)
            GROUP BY u.id
            """,
            (person_identifier,),
        ).fetchone()

    if row is None and email:
        row = conn.execute(
            """
            SELECT
                u.*,
                COUNT(fs.id) AS sample_count
            FROM users u
            LEFT JOIN face_samples fs ON fs.user_id = u.id
            WHERE lower(u.email) = lower(?)
            GROUP BY u.id
            """,
            (email,),
        ).fetchone()

    if row is None and name:
        row = conn.execute(
            """
            SELECT
                u.*,
                COUNT(fs.id) AS sample_count
            FROM users u
            LEFT JOIN face_samples fs ON fs.user_id = u.id
            WHERE lower(u.name) = lower(?)
            GROUP BY u.id
            """,
            (name,),
        ).fetchone()

    conn.close()
    if not row:
        return None
    return _user_from_row(row)


def create_user(name: str, person_identifier: str, email: str) -> int:
    conn = get_connection()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO users (name, person_identifier, email, image_path, features, created_at, updated_at)
            VALUES (?, NULLIF(?, ''), NULLIF(?, ''), '[]', '{}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (name, person_identifier, email),
        )
        user_id = cursor.lastrowid
    conn.close()
    return int(user_id)


def update_user_details(user_id: int, name: str, person_identifier: str, email: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            UPDATE users
            SET name = ?,
                person_identifier = NULLIF(?, ''),
                email = NULLIF(?, ''),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (name, person_identifier, email, user_id),
        )
    conn.close()


def update_user_average_eye_distance(user_id: int, average_eye_distance: float) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            UPDATE users
            SET average_eye_distance = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (average_eye_distance, user_id),
        )
    conn.close()


def update_user_feature_summary(user_id: int, feature_summary: Dict[str, Any]) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            UPDATE users
            SET features = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (json.dumps(feature_summary), user_id),
        )
    conn.close()


def clear_legacy_user_data(user_id: int) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            UPDATE users
            SET image_path = '[]',
                features = '{}',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (user_id,),
        )
    conn.close()


def delete_user_by_id(user_id: int) -> bool:
    conn = get_connection()
    with conn:
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        deleted = cursor.rowcount > 0
    conn.close()
    return bool(deleted)


def _valid_eye_distances(samples: List[Dict[str, Any]]) -> List[float]:
    valid_values: List[float] = []
    for sample in samples:
        try:
            value = float(sample["eye_distance"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0.05 <= value <= 0.8:
            valid_values.append(value)
    return valid_values


def replace_face_samples(user_id: int, samples: List[Dict[str, Any]]) -> None:
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM face_samples WHERE user_id = ?", (user_id,))
        for index, sample in enumerate(samples, start=1):
            conn.execute(
                """
                INSERT INTO face_samples (user_id, sample_index, feature_blob, eye_distance)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, index, sample["feature_blob"], sample["eye_distance"]),
            )

        valid_eye_values = _valid_eye_distances(samples)
        average_eye_distance = 0.0
        if valid_eye_values:
            average_eye_distance = sum(valid_eye_values) / len(valid_eye_values)
        conn.execute(
            """
            UPDATE users
            SET average_eye_distance = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (average_eye_distance, user_id),
        )

    conn.close()


def get_face_samples_for_user(user_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, user_id, sample_index, feature_blob, eye_distance, created_at
        FROM face_samples
        WHERE user_id = ?
        ORDER BY sample_index ASC
        """,
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_auth_log(
    status: str,
    message: str,
    matched_user_id: Optional[int] = None,
    matched_name: Optional[str] = None,
    pca_distance: Optional[float] = None,
    eye_difference: Optional[float] = None,
    correlation: Optional[float] = None,
) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO auth_logs (
                matched_user_id,
                matched_name,
                status,
                pca_distance,
                eye_difference,
                correlation,
                message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (matched_user_id, matched_name, status, pca_distance, eye_difference, correlation, message),
        )
    conn.close()


def list_auth_logs(limit: int = 200) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, matched_user_id, matched_name, status, pca_distance, eye_difference, correlation, message, created_at
        FROM auth_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
