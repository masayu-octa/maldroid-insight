"""Akun pengguna: hash password (PBKDF2-SHA256 + salt acak) dan akses tabel users. Tanpa Streamlit."""
import hashlib
import hmac
import os
from datetime import datetime

from models import db

ITER = 200_000
PLANS = ("free", "pro")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITER)
    return f"pbkdf2${ITER}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, it, salt, dk = stored.split("$")
        new = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(new.hex(), dk)
    except (ValueError, AttributeError):
        return False


def _row(r):
    return None if r is None else {"id": r["id"], "email": r["email"], "name": r["name"], "plan": r["plan"]}


def get_by_email(email: str):
    with db.connect() as con:
        return con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def create_user(email: str, name: str, password: str):
    with db.connect() as con:
        cur = con.execute("INSERT INTO users (email, name, pw_hash, plan, created_at) VALUES (?,?,?,?,?)",
                          (email, name, hash_password(password), "free", datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid


def get(user_id: int):
    with db.connect() as con:
        return _row(con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


def set_plan(user_id: int, plan: str):
    if plan not in PLANS:
        raise ValueError(plan)
    with db.connect() as con:
        con.execute("UPDATE users SET plan = ? WHERE id = ?", (plan, user_id))