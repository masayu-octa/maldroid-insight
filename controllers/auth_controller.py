"""Alur daftar & masuk. Mengembalikan (data, kunci_error) agar view yang menerjemahkan."""
import re
import sqlite3

from models import auth

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PW = 8
MAX_TRIES = 5


def register(email, name, pw, pw2):
    email = (email or "").strip().lower()
    name = (name or "").strip()
    if not EMAIL_RE.match(email):
        return None, "auth_err_email"
    if len(pw or "") < MIN_PW:
        return None, "auth_err_pw_short"
    if pw != pw2:
        return None, "auth_err_pw_match"
    try:
        uid = auth.create_user(email, name or email.split("@")[0], pw)
    except sqlite3.IntegrityError:
        return None, "auth_err_exists"
    return auth.get(uid), None


def login(email, pw):
    row = auth.get_by_email((email or "").strip().lower())
    # verifikasi tetap dijalankan walau email tak ada, agar waktu respons sama
    ok = auth.verify_password(pw or "", row["pw_hash"] if row else "pbkdf2$1$00$00")
    if not row or not ok:
        return None, "auth_err_wrong"
    return auth.get(row["id"]), None


def refresh(user_id):
    return auth.get(user_id)