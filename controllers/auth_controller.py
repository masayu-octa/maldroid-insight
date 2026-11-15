"""Alur daftar, masuk, dan pengaturan akun. Mengembalikan (data, kunci_error) agar view yang menerjemahkan."""
import re
import sqlite3

from controllers import maintenance_controller
from models import auth

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PW = 8
MAX_TRIES = 5
MAX_NAME = 60


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


def update_name(user_id, name):
    name = (name or "").strip()
    if not name or len(name) > MAX_NAME:
        return None, "pr_err_name"
    auth.set_name(user_id, name)
    return auth.get(user_id), None


def change_password(user_id, old, new, new2):
    row = auth.get_row(user_id)
    if not row or not auth.verify_password(old or "", row["pw_hash"]):
        return False, "pr_err_old"
    if len(new or "") < MIN_PW:
        return False, "auth_err_pw_short"
    if new != new2:
        return False, "auth_err_pw_match"
    if new == old:
        return False, "pr_err_same"
    auth.set_password(user_id, new)
    return True, None


def change_plan(user_id, plan):
    """Simulasi langganan: hanya mengubah kolom plan (tidak ada pembayaran)."""
    auth.set_plan(user_id, plan)
    return auth.get(user_id)


def delete_account(user_id, pw):
    """Hapus data milik akun (dataset, run, file) lalu akunnya. Pemilik data saat ini harus akun ini."""
    row = auth.get_row(user_id)
    if not row or not auth.verify_password(pw or "", row["pw_hash"]):
        return False, "pr_err_old"
    maintenance_controller.delete_my_data()
    auth.delete_user(user_id)
    return True, None