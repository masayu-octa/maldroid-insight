"""Alur pembayaran paket Pro (simulasi) dan pengelolaan langganan. Mengembalikan (data, kunci_error)."""
from datetime import datetime, timedelta

from config import PERIOD_DAYS, PRO_PRICES
from models import auth, db, payments


def price(period):
    return PRO_PRICES[period]


def yearly_saving():
    return round(PRO_PRICES["monthly"] * 12 - PRO_PRICES["yearly"], 2)


def money(value):
    return f"${value:,.2f}"


def validate(name, number, expiry, cvc):
    if len((name or "").strip()) < 2:
        return "co_err_name"
    if not payments.luhn_ok(number):
        return "co_err_card"
    if not payments.expiry_valid(expiry):
        return "co_err_exp"
    if not 3 <= len(payments.digits(cvc)) <= 4:
        return "co_err_cvc"
    return None


def pay(user_id, period, name, number, expiry, cvc):
    """Proses pembayaran. Berhasil: paket menjadi Pro dan transaksi tercatat."""
    err = validate(name, number, expiry, cvc)
    if err:
        return None, err
    outcome, brand = payments.process(number)
    if outcome == "unsupported":
        return None, "co_err_testonly"
    last4 = payments.digits(number)[-4:]
    amount = price(period)
    if outcome != "ok":
        db.add_payment(user_id, "pro", period, amount, brand, last4, "failed", outcome)
        return None, "co_err_declined" if outcome == "declined" else "co_err_funds"
    pid = db.add_payment(user_id, "pro", period, amount, brand, last4, "paid")
    auth.set_plan(user_id, "pro")
    return next(p for p in db.list_payments(user_id) if p["id"] == pid), None


def cancel(user_id):
    """Batalkan langganan: kembali ke Free. Data dan riwayat tagihan tetap ada."""
    auth.set_plan(user_id, "free")
    return auth.get(user_id)


def history(user_id):
    return db.list_payments(user_id)


def renewal(user_id):
    """(tanggal perpanjangan, pembayaran terakhir yang berhasil) atau (None, None)."""
    last = next((p for p in db.list_payments(user_id) if p["status"] == "paid"), None)
    if not last:
        return None, None
    start = datetime.strptime(last["created_at"], "%Y-%m-%d %H:%M")
    return (start + timedelta(days=PERIOD_DAYS[last["period"]])).strftime("%Y-%m-%d"), last


def receipt_text(p, t):
    """Isi struk dalam teks biasa untuk diunduh."""
    lines = [t("co_rc_title"), "-" * 32,
             f'{t("co_rc_no")}: {p["receipt"]}', f'{t("co_rc_date")}: {p["created_at"]}',
             f'{t("co_rc_plan")}: {t("auth_plan_pro")}', f'{t("co_rc_billing")}: {t("pp_" + p["period"])}',
             f'{t("co_rc_amount")}: {money(p["amount"])}', f'{t("co_rc_card")}: {p["brand"]} **** {p["last4"]}',
             f'{t("co_rc_status")}: {t("pp_status_" + p["status"])}']
    return "\n".join(lines) + "\n"