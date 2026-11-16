"""Gerbang pembayaran simulasi: validasi kartu dan hasil transaksi. Tanpa Streamlit.
Hanya nomor kartu uji yang diproses; nomor lain ditolak. Nomor kartu tidak pernah disimpan."""
from datetime import date

# nomor kartu uji -> (merek, hasil)
TEST_CARDS = {
    "4242424242424242": ("Visa", "ok"),
    "5555555555554444": ("Mastercard", "ok"),
    "4000000000000002": ("Visa", "declined"),
    "4000000000009995": ("Visa", "funds"),
}


def digits(text):
    return "".join(ch for ch in (text or "") if ch.isdigit())


def luhn_ok(number):
    n = digits(number)
    if not 13 <= len(n) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(n)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def parse_expiry(text):
    """'MM/YY' atau 'MMYY' -> (bulan, tahun 4 digit) atau None."""
    d = digits(text)
    if len(d) != 4:
        return None
    month, year = int(d[:2]), 2000 + int(d[2:])
    return (month, year) if 1 <= month <= 12 else None


def expiry_valid(text, today=None):
    p = parse_expiry(text)
    if not p:
        return False
    today = today or date.today()
    return (p[1], p[0]) >= (today.year, today.month)


def process(number):
    """(hasil, merek): hasil = ok | declined | funds | unsupported."""
    return TEST_CARDS.get(digits(number), ("Card", "unsupported"))[::-1]


def format_number(number):
    d = digits(number)
    return " ".join(d[i:i + 4] for i in range(0, len(d), 4))