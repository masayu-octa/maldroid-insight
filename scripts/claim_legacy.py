"""Pindahkan data lama (sebelum sistem pemilik ada) ke sebuah akun.
Pakai: python scripts\\claim_legacy.py emailmu@contoh.com"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import auth, db  # noqa: E402

if len(sys.argv) != 2:
    sys.exit("Pakai: python scripts\\claim_legacy.py emailmu@contoh.com")
db.init_db()
row = auth.get_by_email(sys.argv[1].strip().lower())
if not row:
    sys.exit("Akun dengan email itu belum terdaftar. Daftar dulu di aplikasi.")
d, r = db.claim("legacy", f"u:{row['id']}")
print(f"Dipindahkan ke akun {row['email']}: {d} dataset, {r} run.")