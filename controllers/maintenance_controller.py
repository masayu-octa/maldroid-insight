"""Pembersihan data tamu yang sudah lewat masa simpan."""
import time

from config import GUEST_TTL_HOURS
from models import artifacts, db
from models import datasets as M

_last = [0.0]


def purge_guests(every_seconds=600):
    """Hapus dataset/run tamu yang lebih tua dari GUEST_TTL_HOURS (dijalankan paling sering tiap 10 menit)."""
    if time.time() - _last[0] < every_seconds:
        return
    _last[0] = time.time()
    datasets, run_ids = db.expired_guest_items(GUEST_TTL_HOURS)
    for rid in run_ids:
        artifacts.delete_run_files(rid)
    for d in datasets:
        M.delete_frame(d["file"])
    db.delete_raw([d["id"] for d in datasets], run_ids)