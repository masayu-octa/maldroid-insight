"""Konfigurasi global: path, bahasa, palet warna."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                    # CSV contoh (CICMalDroid2020, KronoDroid)
STORAGE = ROOT / "storage"                  # dibuat otomatis
DATASET_DIR = STORAGE / "datasets"
DB_PATH = STORAGE / "maldroid.db"
I18N_DIR = ROOT / "i18n"
CSS_FILE = ROOT / "views" / "theme.css"

SEED = 42
DEFAULT_LANG = "EN"
LANGS = {"EN": "English", "ID": "Bahasa Indonesia", "MS": "Bahasa Melayu", "ZH": "中文 (简体)"}

CLASS_COLORS = {
    "Benign": "#3A6EA5", "Malware": "#924A2F",
    "Adware": "#D2A35C", "Banking": "#B5603F", "SMS": "#233A6E", "Riskware": "#7A8DB3",
}
EXTRA_COLORS = ["#5B94BD", "#C9983A", "#6B4E71", "#3C7A5A", "#8C6D46", "#4C6A92"]
BLUE_SCALE = ["#DCE7F1", "#8FB4D1", "#3A6EA5", "#141F52"]