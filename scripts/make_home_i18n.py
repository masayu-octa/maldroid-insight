import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / "i18n"
T = {"en": ("Home", "Back to Home"), "id": ("Beranda", "Kembali ke Beranda"),
     "ms": ("Laman Utama", "Kembali ke Laman Utama"), "zh": ("首页", "返回首页")}
for lang, (a, b) in T.items():
    (root / f"{lang}_zhome.json").write_text(
        json.dumps({"nav_home": a, "lock_back": b}, ensure_ascii=False, indent=1), encoding="utf-8")