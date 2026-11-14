import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / "i18n"
T = {"en": ("Log in", "Log out"), "id": ("Masuk", "Keluar"),
     "ms": ("Log masuk", "Log keluar"), "zh": ("登录", "退出")}
for lang, (a, b) in T.items():
    (root / f"{lang}_auth.json").write_text(
        json.dumps({"auth_login": a, "auth_logout": b}, ensure_ascii=False, indent=1), encoding="utf-8")