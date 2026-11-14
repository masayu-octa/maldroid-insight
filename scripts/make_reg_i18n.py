import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / "i18n"
T = {
"en": {"auth_reg_done": "Account created. Please log in with your email and password."},
"id": {"auth_reg_done": "Akun berhasil dibuat. Silakan login dengan email dan password kamu."},
"ms": {"auth_reg_done": "Akaun berjaya dicipta. Sila log masuk dengan e-mel dan kata laluan anda."},
"zh": {"auth_reg_done": "账户已创建。请使用您的邮箱和密码登录。"},
}
for lang, d in T.items():
    (root / f"{lang}_zzreg.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")