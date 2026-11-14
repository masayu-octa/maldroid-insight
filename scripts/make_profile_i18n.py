import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / "i18n"
T = {
"en": {"pi_btn": "Plans & pricing", "pi_title": "Plans and limits", "pi_col_feat": "Feature",
       "pi_col_guest": "Guest", "pi_col_user": "Account", "pi_datasets": "Datasets kept",
       "pi_runs": "Runs kept", "pi_mb": "Max file size", "pi_rows": "Rows per run",
       "pi_keep": "Data stored for", "pi_forever": "Permanent", "pi_ai": "Prediction and AI chat",
       "pi_yes": "Yes", "pi_no": "No",
       "pi_pro": "A paid plan with higher limits is planned. For now every registered account is free."},
"id": {"pi_btn": "Info paket", "pi_title": "Paket dan batas pemakaian", "pi_col_feat": "Fitur",
       "pi_col_guest": "Tamu", "pi_col_user": "Akun", "pi_datasets": "Dataset tersimpan",
       "pi_runs": "Run tersimpan", "pi_mb": "Ukuran file maks.", "pi_rows": "Baris per run",
       "pi_keep": "Data disimpan selama", "pi_forever": "Permanen", "pi_ai": "Prediksi dan chat AI",
       "pi_yes": "Ya", "pi_no": "Tidak",
       "pi_pro": "Paket berbayar dengan batas lebih tinggi sedang direncanakan. Untuk saat ini semua akun terdaftar gratis."},
"ms": {"pi_btn": "Maklumat pelan", "pi_title": "Pelan dan had penggunaan", "pi_col_feat": "Ciri",
       "pi_col_guest": "Tetamu", "pi_col_user": "Akaun", "pi_datasets": "Set data disimpan",
       "pi_runs": "Larian disimpan", "pi_mb": "Saiz fail maks.", "pi_rows": "Baris setiap larian",
       "pi_keep": "Data disimpan selama", "pi_forever": "Kekal", "pi_ai": "Ramalan dan sembang AI",
       "pi_yes": "Ya", "pi_no": "Tidak",
       "pi_pro": "Pelan berbayar dengan had lebih tinggi sedang dirancang. Buat masa ini semua akaun berdaftar adalah percuma."},
"zh": {"pi_btn": "套餐信息", "pi_title": "套餐与使用限制", "pi_col_feat": "项目",
       "pi_col_guest": "访客", "pi_col_user": "账户", "pi_datasets": "可保存的数据集",
       "pi_runs": "可保存的运行", "pi_mb": "文件大小上限", "pi_rows": "每次运行的行数",
       "pi_keep": "数据保存时间", "pi_forever": "永久", "pi_ai": "预测与 AI 聊天",
       "pi_yes": "是", "pi_no": "否",
       "pi_pro": "我们正在规划限制更高的付费套餐。目前所有注册账户均免费。"},
}
for lang, d in T.items():
    (root / f"{lang}_zzprofile.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")