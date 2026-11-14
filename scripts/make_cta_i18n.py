import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / "i18n"
T = {
"en": {"ds_next_btn": "Next: run the pipeline", "pl_open_explain": "See why: explain this run",
       "pl_open_explain_lock": "Explain this run (login needed)"},
"id": {"ds_next_btn": "Lanjut: jalankan pipeline", "pl_open_explain": "Lihat penjelasan run ini",
       "pl_open_explain_lock": "Jelaskan run ini (perlu login)"},
"ms": {"ds_next_btn": "Seterusnya: jalankan pipeline", "pl_open_explain": "Lihat penjelasan larian ini",
       "pl_open_explain_lock": "Terangkan larian ini (perlu log masuk)"},
"zh": {"ds_next_btn": "下一步：运行流程", "pl_open_explain": "查看此次运行的解释",
       "pl_open_explain_lock": "解释此次运行（需登录）"},
}
for lang, d in T.items():
    (root / f"{lang}_zzcta.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")