import json, pathlib
T = {
 "en": dict(ex_view="Section", ex_tab_explain="Explanation and AI chat", ex_tab_compare="Model comparison"),
 "id": dict(ex_view="Bagian", ex_tab_explain="Penjelasan dan Chatbot", ex_tab_compare="Perbandingan model"),
 "ms": dict(ex_view="Bahagian", ex_tab_explain="Penerangan dan Chatbot", ex_tab_compare="Perbandingan model"),
 "zh": dict(ex_view="部分", ex_tab_explain="解释与 AI 聊天", ex_tab_compare="模型比较"),
}
out = pathlib.Path(__file__).resolve().parent.parent / "i18n"
for k, v in T.items():
    (out / f"{k}_extab.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")