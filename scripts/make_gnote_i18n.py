import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / "i18n"
T = {
"en": {"gn_title": "Guest mode", "gn_hide": "Hide", "gn_show": "Guest info"},
"id": {"gn_title": "Mode tamu", "gn_hide": "Sembunyikan", "gn_show": "Info tamu"},
"ms": {"gn_title": "Mod tetamu", "gn_hide": "Sembunyikan", "gn_show": "Maklumat tetamu"},
"zh": {"gn_title": "访客模式", "gn_hide": "隐藏", "gn_show": "访客说明"},
}
for lang, d in T.items():
    (root / f"{lang}_zzgnote.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")