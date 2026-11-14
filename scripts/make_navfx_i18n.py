import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / "i18n"
T = {"en": ("Features", "Create free account", "Try Explainability"),
     "id": ("Fitur", "Buat akun gratis", "Coba Explainability"),
     "ms": ("Ciri", "Cipta akaun percuma", "Cuba Explainability"),
     "zh": ("功能", "创建免费账户", "体验可解释性")}
for lang, (a, b, c) in T.items():
    (root / f"{lang}_navfx.json").write_text(
        json.dumps({"nav_features": a, "lp_cta_signup": b, "lp_cta_explore": c}, ensure_ascii=False, indent=1),
        encoding="utf-8")