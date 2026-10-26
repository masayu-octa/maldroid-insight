"""Mengisi kunci terjemahan yang belum ada, dari i18n/en.json, memakai mesin offline (argostranslate).

Pemakaian (dari folder proyek):
    pip install argostranslate
    python scripts/translate_missing.py          # semua bahasa
    python scripts/translate_missing.py ID ZH    # hanya bahasa tertentu

Aturan:
- Hanya kunci yang BELUM ada di file bahasa yang diterjemahkan; terjemahan yang sudah ada tidak diubah.
- Placeholder seperti {n} dan {name} wajib tetap utuh; kalau rusak, kunci itu dilewati (pakai Inggris) dan dilaporkan.
- Hasil mesin tetap perlu diperiksa sekilas, terutama istilah teknis.
"""
import json
import re
import sys
from pathlib import Path

I18N = Path(__file__).resolve().parent.parent / "i18n"
CODES = {"ID": "id", "MS": "ms", "ZH": "zh"}          # kode bahasa Argos
PLACEHOLDER = re.compile(r"\{[a-z_]+(?::[^}]*)?\}")


def ensure_pair(to_code):
    import argostranslate.package as pkg
    import argostranslate.translate as tr
    langs = {l.code: l for l in tr.get_installed_languages()}
    if "en" in langs and to_code in langs and langs["en"].get_translation(langs[to_code]):
        return True
    pkg.update_package_index()
    cand = [p for p in pkg.get_available_packages() if p.from_code == "en" and p.to_code == to_code]
    if not cand:
        return False
    pkg.install_from_path(cand[0].download())
    return True


def main(targets):
    import argostranslate.translate as tr
    en = json.loads((I18N / "en.json").read_text(encoding="utf-8"))
    for lang in targets:
        code = CODES[lang]
        if not ensure_pair(code):
            print(f"[{lang}] pasangan en->{code} tidak tersedia di Argos; dilewati.")
            continue
        path = I18N / f"{lang.lower()}.json"
        cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        todo = [k for k in en if not cur.get(k)]
        bad = []
        for k in todo:
            out = tr.translate(en[k], "en", code)
            if sorted(PLACEHOLDER.findall(out)) != sorted(PLACEHOLDER.findall(en[k])):
                bad.append(k)
                continue
            cur[k] = out
        path.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{lang}] diterjemahkan {len(todo) - len(bad)}, dilewati (placeholder rusak) {len(bad)}: {bad}")


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or list(CODES))