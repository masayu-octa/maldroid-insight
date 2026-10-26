"""Terjemahan dari file JSON. Menambah bahasa = tambah file <kode>.json.
File per fitur memakai awalan yang sama, misalnya en.json (umum) dan en_pipeline.json (Pipeline Runner);
semuanya digabung per bahasa."""
import json

import streamlit as st

from config import DEFAULT_LANG, I18N_DIR, LANGS


def _files(code):
    c = code.lower()
    return [p for p in [I18N_DIR / f"{c}.json", *sorted(I18N_DIR.glob(f"{c}_*.json"))] if p.exists()]


def _stamp():
    return tuple((p.name, p.stat().st_mtime_ns) for c in LANGS for p in _files(c))


@st.cache_data(show_spinner=False)
def _load(stamp):
    out = {}
    for code in LANGS:
        merged = {}
        for p in _files(code):
            merged.update(json.loads(p.read_text(encoding="utf-8")))
        out[code] = merged
    return out


def t(key, **kw):
    lang = st.session_state.get("lang", DEFAULT_LANG)
    data = _load(_stamp())
    s = data.get(lang, {}).get(key) or data.get("EN", {}).get(key) or key
    return s.format(**kw) if kw else s