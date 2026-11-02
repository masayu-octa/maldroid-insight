"""Controller asisten AI: membaca kunci dari st.secrets, menyimpan hasil (cache), menghitung batas pertanyaan."""
import hashlib

import streamlit as st

from config import DEFAULT_LANG
from models import ai

FREE_QUESTIONS = 5            # batas pertanyaan chatbot per sesi (paket gratis, disimulasikan)
DEFAULT_MODEL = "gemini-3.8-flash"


def config():
    """(kunci, model) dari .streamlit/secrets.toml; kunci None bila belum diisi."""
    try:
        key = st.secrets.get("GEMINI_API_KEY") or None
        model = st.secrets.get("GEMINI_MODEL") or DEFAULT_MODEL
    except Exception:
        key, model = None, DEFAULT_MODEL
    return key, model


def available():
    return config()[0] is not None


def lang():
    return st.session_state.get("lang", DEFAULT_LANG)


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=64)
def _explain_cached(context_hash, context, language, model):
    key, _ = config()
    return ai.explain(key, model, context, language)


def explain(context):
    """Ringkasan otomatis (disimpan 24 jam per isi hasil, bahasa, dan model). Melempar AIError bila gagal."""
    key, model = config()
    if key is None:
        raise ai.AIError("ai_err_nokey")
    h = hashlib.sha1(context.encode("utf-8")).hexdigest()
    return _explain_cached(h, context, lang(), model)


def ask(context, history, question):
    key, model = config()
    if key is None:
        raise ai.AIError("ai_err_nokey")
    return ai.chat(key, model, context, history, question, lang())