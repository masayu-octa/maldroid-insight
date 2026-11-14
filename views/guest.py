"""Identitas pemilik data: user login -> "u:<id>", tamu -> "g:<token di browser>"."""
import re

import streamlit as st
import streamlit.components.v1 as components

from config import ROOT

_token_component = components.declare_component("guest_token", path=str(ROOT / "components" / "guest_token"))
_TOKEN = re.compile(r"^[0-9a-f]{32}$")


def resolve_owner():
    user = st.session_state.get("user")
    if user:
        return f"u:{user['id']}"
    tok = _token_component(key="gt", default=None)
    if not (isinstance(tok, str) and _TOKEN.match(tok)):
        # Token asli dari browser belum diterima: hentikan run ini.
        # Komponen akan membalas dalam beberapa milidetik dan Streamlit otomatis menjalankan ulang.
        st.stop()
    return "g:" + tok