"""Halaman yang belum dibangun di struktur baru."""
import streamlit as st

from i18n import t
from views.components import hero


def make(key, stage):
    def _render():
        hero(t("nav_" + key), t("soon_sub", stage=t("stage", n=stage)))
        st.info(t("soon_info"))
    _render.__name__ = f"render_{key}"
    return _render