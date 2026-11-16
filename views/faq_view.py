"""Halaman FAQ: pertanyaan yang sering diajukan."""
import streamlit as st

from i18n import t
from views.components import hero

FAQ = ["install", "format", "toogood", "which", "av", "store"]


def render_faq():
    hero(t("nav_faq"), t("faq_sub"))
    for q in FAQ:
        with st.container(border=True):
            st.markdown(f"**{t('lp_faq_' + q + '_q')}**")
            st.markdown(t("lp_faq_" + q + "_a"))