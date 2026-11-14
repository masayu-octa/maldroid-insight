"""Halaman 'Perbandingan model' (anak menu Explainability)."""
import streamlit as st

from controllers import compare_controller as cc
from controllers import pipeline_controller as pc
from i18n import t
from views.components import apply_pending, hero, page_link
from views.models_view import _compare, _label, _single


def render_compare():
    apply_pending()
    hero(t("ex_tab_compare"), t("ex_cmp_help"))
    runs = cc.all_runs()
    if not runs:
        st.info(t("ex_none"))
        page_link("pipeline", t("nav_pipeline"), ":material/tune:")
        return
    names = cc.dataset_names()
    ids = [r["id"] for r in runs]
    by_id = {r["id"]: r for r in runs}
    if st.session_state.get("cmp_run") not in ids:
        st.session_state["cmp_run"] = ids[0]
    rid = st.selectbox(t("pl_view_run"), ids, key="cmp_run", format_func=lambda i: _label(by_id[i], names))
    st.session_state["mc_run"] = rid
    tab1, tab2 = st.tabs([t("mc_tab_run"), t("mc_tab_cmp")])
    with tab1:
        _single(pc.load_run(rid))
    with tab2:
        _compare(ids, by_id, names)