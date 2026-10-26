"""Halaman SUS: kuesioner kegunaan sistem (10 butir) dan rekap skor responden."""
import plotly.graph_objects as go
import streamlit as st

from controllers import sus_controller as ctl
from i18n import t
from models import sus as S
from views.components import apply_pending, chip, flash, hero, show, style


def render_sus():
    apply_pending()
    hero(t("nav_sus"), t("sus_sub"))
    flash()
    st.info(t("sus_intro"))

    with st.form("sus_form", clear_on_submit=False):
        st.text_input(t("sus_name"), key="sus_name")
        st.caption(t("sus_scale"))
        for i in range(1, S.N_ITEMS + 1):
            st.radio(f"{i}. {t(f'sus_q{i}')}", [1, 2, 3, 4, 5], index=None, horizontal=True, key=f"sus_q{i}")
        sent = st.form_submit_button(t("sus_submit"), type="primary")
    if sent:
        answers = [st.session_state.get(f"sus_q{i}") for i in range(1, S.N_ITEMS + 1)]
        if any(a is None for a in answers):
            st.warning(t("sus_incomplete"))
        else:
            sc = ctl.submit(st.session_state.get("sus_name", ""), answers)
            st.session_state["_flash"] = t("sus_thanks", s=sc)
            st.session_state["_pending"] = {"sus_name": "", **{f"sus_q{i}": None for i in range(1, S.N_ITEMS + 1)}}
            st.rerun()

    rows, mean = ctl.summary()
    st.markdown(f"### {t('sus_res_title')}")
    if not rows:
        st.info(t("sus_none"))
        return
    g = S.grade(mean)
    k1, k2 = st.columns(2)
    k1.metric(t("sus_n"), len(rows))
    k2.metric(t("sus_mean"), f"{mean:.1f}")
    st.markdown(chip(t("sus_g_" + g), {"good": "ok", "ok": "warn", "poor": "bad"}[g]), unsafe_allow_html=True)
    st.caption(t("sus_bench"))
    fig = go.Figure(go.Bar(x=[f"#{r['id']}" for r in rows], y=[r["score"] for r in rows], marker_color="#3A6EA5"))
    fig.add_hline(y=68, line_dash="dash", line_color="#924A2F", annotation_text="68")
    fig.update_layout(yaxis=dict(range=[0, 100], title="SUS"))
    show(style(fig))
    df = ctl.table(rows)
    st.dataframe(df, hide_index=True)
    st.download_button(t("mc_download"), df.to_csv(index=False).encode("utf-8"),
                       file_name="sus_responses.csv", mime="text/csv", key="sus_dl")
    with st.expander(t("sus_manage")):
        rid = st.selectbox(t("sus_pick"), [r["id"] for r in rows], key="sus_del_id")
        if st.button(t("sus_delete"), key="sus_del"):
            ctl.delete(rid)
            st.rerun()