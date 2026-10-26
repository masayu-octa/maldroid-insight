"""Halaman Dashboard: tujuan alat, alur kerja, ringkasan sesi, glosarium."""
import pandas as pd
import streamlit as st

from controllers import dataset_controller as ctl
from i18n import t
from views.components import chip, hero, page_link

STEPS = [("1", "datasets"), ("2", "datasets"), ("3", "pipeline"),
         ("4", "models"), ("5", "explain"), ("6", "predict")]
GLOSSARY = ["log1p", "scaler", "smote", "leak", "holdout", "cv", "shap"]


def render_home():
    hero(t("nav_home"), t("hm_sub"))
    s = ctl.summary()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("hm_kpi_datasets"), s["n_datasets"])
    k2.metric(t("hm_kpi_samples"), f"{s['n_samples']:,}")
    k3.metric(t("hm_kpi_runs"), s["n_runs"])
    k4.metric(t("hm_kpi_best"), f"{s['best_acc'] * 100:.2f}%" if s["best_acc"] else "-")

    st.markdown(f"### {t('hm_what_title')}")
    st.markdown(f'<div class="lead">{t("hm_what")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prov">{t("hm_pipe")}</div>', unsafe_allow_html=True)

    st.markdown(f"### {t('hm_flow_title')}")
    has_ds, has_run = s["n_datasets"] > 0, s["n_runs"] > 0
    ready = {"1": has_ds, "2": has_ds, "3": has_run, "4": has_run, "5": has_run, "6": has_run}
    for row in (STEPS[:3], STEPS[3:]):
        cols = st.columns(3)
        for col, (n, page) in zip(cols, row):
            with col, st.container(border=True):
                st.markdown(f'<span class="stepno">{n}</span><b>{t(f"hm_s{n}_t")}</b> '
                            + chip(t("hm_done") if ready[n] else t("hm_todo"),
                                   "ok" if ready[n] else "todo"), unsafe_allow_html=True)
                st.caption(t(f"hm_s{n}_d"))
                page_link(page, t("hm_open"))
    st.caption(t("hm_cross"))
    page_link("cross", t("nav_cross"))

    st.markdown(f"### {t('hm_mine_title')}")
    if not s["datasets"]:
        st.info(t("hm_none"))
        page_link("datasets", t("hm_start"))
    else:
        st.dataframe(pd.DataFrame({
            t("ds_col_name"): [d["name"] for d in s["datasets"]],
            t("ds_col_source"): [t("src_" + d["source"]) for d in s["datasets"]],
            t("ds_col_rows"): [d["n_rows"] for d in s["datasets"]],
            t("ds_col_feats"): [d["n_features"] for d in s["datasets"]],
            t("ds_col_classes"): [len(d["class_counts"]) for d in s["datasets"]],
        }), hide_index=True)

    with st.expander(t("hm_gloss_title")):
        for g in GLOSSARY:
            st.markdown(f"**{t(f'gl_{g}_t')}**: {t(f'gl_{g}_d')}")