"""Halaman Explainability: fitur apa yang diandalkan model, dan mengapa satu sampel diputuskan begitu."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from controllers import compare_controller as cc
from controllers import explain_controller as ctl
from controllers import pipeline_controller as pc
from i18n import t
from models import explain as E
from views.components import apply_pending, hero, page_link, show, style
from views.models_view import _label


def render_explain():
    apply_pending()
    hero(t("nav_explain"), t("ex_sub"))
    runs = ctl.runs_with_models()
    if not runs:
        st.info(t("ex_none"))
        page_link("pipeline", t("nav_pipeline"), ":material/tune:")
        return
    st.info(t("ex_intro"))

    names = cc.dataset_names()
    ids = [r["id"] for r in runs]
    by_id = {r["id"]: r for r in runs}
    if st.session_state.get("ex_run") not in ids:
        st.session_state["ex_run"] = ids[0]
    c1, c2 = st.columns([3, 1])
    rid = c1.selectbox(t("pl_view_run"), ids, key="ex_run", format_func=lambda i: _label(by_id[i], names))
    run = pc.load_run(rid)
    model = c2.selectbox(t("pl_col_model"), list(run["holdout"]), key=f"ex_model_{rid}")
    feats = run["summary"]["features"]
    classes = run["summary"]["classes"]

    # ---------- 1. fitur yang diandalkan ----------
    st.markdown(f"### {t('ex_global_title')}")
    st.caption(t("ex_global_help"))
    builtin = pd.Series(run["importance"][model]) if run["importance"].get(model) else None
    sources = ["perm"] + (["builtin"] if builtin is not None else [])
    src = st.radio(t("ex_source"), sources, horizontal=True, key=f"ex_src_{rid}_{model}",
                   format_func=lambda k: t("ex_src_" + k))
    if src == "perm":
        with st.spinner(t("ex_computing")):
            perm = ctl.permutation(rid, model)
        raw = perm["importance"]
        axis = t("ex_axis_perm")
        values = raw * 100
    else:
        raw = builtin
        axis = t("ex_axis_builtin")
        values = raw * 100
    total = float(raw.sum())
    share = raw / total if total > 0 else raw
    top = values.sort_values(ascending=False).head(15)[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=list(top.index), orientation="h", marker_color="#3A6EA5"))
    fig.update_layout(xaxis_title=axis, height=max(280, 26 * len(top) + 90))
    show(style(fig))

    st.markdown(f"##### {t('ex_plain_title')}")
    for key, kw in E.narrative(share, builtin if src == "perm" else None):
        st.markdown(f"- {t(key, **kw)}")
    if src == "perm":
        st.caption(t("ex_perm_note"))

    # ---------- 2. satu sampel ----------
    st.markdown(f"### {t('ex_local_title')}")
    st.caption(t("ex_local_help"))
    X, y = ctl.load_test(rid)
    wrong = ctl.misclassified(rid, model)
    mode = st.radio(t("ex_pick"), ["any", "wrong"], horizontal=True, key=f"ex_pick_{rid}_{model}",
                    format_func=lambda k: t("ex_pick_" + k, n=len(wrong)))
    if mode == "wrong":
        if not wrong:
            st.success(t("ex_no_wrong"))
            return
        row = st.selectbox(t("ex_sample"), wrong, key=f"ex_row_w_{rid}_{model}")
    else:
        row = int(st.number_input(t("ex_sample"), 0, len(y) - 1, 0, key=f"ex_row_{rid}_{model}"))
    true, pred, proba = ctl.sample_view(rid, model, row)
    k1, k2, k3 = st.columns(3)
    k1.metric(t("ex_true"), true)
    k2.metric(t("ex_pred"), pred)
    k3.metric(t("ex_conf"), f"{proba.max() * 100:.1f}%")
    if true != pred:
        st.warning(t("ex_wrong_note"))

    cand = list(share.sort_values(ascending=False).index[:15])
    cls_idx = int(proba.argmax())
    base, eff = ctl.local(rid, model, row, cls_idx, cand)
    eff = eff.reindex(eff.abs().sort_values(ascending=False).index).head(10)[::-1] * 100
    fig = go.Figure(go.Bar(x=eff.values, y=list(eff.index), orientation="h",
                           marker_color=["#3C7A5A" if v >= 0 else "#924A2F" for v in eff.values]))
    fig.update_layout(xaxis_title=t("ex_axis_local", c=pred), height=max(280, 26 * len(eff) + 90))
    show(style(fig))
    pos = [f for f, v in eff[::-1].items() if v > 0.5][:3]
    neg = [f for f, v in eff[::-1].items() if v < -0.5][:3]
    st.write(t("ex_loc_pos", c=pred, f=", ".join(pos)) if pos else t("ex_loc_none"))
    if neg:
        st.write(t("ex_loc_neg", f=", ".join(neg)))
    st.caption(t("ex_local_note"))

    # ---------- 3. AI opsional ----------
    with st.expander(t("ex_ai_title")):
        st.markdown(t("ex_ai_help"))
        lines = "\n".join(f"- {f}: {share[f] * 100:.1f}%" for f in share.sort_values(ascending=False).index[:8])
        st.code(t("ex_ai_prompt", ds=run["dataset"], m=model, cl=", ".join(classes), n=len(feats), lines=lines),
                language="text")
    with st.expander(t("ex_how_title")):
        st.markdown(t("ex_how"))