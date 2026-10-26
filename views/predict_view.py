"""Halaman Prediction: memakai model tersimpan pada CSV baru, dengan pencocokan kolom."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from controllers import compare_controller as cc
from controllers import explain_controller as ctl
from controllers import pipeline_controller as pc
from i18n import t
from models import datasets as M
from models import predict as PR
from views.components import apply_pending, hero, page_link, show, style
from views.models_view import _label

NONE = "-"


def render_predict():
    apply_pending()
    hero(t("nav_predict"), t("pr_sub"))
    runs = ctl.runs_with_models()
    if not runs:
        st.info(t("ex_none"))
        page_link("pipeline", t("nav_pipeline"), ":material/tune:")
        return
    names = cc.dataset_names()
    ids = [r["id"] for r in runs]
    by_id = {r["id"]: r for r in runs}
    c1, c2 = st.columns([3, 1])
    if st.session_state.get("pr_run") not in ids:
        st.session_state["pr_run"] = ids[0]
    rid = c1.selectbox(t("pl_view_run"), ids, key="pr_run", format_func=lambda i: _label(by_id[i], names))
    run = pc.load_run(rid)
    model = c2.selectbox(t("pl_col_model"), list(run["holdout"]), key=f"pr_model_{rid}")
    feats = run["summary"]["features"]
    classes = run["summary"]["classes"]
    imp = run["importance"].get(model) or {}

    st.markdown(f'<div class="prov">{t("pr_expect", n=len(feats), c=", ".join(classes))}</div>',
                unsafe_allow_html=True)
    with st.expander(t("pr_feat_list")):
        st.write(", ".join(feats))

    file = st.file_uploader(t("pr_upload"), type=["csv"], key="pr_file")
    if not file:
        st.info(t("pr_hint"))
        return
    try:
        df = ctl.read_csv(file)
    except M.DatasetError as e:
        st.error(t(e.key, **e.kw))
        return
    if df.empty:
        st.error(t("pr_empty"))
        return

    mapping = PR.auto_match(feats, df.columns)
    unmatched = sorted([f for f in feats if f not in mapping], key=lambda f: -imp.get(f, 0))
    if unmatched:
        with st.expander(t("pr_manual", n=len(unmatched))):
            st.caption(t("pr_manual_help"))
            free = [c for c in df.columns if c not in set(mapping.values())]
            for f in unmatched[:25]:
                pick = st.selectbox(f, [NONE] + free, key=f"pr_map_{rid}_{model}_{f}")
                if pick != NONE:
                    mapping[f] = pick
            if len(unmatched) > 25:
                st.caption(t("pr_manual_more", n=len(unmatched) - 25))

    plain, weighted = PR.coverage(mapping, feats, imp)
    k1, k2, k3 = st.columns(3)
    k1.metric(t("pr_matched"), f"{len(mapping)} / {len(feats)}")
    k2.metric(t("pr_cov"), f"{plain * 100:.0f}%")
    k3.metric(t("pr_cov_w"), f"{weighted * 100:.0f}%" if weighted is not None else "-")
    if plain < 0.5:
        st.error(t("pr_err_cov"))
        st.caption(t("pr_err_cov_hint", cols=", ".join(map(str, df.columns[:12]))))
        return
    if plain < 1:
        st.caption(t("pr_fill_note", n=len(feats) - len(mapping)))
    if weighted is not None and weighted < 0.7:
        st.warning(t("pr_warn_w"))

    with st.expander(t("pr_label_opt")):
        st.caption(t("pr_label_help"))
        lab_col = st.selectbox(t("pr_label_col"), [NONE] + list(df.columns), key=f"pr_lab_{rid}")
        benign_vals = []
        if lab_col != NONE:
            vals = sorted(df[lab_col].astype(str).unique())[:50]
            default = [v for v in vals if "benign" in v.lower() or v.lower() in ("0", "clean", "legitimate")]
            benign_vals = st.multiselect(t("pr_benign_vals"), vals, default=default, key=f"pr_ben_{rid}_{lab_col}")

    sig = (rid, model, file.name, file.size, tuple(sorted(mapping.items())))
    if st.button(t("pr_run"), type="primary", key="pr_go"):
        with st.spinner(t("ex_computing")):
            out, bad = ctl.predict(rid, model, df, mapping)
        st.session_state["pr_out"] = (sig, out, bad)
    saved = st.session_state.get("pr_out")
    if not saved or saved[0] != sig:
        return
    _, out, bad = saved
    if bad:
        st.caption(t("pr_bad_cells", n=bad))

    counts = out["prediction"].value_counts().reindex(classes).fillna(0)
    st.markdown(f"##### {t('pr_res_title', n=len(out))}")
    fig = go.Figure(go.Bar(x=list(counts.index), y=counts.values, marker_color="#3A6EA5"))
    fig.update_layout(yaxis_title=t("pr_count"))
    show(style(fig))
    low = int((out["confidence_%"] < 60).sum())
    if low:
        st.caption(t("pr_low_conf", n=low))

    if lab_col != NONE and benign_vals:
        truth = df[lab_col].astype(str).map(lambda v: "Benign" if v in benign_vals else (
            "Malware" if run["config"]["mode"] == "binary" else v))
        acc = float((truth.to_numpy() == out["prediction"].to_numpy()).mean())
        st.metric(t("pr_acc_on_file"), f"{acc * 100:.2f}%")
        st.caption(t("pr_acc_note"))

    result = pd.concat([df.reset_index(drop=True), out], axis=1)
    st.dataframe(result.head(200))
    st.caption(t("pr_show_n", n=min(200, len(result)), total=len(result)))
    st.download_button(t("mc_download"), result.to_csv(index=False).encode("utf-8"),
                       file_name=f"prediction_run{rid}_{model}.csv", mime="text/csv", key="pr_dl")