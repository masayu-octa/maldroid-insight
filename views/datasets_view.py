"""Halaman Datasets: tambah (contoh / unggah CSV), lihat kualitas, jelajahi fitur, hapus."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from controllers import dataset_controller as ctl
from i18n import t
from models import datasets as M
from views.components import (apply_pending, chip, class_colors, flash, hero, level, show, style)


def _goto_add():
    st.session_state["ds_view"] = "add"


def render_datasets():
    apply_pending()
    hero(t("nav_datasets"), t("ds_sub"))
    flash()
    view = st.radio(t("view"), ["mine", "add"], horizontal=True, key="ds_view",
                    format_func=lambda k: t("ds_view_" + k), label_visibility="collapsed")
    if view == "mine":
        _mine()
    else:
        _add()


# ---------------- daftar & inspeksi ----------------
def _mine():
    datasets = ctl.list_datasets()
    if not datasets:
        st.info(t("ds_none"))
        st.button(t("hm_start"), on_click=_goto_add, type="primary")
        return
    ids = [d["id"] for d in datasets]
    by_id = {d["id"]: d for d in datasets}
    ds_id = st.selectbox(t("ds_pick"), ids, key="ds_pick",
                         format_func=lambda i: by_id[i]["name"])
    meta = by_id[ds_id]
    df, feats, info = ctl.load(ds_id)

    counts = df["Label"].value_counts()
    raw = meta["raw_rows"] or len(df)
    dup_pct, na_pct = info["dup_removed"] / raw * 100, info["na_removed"] / raw * 100
    imb = counts.max() / counts.min()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("m_samples"), f"{len(df):,}")
    k2.metric(t("m_features"), len(feats))
    k3.metric(t("m_classes"), len(counts))
    k4.metric(t("m_ratio"), f"{imb:.2f} : 1")

    q1, q2, q3 = st.columns(3)
    q1.markdown(f"**{t('q_dup')}** {chip(f'{dup_pct:.2f}%', level(dup_pct, 1, 5))}",
                unsafe_allow_html=True)
    q2.markdown(f"**{t('q_na')}** {chip(f'{na_pct:.2f}%', level(na_pct, 1, 5))}",
                unsafe_allow_html=True)
    q3.markdown(f"**{t('q_imb')}** {chip(f'{imb:.2f}:1', level(imb, 1.5, 4))}",
                unsafe_allow_html=True)
    src = f"{t('src_' + meta['source'])} · {t('ds_label_col')}: {meta['label_col']}"
    st.markdown(f'<div class="prov">{t("prov", src=src, raw=f"{raw:,}", ts=meta["created_at"])}</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([t("tab_dist"), t("tab_feat"), t("tab_raw")])
    cc, cn, cp = t("col_class"), t("col_count"), t("col_pct")
    cmap = class_colors(counts.index)

    with tab1:
        c1, c2 = st.columns(2)
        ctype = c1.radio(t("view"), ["bar", "donut"], horizontal=True,
                         format_func=lambda k: t(k), key=f"ctype_{st.session_state.lang}")
        as_pct = c2.toggle(t("as_pct"), value=False, key="pct")
        dist = counts.rename_axis(cc).reset_index(name=cn)
        dist[cp] = (dist[cn] / dist[cn].sum() * 100).round(2)
        if ctype == "bar":
            ycol = cp if as_pct else cn
            show(style(px.bar(dist, x=cc, y=ycol, color=cc, color_discrete_map=cmap, text=ycol)))
        else:
            show(style(px.pie(dist, names=cc, values=cn, hole=0.55, color=cc,
                              color_discrete_map=cmap), legend=True))
        st.caption(t("ds_dist_help"))

    with tab2:
        scores = ctl.separation_scores(ds_id)
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown(f"##### {t('feat_top_title')}")
            top_n = st.slider(t("top_n"), 5, min(30, len(scores)), min(10, len(scores)), key="topn")
            top = scores.head(top_n).rename_axis(t("col_feat")).reset_index(name=t("col_score"))
            fig = px.bar(top, x=t("col_score"), y=t("col_feat"), color=t("col_score"),
                         orientation="h", color_continuous_scale=[[0, "#DCE7F1"], [1, "#141F52"]])
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(autorange="reversed")
            show(style(fig))
            st.caption(t("ds_sep_help"))
        with right:
            st.markdown(f"##### {t('feat_dist_title')}")
            feat = st.selectbox(t("pick_feat"), list(scores.index), key=f"feat_{ds_id}")
            kind = st.radio(t("chart_kind"), ["box", "violin", "hist"], horizontal=True,
                            format_func=lambda k: t(k), key=f"gkind_{st.session_state.lang}")
            use_log = st.checkbox(t("log1p"), value=True, key="log")
            vals = df[feat].astype(float)
            if use_log:
                vals = np.log1p(np.abs(vals))
            pdf = pd.DataFrame({cc: df["Label"], t("col_value"): vals})
            pdf = pdf.sample(min(len(pdf), 20000), random_state=42)
            y = t("col_value")
            if kind == "box":
                fig = px.box(pdf, x=cc, y=y, color=cc, color_discrete_map=cmap)
            elif kind == "violin":
                fig = px.violin(pdf, x=cc, y=y, color=cc, box=True, color_discrete_map=cmap)
            else:
                fig = px.histogram(pdf, x=y, color=cc, barmode="overlay", opacity=0.65,
                                   nbins=40, color_discrete_map=cmap)
            show(style(fig, legend=(kind == "hist")))
            st.caption(t("sep_caption", f=feat, s=scores[feat]))

    with tab3:
        f1, f2, f3 = st.columns([1, 2, 1])
        all_cls = sorted(df["Label"].unique())
        sel_cls = f1.multiselect(t("filter_cls"), all_cls, default=all_cls, key=f"cls_{ds_id}")
        sel_cols = f2.multiselect(t("filter_cols"), feats, default=feats[:8], key=f"cols_{ds_id}")
        n_rows = f3.slider(t("n_rows"), 10, 500, 50, key="rows")
        st.dataframe(df[df["Label"].isin(sel_cls)][["Label"] + sel_cols].head(n_rows))

    with st.expander(t("ds_delete_title")):
        ok = st.checkbox(t("ds_delete_confirm", name=meta["name"]), key=f"del_ok_{ds_id}")
        if st.button(t("ds_delete"), disabled=not ok, key=f"del_{ds_id}"):
            ctl.delete(ds_id)
            st.session_state["_flash"] = t("ds_deleted", name=meta["name"])
            st.rerun()


# ---------------- tambah dataset ----------------
def _add():
    st.markdown(f"#### {t('ad_example_title')}")
    st.caption(t("ad_example_note"))
    status = ctl.example_status()
    cols = st.columns(len(status))
    for col, (key, s) in zip(cols, status.items()):
        with col, st.container(border=True):
            st.markdown(f"**{s['name']}**")
            if s["added"]:
                st.markdown(chip(t("ad_exists"), "ok"), unsafe_allow_html=True)
            elif s["missing"]:
                st.caption(t("ad_err_file", f=", ".join(s["missing"])))
            if st.button(t("ad_add_btn"), key=f"ex_{key}", disabled=s["added"] or bool(s["missing"])):
                try:
                    with st.spinner(t("ad_working")):
                        new_id = ctl.add_example(key)
                except M.DatasetError as e:
                    st.error(t(e.key, **e.kw))
                else:
                    _done(new_id, s["name"])

    st.markdown(f"#### {t('ad_upload_title')}")
    st.caption(t("ad_upload_hint"))
    file = st.file_uploader(t("ad_file"), type=["csv"], key="ds_file")
    if file is None:
        return
    try:
        df = ctl.read_upload(file)
    except M.DatasetError as e:
        st.error(t(e.key, **e.kw))
        return
    fk = f"{file.name}_{file.size}"
    with st.expander(t("ad_preview", r=f"{len(df):,}", c=df.shape[1])):
        st.dataframe(df.head(10))

    cols = list(df.columns)
    name = st.text_input(t("ad_name"), value=Path(file.name).stem, key=f"ad_name_{fk}")
    label_col = st.selectbox(t("ad_label_col"), cols, key=f"ad_label_{fk}",
                             index=cols.index(M.suggest_label_column(df)))
    values = M.label_values(df, label_col)
    if len(values) < 2 or len(values) > M.MAX_LABEL_VALUES:
        st.error(t("ad_err_label" if len(values) < 2 else "ad_err_label_many", n=M.MAX_LABEL_VALUES))
        return
    labels = M.label_strings(df[label_col])
    st.dataframe(labels.value_counts().rename_axis(t("col_class")).reset_index(name=t("col_count")),
                 hide_index=True, height=min(240, 38 + 35 * len(values)))
    benign = st.multiselect(t("ad_benign"), values, key=f"ad_benign_{fk}_{label_col}",
                            default=M.suggest_benign(values), help=t("ad_benign_help"))
    cand, ignored = M.numeric_candidates(df, label_col)
    if ignored:
        st.caption(t("ad_warn_dropped", n=len(ignored)))
    exclude = st.multiselect(t("ad_exclude"), cand, key=f"ad_excl_{fk}_{label_col}",
                             help=t("ad_exclude_help"))
    feats = [c for c in cand if c not in exclude]
    dedup = st.checkbox(t("ad_dedup"), value=True, key=f"ad_dedup_{fk}")
    n_ben = int(labels.isin(benign).sum())
    st.markdown(f'<div class="prov">{t("ad_summary", nf=len(feats), nb=f"{n_ben:,}", nm=f"{int(labels.notna().sum()) - n_ben:,}")}</div>',
                unsafe_allow_html=True)

    if st.button(t("ad_register"), type="primary", key=f"ad_go_{fk}"):
        try:
            with st.spinner(t("ad_working")):
                new_id, warns = ctl.register_upload(df, name, label_col, benign, feats, dedup)
        except M.DatasetError as e:
            st.error(t(e.key, **e.kw))
        else:
            for k, kw in warns:
                st.warning(t(k, **kw))
            _done(new_id, name.strip())


def _done(new_id, name):
    st.session_state["_flash"] = t("ad_ok", name=name)
    st.session_state["_pending"] = {"ds_view": "mine", "ds_pick": new_id}
    st.rerun()