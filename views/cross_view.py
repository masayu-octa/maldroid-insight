"""Halaman Cross-Dataset: latih di satu dataset, uji di dataset lain (kolom bernama sama)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from controllers import cross_controller as ctl
from controllers import dataset_controller as datasets_ctl
from i18n import t
from models import cross as X
from models import pipeline as P
from views.components import apply_pending, flash, hero, page_link, show, style
from views.models_view import _confusion


def render_cross():
    apply_pending()
    hero(t("nav_cross"), t("cr_sub"))
    flash()
    st.info(t("cr_purpose"))
    datasets = datasets_ctl.list_datasets()
    if len(datasets) < 2:
        st.warning(t("cr_need_two"))
        page_link("datasets", t("hm_start"), ":material/dataset:")
        return
    by_id = {d["id"]: d for d in datasets}
    ids = list(by_id)
    fmt = lambda i: by_id[i]["name"]            # noqa: E731

    c1, c2 = st.columns(2)
    train_id = c1.selectbox(t("cr_train"), ids, key="cr_train", format_func=fmt)
    test_opts = [i for i in ids if i != train_id]
    test_id = c2.selectbox(t("cr_test"), test_opts, key=f"cr_test_{train_id}", format_func=fmt)

    cols_all = ctl.shared(train_id, test_id)
    st.markdown(f'<div class="prov">{t("cr_shared_n", n=len(cols_all), a=len(by_id[train_id]["features"]), b=len(by_id[test_id]["features"]))}</div>',
                unsafe_allow_html=True)
    if not cols_all:
        st.error(t("cr_err_shared"))
        st.caption(t("cr_shared_hint"))
        return
    if len(cols_all) < 5:
        st.warning(t("cr_few"))
    with st.expander(t("cr_shared_list")):
        excl = st.multiselect(t("pl_excl"), cols_all, key=f"cr_excl_{train_id}_{test_id}", help=t("pl_excl_help"))
        st.caption(t("cr_shared_note"))
    cols = [c for c in cols_all if c not in set(excl)]

    left, right = st.columns(2)
    with left:
        avail = P.available_models()
        default = [m for m in ["RandomForest", "LightGBM"] if m in avail][:2]
        models = st.multiselect(t("pl_models"), avail, default=default, key="cr_models")
    with right:
        smote = st.toggle(t("pl_smote"), value=True, key="cr_smote")
        speed = st.radio(t("pl_speed"), ["fast", "full"], horizontal=True, key="cr_speed",
                         format_func=lambda k: t("pl_" + k))
        n_rows = int(st.number_input(t("pl_maxrows"), min_value=2000, max_value=200000, value=20000,
                                     step=2000, key="cr_rows"))

    if st.button(t("pl_run"), type="primary", key="cr_go"):
        if not models:
            st.warning(t("pl_no_model"))
        else:
            bar, status = st.progress(0.0), st.empty()

            def cb(done, tot, msg):
                bar.progress(done / tot if tot else 1.0)
                status.markdown(f"**{msg}**")
            try:
                rid = ctl.execute(train_id, test_id, {"columns": cols, "models": models, "smote": smote,
                                                      "fast": speed == "fast", "n_rows": n_rows}, cb)
            except ctl.CrossError as e:
                bar.empty()
                status.empty()
                st.error(t(e.key, **e.kw))
            else:
                st.session_state["_flash"] = t("cr_done", id=rid)
                st.session_state["_pending"] = {"cr_view": rid}
                st.rerun()
    _results()


def _results():
    runs = ctl.list_runs()
    st.markdown(f"### {t('cr_res_title')}")
    if not runs:
        st.info(t("cr_no_runs"))
        return
    ids = [r["id"] for r in runs]
    by_id = {r["id"]: r for r in runs}
    if st.session_state.get("cr_view") not in ids:
        st.session_state["cr_view"] = ids[0]
    rid = st.selectbox(t("pl_view_run"), ids, key="cr_view",
                       format_func=lambda i: f"#{i} · {by_id[i]['train']} → {by_id[i]['test']} · {by_id[i]['created_at']}")
    run = by_id[rid]
    res = {k: v for k, v in run["results"].items() if not k.startswith("_")}
    meta = run["results"].get("_meta", {})
    st.markdown(f'<div class="prov">{t("cr_cfg", id=rid, a=run["train"], b=run["test"], n=len(run["config"]["columns"]), rows=meta.get("n_cross", 0))}</div>',
                unsafe_allow_html=True)

    df = pd.DataFrame({
        t("pl_col_model"): list(res),
        t("cr_in_f1"): [round(v["in"]["f1_macro"] * 100, 2) for v in res.values()],
        t("cr_x_f1"): [round(v["cross"]["f1_macro"] * 100, 2) for v in res.values()],
        t("cr_drop"): [round((v["in"]["f1_macro"] - v["cross"]["f1_macro"]) * 100, 2) for v in res.values()],
        t("cr_x_acc"): [round(v["cross"]["acc"] * 100, 2) for v in res.values()],
        t("cr_x_rec"): [round(v["cross"]["rec"] * 100, 2) for v in res.values()],
        t("cr_x_spec"): [round(v["cross"]["spec"] * 100, 2) for v in res.values()]})
    fig = go.Figure()
    fig.add_bar(x=df[t("pl_col_model")], y=df[t("cr_in_f1")], name=t("cr_in_f1"), marker_color="#3A6EA5")
    fig.add_bar(x=df[t("pl_col_model")], y=df[t("cr_x_f1")], name=t("cr_x_f1"), marker_color="#924A2F")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 100.5], title="%"))
    show(style(fig, legend=True))
    st.dataframe(df, hide_index=True)
    st.download_button(t("mc_download"), df.to_csv(index=False).encode("utf-8"),
                       file_name=f"cross{rid}.csv", mime="text/csv", key=f"dl_cr_{rid}")

    best = max(res, key=lambda k: res[k]["cross"]["f1_macro"])
    b = res[best]
    drop = b["in"]["f1_macro"] - b["cross"]["f1_macro"]
    st.markdown(f"##### {t('cr_read_title')}")
    st.write(t("cr_read_" + X.verdict(drop), m=best, a=run["train"], b=run["test"],
               i=b["in"]["f1_macro"] * 100, x=b["cross"]["f1_macro"] * 100, d=drop * 100))
    if b["cross"]["rec"] < 0.7 <= b["cross"]["spec"]:
        st.write(t("cr_read_missed", r=b["cross"]["rec"] * 100))
    elif b["cross"]["spec"] < 0.7 <= b["cross"]["rec"]:
        st.write(t("cr_read_falsealarm", s=b["cross"]["spec"] * 100))
    if abs(meta.get("malware_share_train", 0) - meta.get("malware_share_test", 0)) > 0.25:
        st.caption(t("cr_read_prior", a=meta["malware_share_train"] * 100, b=meta["malware_share_test"] * 100))
    st.caption(t("cr_read_caveat"))

    st.markdown(f"##### {t('mc_cm_title')} · {run['test']}")
    st.caption(t("cr_cm_help"))
    m = st.selectbox(t("pl_col_model"), list(res), index=list(res).index(best), key=f"cr_cm_{rid}")
    _confusion(res[m]["cross"]["cm"], ["Benign", "Malware"])

    if st.button(t("cr_delete"), key=f"cr_del_{rid}"):
        ctl.delete_run(rid)
        st.session_state.pop("cr_view", None)
        st.rerun()