"""Halaman Pipeline Runner: pilih dataset, fitur, model, evaluasi; jalankan; lihat hasil."""
import pandas as pd
import streamlit as st

from controllers import dataset_controller as datasets_ctl
from controllers import pipeline_controller as ctl
from i18n import t
from models import pipeline as P
from views.components import apply_pending, flash, hero, page_link


def render_pipeline():
    apply_pending()
    hero(t("nav_pipeline"), t("pl_sub"))
    flash()
    datasets = datasets_ctl.list_datasets()
    if not datasets:
        st.info(t("ds_none"))
        page_link("datasets", t("hm_start"))
        return

    by_id = {d["id"]: d for d in datasets}
    ds_id = st.selectbox(t("dataset"), list(by_id), key="pl_ds", format_func=lambda i: by_id[i]["name"])
    meta = by_id[ds_id]
    df, feats, _ = datasets_ctl.load(ds_id)
    classes_n = df["Label"].nunique()
    others = [d for d in datasets if d["id"] != ds_id]

    left, right = st.columns(2)
    with left:
        scenarios = ["all", "exclude"] + (["shared"] if others else [])
        scen = st.radio(t("pl_scenario"), scenarios, key=f"pl_scen_{ds_id}", horizontal=True,
                        format_func=lambda k: t("pl_scn_" + k))
        excluded, shared_with = [], None
        if scen == "exclude":
            excluded = st.multiselect(t("pl_excl"), feats, key=f"pl_excl_{ds_id}", help=t("pl_excl_help"))
        elif scen == "shared":
            shared_with = st.selectbox(t("pl_shared_with"), [d["id"] for d in others],
                                       key=f"pl_shared_{ds_id}", format_func=lambda i: by_id[i]["name"])
        try:
            cols = ctl.feature_columns(ds_id, scen, excluded, shared_with)
        except ctl.PipelineError as e:
            st.error(t(e.key, **e.kw))
            return
        st.caption(t("pl_n_feats", n=len(cols), total=len(feats)))

        if classes_n > 2:
            mode = st.radio(t("pl_labels"), ["orig", "binary"], horizontal=True, key=f"pl_mode_{ds_id}",
                            format_func=lambda k: t("pl_lab_" + k, n=classes_n))
        else:
            mode = "binary"
            st.caption(t("pl_two_classes"))
        avail = P.available_models()
        default = [m for m in P.DEFAULT_MODELS if m in avail][:2]
        models = st.multiselect(t("pl_models"), avail, default=default, key="pl_models",
                                help=t("pl_models_help"))
        if P.missing_libs():
            st.caption(t("pl_missing_lib", pip=" ".join(P.missing_libs())))
    with right:
        e1, e2 = st.columns(2)
        do_ho = e1.checkbox(t("pl_holdout"), value=True, key="pl_ho")
        do_cv = e2.checkbox(t("pl_cv"), value=True, key="pl_cvc")
        folds = st.slider(t("pl_folds"), 3, 10, 5, key="pl_folds", disabled=not do_cv)
        use_smote = st.toggle(t("pl_smote"), value=True, key="pl_smote")
        speed = st.radio(t("pl_speed"), ["fast", "full"], horizontal=True, key="pl_speed",
                         format_func=lambda k: t("pl_" + k))
        total = meta["n_rows"]
        if total > 2000:
            n_used = int(st.number_input(t("pl_maxrows"), min_value=2000, max_value=total,
                                         value=min(20000, total), step=1000, key=f"pl_rows_{ds_id}"))
        else:
            n_used = total
        st.caption(t("pl_rows_used", n=f"{n_used:,}", total=f"{total:,}"))
        with st.expander(t("pl_hyper_title")):
            st.dataframe(pd.DataFrame({
                t("pl_col_model"): list(P.HYPERS),
                t("pl_fast"): [v[0] for v in P.HYPERS.values()],
                t("pl_full"): [v[1] for v in P.HYPERS.values()]}), hide_index=True)

    top = datasets_ctl.separation_scores(ds_id)
    top = top[top.index.isin(cols)]
    if len(top) and top.iloc[0] > 2.0:
        st.warning(t("pl_leak_hint", f=top.index[0], s=float(top.iloc[0])))

    steps = [("log1p", True), ("RobustScaler", True), ("SMOTE", use_smote), ("Model", True), ("Evaluate", True)]
    chips = '<span class="arrow">→</span>'.join(
        f'<span class="step{"" if on else " off"}">{n}</span>' for n, on in steps)
    st.markdown(f'<div class="steps">{chips}</div>', unsafe_allow_html=True)

    if st.button(t("pl_run"), type="primary", key="pl_go"):
        if not models:
            st.warning(t("pl_no_model"))
        elif not (do_ho or do_cv):
            st.warning(t("pl_no_eval"))
        else:
            bar, status = st.progress(0.0), st.empty()

            def cb(done, tot, msg):
                bar.progress(done / tot if tot else 1.0)
                status.markdown(f"**{msg}**")

            cfg = {"scenario": scen, "excluded": excluded, "shared_with": shared_with, "mode": mode,
                   "models": models, "holdout": do_ho, "cv": do_cv, "folds": folds,
                   "smote": use_smote, "fast": speed == "fast", "n_rows": n_used}
            try:
                run_id = ctl.execute(ds_id, cfg, cb)
            except ctl.PipelineError as e:
                bar.empty()
                status.empty()
                st.error(t(e.key, **e.kw))
            else:
                st.session_state["_flash"] = t("pl_done", id=run_id)
                st.session_state["_pending"] = {"pl_view": run_id, "mc_run": run_id}
                st.rerun()

    _history(ds_id)


def _history(ds_id):
    runs = ctl.list_runs(ds_id)
    st.markdown(f"### {t('pl_res_title')}")
    if not runs:
        st.info(t("pl_no_runs"))
        return
    ids = [r["id"] for r in runs]
    by_id = {r["id"]: r for r in runs}
    if st.session_state.get("pl_view") not in ids:
        st.session_state["pl_view"] = ids[0]
    c1, c2 = st.columns([3, 1])
    rid = c1.selectbox(t("pl_view_run"), ids, key="pl_view",
                       format_func=lambda i: f"#{i} · {by_id[i]['created_at']} · "
                                             f"{t('pl_sc_' + by_id[i]['config']['scenario'], n=by_id[i]['summary']['n_features'])}")
    with c2:
        st.write("")
        page_link("models", t("pl_open_cmp"), ":material/leaderboard:")
    show_run(ctl.load_run(rid))

    with st.expander(t("pl_hist", n=len(runs))):
        st.dataframe(pd.DataFrame({
            t("pl_col_run"): ids,
            t("pl_col_when"): [r["created_at"] for r in runs],
            t("pl_scenario"): [t("pl_sc_" + r["config"]["scenario"], n=r["summary"]["n_features"]) for r in runs],
            t("pl_labels"): [t("pl_lab_" + r["config"]["mode"], n=len(r["summary"]["classes"])) for r in runs],
            t("pl_models"): [", ".join(r["config"]["models"]) for r in runs],
            t("pl_col_best"): [round(r["best_acc"] * 100, 2) if r["best_acc"] is not None else None for r in runs],
        }), hide_index=True)
        if st.button(t("pl_delete_run"), key=f"pl_del_{rid}"):
            ctl.delete_run(rid)
            st.session_state.pop("pl_view", None)
            st.session_state["_flash"] = t("pl_run_deleted", id=rid)
            st.rerun()


def show_run(run):
    cfg, summ = run["config"], run["summary"]
    fs = t("pl_sc_" + cfg["scenario"], n=summ["n_features"])
    lab = t("pl_lab_" + cfg["mode"], n=len(summ["classes"]))
    smote = t("pl_on") if cfg["smote"] else t("pl_off")
    rows_txt = f"{summ['n_rows']:,}"
    cfg_txt = t("pl_cfg", id=run["id"], ds=run["dataset"], fs=fs, lab=lab, rows=rows_txt, smote=smote)
    st.markdown(f'<div class="prov">{cfg_txt}</div>', unsafe_allow_html=True)
    cm_col = t("pl_col_model")
    if run["holdout"]:
        st.markdown(f"##### {t('pl_res_holdout')}")
        ho = run["holdout"]
        st.dataframe(pd.DataFrame({
            cm_col: list(ho),
            "Acc (%)": [round(v["acc"] * 100, 2) for v in ho.values()],
            "Prec (%)": [round(v["prec"] * 100, 2) for v in ho.values()],
            "Rec (%)": [round(v["rec"] * 100, 2) for v in ho.values()],
            "F1 (%)": [round(v["f1"] * 100, 2) for v in ho.values()],
            t("pl_col_time"): [round(v["time"], 1) for v in ho.values()]}), hide_index=True)
        best = max(ho, key=lambda k: ho[k]["acc"])
        st.caption(t("pl_best", m=best, v=ho[best]["acc"] * 100))
    if run["cv"]:
        st.markdown(f"##### {t('pl_res_cv')}")
        cv = run["cv"]
        st.dataframe(pd.DataFrame({
            cm_col: list(cv),
            "Acc (%)": [round(v["acc"] * 100, 2) for v in cv.values()],
            t("pl_col_std"): [round(v["std"] * 100, 3) for v in cv.values()],
            "F1 (%)": [round(v["f1"] * 100, 2) for v in cv.values()],
            t("pl_col_folds"): [", ".join(f"{a * 100:.2f}" for a in v["folds"]) for v in cv.values()]}),
            hide_index=True)
        st.caption(t("pl_cv_note"))
    if run["importance"]:
        name = next(iter(run["importance"]))
        st.markdown(f"##### {t('pl_top_feats', m=name)}")
        top = list(run["importance"][name].items())[:8]
        st.dataframe(pd.DataFrame({t("col_feat"): [k for k, _ in top],
                                   t("pl_col_importance"): [v for _, v in top]}), hide_index=True)