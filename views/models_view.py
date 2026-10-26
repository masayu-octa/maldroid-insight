"""Halaman Model Comparison: membaca run yang tersimpan di database."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import BLUE_SCALE, EXTRA_COLORS
from controllers import compare_controller as cc
from controllers import pipeline_controller as pc
from i18n import t
from views.components import apply_pending, hero, page_link, show, style

CM_SCALE = [[0, "#FBF6E9"], [1, "#3A6EA5"]]


def _label(r, names):
    n = r["summary"]["n_features"]
    return (f"#{r['id']} · {names.get(r['dataset_id'], '?')} · "
            f"{t('pl_sc_' + r['config']['scenario'], n=n)} · "
            f"{t('pl_lab_' + r['config']['mode'], n=len(r['summary']['classes']))}")


def render_models():
    apply_pending()
    hero(t("nav_models"), t("mc_sub"))
    runs = cc.all_runs()
    if not runs:
        st.info(t("mc_none"))
        page_link("pipeline", t("nav_pipeline"), ":material/tune:")
        return
    names = cc.dataset_names()
    ids = [r["id"] for r in runs]
    by_id = {r["id"]: r for r in runs}

    tab1, tab2 = st.tabs([t("mc_tab_run"), t("mc_tab_cmp")])
    with tab1:
        if st.session_state.get("mc_run") not in ids:
            st.session_state["mc_run"] = ids[0]
        rid = st.selectbox(t("pl_view_run"), ids, key="mc_run", format_func=lambda i: _label(by_id[i], names))
        _single(pc.load_run(rid))
    with tab2:
        _compare(ids, by_id, names)


# ---------------- satu run ----------------
def _single(run):
    ho, cv = run["holdout"], run["cv"]
    classes = run["summary"]["classes"]
    cfg = run["config"]
    ex = ", ".join(cfg.get("excluded") or []) if cfg["scenario"] == "exclude" else ""
    txt = t("pl_cfg", id=run["id"], ds=run["dataset"],
            fs=t("pl_sc_" + cfg["scenario"], n=run["summary"]["n_features"]),
            lab=t("pl_lab_" + cfg["mode"], n=len(classes)), rows=f"{run['summary']['n_rows']:,}",
            smote=t("pl_on") if cfg["smote"] else t("pl_off"))
    st.markdown(f'<div class="prov">{txt}{(" · " + t("mc_excluded") + ": " + ex) if ex else ""}</div>',
                unsafe_allow_html=True)

    main = ho or cv
    best = max(main, key=lambda k: main[k]["acc"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("mc_best_model"), best)
    k2.metric(t("mc_acc"), f"{main[best]['acc'] * 100:.2f}%")
    k3.metric(t("mc_f1m"), f"{main[best]['f1_macro'] * 100:.2f}%")
    k4.metric(t("mc_nmodels"), len(main))
    page_link("pipeline", t("mc_back"), ":material/tune:")

    if ho:
        st.markdown(f"##### {t('pl_res_holdout')}")
        fig = go.Figure()
        fig.add_bar(x=list(ho), y=[v["acc"] * 100 for v in ho.values()], name=t("mc_acc"), marker_color="#3A6EA5")
        fig.add_bar(x=list(ho), y=[v["f1_macro"] * 100 for v in ho.values()], name=t("mc_f1m"), marker_color="#D2A35C")
        lo = min(min(v["acc"], v["f1_macro"]) for v in ho.values()) * 100
        fig.update_layout(barmode="group", yaxis=dict(range=[max(0, lo - 5), 100.5], title="%"))
        show(style(fig, legend=True))
        df = pd.DataFrame({
            t("pl_col_model"): list(ho),
            "Acc (%)": [round(v["acc"] * 100, 2) for v in ho.values()],
            "Prec (%)": [round(v["prec"] * 100, 2) for v in ho.values()],
            "Rec (%)": [round(v["rec"] * 100, 2) for v in ho.values()],
            "F1 (%)": [round(v["f1"] * 100, 2) for v in ho.values()],
            "F1 macro (%)": [round(v["f1_macro"] * 100, 2) for v in ho.values()],
            t("pl_col_time"): [round(v["time"], 1) for v in ho.values()]})
        st.dataframe(df, hide_index=True)
        st.download_button(t("mc_download"), df.to_csv(index=False).encode("utf-8"),
                           file_name=f"run{run['id']}_holdout.csv", mime="text/csv", key=f"dl_ho_{run['id']}")
    if cv:
        st.markdown(f"##### {t('pl_res_cv')}")
        fig = go.Figure(go.Bar(x=list(cv), y=[v["acc"] * 100 for v in cv.values()], marker_color="#3A6EA5",
                               error_y=dict(type="data", array=[v["std"] * 100 for v in cv.values()],
                                            color="#141F52")))
        lo = min(v["acc"] - v["std"] for v in cv.values()) * 100
        fig.update_layout(yaxis=dict(range=[max(0, lo - 5), 100.5], title="%"))
        show(style(fig))
        st.caption(t("mc_cv_note"))
        df = pd.DataFrame({
            t("pl_col_model"): list(cv),
            "Acc (%)": [round(v["acc"] * 100, 2) for v in cv.values()],
            t("pl_col_std"): [round(v["std"] * 100, 3) for v in cv.values()],
            "F1 macro (%)": [round(v["f1_macro"] * 100, 2) for v in cv.values()],
            t("pl_col_folds"): [", ".join(f"{a * 100:.2f}" for a in v["folds"]) for v in cv.values()]})
        st.dataframe(df, hide_index=True)
        st.download_button(t("mc_download"), df.to_csv(index=False).encode("utf-8"),
                           file_name=f"run{run['id']}_cv.csv", mime="text/csv", key=f"dl_cv_{run['id']}")

    cms = {m: v["cm"] for m, v in ho.items() if v.get("cm")}
    if cms:
        st.markdown(f"##### {t('mc_cm_title')}")
        st.caption(t("mc_cm_help"))
        model = st.selectbox(t("pl_col_model"), list(cms), key=f"mc_cm_{run['id']}")
        _confusion(cms[model], classes)

    if run["importance"]:
        st.markdown(f"##### {t('mc_imp_title')}")
        m = st.selectbox(t("pl_col_model"), list(run["importance"]), key=f"mc_imp_{run['id']}")
        top = list(run["importance"][m].items())[:15][::-1]
        fig = go.Figure(go.Bar(x=[v * 100 for _, v in top], y=[k for k, _ in top], orientation="h",
                               marker_color="#3A6EA5"))
        fig.update_layout(xaxis_title="%", height=max(260, 24 * len(top) + 80))
        show(style(fig))


def _confusion(cm, classes):
    a = pd.DataFrame(cm).to_numpy(dtype=float)
    norm = a / a.sum(axis=1, keepdims=True).clip(min=1)
    fig = go.Figure(go.Heatmap(
        z=norm * 100, x=classes, y=classes, colorscale=CM_SCALE, zmin=0, zmax=100, showscale=False,
        text=[[f"{int(a[i][j]):,}<br>{norm[i][j] * 100:.1f}%" for j in range(len(classes))]
              for i in range(len(classes))], texttemplate="%{text}"))
    fig.update_layout(xaxis_title=t("mc_pred"), yaxis_title=t("mc_true"),
                      yaxis=dict(autorange="reversed"), height=max(320, 70 * len(classes) + 120))
    show(style(fig))
    rec = {classes[i]: (norm[i][i] * 100 if a[i].sum() else 0) for i in range(len(classes))}
    weakest = min(rec, key=rec.get)
    st.caption(t("mc_cm_weak", c=weakest, v=rec[weakest]))


# ---------------- banding antar run ----------------
def _compare(ids, by_id, names):
    st.markdown(t("mc_cmp_help"))
    cur = st.session_state.get("mc_run")
    default = [cur] if cur in ids else ids[:1]
    same = [i for i in ids if by_id[i]["dataset_id"] == by_id[default[0]]["dataset_id"] and i not in default]
    default += same[:2]
    pick = st.multiselect(t("mc_pick_runs"), ids, default=default, key="mc_cmp_runs",
                          format_func=lambda i: _label(by_id[i], names))
    if len(pick) < 2:
        st.info(t("mc_pick_two"))
        return
    key = st.radio(t("mc_metric"), ["ho_acc", "ho_f1m", "cv_acc", "cv_f1m"], horizontal=True, key="mc_metric",
                   format_func=lambda k: t("mc_m_" + k))
    df, runs = cc.pivot(pick, key)
    ds = {by_id[i]["dataset_id"] for i in pick}
    modes = {by_id[i]["config"]["mode"] for i in pick}
    if len(ds) > 1 or len(modes) > 1:
        st.warning(t("mc_not_comparable"))
    cols = [f"#{i}" for i in pick]
    if df[cols].isna().all().all():
        st.info(t("mc_no_metric"))
        return
    fig = go.Figure()
    palette = ["#3A6EA5", "#D2A35C", "#924A2F", "#233A6E"] + EXTRA_COLORS
    for n, c in enumerate(cols):
        fig.add_bar(x=df["Model"], y=df[c], name=c, marker_color=palette[n % len(palette)])
    vals = df[cols].stack()
    fig.update_layout(barmode="group", yaxis=dict(range=[max(0, vals.min() - 5), 100.5], title="%"))
    show(style(fig, legend=True))
    out = df.copy()
    if len(cols) == 2:
        out[t("mc_delta", a=cols[1], b=cols[0])] = (df[cols[1]] - df[cols[0]]).round(2)
    st.dataframe(out, hide_index=True)
    st.download_button(t("mc_download"), out.to_csv(index=False).encode("utf-8"),
                       file_name="run_comparison.csv", mime="text/csv", key="dl_cmp")

    st.markdown(f"##### {t('mc_cfg_title')}")
    cfg = pd.DataFrame({
        t("pl_col_run"): cols,
        t("dataset"): [names.get(by_id[i]["dataset_id"], "?") for i in pick],
        t("pl_scenario"): [t("pl_scn_" + by_id[i]["config"]["scenario"]) for i in pick],
        t("mc_excluded"): [", ".join(by_id[i]["config"].get("excluded") or []) or "-" for i in pick],
        t("pl_n_feats_short"): [by_id[i]["summary"]["n_features"] for i in pick],
        t("pl_labels"): [t("pl_lab_" + by_id[i]["config"]["mode"], n=len(by_id[i]["summary"]["classes"])) for i in pick],
        t("mc_rows"): [by_id[i]["summary"]["n_rows"] for i in pick],
        "SMOTE": [t("pl_on") if by_id[i]["config"]["smote"] else t("pl_off") for i in pick],
        t("pl_speed"): [t("pl_fast") if by_id[i]["config"]["fast"] else t("pl_full") for i in pick]})
    st.dataframe(cfg, hide_index=True)