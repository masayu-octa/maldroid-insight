"""Halaman Overview: rangkuman visual seluruh pemakaian (dataset, run, akurasi model, batas paket)."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from controllers import overview_controller as ctl
from i18n import t
from views.components import class_colors, hero, page_link, show, style

BLUE = ["#141F52", "#254F73", "#5B94BD", "#2F8F9D", "#C9983A", "#3C7A5A", "#8A5FA8", "#B0583A"]


def render_overview():
    hero(t("nav_overview"), t("nd_overview"))
    d = ctl.overview()
    datasets, runs, res = d["datasets"], d["runs"], d["results"]
    if not datasets:
        st.info(t("ov_none"))
        page_link("datasets_add", t("hm_start"), ":material/upload_file:", cta=True)
        return

    best = f"{res['acc'].max():.2f}%" if len(res) else "-"
    avg = f"{res['acc'].mean():.2f}%" if len(res) else "-"
    kpis = [(t("hm_kpi_datasets"), len(datasets), "#254F73"),
            (t("hm_kpi_samples"), f"{sum(x['n_rows'] for x in datasets):,}", "#2F8F9D"),
            (t("hm_kpi_runs"), len(runs), "#C9983A"),
            (t("ov_kpi_best"), best, "#3C7A5A"), (t("ov_kpi_avg"), avg, "#8A5FA8")]
    cards = "".join(f'<div class="kpi" style="--c:{c}"><div class="kpi-l">{l}</div><div class="kpi-v">{v}</div></div>'
                    for l, v, c in kpis)
    st.markdown(f'<div class="kpis">{cards}</div>', unsafe_allow_html=True)

    if not len(runs):
        st.info(t("ov_no_runs"))
        page_link("pipeline", t("nav_pipeline"), ":material/tune:", cta=True)
    else:
        a, b = st.columns(2)
        with a:
            st.markdown(f"##### {t('ov_best_ds')}")
            g = res.groupby("dataset", as_index=False)["acc"].max()
            fig = px.bar(g, x="dataset", y="acc", text=g["acc"].round(2), color_discrete_sequence=["#254F73"])
            fig.update_yaxes(range=[0, 100], title=t("ov_acc"))
            fig.update_xaxes(title=None)
            show(style(fig.update_layout(height=340)))
        with b:
            st.markdown(f"##### {t('ov_by_model')}")
            g = res.groupby("model", as_index=False).agg(best=("acc", "max"), avg=("acc", "mean"))
            fig = go.Figure()
            fig.add_trace(go.Bar(name=t("ov_best"), x=g["model"], y=g["best"], marker_color="#254F73"))
            fig.add_trace(go.Bar(name=t("ov_avg"), x=g["model"], y=g["avg"], marker_color="#C9983A"))
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 100], title=t("ov_acc"))
            show(style(fig.update_layout(height=340), legend=True))
        c, e = st.columns(2)
        with c:
            st.markdown(f"##### {t('ov_over_time')}")
            g = runs.groupby("date", as_index=False).size().rename(columns={"size": t("ov_runs")})
            fig = px.bar(g, x="date", y=t("ov_runs"), color_discrete_sequence=["#5B94BD"])
            fig.update_xaxes(title=None, type="category")
            fig.update_yaxes(dtick=1)
            show(style(fig.update_layout(height=340)))
        with e:
            st.markdown(f"##### {t('ov_f1_acc')}")
            fig = px.scatter(res, x="acc", y="f1", color="model", hover_data=["dataset", "run"],
                             color_discrete_sequence=BLUE)
            fig.update_xaxes(title=t("ov_acc"))
            fig.update_yaxes(title="F1 (%)")
            show(style(fig.update_layout(height=340), legend=True))

    st.markdown(f"##### {t('ov_classes')}")
    classes = sorted({c for x in datasets for c in x["class_counts"]})
    colors = class_colors(classes)
    names = [x["name"] for x in datasets]
    fig = go.Figure()
    for c in classes:
        fig.add_trace(go.Bar(name=c, y=names, x=[x["class_counts"].get(c, 0) for x in datasets],
                             orientation="h", marker_color=colors[c]))
    fig.update_layout(barmode="stack", height=max(200, 70 * len(names) + 90))
    show(style(fig, legend=True))

    if len(runs):
        st.markdown(f"##### {t('ov_latest')}")
        st.dataframe(runs.head(8).rename(columns={"run": t("pl_col_run"), "date": t("pl_col_when"),
                                                  "dataset": t("dataset"), "models": t("pl_models"),
                                                  "best": t("pl_col_best")}).drop(columns=["created"]),
                     hide_index=True, use_container_width=True)

    st.markdown(f"##### {t('ov_usage')}")
    lim, used = d["limits"], d["used"]
    u1, u2 = st.columns(2)
    for col, key, label in ((u1, "datasets", t("ov_u_datasets")), (u2, "runs", t("ov_u_runs"))):
        with col:
            st.markdown(f"**{label}**: {used[key]} / {lim[key]}")
            st.progress(min(used[key] / lim[key], 1.0))
    st.page_link(st.session_state["_plans_page"], label=t("pi_btn"), icon=":material/workspace_premium:")