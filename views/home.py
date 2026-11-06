"""Halaman Dashboard (landing page): judul, tujuan, cara memakai, panduan tiap fitur, ringkasan ruang kerja."""
import plotly.graph_objects as go
import streamlit as st

from controllers import dataset_controller as ctl
from i18n import t
from views.components import chip, class_colors, page_link, show, style

FEATURES = ["datasets", "pipeline", "explain", "predict", "sus"]
FEATURE_ICON = {"datasets": ":material/dataset:", "pipeline": ":material/tune:",
                "models": ":material/leaderboard:", "cross": ":material/compare_arrows:",
                "explain": ":material/manage_search:", "predict": ":material/online_prediction:",
                "sus": ":material/rate_review:"}
# (kunci langkah, halaman tujuan, butuh: None | "ds" | "run" | "two")
STEPS = [("prep", "datasets", None), ("add", "datasets", None), ("check", "datasets", "ds"),
         ("run", "pipeline", "ds"),
         ("explain", "explain", "run"), ("predict", "predict", "run"), ("sus", "sus", None)]
ACCENT = {"datasets": "#254F73", "pipeline": "#3C7A5A", "models": "#C9983A", "cross": "#6B7FB0",
          "explain": "#B0583A", "predict": "#2F8F9D", "sus": "#8A5FA8"}
GLOSSARY = ["log1p", "scaler", "smote", "leak", "holdout", "cv", "perm", "f1m"]
FAQ = ["install", "format", "toogood", "which", "av", "store"]


def render_home():
    s = ctl.summary()
    _landing()
    _pillars()
    _progress(s)
    _how_to(s)
    _feature_cards()
    _feature_guide()
    _pipeline_section()
    _workspace(s)
    _help_sections()


def _landing():
    st.markdown(
        f'<div class="landing"><div class="landing-kicker">{t("lp_kicker")}</div>'
        f'<div class="landing-title">MalDroid Insight</div><p>{t("lp_tagline")}</p></div>',
        unsafe_allow_html=True)
    with st.container(key="cta"):
        c1, c2, c3 = st.columns(3)
        with c1:
            page_link("datasets", t("lp_cta_start"), ":material/upload_file:")
        with c2:
            page_link("pipeline", t("lp_cta_run"), ":material/tune:")
        with c3:
            page_link("sus", t("lp_cta_rate"), ":material/rate_review:")
    st.markdown(f'<div class="lead">{t("hm_what")}</div>', unsafe_allow_html=True)


def _pillars():
    cards = "".join(
        f'<div class="pillar"><b>{t(f"lp_p{n}_t")}</b><div>{t(f"lp_p{n}_d")}</div></div>'
        for n in ("1", "2", "3"))
    st.markdown(f'<div class="pillars">{cards}</div>', unsafe_allow_html=True)


def _done_flags(s):
    ds, run = s["n_datasets"] > 0, s["n_runs"] > 0
    return {"prep": ds, "add": ds, "check": ds, "run": run,
            "explain": run, "predict": run, "sus": False}


def _progress(s):
    flags = _done_flags(s)
    n, total = sum(flags.values()), len(flags)
    pct = int(n / total * 100)
    st.markdown(
        f'<div class="prog"><div class="prog-top"><b>{t("lp_prog_title")}</b>'
        f'<span>{t("lp_prog_val", n=n, total=total)}</span></div>'
        f'<div class="prog-bar"><div class="prog-fill" style="width:{pct}%"></div></div></div>',
        unsafe_allow_html=True)


def _feature_cards():
    cards = "".join(
        f'<div class="fcard" style="--c:{ACCENT[f]}"><div class="fcard-name">{t("nav_" + f)}</div>'
        f'<div class="fcard-desc">{t(f"lp_{f}_what")}</div></div>' for f in FEATURES)
    st.markdown(f'<div class="fcards">{cards}</div>', unsafe_allow_html=True)


def _how_to(s):
    st.markdown(f"### {t('lp_how_title')}")
    st.caption(t("lp_how_sub"))
    ready = {None: None, "ds": s["n_datasets"] > 0, "run": s["n_runs"] > 0, "two": s["n_datasets"] > 1}
    with st.container(key="steps2"):
        for row in range(0, len(STEPS), 2):
            chunk = STEPS[row:row + 2]
            cols = st.columns(2) if len(chunk) == 2 else st.columns([1, 2, 1])[1:2]   # kartu tunggal di tengah
            for col, (i, (key, page, need)) in zip(cols, enumerate(chunk, row + 1)):
                with col, st.container(border=True):
                    st.markdown(f'<span class="stepmark c-{page}"></span>', unsafe_allow_html=True)
                    badge = ""
                    if ready[need] is not None:
                        badge = " " + chip(t("hm_done") if ready[need] else t("hm_todo"),
                                           "ok" if ready[need] else "todo")
                    st.markdown(
                        f'<span class="stepno" style="background:{ACCENT[page]}">{i}</span>'
                        f'<b>{t("lp_step_" + key + "_t")}</b>{badge}', unsafe_allow_html=True)
                    st.caption(t("lp_step_" + key + "_d"))
                    page_link(page, t("hm_open"), FEATURE_ICON[page])
    st.divider()


def _feature_guide():
    st.markdown(f"### {t('lp_feat_title')}")
    st.caption(t("lp_feat_sub"))
    tabs = st.tabs([t("nav_" + f) for f in FEATURES])
    for tab, f in zip(tabs, FEATURES):
        with tab:
            st.markdown(f'<div class="fhead" style="--c:{ACCENT[f]}">{t(f"lp_{f}_what")}</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{t('lp_h_do')}**")
                st.markdown(t(f"lp_{f}_do"))
            with c2:
                st.markdown(f"**{t('lp_h_get')}**")
                st.markdown(t(f"lp_{f}_get"))
            st.info(t(f"lp_{f}_note"))
            page_link(f, t("lp_goto", name=t("nav_" + f)), FEATURE_ICON[f])


FLOW = [("log1p", "log1p", "#254F73"), ("scaler", "RobustScaler", "#2F8F9D"), ("smote", "SMOTE", "#C9983A"),
        ("model", "Model", "#3C7A5A"), ("eval", "Evaluate", "#8A5FA8")]


def _flow_css():
    """CSS bulatan pipeline yang bisa diklik (disisipkan dari sini, tidak bergantung pada theme.css)."""
    css = """
.st-key-flowrow [data-testid="stHorizontalBlock"]{align-items:center;gap:0.4rem;}
.st-key-flowrow [data-testid="stColumn"]{display:flex;justify-content:center;align-items:center;}
.st-key-flowrow [data-testid="stColumn"] [data-testid="stVerticalBlock"]{align-items:center;width:100%;}
.st-key-flowrow [data-testid="stElementContainer"],.st-key-flowrow [data-testid="stButton"]{width:auto !important;}
.st-key-flowrow button{border:2px solid var(--c) !important;border-radius:999px !important;background:#FFFFFF !important;
  padding:8px 20px 8px 10px !important;min-height:0;gap:8px;width:auto !important;justify-content:center;
  transition:transform .18s ease,box-shadow .18s ease,background .18s ease;}
.st-key-flowrow button *{color:var(--c) !important;font-weight:700 !important;font-size:1rem !important;}
.st-key-flowrow button:hover{transform:translateY(-3px);box-shadow:0 8px 16px rgba(20,31,82,.18);}
.flow-arrow{color:#7A8DB3;font-size:1.4rem;text-align:center;width:100%;}
.flowinfo{border-left:6px solid var(--c);background:linear-gradient(90deg,#F3EBDA,#FBF7EC);border-radius:10px;padding:12px 18px;margin:8px 0 16px;}
.flowinfo b{color:var(--c);font-size:1.02rem;}
.flowinfo p{margin:4px 0 0;color:#2B3566;}
"""
    for i, (k, _, c) in enumerate(FLOW, 1):
        for state in ("on", "off"):
            cls = f".st-key-fl_{k}_{state}"
            css += f"{cls}{{--c:{c};}}\n"
            css += (f"{cls} button::before{{content:\"{i}\";display:inline-flex;align-items:center;justify-content:center;"
                    f"width:24px;height:24px;border-radius:50%;background:{c};color:#FFFFFF;font-size:.8rem;font-weight:700;}}\n")
        css += (f".st-key-fl_{k}_on button{{background:{c} !important;transform:scale(1.07);"
                f"box-shadow:0 8px 16px rgba(20,31,82,.22);}}\n"
                f".st-key-fl_{k}_on button *{{color:#FFFFFF !important;}}\n"
                f".st-key-fl_{k}_on button::before{{background:#FFFFFF;color:{c};}}\n")
    return "<style>" + css + "</style>"


def _pick_step(k):
    st.session_state["flow_sel"] = k


def _pipeline_section():
    st.markdown(f"### {t('lp_pipe_title')}")
    st.markdown(t("hm_pipe"))
    sel = st.session_state.get("flow_sel", FLOW[0][0])
    st.markdown(_flow_css(), unsafe_allow_html=True)
    with st.container(key="flowrow"):
        widths = [1.5, 2.5, 1.8, 1.7, 2.1]          # lebar kolom mengikuti panjang teks
        cols = st.columns([1.1] + [x for w in widths for x in (w, 0.45)][:-1] + [1.1])
        for i, (k, name, _) in enumerate(FLOW):
            with cols[i * 2 + 1]:
                st.button(name, key=f"fl_{k}_{'on' if k == sel else 'off'}", on_click=_pick_step, args=(k,))
            if i < len(FLOW) - 1:
                with cols[i * 2 + 2]:
                    st.markdown('<div class="flow-arrow">&#10140;</div>', unsafe_allow_html=True)
    color = {k: c for k, _, c in FLOW}
    st.markdown(
        f'<div class="flowinfo" style="--c:{color[sel]}"><b>{t("pl_step_" + sel + "_t")}</b>'
        f'<p>{t("pl_step_" + sel)}</p></div>', unsafe_allow_html=True)
    st.markdown(f"**{t('lp_fastfull_t')}**: {t('lp_fastfull_d')}")

def _workspace(s):
    st.markdown(f"### {t('lp_ws_title')}")
    best = f"{s['best_acc'] * 100:.2f}%" if s["best_acc"] else "-"
    kpis = [(t("hm_kpi_datasets"), s["n_datasets"], "#254F73"),
            (t("hm_kpi_samples"), f"{s['n_samples']:,}", "#2F8F9D"),
            (t("hm_kpi_runs"), s["n_runs"], "#C9983A"), (t("hm_kpi_best"), best, "#3C7A5A")]
    cards = "".join(f'<div class="kpi" style="--c:{c}"><div class="kpi-l">{l}</div><div class="kpi-v">{v}</div></div>'
                    for l, v, c in kpis)
    st.markdown(f'<div class="kpis">{cards}</div>', unsafe_allow_html=True)
    if not s["datasets"]:
        st.info(t("hm_none"))
        page_link("datasets", t("hm_start"))
        return
    classes = sorted({c for d in s["datasets"] for c in d["class_counts"]})
    colors = class_colors(classes)
    names = [d["name"] for d in s["datasets"]]
    fig = go.Figure()
    for c in classes:
        fig.add_trace(go.Bar(name=c, y=names, x=[d["class_counts"].get(c, 0) for d in s["datasets"]],
                             orientation="h", marker_color=colors[c]))
    fig.update_layout(barmode="stack", title=t("lp_ws_chart"), height=max(200, 70 * len(names) + 90))
    show(style(fig, legend=True))


def _help_sections():
    st.markdown(f"### {t('lp_good_title')}")
    st.markdown(t("lp_good"))
    with st.expander(t("lp_faq_title")):
        for q in FAQ:
            st.markdown(f"**{t('lp_faq_' + q + '_q')}**")
            st.markdown(t("lp_faq_" + q + "_a"))
    with st.expander(t("hm_gloss_title")):
        for g in GLOSSARY:
            st.markdown(f"**{t(f'gl_{g}_t')}**: {t(f'gl_{g}_d')}")