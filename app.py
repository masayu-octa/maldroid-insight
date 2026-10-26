"""Titik masuk: konfigurasi halaman, tema, navigasi, pemilih bahasa."""
import streamlit as st

st.set_page_config(page_title="MalDroid Insight", page_icon=":material/shield:", layout="wide")

from config import DEFAULT_LANG, LANGS          # noqa: E402
from i18n import t                              # noqa: E402
from models import db                           # noqa: E402
from views import components as ui              # noqa: E402
from views.datasets_view import render_datasets  # noqa: E402
from views.home import render_home              # noqa: E402
from views.pipeline_view import render_pipeline  # noqa: E402
from views.models_view import render_models      # noqa: E402
from views.cross_view import render_cross        # noqa: E402
from views.explain_view import render_explain    # noqa: E402
from views.predict_view import render_predict    # noqa: E402
from views.sus_view import render_sus            # noqa: E402

db.init_db()
st.session_state.setdefault("lang", DEFAULT_LANG)
ui.inject_css()

# (kunci, fungsi halaman, ikon)
PAGES = [
    ("home",     render_home,     ":material/space_dashboard:"),
    ("datasets", render_datasets, ":material/dataset:"),
    ("pipeline", render_pipeline, ":material/tune:"),
    ("models",   render_models,   ":material/leaderboard:"),
    ("cross",    render_cross,    ":material/compare_arrows:"),
    ("explain",  render_explain,  ":material/manage_search:"),
    ("predict",  render_predict,  ":material/online_prediction:"),
    ("sus",      render_sus,      ":material/rate_review:"),
]
pages = {key: st.Page(fn, title=t("nav_" + key), icon=icon, url_path=key, default=(key == "home"))
         for key, fn, icon in PAGES}
ui.NAV.update(pages)
nav = st.navigation(list(pages.values()), position="top")
nav.run()

with st.container(key="langbox"):
    st.selectbox("Language", list(LANGS), key="lang", format_func=lambda k: LANGS[k],
                 label_visibility="collapsed")