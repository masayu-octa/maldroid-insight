"""Titik masuk: konfigurasi halaman, tema, navigasi, pemilih bahasa."""
import streamlit as st

st.set_page_config(page_title="MalDroid Insight", page_icon=":material/shield:", layout="wide")

from config import DEFAULT_LANG, LANGS          # noqa: E402
from i18n import t                              # noqa: E402
from models import db                           # noqa: E402
from views import components as ui              # noqa: E402
from views import placeholder                   # noqa: E402
from views.datasets_view import render_datasets  # noqa: E402
from views.home import render_home              # noqa: E402
from views.pipeline_view import render_pipeline  # noqa: E402

db.init_db()
st.session_state.setdefault("lang", DEFAULT_LANG)
ui.inject_css()

# (kunci, fungsi halaman, ikon). Halaman placeholder diganti saat tahapnya selesai.
PAGES = [
    ("home",     render_home,                       ":material/space_dashboard:"),
    ("datasets", render_datasets,                   ":material/dataset:"),
        ("pipeline", render_pipeline,                   ":material/tune:"),
    ("models",   placeholder.make("models", 3),     ":material/leaderboard:"),
    ("cross",    placeholder.make("cross", 3),      ":material/compare_arrows:"),
    ("explain",  placeholder.make("explain", 4),    ":material/manage_search:"),
    ("predict",  placeholder.make("predict", 4),    ":material/online_prediction:"),
]
pages = {key: st.Page(fn, title=t("nav_" + key), icon=icon, url_path=key, default=(key == "home"))
         for key, fn, icon in PAGES}
ui.NAV.update(pages)
nav = st.navigation(list(pages.values()), position="top")
nav.run()

with st.container(key="langbox"):
    st.selectbox("Language", list(LANGS), key="lang", format_func=lambda k: LANGS[k],
                 label_visibility="collapsed")