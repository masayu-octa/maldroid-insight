import streamlit as st

st.set_page_config(page_title="MalDroid Insight", page_icon=":material/shield:", layout="wide")

from config import DEFAULT_LANG, LANGS
from controllers import maintenance_controller
from i18n import t
from models import db
from views import components as ui
from views import guest
from views import nav as menu
from views.auth_view import render_auth, render_locked
from views.checkout_view import render_checkout
from views.compare_page import render_compare
from views.datasets_view import render_add, render_mine
from views.explain_view import render_explain
from views.faq_view import render_faq
from views.home import render_home
from views.overview_view import render_overview
from views.pipeline_view import render_pipeline
from views.plans_view import render_plans
from views.predict_view import render_predict
from views.profile_view import render_profile
from views.sus_view import render_sus

db.init_db()
st.session_state.setdefault("lang", DEFAULT_LANG)
st.session_state.setdefault("user", None)      # None = tamu; kalau login: {id, email, name, plan, created_at}
ui.inject_css()

FUNCS = {"home": render_home, "overview": render_overview, "datasets_mine": render_mine, "datasets_add": render_add,
         "pipeline": render_pipeline, "explain_main": render_explain, "compare": render_compare,
         "predict": render_predict, "sus": render_sus, "faq": render_faq, "auth": render_auth,
         "profile": render_profile, "plans": render_plans, "checkout": render_checkout}
HIDDEN = {"auth", "profile", "plans", "checkout"}          # halaman yang tidak muncul di menu (dibuka lewat tombol)


def gate(key, fn):
    """Tamu yang membuka halaman terkunci (lewat menu atau URL) melihat layar 'butuh login'."""
    def run():
        if key in menu.LOCKED and st.session_state.get("user") is None:
            render_locked(key)
        else:
            fn()
    run.__name__ = "page_" + key
    return run


pages = {k: st.Page(gate(k, fn), title=menu.label(k), icon=menu.ICON[k], url_path=k, default=(k == "home"),
                    visibility="hidden" if k in HIDDEN else "visible")
         for k, fn in FUNCS.items()}
# alias agar st.page_link("datasets"/"explain", ...) di halaman lain tetap jalan
ui.NAV.update(pages)
ui.NAV["datasets"] = pages["datasets_mine"]
ui.NAV["explain"] = pages["explain_main"]
st.session_state["_home_page"] = pages["home"]
st.session_state["_auth_page"] = pages["auth"]
st.session_state["_plans_page"] = pages["plans"]
st.session_state["_checkout_page"] = pages["checkout"]
logged_in = st.session_state["user"] is not None

if logged_in:
    nav = st.navigation([p for k, p in pages.items() if k != "auth"], position="hidden")
else:
    nav = st.navigation(list(pages.values()), position="hidden")
# pemilik data + paketnya. Harus setelah st.navigation: untuk tamu, run berhenti di sini sampai token browser
# diterima, dan alamat halaman (mis. /pipeline) tetap terjaga setelah refresh.
db.set_owner(guest.resolve_owner(), (st.session_state["user"] or {}).get("plan"))
maintenance_controller.purge_guests()          # hapus data tamu yang lewat 24 jam
if logged_in:
    menu.sidebar(pages, nav)
else:
    menu.guest_topnav(pages)
    with st.container(key="authbox"):          # tombol Masuk / Daftar di ujung kanan navbar
        st.page_link(pages["auth"], label=t("nav_auth"), icon=menu.ICON["auth"])
nav.run()

with st.container(key="langbox"):
    st.selectbox("Language", list(LANGS), key="lang", format_func=lambda k: LANGS[k], label_visibility="collapsed")
    if not logged_in:      # setelah logout Streamlit menyisakan sidebar kosong: sembunyikan untuk tamu
        st.markdown('<style>[data-testid="stSidebar"],[data-testid="stExpandSidebarButton"]{display:none !important;}'
                    '.stApp:has([data-testid="stSidebar"]) [data-testid="stHeader"]::before{left:28px !important;}</style>', unsafe_allow_html=True)