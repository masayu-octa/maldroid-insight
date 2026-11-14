"""Susunan menu: navbar atas (tamu) dan sidebar kiri dengan grup dropdown (sudah login)."""
import streamlit as st

from i18n import t
from views.auth_view import profile_bottom, profile_top

ICON = {"home": ":material/space_dashboard:", "datasets": ":material/dataset:", "pipeline": ":material/tune:",
        "explain": ":material/manage_search:", "predict": ":material/online_prediction:",
        "sus": ":material/rate_review:", "datasets_mine": ":material/folder_open:",
        "datasets_add": ":material/upload_file:", "explain_main": ":material/psychology:",
        "compare": ":material/compare_arrows:", "auth": ":material/login:"}
MENU = [("home", None), ("datasets", ["datasets_mine", "datasets_add"]), ("pipeline", None),
        ("explain", ["explain_main", "compare"]), ("predict", None), ("sus", None)]
# halaman yang butuh login (tamu melihat layar "terkunci")
LOCKED = {"predict", "explain_main", "compare"}
# kolom menu "Features" saat kursor di atasnya: (judul kolom, [halaman])
MEGA = [("nav_col_data", ["datasets_mine", "datasets_add"]),
        ("nav_col_models", ["pipeline", "explain_main", "compare"]),
        ("nav_col_use", ["predict", "sus"])]
LABEL = {"datasets_mine": "ds_view_mine", "datasets_add": "ds_view_add",
         "explain_main": "ex_tab_explain", "compare": "ex_tab_compare"}


def label(key):
    return t(LABEL.get(key, "nav_" + key))


CHEV = ('<svg width="18" height="18" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5" fill="none" '
        'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def guest_topnav(pages):
    """Navbar atas (tamu) di tengah. Menu 'Features' terbuka saat kursor di atasnya (CSS :hover)."""
    with st.container(key="topnav"):
        st.page_link(pages["home"], label=label("home"), icon=ICON["home"])
        with st.container(key="navfeat"):
            st.markdown(f'<span class="nf-trigger">{t("nav_features")}{CHEV}</span>', unsafe_allow_html=True)
            with st.container(key="mega"):
                for col, (head, keys) in zip(st.columns(3), MEGA):
                    with col:
                        st.markdown(f'<div class="mg-head">{t(head)}</div>', unsafe_allow_html=True)
                        for k in keys:
                            lock = k in LOCKED
                            with st.container(key=f"mi_{k}"):      # satu kotak per fitur (nama + deskripsi)
                                st.page_link(pages[k], label=label(k), icon=":material/lock:" if lock else ICON[k])
                                st.markdown(f'<div class="mg-desc">{t("nd_" + k)}</div>', unsafe_allow_html=True)


def sidebar(pages, current):
    """Sidebar kiri: profil di atas, menu di tengah, info paket dan keluar di bawah."""
    with st.sidebar:
        profile_top()
        for key, kids in MENU:
            if not kids:
                st.page_link(pages[key], label=label(key), icon=ICON[key])
                continue
            with st.expander(label(key), icon=ICON[key], expanded=current in [pages[k] for k in kids]):
                for k in kids:
                    st.page_link(pages[k], label=label(k), icon=ICON[k])
        st.divider()
        profile_bottom()