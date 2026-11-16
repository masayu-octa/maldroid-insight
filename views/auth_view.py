"""Halaman Masuk / Daftar, profil di sidebar, kartu terkunci, dan kartu info tamu."""
import html

import streamlit as st

from controllers import auth_controller as ctl
from controllers.auth_controller import MAX_TRIES
from i18n import t
from models import db
from views.components import hero

LOCK = ('<svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="#254F73" stroke-width="1.8" '
        'stroke-linecap="round"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 018 0v3"/></svg>')

_INFO = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#C9983A" stroke-width="2" '
         'stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>')


def current_user():
    return st.session_state.get("user")


_WORK_PREFIX = ("pl_", "mc_", "ex_", "pr_", "cr_", "cmp_", "ds_", "gd_", "ad_", "_pl_", "_ex_", "_upload",
                "_pred_upload", "_pending", "_next_pipeline", "_flash", "_receipt")


def _reset_workspace():
    """Hapus pilihan dataset/run dan hasil sementara milik sesi sebelumnya."""
    for k in [k for k in st.session_state if str(k).startswith(_WORK_PREFIX)]:
        del st.session_state[k]


def logout():
    _reset_workspace()
    st.session_state["user"] = None


def _start(user):
    # akun selalu mulai dari ruang kerja milik akun itu sendiri; data tamu tidak ikut pindah
    _reset_workspace()
    st.session_state["user"] = user
    st.session_state["_auth_fail"] = 0
    st.switch_page(st.session_state["_home_page"])


def render_auth():
    hero(t("auth_title"), t("auth_sub"))
    # sisa dari pendaftaran: isi email di form login dan tampilkan pesan sukses
    if "_li_prefill" in st.session_state:
        st.session_state["li_email"] = st.session_state.pop("_li_prefill")
    reg_msg = st.session_state.pop("_reg_msg", None)
    left, mid, right = st.columns([1, 1.6, 1])
    with mid:
        if reg_msg:
            st.success(reg_msg)
        tab_in, tab_up = st.tabs([t("auth_tab_login"), t("auth_tab_reg")])
        with tab_in:
            with st.form("login_form"):
                email = st.text_input(t("auth_email"), key="li_email")
                pw = st.text_input(t("auth_password"), type="password", key="li_pw")
                go = st.form_submit_button(t("auth_submit_login"), type="primary", use_container_width=True)
            if go:
                if st.session_state.get("_auth_fail", 0) >= MAX_TRIES:
                    st.error(t("auth_err_locked"))
                else:
                    user, err = ctl.login(email, pw)
                    if err:
                        st.session_state["_auth_fail"] = st.session_state.get("_auth_fail", 0) + 1
                        st.error(t(err))
                    else:
                        _start(user)
        with tab_up:
            with st.form("reg_form"):
                name = st.text_input(t("auth_name"), key="rg_name")
                email = st.text_input(t("auth_email"), key="rg_email")
                pw = st.text_input(t("auth_password"), type="password", key="rg_pw", help=t("auth_pw_help"))
                pw2 = st.text_input(t("auth_password2"), type="password", key="rg_pw2")
                go = st.form_submit_button(t("auth_submit_reg"), type="primary", use_container_width=True)
            if go:
                user, err = ctl.register(email, name, pw, pw2)
                if err:
                    st.error(t(err))
                else:
                    # akun sudah dibuat, tapi pengguna harus login sendiri
                    st.session_state["_li_prefill"] = user["email"]
                    st.session_state["_reg_msg"] = t("auth_reg_done")
                    for k in ("rg_name", "rg_email", "rg_pw", "rg_pw2"):
                        st.session_state.pop(k, None)
                    st.rerun()


def initials(name):
    parts = [p for p in (name or "?").split() if p]
    return ("".join(p[0] for p in parts[:2]) or "?").upper()


def profile_top(pages):
    """Menu profil di atas sidebar: seluruh kartu bisa diklik menuju halaman Profil."""
    u = current_user()
    if not u:
        return
    with st.container(key="profcard"):
        st.markdown(
            f'<div class="pr-card"><div class="pr-av">{html.escape(initials(u["name"]))}</div>'
            f'<div class="pr-info"><b>{html.escape(u["name"] or "")}</b>'
            f'<span>{html.escape(u["email"])}</span>'
            f'<em>{t("auth_plan_" + u["plan"])}</em></div></div>', unsafe_allow_html=True)
        st.page_link(pages["profile"], label=t("pr_open"), icon=":material/settings:")


def profile_bottom(pages):
    """Bagian bawah sidebar: info paket dan keluar."""
    if not current_user():
        return
    if st.button(t("pi_btn"), key="sb_plans", icon=":material/workspace_premium:", use_container_width=True):
        st.switch_page(pages["plans"])
    st.button(t("auth_logout"), on_click=logout, key="sb_logout", icon=":material/logout:",
              use_container_width=True)


def locked_card(key=None):
    """Kartu 'butuh login' untuk tamu (dipakai halaman terkunci dan panel AI)."""
    with st.container(border=True, key="lockcard"):
        st.markdown(f'<div class="lk-ico">{LOCK}</div>'
                    f'<div class="lk-title">{t("lock_title")}</div><p class="lk-text">{t("lock_text")}</p>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.page_link(st.session_state["_auth_page"], label=t("lp_cta_signup"), icon=":material/person_add:")
        with c2:
            st.page_link(st.session_state["_home_page"], label=t("lock_back"), icon=":material/space_dashboard:")


def render_locked(key):
    from views import nav as menu
    hero(menu.label(key), t("nd_" + key))
    locked_card(key)


def _gn_set(hidden):
    st.session_state["_gn_hidden"] = hidden


def guest_notice():
    """Kartu info untuk tamu di halaman yang membuat data. Bisa disembunyikan dan dimunculkan lagi."""
    if st.session_state.get("user") is not None:
        return
    from config import GUEST_TTL_HOURS
    if st.session_state.get("_gn_hidden"):
        with st.container(key="gnote_min"):
            st.button(t("gn_show"), key="gn_show_btn", icon=":material/info:", on_click=_gn_set, args=(False,))
        return
    with st.container(key="gnote"):
        c1, c2 = st.columns([8, 2], vertical_alignment="center")
        with c1:
            st.markdown(f'<div class="gn-title">{_INFO}<span>{t("gn_title")}</span></div>'
                        f'<div class="gn-text">{t("guest_note", h=GUEST_TTL_HOURS)}</div>',
                        unsafe_allow_html=True)
        with c2:
            st.button(t("gn_hide"), key="gn_hide_btn", on_click=_gn_set, args=(True,), use_container_width=True)