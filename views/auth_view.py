"""Halaman Masuk / Daftar, kotak akun di sidebar, kartu terkunci, dan kartu info tamu."""
import streamlit as st
import html

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


def logout():
    st.session_state["user"] = None


def _start(user):
    # data milik tamu di browser ini ikut pindah ke akun
    moved = db.claim(db.get_owner(), f"u:{user['id']}") if db.owner_kind() == "guest" else (0, 0)
    st.session_state["user"] = user
    st.session_state["_auth_fail"] = 0
    if moved[0] or moved[1]:
        st.toast(t("auth_claimed", d=moved[0], r=moved[1]))
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

def _initials(name):
    parts = [p for p in (name or "?").split() if p]
    return ("".join(p[0] for p in parts[:2]) or "?").upper()


def profile_top():
    """Menu profil di atas sidebar: avatar, nama, email, paket."""
    u = current_user()
    if not u:
        return
    st.markdown(
        f'<div class="pr-card"><div class="pr-av">{html.escape(_initials(u["name"]))}</div>'
        f'<div class="pr-info"><b>{html.escape(u["name"] or "")}</b>'
        f'<span>{html.escape(u["email"])}</span>'
        f'<em>{t("auth_plan_" + u["plan"])}</em></div></div>', unsafe_allow_html=True)


def plans_dialog():
    """Jendela informasi paket: perbandingan batas tamu vs akun."""
    from config import GUEST_TTL_HOURS, LIMITS
    import pandas as pd

    @st.dialog(t("pi_title"))
    def _body():
        g, u = LIMITS["guest"], LIMITS["user"]
        rows = [
            (t("pi_datasets"), g["datasets"], u["datasets"]),
            (t("pi_runs"), g["runs"], u["runs"]),
            (t("pi_mb"), f'{g["mb"]} MB', f'{u["mb"]} MB'),
            (t("pi_rows"), f'{g["rows"]:,}', f'{u["rows"]:,}'),
            (t("pi_keep"), f"{GUEST_TTL_HOURS} h", t("pi_forever")),
            (t("pi_ai"), t("pi_no"), t("pi_yes")),
        ]
        df = pd.DataFrame(rows, columns=[t("pi_col_feat"), t("pi_col_guest"), t("pi_col_user")]).astype(str)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.info(t("pi_pro"))

    _body()


def profile_bottom():
    """Bagian bawah sidebar: tombol info paket dan keluar."""
    if not current_user():
        return
    if st.button(t("pi_btn"), key="sb_plans", icon=":material/workspace_premium:", use_container_width=True):
        plans_dialog()
    st.button(t("auth_logout"), on_click=logout, key="sb_logout", icon=":material/logout:",
              use_container_width=True)


def account_box():
    """Kompatibilitas lama: sekarang sama dengan bagian bawah."""
    profile_bottom()

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