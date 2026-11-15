"""Halaman Profil: ringkasan akun, ubah nama, ganti password, pemakaian, hapus akun."""
import html

import streamlit as st

from config import LIMITS
from controllers import auth_controller as ctl
from i18n import t
from models import db
from views.auth_view import current_user, initials, logout
from views.components import hero


def render_profile():
    u = current_user()
    hero(t("nav_profile"), t("nd_profile"))
    since = (u.get("created_at") or "")[:10] or "-"
    st.markdown(
        f'<div class="pr-big"><div class="pr-av pr-av-l">{html.escape(initials(u["name"]))}</div>'
        f'<div class="pr-info"><b>{html.escape(u["name"] or "")}</b><span>{html.escape(u["email"])}</span>'
        f'<em>{t("auth_plan_" + u["plan"])}</em></div>'
        f'<div class="pr-since">{t("pr_since", d=since)}</div></div>', unsafe_allow_html=True)

    tab_acc, tab_sec, tab_use = st.tabs([t("pr_tab_account"), t("pr_tab_security"), t("pr_tab_usage")])
    with tab_acc:
        with st.form("pr_name_form"):
            name = st.text_input(t("auth_name"), value=u["name"] or "", key="pr_name")
            st.text_input(t("auth_email"), value=u["email"], disabled=True, key="pr_email")
            if st.form_submit_button(t("pr_save"), type="primary"):
                user, err = ctl.update_name(u["id"], name)
                if err:
                    st.error(t(err))
                else:
                    st.session_state["user"] = user
                    st.toast(t("pr_saved"))
                    st.rerun()
        st.page_link(st.session_state["_plans_page"], label=t("pi_btn"), icon=":material/workspace_premium:")
    with tab_sec:
        with st.form("pr_pw_form", clear_on_submit=True):
            old = st.text_input(t("pr_old_pw"), type="password", key="pr_old")
            new = st.text_input(t("pr_new_pw"), type="password", key="pr_new", help=t("auth_pw_help"))
            new2 = st.text_input(t("auth_password2"), type="password", key="pr_new2")
            if st.form_submit_button(t("pr_change_pw"), type="primary"):
                ok, err = ctl.change_password(u["id"], old, new, new2)
                if err:
                    st.error(t(err))
                else:
                    st.success(t("pr_pw_changed"))
        with st.expander(t("pr_danger")):
            st.warning(t("pr_delete_warn"))
            with st.form("pr_del_form"):
                pw = st.text_input(t("pr_confirm_pw"), type="password", key="pr_del_pw")
                sure = st.checkbox(t("pr_delete_ok"), key="pr_del_ok")
                if st.form_submit_button(t("pr_delete_btn")):
                    if not sure:
                        st.error(t("pr_err_confirm"))
                    else:
                        ok, err = ctl.delete_account(u["id"], pw)
                        if err:
                            st.error(t(err))
                        else:
                            logout()
                            st.toast(t("pr_deleted"))
                            st.switch_page(st.session_state["_home_page"])
    with tab_use:
        lim = LIMITS.get(db.tier(), LIMITS["guest"])
        for key, label in (("datasets", t("ov_u_datasets")), ("runs", t("ov_u_runs"))):
            n = db.count(key)
            st.markdown(f"**{label}**: {n} / {lim[key]}")
            st.progress(min(n / lim[key], 1.0))
        st.caption(t("pr_limits", mb=lim["mb"], rows=f'{lim["rows"]:,}'))