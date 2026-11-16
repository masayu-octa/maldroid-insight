"""Halaman Paket & harga: perbandingan paket, kelola langganan, riwayat tagihan."""
import pandas as pd
import streamlit as st

from config import GUEST_TTL_HOURS, LIMITS, PRO_PRICES
from controllers import ai_controller as actl
from controllers import payment_controller as pay
from i18n import t
from views.auth_view import current_user
from views.components import hero


def _features(lim, ai_line):
    return [t("pp_f_datasets", n=lim["datasets"]), t("pp_f_runs", n=lim["runs"]),
            t("pp_f_mb", n=lim["mb"]), t("pp_f_rows", n=f'{lim["rows"]:,}'),
            t("pp_f_keep"), t("pp_f_pred"), ai_line]


def _cancel_dialog():
    @st.dialog(t("pp_cancel_title"))
    def body():
        st.write(t("pp_cancel_text", n=actl.FREE_QUESTIONS))
        c1, c2 = st.columns(2)
        if c1.button(t("pp_cancel_keep"), use_container_width=True):
            st.rerun()
        if c2.button(t("pp_cancel_confirm"), type="primary", use_container_width=True):
            st.session_state["user"] = pay.cancel(current_user()["id"])
            st.session_state["_flash"] = t("pp_cancelled")
            st.rerun()
    body()


def _go_checkout():
    st.session_state.pop("_receipt", None)
    st.switch_page(st.session_state["_checkout_page"])


def _plan_card(col, plan, user):
    lim = LIMITS["user" if plan == "free" else "pro"]
    here = user["plan"] == plan
    ai_line = t("pp_f_ai_free", n=actl.FREE_QUESTIONS) if plan == "free" else t("pp_f_ai_pro")
    price = "$0" if plan == "free" else pay.money(PRO_PRICES["monthly"])
    with col, st.container(border=True, key=f"plan_{plan}"):
        badge = f'<span class="pp-badge">{t("pp_badge_now")}</span>' if here else ""
        items = "".join(f"<li>{x}</li>" for x in _features(lim, ai_line))
        yearly = (f'<div class="pp-year">{t("pp_year_hint", price=pay.money(PRO_PRICES["yearly"]))}</div>'
                  if plan == "pro" else '<div class="pp-year">&nbsp;</div>')
        st.markdown(
            f'<div class="pp-name">{t("auth_plan_" + plan)}{badge}</div>'
            f'<div class="pp-price">{price}<span>{t("pp_month")}</span></div>{yearly}'
            f'<div class="pp-tag">{t("pp_tag_" + plan)}</div><ul class="pp-list">{items}</ul>',
            unsafe_allow_html=True)
        if here:
            st.button(t("pp_btn_current"), key=f"pp_btn_{plan}", disabled=True, use_container_width=True)
        elif plan == "pro":
            if st.button(t("pp_btn_upgrade"), key="pp_btn_pro", type="primary", use_container_width=True):
                _go_checkout()
        else:
            if st.button(t("pp_btn_down"), key="pp_btn_free", use_container_width=True):
                _cancel_dialog()


def _manage(user):
    renews, last = pay.renewal(user["id"])
    with st.container(border=True, key="pp_manage"):
        st.markdown(f'<div class="pp-m-title">{t("pp_manage")}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        with c1:
            if renews:
                st.markdown(f'<div class="pp-m-line">{t("pp_renews", d=renews)}</div>'
                            f'<div class="pp-m-sub">{t("pp_card_on", brand=last["brand"], last4=last["last4"])}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pp-m-line">{t("auth_plan_pro")}</div>', unsafe_allow_html=True)
        with c2:
            if st.button(t("pp_cancel_btn"), key="pp_cancel_btn", use_container_width=True):
                _cancel_dialog()


def _history(user):
    rows = pay.history(user["id"])
    if not rows:
        return
    st.markdown(f"##### {t('pp_history')}")
    df = pd.DataFrame([{t("pp_h_date"): r["created_at"], t("pp_h_receipt"): r["receipt"],
                        t("pp_h_period"): t("pp_" + r["period"]), t("pp_h_amount"): pay.money(r["amount"]),
                        t("pp_h_card"): f'{r["brand"]} **** {r["last4"]}',
                        t("pp_h_status"): t("pp_status_" + r["status"])} for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_plans():
    u = current_user()
    hero(t("nav_plans"), t("nd_plans"))
    msg = st.session_state.pop("_flash", None)
    if msg:
        st.success(msg)
    st.markdown(f'<div class="pp-now">{t("pp_current", plan=t("auth_plan_" + u["plan"]))}</div>',
                unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    _plan_card(cols[0], "free", u)
    _plan_card(cols[1], "pro", u)

    if u["plan"] == "pro":
        _manage(u)
    _history(u)

    st.markdown(f"##### {t('pp_compare')}")
    g, f, p = LIMITS["guest"], LIMITS["user"], LIMITS["pro"]
    rows = [(t("pi_datasets"), g["datasets"], f["datasets"], p["datasets"]),
            (t("pi_runs"), g["runs"], f["runs"], p["runs"]),
            (t("pi_mb"), f'{g["mb"]} MB', f'{f["mb"]} MB', f'{p["mb"]} MB'),
            (t("pi_rows"), f'{g["rows"]:,}', f'{f["rows"]:,}', f'{p["rows"]:,}'),
            (t("pi_keep"), f"{GUEST_TTL_HOURS} h", t("pi_forever"), t("pi_forever")),
            (t("pi_pred"), t("pi_no"), t("pi_yes"), t("pi_yes")),
            (t("pi_ai"), t("pi_no"), t("pi_ai_free", n=actl.FREE_QUESTIONS), t("pi_ai_pro"))]
    df = pd.DataFrame(rows, columns=[t("pi_col_feat"), t("pi_col_guest"), t("auth_plan_free"),
                                     t("auth_plan_pro")]).astype(str)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.markdown(f"##### {t('pp_faq')}")
    for n in ("1", "2", "3"):
        st.markdown(f"**{t('pp_q' + n)}**")
        st.markdown(t("pp_a" + n))