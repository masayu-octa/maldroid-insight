"""Halaman Paket & harga: perbandingan Free vs Pro, ganti paket (simulasi, tanpa pembayaran)."""
import pandas as pd
import streamlit as st

from config import GUEST_TTL_HOURS, LIMITS, PRO_PRICE
from controllers import auth_controller as ctl
from i18n import t
from views.auth_view import current_user
from views.components import hero


def _switch(plan):
    user = ctl.change_plan(current_user()["id"], plan)
    st.session_state["user"] = user
    st.toast(t("pp_switched", plan=t("auth_plan_" + plan)))


def _features(lim):
    return [t("pp_f_datasets", n=lim["datasets"]), t("pp_f_runs", n=lim["runs"]),
            t("pp_f_mb", n=lim["mb"]), t("pp_f_rows", n=f'{lim["rows"]:,}'),
            t("pp_f_keep"), t("pp_f_ai")]


def render_plans():
    u = current_user()
    hero(t("nav_plans"), t("nd_plans"))
    st.markdown(f'<div class="pp-now">{t("pp_current", plan=t("auth_plan_" + u["plan"]))}</div>',
                unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for col, plan in zip(cols, ("free", "pro")):
        lim = LIMITS["user" if plan == "free" else "pro"]
        price = "$0" if plan == "free" else PRO_PRICE
        here = u["plan"] == plan
        with col, st.container(border=True, key=f"plan_{plan}"):
            badge = f'<span class="pp-badge">{t("pp_badge_now")}</span>' if here else ""
            items = "".join(f"<li>{x}</li>" for x in _features(lim))
            st.markdown(
                f'<div class="pp-name">{t("auth_plan_" + plan)}{badge}</div>'
                f'<div class="pp-price">{price}<span>{t("pp_month")}</span></div>'
                f'<div class="pp-tag">{t("pp_tag_" + plan)}</div><ul class="pp-list">{items}</ul>',
                unsafe_allow_html=True)
            if here:
                st.button(t("pp_btn_current"), key=f"pp_btn_{plan}", disabled=True, use_container_width=True)
            elif plan == "pro":
                st.button(t("pp_btn_upgrade"), key="pp_btn_pro", type="primary", use_container_width=True,
                          on_click=_switch, args=("pro",))
            else:
                st.button(t("pp_btn_down"), key="pp_btn_free", use_container_width=True,
                          on_click=_switch, args=("free",))
    st.caption(t("pp_demo"))
    if u["plan"] == "free":
        st.caption(t("pp_note_up"))

    st.markdown(f"##### {t('pp_compare')}")
    g, f, p = LIMITS["guest"], LIMITS["user"], LIMITS["pro"]
    rows = [(t("pi_datasets"), g["datasets"], f["datasets"], p["datasets"]),
            (t("pi_runs"), g["runs"], f["runs"], p["runs"]),
            (t("pi_mb"), f'{g["mb"]} MB', f'{f["mb"]} MB', f'{p["mb"]} MB'),
            (t("pi_rows"), f'{g["rows"]:,}', f'{f["rows"]:,}', f'{p["rows"]:,}'),
            (t("pi_keep"), f"{GUEST_TTL_HOURS} h", t("pi_forever"), t("pi_forever")),
            (t("pi_ai"), t("pi_no"), t("pi_yes"), t("pi_yes"))]
    df = pd.DataFrame(rows, columns=[t("pi_col_feat"), t("pi_col_guest"), t("auth_plan_free"),
                                     t("auth_plan_pro")]).astype(str)
    st.dataframe(df, hide_index=True, use_container_width=True)
    with st.expander(t("pp_faq")):
        for n in ("1", "2", "3"):
            st.markdown(f"**{t('pp_q' + n)}**")
            st.markdown(t("pp_a" + n))