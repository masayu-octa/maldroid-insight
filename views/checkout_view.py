"""Halaman Checkout: pilih periode tagihan, isi data kartu, bayar paket Pro; layar struk setelah berhasil."""
import time

import pandas as pd
import streamlit as st

from config import LIMITS, PRO_PRICES
from controllers import ai_controller as actl
from controllers import payment_controller as pay
from i18n import t
from models import auth, payments
from views.auth_view import current_user
from views.components import hero, page_link


def _summary(period):
    amount = PRO_PRICES[period]
    lim = LIMITS["pro"]
    save = ""
    if period == "yearly":
        save = f'<div class="co-save">{t("co_save", amount=pay.money(pay.yearly_saving()))}</div>'
    inc = [t("pp_f_datasets", n=lim["datasets"]), t("pp_f_runs", n=lim["runs"]),
           t("pp_f_mb", n=lim["mb"]), t("pp_f_rows", n=f'{lim["rows"]:,}'), t("pp_f_ai_pro")]
    items = "".join(f"<li>{x}</li>" for x in inc)
    with st.container(border=True, key="co_sum"):
        st.markdown(
            f'<div class="co-h">{t("co_sum_title")}</div>'
            f'<div class="co-row"><span>{t("co_row_plan")}</span><b>{t("auth_plan_pro")}</b></div>'
            f'<div class="co-row"><span>{t("co_row_period")}</span><b>{t("pp_" + period)}</b></div>'
            f'<div class="co-row total"><span>{t("co_total")}</span><b>{pay.money(amount)}</b></div>{save}'
            f'<div class="co-inc">{t("co_includes")}</div><ul class="co-list">{items}</ul>',
            unsafe_allow_html=True)


def _test_cards():
    with st.expander(t("co_test_title")):
        st.caption(t("co_test_note"))
        names = {"ok": t("co_test_ok"), "declined": t("co_test_declined"), "funds": t("co_test_funds")}
        rows = [{t("co_number"): payments.format_number(n), t("co_test_result"): names[o]}
                for n, (_, o) in payments.TEST_CARDS.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _success(p):
    hero(t("co_ok_title"), t("co_ok_sub"))
    left, mid, right = st.columns([1, 1.5, 1])
    with mid:
        rows = [(t("co_rc_no"), p["receipt"]), (t("co_rc_date"), p["created_at"]),
                (t("co_rc_plan"), t("auth_plan_pro")), (t("co_rc_billing"), t("pp_" + p["period"])),
                (t("co_rc_amount"), pay.money(p["amount"])),
                (t("co_rc_card"), f'{p["brand"]} **** {p["last4"]}'),
                (t("co_rc_status"), t("pp_status_paid"))]
        body = "".join(f'<div class="co-row"><span>{k}</span><b>{v}</b></div>' for k, v in rows)
        with st.container(border=True, key="co_receipt"):
            st.markdown(f'<div class="co-h">{t("co_rc_title")}</div>{body}', unsafe_allow_html=True)
        st.download_button(t("co_download"), data=pay.receipt_text(p, t), file_name=f'{p["receipt"]}.txt',
                           mime="text/plain", on_click="ignore", use_container_width=True)
        page_link("overview", t("co_go_overview"), ":material/monitoring:", cta=True)
        st.page_link(st.session_state["_plans_page"], label=t("co_go_plans"), icon=":material/workspace_premium:")


def render_checkout():
    u = current_user()
    receipt = st.session_state.get("_receipt")
    if receipt:
        _success(receipt)
        return
    hero(t("nav_checkout"), t("nd_checkout"))
    if u["plan"] == "pro":
        st.info(t("co_already"))
        st.page_link(st.session_state["_plans_page"], label=t("co_go_plans"), icon=":material/workspace_premium:")
        return

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown(f'<div class="co-h">{t("co_period")}</div>', unsafe_allow_html=True)
        period = st.segmented_control(
            t("co_period"), ["monthly", "yearly"], default="monthly", key="co_period", label_visibility="collapsed",
            format_func=lambda k: t("co_monthly") if k == "monthly"
            else t("co_yearly", amount=pay.money(pay.yearly_saving()))) or "monthly"
        with st.container(border=True, key="co_form_box"):
            st.markdown(f'<div class="co-h">{t("co_card_title")}</div>', unsafe_allow_html=True)
            with st.form("co_form", border=False):
                name = st.text_input(t("co_name"), key="co_name", max_chars=60)
                number = st.text_input(t("co_number"), key="co_number", max_chars=23, placeholder="4242 4242 4242 4242")
                c1, c2 = st.columns(2)
                expiry = c1.text_input(t("co_exp"), key="co_exp", max_chars=5, placeholder="MM/YY")
                cvc = c2.text_input(t("co_cvc"), key="co_cvc", max_chars=4, type="password", placeholder="123")
                go = st.form_submit_button(t("co_pay", amount=pay.money(PRO_PRICES[period])), type="primary",
                                           use_container_width=True)
            if go:
                err = pay.validate(name, number, expiry, cvc)
                if err:
                    st.error(t(err))
                else:
                    with st.spinner(t("co_processing")):
                        time.sleep(1.4)
                        payment, err = pay.pay(u["id"], period, name, number, expiry, cvc)
                    if err:
                        st.error(t(err))
                    else:
                        for k in ("co_name", "co_number", "co_exp", "co_cvc"):
                            st.session_state.pop(k, None)
                        st.session_state["user"] = auth.get(u["id"])
                        st.session_state["_receipt"] = payment
                        st.rerun()
        _test_cards()
    with right:
        _summary(period)