"""Panduan dataset di halaman Datasets: template (unduh + visual), ketentuan, dan daftar galat."""
import pandas as pd
import streamlit as st

from controllers import dataset_controller as ctl
from i18n import t
from models import datasets as M
from models import templates as T

ROLE_STYLE = {"id": "#E3E7F0", "feature": "#DCE7F1", "label": "#F4E2B8"}
ERROR_KEYS = [("ad_err_read", {"err": "..."}), ("ad_err_label", {}), ("ad_err_label_many", {"n": M.MAX_LABEL_VALUES}),
              ("ad_err_benign", {}), ("ad_err_feats", {}), ("ad_err_empty", {}), ("ad_err_name", {}),
              ("ad_err_noname", {}), ("pl_err_small", {"c": "N", "k": "K"}),
              ("ad_warn_dropped", {"n": "N"}), ("ad_warn_const", {"n": "N"}), ("ad_warn_small", {"n": M.MIN_CLASS_SAMPLES})]


def render_guide():
    st.markdown(f"#### {t('gd_title')}")
    st.caption(t("gd_sub"))
    tab_t, tab_r, tab_e = st.tabs([t("gd_tab_template"), t("gd_tab_req"), t("gd_tab_err")])
    with tab_t:
        _template()
    with tab_r:
        _requirements()
    with tab_e:
        _errors()


def _legend():
    chips = (f'<span class="chip id">{t("gd_role_id")}</span>'
             f'<span class="chip feat">{t("gd_role_feature")}</span>'
             f'<span class="chip lab">{t("gd_role_label")}</span>')
    st.markdown(f'<div class="legend">{chips}</div>', unsafe_allow_html=True)


def _template():
    st.markdown(t("gd_tpl_intro"))
    _legend()
    kind = st.radio(t("gd_kind"), list(T.KINDS), horizontal=True, key="gd_kind",
                    format_func=lambda k: t("gd_kind_" + k, c=", ".join(T.KINDS[k])))
    df = ctl.template_df(kind)

    def color(col):
        role = "id" if col.name == T.ID_COL else ("label" if col.name == T.LABEL_COL else "feature")
        return [f"background-color: {ROLE_STYLE[role]}"] * len(col)

    st.dataframe(df.head(10).style.apply(color, axis=0), hide_index=True)
    st.caption(t("gd_tpl_caption", r=len(df), c=df.shape[1]))

    guide = pd.DataFrame([{
        t("gd_col_name"): name, t("gd_col_role"): t("gd_role_" + role),
        t("gd_col_meaning"): t("tp_grp_" + grp) if grp else t("gd_meaning_" + role)}
        for name, role, grp in T.column_guide(kind)])
    with st.expander(t("gd_cols_title")):
        st.dataframe(guide, hide_index=True)

    c1, c2 = st.columns(2)
    c1.download_button(t("gd_dl_sample"), ctl.template_csv(kind), file_name=f"maldroid_template_{kind}.csv",
                       mime="text/csv", key=f"gd_dl_{kind}", type="primary")
    c2.download_button(t("gd_dl_blank"), ctl.template_csv(kind, blank=True),
                       file_name=f"maldroid_template_{kind}_blank.csv", mime="text/csv", key=f"gd_dlb_{kind}")
    st.caption(t("gd_tpl_synthetic"))

    with st.expander(t("gd_api_title")):
        st.markdown(t("gd_api_help"))
        ds_id = st.text_input(t("gd_api_id"), value="builetrongduc/CICMalDroid", key="gd_api_id")
        if st.button(t("gd_api_fetch"), key="gd_api_go"):
            try:
                with st.spinner(t("ad_working")):
                    df_r, info = ctl.fetch_online_sample(ds_id)
            except T.RemoteError as e:
                st.session_state.pop("gd_api_res", None)
                st.error(t(e.key, **e.kw))
            else:
                st.session_state["gd_api_res"] = (ds_id, df_r, info)
        res = st.session_state.get("gd_api_res")
        if res:
            rid, df_r, info = res
            st.success(t("gd_api_ok", ds=rid, r=len(df_r), c=info["n_cols"], s=info["split"]))
            st.dataframe(df_r.iloc[:10, :14], hide_index=True)
            if df_r.shape[1] > 14:
                st.caption(t("gd_api_more", n=df_r.shape[1] - 14))
            st.download_button(t("gd_api_dl"), T.to_csv(df_r), file_name="online_sample.csv", mime="text/csv",
                               key="gd_api_dl")


def _requirements():
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### {t('gd_must_title')}")
        st.markdown(t("gd_must"))
    with c2:
        st.markdown(f"##### {t('gd_auto_title')}")
        st.markdown(t("gd_auto"))
    st.markdown(f"##### {t('gd_tips_title')}")
    st.markdown(t("gd_tips"))
    st.markdown(f"##### {t('gd_limits_title')}")
    st.dataframe(pd.DataFrame({
        t("gd_lim_item"): [t("gd_lim1"), t("gd_lim2"), t("gd_lim3"), t("gd_lim4")],
        t("gd_lim_value"): [t("gd_lim1_v"), t("gd_lim2_v", n=M.MAX_LABEL_VALUES),
                            t("gd_lim3_v", n=M.MIN_CLASS_SAMPLES), t("gd_lim4_v")]}), hide_index=True)


def _errors():
    st.markdown(t("gd_err_intro"))
    rows = []
    for key, kw in ERROR_KEYS:
        rows.append({t("gd_err_col_msg"): t(key, **kw), t("gd_err_col_why"): t("gd_e_" + key),
                     t("gd_err_col_fix"): t("gd_f_" + key)})
    st.table(pd.DataFrame(rows))
    st.markdown(f"##### {t('gd_unusable_title')}")
    st.markdown(t("gd_unusable"))