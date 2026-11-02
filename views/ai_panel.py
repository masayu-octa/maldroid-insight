"""Panel AI Assistant di halaman Explainability: kotak percakapan (penjelasan otomatis + chatbot Gemini)."""
import streamlit as st

from controllers import ai_controller as actl
from i18n import t
from models import ai
from models import explain as E

BOX_HEIGHT = 520        # tinggi kotak percakapan (px); isi menggulir bila lebih panjang


def _run_context(run, model, share, classes, feats, sample):
    m = run["holdout"].get(model, {})
    info = {"dataset": run["dataset"], "model": model, "classes": classes, "n_features": len(feats),
            "accuracy": m.get("acc"), "f1": m.get("f1_macro")}
    top = [(f, float(share[f]) * 100) for f in share.sort_values(ascending=False).index[:8]]
    return ai.build_context(info, top, sample)


def _auto_explanation(rid, model, base_ctx):
    """(teks AI atau None, kunci galat atau None). Kegagalan diingat agar halaman tidak menunggu tiap klik."""
    if not actl.available():
        return None, "ai_err_nokey"
    fail_key = f"ai_fail_{rid}_{model}_{actl.lang()}"
    if fail_key in st.session_state:
        return None, st.session_state[fail_key]
    try:
        with st.spinner(t("ai_thinking")):
            return actl.explain(base_ctx), None
    except ai.AIError as e:
        st.session_state[fail_key] = e.key
        return None, e.key


def render_ai_panel(rid, model, run, share, classes, feats, builtin, src, sample=None):
    st.markdown(f"### {t('ai_title')}")
    
    base_ctx = _run_context(run, model, share, classes, feats, None)
    text, err = _auto_explanation(rid, model, base_ctx)

    hist_key = f"ai_chat_{rid}_{model}"
    history = st.session_state.setdefault(hist_key, [])
    used = st.session_state.get("ai_used", 0)
    left = max(0, actl.FREE_QUESTIONS - used)
    can_chat = actl.available()

    # ---------- bilah atas: sampel + sisa pertanyaan + hapus ----------
    use_sample = False
    top1, top2, top3 = st.columns([3, 2, 1])
    if sample and can_chat:
        use_sample = top1.checkbox(t("ai_use_sample"), value=True, key=f"ai_use_sample_{rid}_{model}")
    if can_chat:
        top2.caption(t("ai_remaining", n=left, total=actl.FREE_QUESTIONS))
    if history and top3.button(t("ai_clear"), key=f"ai_clear_{rid}_{model}"):
        st.session_state[hist_key] = []
        st.rerun()

    # ---------- kotak percakapan ----------
    with st.container(height=BOX_HEIGHT, border=True):
        with st.chat_message("assistant"):
            st.markdown(f"**{t('ai_summary_title')}**")
            if text:
                st.markdown(text)
                st.caption(t("ai_badge"))
            else:
                if err and err != "ai_err_nokey":
                    st.warning(t(err))
                    if st.button(t("ai_retry"), key=f"ai_retry_{rid}_{model}"):
                        st.session_state.pop(f"ai_fail_{rid}_{model}_{actl.lang()}", None)
                        st.rerun()
                elif err == "ai_err_nokey":
                    st.info(t("ai_err_nokey"))
                st.caption(t("ai_fallback_note"))
                for key, kw in E.narrative(share, builtin if src == "perm" else None):
                    st.markdown(f"- {t(key, **kw)}")
        for role, msg in history:
            with st.chat_message(role):
                st.markdown(msg)

    # ---------- kolom pertanyaan (tepat di bawah kotak) ----------
    if not can_chat:
        return
    if left == 0:
        st.info(t("ai_limit"))
        return
    with st.container():
        q = st.chat_input(t("ai_chat_hint"), key=f"ai_input_{rid}_{model}")
    if q:
        ctx = _run_context(run, model, share, classes, feats, sample if use_sample else None)
        try:
            with st.spinner(t("ai_thinking")):
                answer = actl.ask(ctx, history, q)
        except ai.AIError as e:
            st.error(t(e.key))
            return
        history += [("user", q), ("assistant", answer)]
        st.session_state["ai_used"] = used + 1
        st.rerun()