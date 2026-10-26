"""Komponen tampilan bersama: CSS, hero, chip, gaya grafik."""
import streamlit as st

from config import CLASS_COLORS, CSS_FILE, EXTRA_COLORS

NAV = {}        # key -> st.Page, diisi app.py agar halaman bisa saling menaut (st.page_link)


def inject_css():
    st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><div class="hero-title">{title}</div><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def chip(text, lvl):
    return f'<span class="chip {lvl}">{text}</span>'


def level(value, warn, bad):
    return "ok" if value < warn else ("warn" if value < bad else "bad")


def class_colors(classes):
    """Warna tetap untuk nama kelas yang dikenal; kelas lain mendapat warna cadangan."""
    out, extra = {}, iter(EXTRA_COLORS * 4)
    for c in classes:
        out[c] = CLASS_COLORS.get(c) or next(extra)
    return out


def style(fig, legend=False):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      font=dict(color="#141F52", size=14), margin=dict(t=30, b=10, l=10, r=10))
    fig.update_yaxes(gridcolor="#E6DEC9")
    return fig


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                      # Streamlit versi lama
        st.plotly_chart(fig, use_container_width=True)


def page_link(key, label, icon=None):
    if key in NAV:
        st.page_link(NAV[key], label=label, icon=icon)


def flash():
    """Pesan sukses sekali tampil (diisi sebelum st.rerun)."""
    msg = st.session_state.pop("_flash", None)
    if msg:
        st.success(msg)


def apply_pending():
    """Terapkan nilai widget yang dijadwalkan sebelum rerun (widget belum dibuat di run ini)."""
    for k, v in st.session_state.pop("_pending", {}).items():
        st.session_state[k] = v