"""Controller dataset: menghubungkan input UI dengan models (db + datasets)."""
import numpy as np
import pandas as pd
import streamlit as st

from models import artifacts
from models import datasets as M
from models import db


def list_datasets():
    return db.list_datasets()


def get_meta(ds_id):
    return db.get_dataset(ds_id)


@st.cache_resource(show_spinner=False, max_entries=4)
def load(ds_id):
    """(df, fitur, info). Dikembalikan tanpa salinan: jangan diubah oleh pemanggil."""
    meta = db.get_dataset(ds_id)
    df = M.load_frame(meta["file"])
    info = {k: meta[k] for k in ("raw_rows", "na_removed", "dup_removed")}
    return df, meta["features"], info


@st.cache_data(show_spinner=False)
def separation_scores(ds_id):
    """Seberapa kuat tiap fitur memisahkan Benign vs bukan-Benign (log1p, selisih rata-rata terstandar)."""
    df, feats, _ = load(ds_id)
    X = np.log1p(np.abs(df[feats].to_numpy(dtype=float)))
    is_ben = (df["Label"] == "Benign").to_numpy()
    a, b = X[is_ben], X[~is_ben]
    pooled = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2) + 1e-9
    score = np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled
    return pd.Series(score, index=feats).sort_values(ascending=False)


def _store(name, source, df, feats, info, label_col):
    if db.name_exists(name):
        raise M.DatasetError("ad_err_name")
    fname = M.save_frame(df)
    try:
        return db.add_dataset({
            "name": name, "source": source, "file": fname, "label_col": label_col,
            "n_rows": len(df), "n_features": len(feats), "features": feats, **info})
    except Exception:
        M.delete_frame(fname)
        raise


def example_status():
    """{key: {name, missing (file yang belum ada), added (sudah terdaftar)}}"""
    names = {d["name"] for d in db.list_datasets()}
    return {k: {"name": v["name"], "missing": M.example_missing(k), "added": v["name"] in names}
            for k, v in M.EXAMPLES.items()}


def add_example(key):
    missing = M.example_missing(key)
    if missing:
        raise M.DatasetError("ad_err_file", f=", ".join(missing))
    name = M.EXAMPLES[key]["name"]
    if db.name_exists(name):
        raise M.DatasetError("ad_err_name")
    df, feats, info, label_col = M.build_example(key)
    return _store(name, "example", df, feats, info, label_col)


# ---- unggahan ----
def read_upload(file):
    """Membaca CSV sekali per file (disimpan di sesi agar tidak dibaca ulang tiap interaksi)."""
    key = (file.name, file.size)
    cached = st.session_state.get("_upload")
    if cached and cached[0] == key:
        return cached[1]
    df = M.read_csv_any(file)
    st.session_state["_upload"] = (key, df)
    return df


def register_upload(df_raw, name, label_col, benign_values, feature_cols, dedup):
    name = name.strip()
    if not name:
        raise M.DatasetError("ad_err_noname")
    if db.name_exists(name):
        raise M.DatasetError("ad_err_name")
    df, feats, info, warnings = M.build_dataset(df_raw, label_col, benign_values,
                                                feature_cols, dedup)
    ds_id = _store(name, "upload", df, feats, info, label_col)
    return ds_id, warnings


def delete(ds_id):
    for run_id in db.run_ids_for_dataset(ds_id):          # berkas model/uji milik run dataset ini
        artifacts.delete_run_files(run_id)
    M.delete_frame(db.delete_dataset(ds_id))
    load.clear()
    separation_scores.clear()


# ---- ringkasan untuk Dashboard ----
def summary():
    ds = db.list_datasets()
    return {"datasets": ds, "n_datasets": len(ds), "n_samples": sum(d["n_rows"] for d in ds),
            "n_runs": db.count("runs"), "best_acc": db.best_holdout_accuracy()}