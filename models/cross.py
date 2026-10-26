"""Uji lintas-dataset (tanpa Streamlit): latih di dataset A, uji di dataset B pada kolom bernama sama.
Label disederhanakan menjadi biner (Benign vs Malware) karena nama kelas malware tiap dataset berbeda."""
import json
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from config import SEED
from models import db
from models import pipeline as P

_DDL = """
CREATE TABLE IF NOT EXISTS cross_runs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    train_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    test_id  INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    config   TEXT,
    results  TEXT,
    seconds  REAL,
    created_at TEXT
);
"""


def _ensure(con):
    con.executescript(_DDL)


# ---------- penyimpanan ----------
def add(train_id, test_id, config, results, seconds):
    with db.connect() as con:
        _ensure(con)
        cur = con.execute(
            "INSERT INTO cross_runs (train_id, test_id, config, results, seconds, created_at) VALUES (?,?,?,?,?,?)",
            (train_id, test_id, json.dumps(config), json.dumps(results), seconds,
             datetime.now().strftime("%Y-%m-%d %H:%M")))
        return cur.lastrowid


def _row(r):
    d = dict(r)
    d["config"] = json.loads(d["config"] or "{}")
    d["results"] = json.loads(d["results"] or "{}")
    return d


def list_all():
    with db.connect() as con:
        _ensure(con)
        return [_row(r) for r in con.execute("SELECT * FROM cross_runs ORDER BY id DESC")]


def delete(run_id):
    with db.connect() as con:
        _ensure(con)
        con.execute("DELETE FROM cross_runs WHERE id = ?", (run_id,))


# ---------- perhitungan ----------
def shared_columns(feats_a, feats_b):
    b = set(feats_b)
    return [c for c in feats_a if c in b]


def binary_xy(df, cols):
    X = np.log1p(np.abs(df[cols].to_numpy(dtype=float)))
    y = (df["Label"].to_numpy() != "Benign").astype(int)       # 1 = Malware
    return X, y


def _sample(X, y, n):
    return P.stratified_sample(X, y, n)


def bin_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),            # recall Malware
        "spec": float(tn / (tn + fp)) if (tn + fp) else 0.0,                    # recall Benign
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }


def run_transfer(df_train, df_test, cols, models, smote=True, fast=True,
                 n_train=20000, n_test=20000, progress=None):
    """Satu kali latih per model pada 80% dataset A.
    Hasil: in-domain (20% sisa A) dan cross (seluruh sampel B)."""
    Xa, ya = _sample(*binary_xy(df_train, cols), n_train)
    Xb, yb = _sample(*binary_xy(df_test, cols), n_test)
    Xtr, Xin, ytr, yin = train_test_split(Xa, ya, test_size=0.2, random_state=SEED, stratify=ya)
    defs = P.get_models(fast, 2)
    out = {}
    for i, name in enumerate(models):
        if progress:
            progress(i, len(models), name)
        pipe = P.make_pipeline(clone(defs[name]), smote)
        pipe.fit(Xtr, ytr)
        out[name] = {"in": bin_metrics(yin, pipe.predict(Xin)),
                     "cross": bin_metrics(yb, pipe.predict(Xb))}
    if progress:
        progress(len(models), len(models), "done")
    out["_meta"] = {"n_train": int(len(ytr)), "n_in": int(len(yin)), "n_cross": int(len(yb)),
                    "malware_share_train": float(ya.mean()), "malware_share_test": float(yb.mean())}
    return out


def verdict(drop):
    """Kategori penurunan F1 (poin 0..1) untuk narasi: good | mid | poor."""
    if drop < 0.05:
        return "good"
    return "mid" if drop < 0.20 else "poor"