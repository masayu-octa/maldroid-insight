"""Prediksi pada CSV baru (tanpa Streamlit): mencocokkan kolom CSV dengan fitur model."""
import re

import numpy as np
import pandas as pd


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def auto_match(feats, csv_cols):
    """{fitur_model: kolom_csv}. Pertama nama persis sama, lalu sama setelah huruf kecil dan tanda baca
    dibuang (hanya bila tidak ambigu)."""
    cols = list(csv_cols)
    mapping = {f: f for f in feats if f in set(cols)}
    left = [c for c in cols if c not in set(mapping.values())]
    by_norm = {}
    for c in left:
        by_norm.setdefault(norm(c), []).append(c)
    for f in feats:
        if f not in mapping:
            cand = by_norm.get(norm(f), [])
            if len(cand) == 1:
                mapping[f] = cand[0]
    return mapping


def build_matrix(df, mapping, feats):
    """log1p(|x|) pada kolom yang cocok; fitur tanpa kolom diisi 0. Mengembalikan (X, jumlah sel tidak valid)."""
    X = np.zeros((len(df), len(feats)), dtype=float)
    bad = 0
    for j, f in enumerate(feats):
        col = mapping.get(f)
        if col is None or col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        bad += int(v.isna().sum() + np.isinf(v.fillna(0)).sum())
        X[:, j] = np.log1p(np.abs(np.nan_to_num(v.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)))
    return X, bad


def coverage(mapping, feats, importance=None):
    """(porsi fitur yang cocok, porsi bobot kepentingan yang cocok atau None), keduanya 0..1."""
    matched = [f for f in feats if f in mapping]
    plain = len(matched) / len(feats) if feats else 0.0
    weighted = None
    if importance:
        tot = sum(importance.get(f, 0) for f in feats)
        if tot > 0:
            weighted = sum(importance.get(f, 0) for f in matched) / tot
    return plain, weighted