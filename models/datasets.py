"""Logika dataset (tanpa Streamlit): baca CSV, validasi, bersihkan, simpan, contoh bawaan."""
import uuid

import numpy as np
import pandas as pd

from config import DATA_DIR, DATASET_DIR

BENIGN_HINTS = {"benign", "benigns", "legitimate", "legit", "goodware", "good", "clean",
                "normal", "safe", "harmless", "trusted", "non-malware", "nonmalware", "0"}
LABEL_NAME_HINTS = ["label", "class", "malware", "category", "target", "type", "family", "y"]
MIN_CLASS_SAMPLES = 10
MAX_LABEL_VALUES = 50


class DatasetError(Exception):
    """Galat yang bisa ditampilkan ke pengguna: key = kunci terjemahan, kw = parameternya."""

    def __init__(self, key, **kw):
        super().__init__(key)
        self.key, self.kw = key, kw


# ---------- membaca & menebak ----------
def read_csv_any(file):
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, low_memory=False)
            if df.shape[1] == 1 and ";" in str(df.columns[0]):      # CSV bertitik koma
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, sep=";", low_memory=False)
            return df
        except UnicodeDecodeError as e:
            last = e
    raise DatasetError("ad_err_read", err=str(last))


def label_strings(s):
    """Label sebagai teks yang konsisten (1.0 -> '1'); NaN tetap NaN."""
    out = pd.Series(np.nan, index=s.index, dtype=object)
    m = s.notna()
    v = s[m]
    if len(v):
        if v.dtype == bool:
            v = v.astype("int64")
        if pd.api.types.is_numeric_dtype(v) and bool((v == np.floor(v)).all()):
            out[m] = v.astype("int64").astype(str)
        else:
            out[m] = v.astype(str).str.strip()
    return out


def suggest_label_column(df):
    lower = {str(c).lower(): c for c in df.columns}
    for h in LABEL_NAME_HINTS:                      # nama persis
        if h in lower:
            return lower[h]
    for h in LABEL_NAME_HINTS[:5]:                  # nama mengandung kata kunci
        for low, col in lower.items():
            if h in low:
                return col
    return df.columns[-1]


def label_values(df, label_col):
    return sorted(label_strings(df[label_col]).dropna().unique().tolist())


def suggest_benign(values):
    return [v for v in values if str(v).strip().lower() in BENIGN_HINTS]


def numeric_candidates(df, label_col):
    """(kolom numerik, kolom yang diabaikan karena bukan numerik)."""
    ok, ignored = [], []
    for c in df.columns:
        if c == label_col:
            continue
        raw = df[c]
        n_raw = int(raw.notna().sum())
        if n_raw == 0:
            ignored.append(c)
            continue
        num = pd.to_numeric(raw, errors="coerce") if raw.dtype != bool else raw.astype("int64")
        (ok if int(num.notna().sum()) >= 0.9 * n_raw else ignored).append(c)
    return ok, ignored


# ---------- membangun dataset bersih ----------
def build_dataset(df_raw, label_col, benign_values, feature_cols, dedup=True):
    """Mengembalikan (df[fitur..., 'Label'], fitur, info, warnings).
    Label: 'Benign' untuk nilai benign; kelas malware memakai nilai aslinya
    (atau 'Malware' bila hanya ada satu nilai malware)."""
    warnings = []
    labels = label_strings(df_raw[label_col])
    uniq = labels.dropna().unique()
    if len(uniq) < 2:
        raise DatasetError("ad_err_label")
    if len(uniq) > MAX_LABEL_VALUES:
        raise DatasetError("ad_err_label_many", n=MAX_LABEL_VALUES)
    benign = [str(b) for b in benign_values if b in set(uniq)]
    if not benign or len(benign) == len(uniq):
        raise DatasetError("ad_err_benign")
    feats = [c for c in feature_cols if c != label_col and c != "Label"]
    if not feats:
        raise DatasetError("ad_err_feats")

    raw_rows = len(df_raw)
    X = df_raw[feats].apply(lambda s: s.astype("int64") if s.dtype == bool
                            else pd.to_numeric(s, errors="coerce"))
    X = X.replace([np.inf, -np.inf], np.nan)
    out = X.copy()
    mal_name = "Malware" if len(uniq) - len(benign) == 1 else None   # satu kelas malware -> "Malware"
    out["Label"] = np.where(labels.isin(benign), "Benign", mal_name or labels)
    out.loc[labels.isna(), "Label"] = np.nan
    out = out.dropna()
    na_removed = raw_rows - len(out)
    if dedup:
        out = out.drop_duplicates()
    dup_removed = raw_rows - na_removed - len(out)
    out = out.reset_index(drop=True)
    if out.empty:
        raise DatasetError("ad_err_empty")

    const = [c for c in feats if out[c].nunique() <= 1]
    if const:
        out = out.drop(columns=const)
        feats = [c for c in feats if c not in const]
        warnings.append(("ad_warn_const", {"n": len(const)}))
    if not feats:
        raise DatasetError("ad_err_feats")
    out[feats] = out[feats].astype("float32")

    counts = out["Label"].value_counts()
    if counts.min() < MIN_CLASS_SAMPLES:
        warnings.append(("ad_warn_small", {"n": MIN_CLASS_SAMPLES}))
    info = {"raw_rows": raw_rows, "na_removed": na_removed, "dup_removed": dup_removed,
            "class_counts": {str(k): int(v) for k, v in counts.items()}}
    return out, feats, info, warnings


# ---------- penyimpanan ----------
def save_frame(df):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    fname = uuid.uuid4().hex[:12] + ".pkl"
    df.to_pickle(DATASET_DIR / fname)         # hanya file yang kita tulis sendiri
    return fname


def load_frame(fname):
    return pd.read_pickle(DATASET_DIR / fname)


def delete_frame(fname):
    if fname:
        (DATASET_DIR / fname).unlink(missing_ok=True)


# ---------- dataset contoh bawaan ----------
EXAMPLES = {
    "cic": {"name": "CICMalDroid2020", "files": ["CICMALDROID2020.csv"]},
    "krono": {"name": "KronoDroid (real-device)",
              "files": ["real_malware_v1.csv", "real_legitimate_v1.csv"]},
}
CIC_LABELS = {1: "Adware", 2: "Banking", 3: "SMS", 4: "Riskware", 5: "Benign"}
KRONO_META = ["Package", "sha256", "EarliestModDate", "HighestModDate", "MalFamily"]


def example_missing(key):
    return [f for f in EXAMPLES[key]["files"] if not (DATA_DIR / f).exists()]


def build_example(key):
    """Mengembalikan (df, fitur, info, label_col). Aturan pembersihan sama seperti versi lama."""
    if key == "cic":
        df = pd.read_csv(DATA_DIR / "CICMALDROID2020.csv")
        raw = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        df["Label"] = df["Class"].map(CIC_LABELS)
        feats = [c for c in df.columns if c not in ("Class", "Label")]
        na, dup, label_col = 0, raw - len(df), "Class"
    else:
        df = pd.concat([pd.read_csv(DATA_DIR / f) for f in EXAMPLES["krono"]["files"]],
                       ignore_index=True)
        raw = len(df)
        df = df.drop(columns=[c for c in KRONO_META if c in df.columns]).dropna()
        na = raw - len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        df["Label"] = df["Malware"].map({0: "Benign", 1: "Malware"})
        feats = [c for c in df.columns if c not in ("Malware", "Label")]
        dup, label_col = raw - na - len(df), "Malware"
    out = df[feats + ["Label"]].copy()
    out[feats] = out[feats].astype("float32")
    counts = out["Label"].value_counts()
    info = {"raw_rows": raw, "na_removed": na, "dup_removed": dup,
            "class_counts": {str(k): int(v) for k, v in counts.items()}}
    return out, feats, info, label_col