from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CIC_FILE = DATA_DIR / "CICMALDROID2020.csv"
KRONO_MAL = DATA_DIR / "real_malware_v1.csv"
KRONO_BEN = DATA_DIR / "real_legitimate_v1.csv"

CIC_LABELS = {1: "Adware", 2: "Banking", 3: "SMS", 4: "Riskware", 5: "Benign"}
KRONO_META = ["Package", "sha256", "EarliestModDate", "HighestModDate", "MalFamily"]


@st.cache_data(show_spinner="Memuat CICMalDroid2020...")
def load_cic():
    df = pd.read_csv(CIC_FILE)
    raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Label"] = df["Class"].map(CIC_LABELS)
    feats = [c for c in df.columns if c not in ("Class", "Label")]
    return df, feats, {"na_removed": 0, "dup_removed": raw - len(df)}


@st.cache_data(show_spinner="Memuat KronoDroid...")
def load_krono():
    df = pd.concat([pd.read_csv(KRONO_MAL), pd.read_csv(KRONO_BEN)], ignore_index=True)
    raw = len(df)
    df = df.drop(columns=[c for c in KRONO_META if c in df.columns])
    df = df.dropna()
    na_removed = raw - len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Label"] = df["Malware"].map({0: "Benign", 1: "Malware"})
    feats = [c for c in df.columns if c not in ("Malware", "Label")]
    return df, feats, {"na_removed": na_removed,
                       "dup_removed": raw - na_removed - len(df)}


def krono_groups(krono_feats, cic_feats):
    """Pisahkan fitur KronoDroid: irisan dengan CICMalDroid2020 (syscall) vs lainnya."""
    cic = set(cic_feats)
    return {
        "Syscall (irisan dengan CIC)": [c for c in krono_feats if c in cic],
        "Lainnya (metadata / permission / VirusTotal)": [c for c in krono_feats if c not in cic],
    }