"""Controller Explainability dan Prediction: memuat model tersimpan dan menjalankan penjelasan/prediksi."""
import numpy as np
import pandas as pd
import streamlit as st

from controllers import pipeline_controller as pc
from models import artifacts, db
from models import datasets as M
from models import explain as E
from models import predict as PR


def runs_with_models():
    """Run yang menyimpan model terlatih (hanya yang memakai hold-out)."""
    names = {d["id"]: d["name"] for d in db.list_datasets()}
    out = []
    for r in pc.list_runs(None):
        if r["config"].get("holdout"):
            r["dataset"] = names.get(r["dataset_id"], "?")
            out.append(r)
    return out


@st.cache_resource(show_spinner=False, max_entries=3)
def load_pipe(run_id, model):
    return artifacts.load_model(run_id, model)


@st.cache_data(show_spinner=False, max_entries=8)
def load_test(run_id):
    X, y = artifacts.load_test(run_id)
    return X, y


@st.cache_data(show_spinner=False, max_entries=12)
def permutation(run_id, model, n_rows=500):
    run = pc.load_run(run_id)
    X, y = load_test(run_id)
    return E.permutation(load_pipe(run_id, model), X, y, run["summary"]["features"], n_rows=n_rows)


def local(run_id, model, row, class_idx, candidates):
    run = pc.load_run(run_id)
    X, _ = load_test(run_id)
    feats = run["summary"]["features"]
    return E.local_effects(load_pipe(run_id, model), X[row], np.median(X, axis=0), feats, candidates, class_idx)


def sample_view(run_id, model, row):
    """(label asli, label prediksi, peluang tiap kelas) untuk satu baris sampel uji."""
    run = pc.load_run(run_id)
    X, y = load_test(run_id)
    classes = run["summary"]["classes"]
    proba = load_pipe(run_id, model).predict_proba(X[row][None, :])[0]
    return classes[int(y[row])], classes[int(np.argmax(proba))], proba


# ---------------- prediksi pada CSV baru ----------------
def read_csv(file):
    key = (file.name, file.size)
    cached = st.session_state.get("_pred_upload")
    if cached and cached[0] == key:
        return cached[1]
    df = M.read_csv_any(file)
    st.session_state["_pred_upload"] = (key, df)
    return df


def predict(run_id, model, df, mapping):
    run = pc.load_run(run_id)
    feats = run["summary"]["features"]
    classes = run["summary"]["classes"]
    X, bad = PR.build_matrix(df, mapping, feats)
    pipe = load_pipe(run_id, model)
    proba = pipe.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    out = pd.DataFrame({"prediction": [classes[i] for i in pred],
                        "confidence_%": np.round(proba.max(axis=1) * 100, 2)})
    for i, c in enumerate(classes):
        out[f"p_{c}_%"] = np.round(proba[:, i] * 100, 2)
    return out, bad


@st.cache_data(show_spinner=False, max_entries=12)
def misclassified(run_id, model):
    """Indeks sampel uji yang salah diprediksi model."""
    X, y = load_test(run_id)
    return [int(i) for i in np.where(load_pipe(run_id, model).predict(X) != y)[0]]