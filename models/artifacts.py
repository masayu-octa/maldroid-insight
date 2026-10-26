"""Berkas besar per run (model terlatih dan sampel uji). Tidak memakai Streamlit."""
import joblib
import numpy as np

from config import STORAGE

RUN_DIR = STORAGE / "runs"


def _model_path(run_id, model):
    return RUN_DIR / f"run{run_id}_{model}.joblib"


def save_models(run_id, fitted):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for name, pipe in fitted.items():
        joblib.dump(pipe, _model_path(run_id, name), compress=3)


def load_model(run_id, model):
    return joblib.load(_model_path(run_id, model))


def save_test(run_id, X, y):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUN_DIR / f"run{run_id}_test.npz", X=X, y=y)


def load_test(run_id):
    z = np.load(RUN_DIR / f"run{run_id}_test.npz")
    return z["X"], z["y"]


def delete_run_files(run_id):
    if RUN_DIR.exists():
        for p in RUN_DIR.glob(f"run{run_id}_*"):
            p.unlink(missing_ok=True)