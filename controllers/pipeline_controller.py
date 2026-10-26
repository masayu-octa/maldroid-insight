"""Controller pipeline: menyiapkan data, menjalankan mesin, menyimpan hasil ke database."""
import time

import numpy as np

from controllers import dataset_controller as datasets_ctl
from models import artifacts, db
from models import pipeline as P


class PipelineError(Exception):
    def __init__(self, key, **kw):
        super().__init__(key)
        self.key, self.kw = key, kw


def feature_columns(ds_id, scenario, excluded=None, shared_with=None):
    """Kolom fitur untuk satu skenario: all | exclude (buang kolom terpilih) | shared (irisan dg dataset lain)."""
    _, feats, _ = datasets_ctl.load(ds_id)
    if scenario == "exclude":
        cols = [c for c in feats if c not in set(excluded or [])]
    elif scenario == "shared":
        other = db.get_dataset(shared_with)
        cols = [c for c in feats if c in set(other["features"])] if other else []
        if not cols:
            raise PipelineError("pl_err_shared")
    else:
        cols = list(feats)
    if not cols:
        raise PipelineError("pl_err_feats")
    return cols


def execute(ds_id, cfg, progress=None):
    """cfg: scenario, excluded, shared_with, mode, models, holdout, cv, folds, smote, fast, n_rows.
    Mengembalikan run_id."""
    df, _, _ = datasets_ctl.load(ds_id)
    cols = feature_columns(ds_id, cfg["scenario"], cfg.get("excluded"), cfg.get("shared_with"))
    X, y, classes = P.prepare_xy(df, cols, cfg["mode"])
    X, y = P.stratified_sample(X, y, cfg["n_rows"])

    smallest = int(np.bincount(y).min())
    need = max(2 if cfg["holdout"] else 0, cfg["folds"] if cfg["cv"] else 0)
    if smallest < need:
        raise PipelineError("pl_err_small", c=smallest, k=need)

    t0 = time.time()
    res = P.run_pipeline(X, y, cols, classes, cfg["models"], holdout=cfg["holdout"], cv=cfg["cv"],
                         folds=cfg["folds"], smote=cfg["smote"], fast=cfg["fast"], progress=progress)
    secs = time.time() - t0

    config = {k: cfg[k] for k in ("scenario", "excluded", "shared_with", "mode", "models",
                                  "holdout", "cv", "folds", "smote", "fast", "n_rows")}
    summary = {"classes": classes, "features": cols, "n_features": len(cols), "n_rows": int(len(y))}
    run_id = db.add_run(ds_id, config, summary, secs)
    for name, m in res["holdout"].items():
        met = {k: float(v) for k, v in m.items() if k != "cm"}
        met["cm"] = np.asarray(m["cm"]).tolist()
        imp = res["importance"].get(name)
        extra = {"importance": {k: float(v) for k, v in imp.head(30).items()}} if imp is not None else {}
        db.add_result(run_id, name, "holdout", met, extra)
    for name, m in res["cv"].items():
        db.add_result(run_id, name, "cv", {k: (v if k == "folds" else float(v)) for k, v in m.items()})
    if res["fitted"]:
        artifacts.save_models(run_id, res["fitted"])
        artifacts.save_test(run_id, res["test"]["X"], res["test"]["y"])
    return run_id


def list_runs(ds_id=None):
    return db.list_runs(ds_id)


def load_run(run_id):
    """Run lengkap: config, summary, dan hasil per model dalam bentuk {model: metrik}."""
    run = db.get_run(run_id)
    if not run:
        return None
    run["holdout"] = {r["model"]: r["metrics"] for r in run["results"] if r["eval"] == "holdout"}
    run["cv"] = {r["model"]: r["metrics"] for r in run["results"] if r["eval"] == "cv"}
    run["importance"] = {r["model"]: r["extra"]["importance"] for r in run["results"]
                         if r["eval"] == "holdout" and r["extra"].get("importance")}
    meta = db.get_dataset(run["dataset_id"])
    run["dataset"] = meta["name"] if meta else "?"
    return run


def delete_run(run_id):
    artifacts.delete_run_files(run_id)
    db.delete_run(run_id)