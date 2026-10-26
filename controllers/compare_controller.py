"""Controller perbandingan model: membaca run dari database dan menyusun tabel perbandingan."""
import pandas as pd

from controllers import pipeline_controller as pc


def all_runs():
    return pc.list_runs(None)


def dataset_names():
    from models import db
    return {d["id"]: d["name"] for d in db.list_datasets()}


def metric_of(run, model, key):
    """Nilai metrik (0..1) satu model pada satu run, atau None. key: ho_acc | ho_f1m | cv_acc | cv_f1m."""
    kind = "holdout" if key.startswith("ho") else "cv"
    m = run[kind].get(model)
    if not m:
        return None
    return m["acc"] if key.endswith("acc") else m["f1_macro"]


def pivot(run_ids, key):
    """DataFrame model x run (persen)."""
    runs = {rid: pc.load_run(rid) for rid in run_ids}
    models = []
    for r in runs.values():
        for m in list(r["holdout"]) + list(r["cv"]):
            if m not in models:
                models.append(m)
    rows = []
    for m in models:
        row = {"Model": m}
        for rid, r in runs.items():
            v = metric_of(r, m, key)
            row[f"#{rid}"] = None if v is None else round(v * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows), runs