"""Ringkasan pemakaian untuk halaman Overview. Tanpa Streamlit."""
import pandas as pd

from config import LIMITS
from models import db


def overview():
    """Data mentah: dataset, run, dan hasil hold-out per model milik pemilik saat ini."""
    datasets = db.list_datasets()
    runs = db.list_runs()
    names = {d["id"]: d["name"] for d in datasets}
    res = []
    for r in runs:
        full = db.get_run(r["id"])
        for x in full["results"] if full else []:
            if x["eval"] == "holdout":
                m = x["metrics"]
                res.append({"run": r["id"], "dataset": names.get(r["dataset_id"], "?"), "model": x["model"],
                            "acc": (m.get("acc") or 0) * 100, "f1": (m.get("f1") or 0) * 100,
                            "date": (r["created_at"] or "")[:10]})
    runs_df = pd.DataFrame([{"run": r["id"], "date": (r["created_at"] or "")[:10], "created": r["created_at"],
                             "dataset": names.get(r["dataset_id"], "?"),
                             "models": ", ".join(r["config"].get("models", [])),
                             "best": round(r["best_acc"] * 100, 2) if r["best_acc"] is not None else None}
                            for r in runs], columns=["run", "date", "created", "dataset", "models", "best"])
    return {"datasets": datasets, "runs": runs_df,
            "results": pd.DataFrame(res, columns=["run", "dataset", "model", "acc", "f1", "date"]),
            "limits": LIMITS.get(db.tier(), LIMITS["guest"]),
            "used": {"datasets": len(datasets), "runs": len(runs)}}