"""Controller uji lintas-dataset."""
import time

from controllers import dataset_controller as datasets_ctl
from models import cross as X
from models import db


class CrossError(Exception):
    def __init__(self, key, **kw):
        super().__init__(key)
        self.key, self.kw = key, kw


def shared(train_id, test_id):
    a, b = db.get_dataset(train_id), db.get_dataset(test_id)
    return X.shared_columns(a["features"], b["features"])


def execute(train_id, test_id, cfg, progress=None):
    """cfg: columns, models, smote, fast, n_rows. Mengembalikan id run lintas-dataset."""
    df_a, _, _ = datasets_ctl.load(train_id)
    df_b, _, _ = datasets_ctl.load(test_id)
    cols = cfg["columns"]
    if not cols:
        raise CrossError("cr_err_shared")
    for df in (df_a, df_b):
        n_mal = int((df["Label"] != "Benign").sum())
        if n_mal < 10 or len(df) - n_mal < 10:
            raise CrossError("cr_err_class")
    t0 = time.time()
    res = X.run_transfer(df_a, df_b, cols, cfg["models"], cfg["smote"], cfg["fast"],
                         cfg["n_rows"], cfg["n_rows"], progress)
    config = {k: cfg[k] for k in ("columns", "models", "smote", "fast", "n_rows")}
    return X.add(train_id, test_id, config, res, time.time() - t0)


def list_runs():
    runs = X.list_all()
    names = {d["id"]: d["name"] for d in db.list_datasets()}
    for r in runs:
        r["train"] = names.get(r["train_id"], "?")
        r["test"] = names.get(r["test_id"], "?")
    return runs


def delete_run(run_id):
    X.delete(run_id)