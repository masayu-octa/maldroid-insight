"""Lapisan database (SQLite). Tidak memakai Streamlit."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH, DATASET_DIR

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS datasets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    source       TEXT NOT NULL,
    file         TEXT NOT NULL,
    label_col    TEXT,
    raw_rows     INTEGER,
    n_rows       INTEGER,
    n_features   INTEGER,
    na_removed   INTEGER,
    dup_removed  INTEGER,
    class_counts TEXT,
    features     TEXT,
    created_at   TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    config     TEXT,
    summary    TEXT,
    seconds    REAL,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS run_results (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    model   TEXT,
    eval    TEXT,
    metrics TEXT,
    extra   TEXT
);
CREATE TABLE IF NOT EXISTS sus_responses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    respondent TEXT,
    answers    TEXT,
    score      REAL,
    created_at TEXT
);
"""

JSON_COLS = ("class_counts", "features")
RUN_JSON_COLS = ("config", "summary")


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.executescript(SCHEMA)


def _dataset_row(r):
    d = dict(r)
    for c in JSON_COLS:
        d[c] = json.loads(d[c]) if d.get(c) else None
    return d


def name_exists(name):
    with connect() as con:
        return con.execute("SELECT 1 FROM datasets WHERE name = ?", (name,)).fetchone() is not None


def add_dataset(meta):
    """meta: name, source, file, label_col, raw_rows, n_rows, n_features, na_removed,
    dup_removed, class_counts (dict), features (list). Mengembalikan id baru."""
    with connect() as con:
        cur = con.execute(
            """INSERT INTO datasets (name, source, file, label_col, raw_rows, n_rows, n_features,
               na_removed, dup_removed, class_counts, features, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (meta["name"], meta["source"], meta["file"], meta.get("label_col"),
             meta["raw_rows"], meta["n_rows"], meta["n_features"], meta["na_removed"],
             meta["dup_removed"], json.dumps(meta["class_counts"]), json.dumps(meta["features"]),
             datetime.now().strftime("%Y-%m-%d %H:%M")))
        return cur.lastrowid


def list_datasets():
    with connect() as con:
        rows = con.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
    return [_dataset_row(r) for r in rows]


def get_dataset(ds_id):
    with connect() as con:
        r = con.execute("SELECT * FROM datasets WHERE id = ?", (ds_id,)).fetchone()
    return _dataset_row(r) if r else None


def delete_dataset(ds_id):
    """Menghapus baris (run terkait ikut terhapus). Mengembalikan nama file data."""
    with connect() as con:
        r = con.execute("SELECT file FROM datasets WHERE id = ?", (ds_id,)).fetchone()
        con.execute("DELETE FROM datasets WHERE id = ?", (ds_id,))
    return r["file"] if r else None


def count(table):
    if table not in ("datasets", "runs", "run_results", "sus_responses"):
        raise ValueError(table)
    with connect() as con:
        return con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]


def best_holdout_accuracy():
    """Akurasi hold-out tertinggi (0..1) dari semua run, atau None."""
    with connect() as con:
        rows = con.execute("SELECT metrics FROM run_results WHERE eval = 'holdout'").fetchall()
    accs = [json.loads(r["metrics"]).get("acc") for r in rows if r["metrics"]]
    accs = [a for a in accs if a is not None]
    return max(accs) if accs else None

# ---------------- run & hasil ----------------
def add_run(dataset_id, config, summary, seconds):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO runs (dataset_id, config, summary, seconds, created_at) VALUES (?,?,?,?,?)",
            (dataset_id, json.dumps(config), json.dumps(summary), seconds,
             datetime.now().strftime("%Y-%m-%d %H:%M")))
        return cur.lastrowid


def add_result(run_id, model, eval_kind, metrics, extra=None):
    with connect() as con:
        con.execute("INSERT INTO run_results (run_id, model, eval, metrics, extra) VALUES (?,?,?,?,?)",
                    (run_id, model, eval_kind, json.dumps(metrics), json.dumps(extra or {})))


def _run_row(r):
    d = dict(r)
    for c in RUN_JSON_COLS:
        d[c] = json.loads(d[c]) if d.get(c) else {}
    return d


def list_runs(dataset_id=None):
    """Run terbaru di atas. Tiap run diberi 'best_acc' (hold-out terbaik, 0..1) bila ada."""
    with connect() as con:
        if dataset_id is None:
            rows = con.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
        else:
            rows = con.execute("SELECT * FROM runs WHERE dataset_id = ? ORDER BY id DESC",
                               (dataset_id,)).fetchall()
        out = []
        for r in rows:
            d = _run_row(r)
            accs = [json.loads(x["metrics"]).get("acc") for x in con.execute(
                "SELECT metrics FROM run_results WHERE run_id = ? AND eval = 'holdout'", (d["id"],))]
            accs = [a for a in accs if a is not None]
            d["best_acc"] = max(accs) if accs else None
            out.append(d)
    return out


def get_run(run_id):
    with connect() as con:
        r = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not r:
            return None
        d = _run_row(r)
        res = con.execute("SELECT model, eval, metrics, extra FROM run_results WHERE run_id = ? ORDER BY id",
                          (run_id,)).fetchall()
    d["results"] = [{"model": x["model"], "eval": x["eval"], "metrics": json.loads(x["metrics"]),
                     "extra": json.loads(x["extra"]) if x["extra"] else {}} for x in res]
    return d


def run_ids_for_dataset(dataset_id):
    with connect() as con:
        return [r["id"] for r in con.execute("SELECT id FROM runs WHERE dataset_id = ?", (dataset_id,))]


def delete_run(run_id):
    with connect() as con:
        con.execute("DELETE FROM runs WHERE id = ?", (run_id,))