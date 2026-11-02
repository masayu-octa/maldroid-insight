"""Template dataset standar (tanpa Streamlit): contoh CSV yang lolos validasi aplikasi,
dan pengambilan contoh baris dari API publik (Hugging Face Datasets Server)."""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ID_COL = "app_id"          # kolom pengenal (opsional, bukan angka sehingga otomatis diabaikan)
LABEL_COL = "label"

# (nama kolom, kelompok). Kelompok dijelaskan lewat kunci terjemahan tp_grp_<kelompok>.
FEATURES = [
    ("perm_READ_SMS", "permission"), ("perm_SEND_SMS", "permission"), ("perm_INTERNET", "permission"),
    ("perm_READ_CONTACTS", "permission"), ("perm_SYSTEM_ALERT_WINDOW", "permission"),
    ("api_getDeviceId", "api"), ("api_sendTextMessage", "api"), ("api_Cipher_getInstance", "api"),
    ("api_DexClassLoader", "api"),
    ("sys_execve", "syscall"), ("sys_openat", "syscall"),
    ("net_bytes_sent", "network"),
]
KINDS = {"binary": ["Benign", "Malware"], "multiclass": ["Benign", "Adware", "Banking", "SMS"]}

_BASE = {f: 2.0 for f, _ in FEATURES}
_BOOST = {
    "Adware": {"perm_INTERNET": 9, "api_DexClassLoader": 7, "net_bytes_sent": 12, "sys_openat": 6},
    "Banking": {"perm_SYSTEM_ALERT_WINDOW": 8, "api_Cipher_getInstance": 8, "perm_READ_CONTACTS": 6,
                "api_getDeviceId": 6},
    "SMS": {"perm_SEND_SMS": 9, "perm_READ_SMS": 8, "api_sendTextMessage": 9, "sys_execve": 5},
}


def build(kind="binary", per_class=40, seed=7):
    """Contoh dataset sintetis (hanya ilustrasi format, bukan data malware nyata)."""
    rng = np.random.default_rng(seed)
    classes = KINDS[kind]
    rows = []
    for cls in classes:
        boost = {}
        if cls == "Malware":                                  # biner: gabungan semua keluarga
            for b in _BOOST.values():
                for f, v in b.items():
                    boost[f] = max(boost.get(f, 0), v * 0.7)
        else:
            boost = _BOOST.get(cls, {})
        for _ in range(per_class):
            r = {f: int(rng.poisson(boost.get(f, _BASE[f]))) for f, _ in FEATURES}
            r[LABEL_COL] = cls
            rows.append(r)
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, ID_COL, [f"app_{i + 1:04d}" for i in range(len(df))])
    return df[[ID_COL] + [f for f, _ in FEATURES] + [LABEL_COL]]


def blank(kind="binary"):
    """Hanya baris judul kolom."""
    return build(kind).head(0)


def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def column_guide(kind="binary"):
    """[(kolom, peran, kelompok)] peran: id | feature | label."""
    out = [(ID_COL, "id", "")]
    out += [(f, "feature", g) for f, g in FEATURES]
    out.append((LABEL_COL, "label", ""))
    return out


# ---------------- API publik ----------------
HF_BASE = "https://datasets-server.huggingface.co"
_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
MAX_BYTES = 8_000_000


class RemoteError(Exception):
    def __init__(self, key, **kw):
        super().__init__(key)
        self.key, self.kw = key, kw


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "maldroid-insight"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise RemoteError("tp_err_http", code=e.code) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RemoteError("tp_err_net") from e
    if len(raw) > MAX_BYTES:
        raise RemoteError("tp_err_big")
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise RemoteError("tp_err_json") from e


def fetch_hf_rows(dataset_id, n=100, base=HF_BASE):
    """Beberapa baris pertama sebuah dataset publik. Mengembalikan (DataFrame, info)."""
    dataset_id = dataset_id.strip()
    if not _ID_RE.match(dataset_id):
        raise RemoteError("tp_err_id")
    splits = _get(f"{base}/splits?" + urllib.parse.urlencode({"dataset": dataset_id})).get("splits") or []
    if not splits:
        raise RemoteError("tp_err_nosplit")
    pick = next((s for s in splits if s.get("split") == "train"), splits[0])
    q = urllib.parse.urlencode({"dataset": dataset_id, "config": pick["config"], "split": pick["split"]})
    data = _get(f"{base}/first-rows?{q}")
    rows = [r["row"] for r in data.get("rows", [])][:n]
    if not rows:
        raise RemoteError("tp_err_norows")
    df = pd.DataFrame(rows)
    return df, {"config": pick["config"], "split": pick["split"], "n_cols": df.shape[1]}