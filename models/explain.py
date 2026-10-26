"""Penjelasan model (tanpa Streamlit): pentingnya fitur lewat permutasi, efek per sampel, narasi berbasis aturan.
Semua metode bersifat model-agnostik, jadi berlaku untuk ke-10 model tanpa pustaka tambahan."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import SEED


def permutation(pipe, X, y, feats, n_rows=500, n_repeats=3):
    """Penurunan F1 makro saat satu fitur diacak. Hasil: DataFrame (importance, std) urut menurun."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=min(n_rows, len(y)), replace=False)
    r = permutation_importance(pipe, X[idx], y[idx], scoring="f1_macro", n_repeats=n_repeats,
                               random_state=SEED, n_jobs=1)
    df = pd.DataFrame({"importance": np.clip(r.importances_mean, 0, None), "std": r.importances_std},
                      index=list(feats))
    return df.sort_values("importance", ascending=False)


def local_effects(pipe, x, ref, feats, candidates, class_idx):
    """Seberapa turun peluang kelas `class_idx` bila tiap fitur kandidat diganti nilai acuan (median).
    Positif = fitur itu mendorong ke kelas tersebut. Mengembalikan (peluang awal, Series efek)."""
    base = pipe.predict_proba(x[None, :])[0][class_idx]
    idx = [feats.index(c) for c in candidates]
    batch = np.tile(x, (len(idx), 1))
    for row, j in enumerate(idx):
        batch[row, j] = ref[j]
    delta = base - pipe.predict_proba(batch)[:, class_idx]
    return float(base), pd.Series(delta, index=candidates)


def narrative(importance, builtin=None):
    """Daftar (kunci_terjemahan, parameter) berisi kesimpulan dari urutan kepentingan fitur.
    importance: Series (jumlah = 1)."""
    imp = importance[importance > 0].sort_values(ascending=False)
    out = []
    if imp.empty:
        return [("ex_nar_none", {})]
    top = list(imp.index[:3])
    shares = [float(imp[c]) * 100 for c in top]
    names = ", ".join(f"{c} ({s:.0f}%)" for c, s in zip(top, shares))
    out.append(("ex_nar_top", {"n": len(top), "names": names}))
    if shares[0] >= 40:
        out.append(("ex_nar_conc", {"f": top[0], "s": shares[0]}))
    else:
        top5 = float(imp.head(5).sum()) * 100
        out.append(("ex_nar_spread" if top5 < 60 else "ex_nar_top5", {"s": top5}))
    unused = int((importance <= 0).sum())
    if unused:
        out.append(("ex_nar_unused", {"n": unused, "total": len(importance)}))
    if builtin is not None and len(builtin):
        a, b = set(imp.index[:5]), set(builtin.sort_values(ascending=False).index[:5])
        out.append(("ex_nar_agree" if len(a & b) >= 3 else "ex_nar_differ", {"k": len(a & b)}))
    return out