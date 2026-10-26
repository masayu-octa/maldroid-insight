"""Mesin pipeline (tanpa Streamlit): log1p -> RobustScaler -> SMOTE (hanya data latih) -> model."""
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import SEED

try:                                    # pustaka opsional: bila tidak terpasang, modelnya disembunyikan
    from lightgbm import LGBMClassifier
except ImportError:                     # pragma: no cover
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:                     # pragma: no cover
    XGBClassifier = None

TREE_MODELS = ["LightGBM", "XGBoost", "RandomForest", "ExtraTrees"]
OTHER_MODELS = ["DecisionTree", "LogisticRegression", "LinearSVM", "KNN", "NaiveBayes", "MLP"]
ALL_MODELS = TREE_MODELS + OTHER_MODELS

DEFAULT_MODELS = ["LightGBM", "XGBoost", "RandomForest", "ExtraTrees"]
OPTIONAL_LIBS = {"LightGBM": ("lightgbm", LGBMClassifier), "XGBoost": ("xgboost", XGBClassifier)}

# Ringkasan pengaturan (Fast, Full) untuk ditampilkan ke pengguna
HYPERS = {
    "LightGBM": ("200 trees, lr 0.1", "1000 trees, lr 0.03"),
    "XGBoost": ("200 trees, lr 0.1", "1000 trees, lr 0.03"),
    "RandomForest": ("150 trees", "700 trees"),
    "ExtraTrees": ("150 trees", "700 trees"),
    "DecisionTree": ("-", "-"),
    "LogisticRegression": ("max_iter 300", "max_iter 1000"),
    "LinearSVM": ("-", "-"),
    "KNN": ("k = 5", "k = 5"),
    "NaiveBayes": ("-", "-"),
    "MLP": ("max_iter 60", "max_iter 200"),
}


def available_models():
    return [m for m in ALL_MODELS if m not in OPTIONAL_LIBS or OPTIONAL_LIBS[m][1] is not None]


def missing_libs():
    """Nama paket pip untuk pustaka opsional yang belum terpasang."""
    return [pkg for pkg, cls in OPTIONAL_LIBS.values() if cls is None]


# ---------- data ----------
def prepare_xy(df, cols, mode="orig"):
    """log1p(|x|) (tanpa parameter yang dipelajari, aman sebelum split) + encode label.
    mode: 'orig' (kelas asli) atau 'binary' (Benign vs Malware)."""
    X = np.log1p(np.abs(df[cols].to_numpy(dtype=float)))
    labels = df["Label"].to_numpy()
    if mode == "binary":
        labels = np.where(labels == "Benign", "Benign", "Malware")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, [str(c) for c in le.classes_]


def stratified_sample(X, y, n):
    if n is None or n >= len(y):
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=int(n), stratify=y, random_state=SEED)
    return X_s, y_s


# ---------- model ----------
def get_models(fast=True, n_classes=2):
    n_big, n_tree = (200, 150) if fast else (1000, 700)
    lr = 0.1 if fast else 0.03
    defs = {}
    if LGBMClassifier is not None:
        defs["LightGBM"] = LGBMClassifier(
            n_estimators=n_big, max_depth=8, learning_rate=lr, num_leaves=127,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.05, reg_lambda=0.5, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=-1)
    if XGBClassifier is not None:
        defs["XGBoost"] = XGBClassifier(
            n_estimators=n_big, max_depth=7, learning_rate=lr, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=5, reg_alpha=0.05, reg_lambda=0.5,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            random_state=SEED, n_jobs=-1)
    defs.update({
        "RandomForest": RandomForestClassifier(
            n_estimators=n_tree, max_features="sqrt", class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=n_tree, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        "LogisticRegression": LogisticRegression(
            max_iter=300 if fast else 1000, class_weight="balanced"),
        "LinearSVM": CalibratedClassifierCV(
            LinearSVC(dual=False, class_weight="balanced"), cv=3),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60 if fast else 200,
                             early_stopping=True, random_state=SEED),
    })
    return defs


def make_pipeline(model, smote=True):
    steps = [("scaler", RobustScaler())]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=5)))
    steps.append(("clf", model))
    return ImbPipeline(steps)


def metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def feature_importance(clf, feats):
    """Bagian kepentingan tiap fitur (jumlah = 1), atau None bila model tidak menyediakannya."""
    if hasattr(clf, "feature_importances_"):
        v = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        v = np.abs(np.asarray(clf.coef_, dtype=float)).mean(axis=0)
    else:
        return None
    total = v.sum()
    if total <= 0:
        return None
    return pd.Series(v / total, index=feats).sort_values(ascending=False)


# ---------- menjalankan ----------
def run_pipeline(X, y, feats, classes, models, holdout=True, cv=True, folds=5,
                 smote=True, fast=True, progress=None):
    """Hold-out 80:20 dan/atau k-fold CV untuk tiap model.
    progress(done, total, teks) dipanggil sebelum tiap langkah."""
    n_classes = len(classes)
    defs = get_models(fast, n_classes)
    total = len(models) * ((1 if holdout else 0) + (folds if cv else 0))
    done = 0
    res = {"holdout": {}, "cv": {}, "importance": {}, "fitted": {}, "test": None,
           "classes": classes, "features": list(feats)}

    if holdout:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)
        res["test"] = {"X": X_te[:2000], "y": y_te[:2000]}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for name in models:
        if holdout:
            if progress:
                progress(done, total, f"{name} · hold-out")
            t0 = time.time()
            pipe = make_pipeline(clone(defs[name]), smote)
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            m = metrics(y_te, pred)
            m["time"] = time.time() - t0
            m["cm"] = confusion_matrix(y_te, pred, labels=list(range(n_classes))).tolist()
            res["holdout"][name] = m
            res["fitted"][name] = pipe
            imp = feature_importance(pipe.named_steps["clf"], feats)
            if imp is not None:
                res["importance"][name] = imp
            done += 1
        if cv:
            t0 = time.time()
            fold_m = []
            for i, (tr, te) in enumerate(skf.split(X, y), 1):
                if progress:
                    progress(done, total, f"{name} · CV {i}/{folds}")
                pipe = make_pipeline(clone(defs[name]), smote)
                pipe.fit(X[tr], y[tr])
                fold_m.append(metrics(y[te], pipe.predict(X[te])))
                done += 1
            fm = pd.DataFrame(fold_m)
            res["cv"][name] = {
                "acc": float(fm["acc"].mean()), "std": float(fm["acc"].std(ddof=0)),
                "prec": float(fm["prec"].mean()), "rec": float(fm["rec"].mean()),
                "f1": float(fm["f1"].mean()), "f1_macro": float(fm["f1_macro"].mean()),
                "folds": fm["acc"].tolist(), "time": time.time() - t0,
            }
    if progress:
        progress(total, total, "done")
    return res