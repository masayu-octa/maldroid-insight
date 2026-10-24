import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

from src import data as D

st.set_page_config(page_title="MalDroid Insight", page_icon="🛡️", layout="wide")

DEFAULT_LANG = "EN"          # EN, ID, MS, ZH
LANGS = {"EN": "English", "ID": "Bahasa Indonesia", "MS": "Bahasa Melayu", "ZH": "中文 (简体)"}
st.session_state.setdefault("lang", DEFAULT_LANG)

# ---------------- TEKS (tambah bahasa baru = tambah satu blok di sini + satu baris di LANGS) ----------------
TEXT = {
    "EN": {
        "nav_overview": "Dataset Overview", "nav_pipeline": "Pipeline Runner",
        "nav_models": "Model Comparison", "nav_cross": "Cross-Dataset Comparison",
        "nav_explain": "Explainability", "nav_predict": "Prediction",
        "ov_title": "Dataset Overview",
        "ov_sub": "Summary of dataset quality and structure before the pipeline runs.",
        "dataset": "Dataset",
        "err_file": "File not found: {f}. Place the CSV in the `data/` folder.",
        "m_samples": "Samples (after cleaning)", "m_features": "Number of features",
        "m_classes": "Number of classes", "m_ratio": "Largest : smallest class",
        "q_dup": "Duplicates removed", "q_na": "Missing rows removed", "q_imb": "Class imbalance",
        "prov": "Source: {src} · Raw rows: {raw} · Cleaning: dropna → drop_duplicates · Loaded: {ts}",
        "tab_dist": "Class distribution", "tab_feat": "Feature explorer",
        "tab_group": "Feature groups", "tab_raw": "Raw data",
        "view": "View", "bar": "Bar", "donut": "Donut", "as_pct": "Show as percentage",
        "col_class": "Class", "col_count": "Count", "col_pct": "Percent",
        "col_feat": "Feature", "col_score": "Score", "col_group": "Group",
        "col_nfeat": "Number of features", "col_value": "Value",
        "feat_top_title": "Features that best separate Benign vs Malware",
        "top_n": "Number of top features",
        "grp_syscall": "Syscall", "grp_non": "Non-syscall",
        "grp_sys_full": "Syscall (shared with CIC)",
        "grp_other_full": "Other (metadata / permission / VirusTotal)",
        "caption_krono": "Brown = not a syscall (metadata/VirusTotal). If many appear at the top, "
                         "label leakage is likely.",
        "feat_dist_title": "Distribution of one feature per class",
        "pick_feat": "Pick a feature (type to search)", "chart_kind": "Chart type",
        "box": "Box", "violin": "Violin", "hist": "Histogram", "log1p": "Apply log1p",
        "sep_caption": "Separation score of `{f}`: {s:.3f} (sample of 20,000 rows)",
        "warn_vt": "Detection_Ratio and TimesSubmitted come from VirusTotal and may leak the "
                   "label. The ablation stage decides which features the pipeline uses.",
        "expander_nonsys": "Show non-syscall feature list",
        "info_cic": "CICMalDroid2020: all features are used as-is as dynamic features.",
        "filter_cls": "Class", "filter_cols": "Columns", "n_rows": "Rows to show",
        "soon_sub": "This module will be built in {stage}.",
        "soon_info": "Waiting for ablation results and the Pipeline Runner.",
        "stage": "Stage {n}",
                # ---- Tahap 7: Pipeline Runner ----
        "pl_sub": "Configure and run the leakage-free pipeline: log1p → RobustScaler → SMOTE (training data only) → model.",
        "pl_scenario": "Feature set",
        "pl_sc_all": "All features ({n})",
        "pl_sc_syscall": "Syscall only, shared with CIC ({n})",
        "pl_sc_no_vt": "Without VirusTotal columns ({n})",
        "pl_cic_na": "Feature-set options apply to KronoDroid only.",
        "pl_labels": "Label mode",
        "pl_lab_orig": "Original classes ({n})",
        "pl_lab_binary": "Binary (Benign vs Malware)",
        "pl_models": "Models",
        "pl_eval": "Evaluation",
        "pl_holdout": "Hold-out 80:20",
        "pl_cv": "Stratified k-fold CV",
        "pl_folds": "Number of folds",
        "pl_smote": "Apply SMOTE (training data only)",
        "pl_speed": "Model size",
        "pl_fast": "Fast (demo)",
        "pl_full": "Paper settings (slow)",
        "pl_maxrows": "Max rows (stratified sample)",
        "pl_rows_used": "Using {n} of {total} rows.",
        "pl_run": "Run pipeline",
        "pl_no_model": "Select at least one model.",
        "pl_no_eval": "Select at least one evaluation method.",
        "pl_done": "Finished in {sec:.1f} s. Run #{id} saved for comparison.",
        "pl_res_title": "Results of the latest run",
        "pl_cfg": "Run #{id} · {ds} · {fs} · {lab} · {rows} rows · SMOTE {smote}",
        "pl_on": "on",
        "pl_off": "off",
        "pl_leak": "All KronoDroid features include VirusTotal-derived columns (Detection_Ratio, TimesSubmitted). Accuracy may be inflated; compare with the other feature sets.",
        "pl_res_holdout": "Hold-out (80:20)",
        "pl_res_cv": "Stratified k-fold cross-validation",
        "pl_col_model": "Model",
        "pl_col_time": "Time (s)",
        "pl_col_std": "Std (%)",
        "pl_col_folds": "Accuracy per fold (%)",
        "pl_best": "Best hold-out accuracy: {m} ({v:.2f}%).",
        "pl_cv_note": "Differences smaller than the fold standard deviation are not meaningful.",
        "pl_top_feats": "Top features (LightGBM split count)",
        "pl_col_importance": "Importance",
        "pl_hist": "All runs in this session ({n})",
        "pl_col_run": "Run",
        "pl_col_best": "Best hold-out (%)",
                # ---- Tahap 8: Model Comparison ----
        "mc_sub": "Compare models, evaluation methods and feature sets using the runs from the Pipeline Runner.",
        "mc_empty": "No runs yet. Open Pipeline Runner and run the pipeline at least once.",
        "mc_pick": "Run to inspect",
        "mc_run_lbl": "Run #{id} · {ds} · {fs} · {lab}",
        "mc_tab_metrics": "Metrics",
        "mc_tab_cv": "Cross-validation",
        "mc_tab_cm": "Confusion matrix",
        "mc_tab_runs": "Across runs",
        "mc_metric": "Metric",
        "mc_no_ho": "This run has no hold-out results.",
        "mc_no_cv": "This run has no cross-validation results.",
        "mc_axis_note": "The Y axis starts near the lowest value so small differences are visible. Check the numbers in the table.",
        "mc_time_title": "Accuracy vs training time",
        "mc_cv_title": "Mean accuracy ± standard deviation",
        "mc_fold_title": "Accuracy of each fold",
        "mc_cm_model": "Model",
        "mc_cm_norm": "Show as % of each true class",
        "mc_cm_true": "True class",
        "mc_cm_pred": "Predicted class",
        "mc_pick_runs": "Runs to compare",
        "mc_source": "Evaluation",
        "mc_runs_title": "Accuracy of each model across runs",
        "mc_runs_note": "Only compare runs with the same number of rows, folds and SMOTE setting; otherwise the difference is not caused by the feature set alone.",
        "mc_download": "Download table (CSV)",
        "mc_download_all": "Download all metrics of selected runs (CSV)",
                # ---- Tahap 9: Cross-Dataset ----
        "cd_sub": "Compare CICMalDroid2020 and KronoDroid: how well does a model trained on one dataset detect malware in the other?",
        "cd_tab_transfer": "Transfer test",
        "cd_tab_runs": "Side by side",
        "cd_intro": "Both datasets are reduced to their shared syscall features and to two classes (Benign vs Malware). A model is trained on one dataset and tested on the other.",
        "cd_shared": "{n} shared syscall features are used.",
        "cd_dir": "Direction",
        "cd_dir_ck": "Train on CICMalDroid2020 → test on KronoDroid",
        "cd_dir_kc": "Train on KronoDroid → test on CICMalDroid2020",
        "cd_baseline": "Include in-domain baseline (train and test on the target dataset)",
        "cd_run": "Run transfer test",
        "cd_no_dir": "Select at least one direction.",
        "cd_done": "Finished in {sec:.1f} s.",
        "cd_res_title": "Transfer results",
        "cd_in": "In-domain",
        "cd_cross": "Cross-dataset",
        "cd_col_dir": "Direction",
        "cd_col_setting": "Setting",
        "cd_rec_ben": "Benign recall (%)",
        "cd_rec_mal": "Malware recall (%)",
        "cd_drop": "Drop (points)",
        "cd_note": "The datasets differ in collection setup and class balance, so some drop is expected. The size of the drop, and which class loses recall, is the finding.",
        "cd_runs_none": "Needs at least one run on each dataset (with hold-out) from the Pipeline Runner.",
        "cd_pick_cic": "CICMalDroid2020 run",
        "cd_pick_kr": "KronoDroid run",
        "cd_incompat": "These runs are not strictly comparable. Use CICMalDroid2020 in binary mode and KronoDroid with the syscall-only feature set.",
        "cd_runs_title": "Hold-out results side by side",
        "cd_download": "Download transfer results (CSV)",
                # ---- Tahap 10: Explainability ----
        "xp_sub": "Explain which features drive a fitted model, globally and for a single sample (SHAP).",
        "xp_empty": "Needs a run with hold-out evaluation. Run the pipeline with the hold-out option first.",
        "xp_missing": "The shap package is not installed. Run: pip install shap",
        "xp_class": "Explain class",
        "xp_n": "Samples to explain",
        "xp_tab_global": "Global importance",
        "xp_tab_dep": "Feature dependence",
        "xp_tab_local": "Single sample",
        "xp_global_title": "Mean |SHAP value| (average impact on the model output)",
        "xp_vt_yes": "VirusTotal column",
        "xp_vt_no": "Other feature",
        "xp_vt_warn": "A VirusTotal column is among the top features. The model may rely on the label source rather than app behaviour; compare with the run without VirusTotal columns.",
        "xp_dep_feat": "Feature",
        "xp_dep_title": "SHAP value vs feature value",
        "xp_dep_x": "Feature value (log1p)",
        "xp_dep_y": "SHAP value",
        "xp_sample": "Sample number",
        "xp_true": "True class",
        "xp_pred": "Predicted class",
        "xp_conf": "Confidence",
        "xp_local_title": "Why this prediction? Top contributions",
        "xp_up": "Toward the explained class",
        "xp_down": "Away from the explained class",
        "xp_note": "SHAP explains the fitted model, not the real-world cause. Feature values are shown after log1p; the model also applies RobustScaler.",
        "xp_computing": "Computing SHAP values...",
                # ---- Tahap 11: Prediction + SUS ----
        "pr_sub": "Predict with a fitted model from the Pipeline Runner, and collect usability feedback (SUS).",
        "pr_tab_pred": "Prediction",
        "pr_tab_sus": "Usability test (SUS)",
        "pr_empty": "Needs a run with hold-out evaluation. Run the pipeline with the hold-out option first.",
        "pr_src": "Input",
        "pr_src_csv": "Upload CSV",
        "pr_src_sample": "Test sample and what-if",
        "pr_upload": "Upload a CSV that has the feature columns of this run",
        "pr_missing": "{n} feature columns are missing, for example: {ex}.",
        "pr_nan": "{n} missing values were replaced with 0.",
        "pr_rows": "{n} rows predicted.",
        "pr_col_pred": "Prediction",
        "pr_col_conf": "Confidence (%)",
        "pr_download": "Download predictions (CSV)",
        "pr_label_acc": "The file has a Label column: agreement with it is {v:.2f}%.",
        "pr_dist": "Predicted class distribution",
        "pr_sample": "Test sample",
        "pr_edit": "What-if: change the top features (raw counts)",
        "pr_orig": "Original prediction",
        "pr_new": "Prediction after changes",
        "pr_note": "Predictions come from the model fitted in the selected run. This is a research prototype, not a security product.",
        "sus_intro": "Rate each statement from 1 (strongly disagree) to 5 (strongly agree).",
        "sus_name": "Respondent ID (optional)",
        "sus_submit": "Save response",
        "sus_incomplete": "Answer all 10 statements.",
        "sus_saved": "Response saved. SUS score: {s:.1f}",
        "sus_results": "Results",
        "sus_n": "Respondents",
        "sus_mean": "Mean SUS score",
        "sus_interp": "Interpretation",
        "sus_below": "Below average (< 68)",
        "sus_above": "Above average (68 to 80.3)",
        "sus_excel": "Excellent (80.3 or higher)",
        "sus_per_item": "Mean score per item",
        "sus_per_resp": "SUS score per respondent",
        "sus_download": "Download responses (CSV)",
        "sus_upload": "Load saved responses (CSV)",
        "sus_load": "Load file",
        "sus_loaded": "Loaded {n} responses.",
        "sus_none": "No responses yet.",
        "sus_note": "Responses live in this session only. Download the CSV before closing the app.",
    },
    "ID": {
        "nav_overview": "Ringkasan Dataset", "nav_pipeline": "Pipeline Runner",
        "nav_models": "Perbandingan Model", "nav_cross": "Lintas Dataset",
        "nav_explain": "Penjelasan Model", "nav_predict": "Prediksi",
        "ov_title": "Ringkasan Dataset",
        "ov_sub": "Ringkasan kualitas dan struktur dataset sebelum pipeline dijalankan.",
        "dataset": "Dataset",
        "err_file": "File tidak ditemukan: {f}. Letakkan CSV di folder `data/`.",
        "m_samples": "Sampel (setelah dibersihkan)", "m_features": "Jumlah fitur",
        "m_classes": "Jumlah kelas", "m_ratio": "Kelas terbesar : terkecil",
        "q_dup": "Duplikat dihapus", "q_na": "Baris missing dihapus",
        "q_imb": "Ketidakseimbangan kelas",
        "prov": "Sumber: {src} · Baris mentah: {raw} · Pembersihan: dropna → drop_duplicates · Dimuat: {ts}",
        "tab_dist": "Distribusi kelas", "tab_feat": "Penjelajah fitur",
        "tab_group": "Kelompok fitur", "tab_raw": "Data mentah",
        "view": "Tampilan", "bar": "Batang", "donut": "Donat", "as_pct": "Tampilkan dalam persen",
        "col_class": "Kelas", "col_count": "Jumlah", "col_pct": "Persen",
        "col_feat": "Fitur", "col_score": "Skor", "col_group": "Kelompok",
        "col_nfeat": "Jumlah fitur", "col_value": "Nilai",
        "feat_top_title": "Fitur paling membedakan Benign vs Malware",
        "top_n": "Jumlah fitur teratas",
        "grp_syscall": "Syscall", "grp_non": "Non-syscall",
        "grp_sys_full": "Syscall (irisan dengan CIC)",
        "grp_other_full": "Lainnya (metadata / permission / VirusTotal)",
        "caption_krono": "Cokelat = bukan syscall (metadata/VirusTotal). Jika banyak di puncak, "
                         "ada indikasi kebocoran label.",
        "feat_dist_title": "Distribusi satu fitur per kelas",
        "pick_feat": "Pilih fitur (ketik untuk mencari)", "chart_kind": "Jenis grafik",
        "box": "Box", "violin": "Violin", "hist": "Histogram", "log1p": "Terapkan log1p",
        "sep_caption": "Skor pemisahan `{f}`: {s:.3f} (sampel 20.000 baris)",
        "warn_vt": "Detection_Ratio dan TimesSubmitted berasal dari VirusTotal dan berpotensi "
                   "membocorkan label. Tahap ablation menentukan fitur yang dipakai pipeline.",
        "expander_nonsys": "Lihat daftar fitur non-syscall",
        "info_cic": "CICMalDroid2020: seluruh fitur dipakai apa adanya sebagai fitur dinamis.",
        "filter_cls": "Kelas", "filter_cols": "Kolom", "n_rows": "Jumlah baris",
        "soon_sub": "Modul ini dikerjakan di {stage}.",
        "soon_info": "Menunggu hasil ablation dan Pipeline Runner.",
        "stage": "Tahap {n}",
                # ---- Tahap 7: Pipeline Runner ----
        "pl_sub": "Atur dan jalankan pipeline bebas kebocoran: log1p → RobustScaler → SMOTE (hanya data latih) → model.",
        "pl_scenario": "Set fitur",
        "pl_sc_all": "Semua fitur ({n})",
        "pl_sc_syscall": "Hanya syscall, irisan dengan CIC ({n})",
        "pl_sc_no_vt": "Tanpa kolom VirusTotal ({n})",
        "pl_cic_na": "Pilihan set fitur hanya berlaku untuk KronoDroid.",
        "pl_labels": "Mode label",
        "pl_lab_orig": "Kelas asli ({n})",
        "pl_lab_binary": "Biner (Benign vs Malware)",
        "pl_models": "Model",
        "pl_eval": "Evaluasi",
        "pl_holdout": "Hold-out 80:20",
        "pl_cv": "k-fold CV berstrata",
        "pl_folds": "Jumlah fold",
        "pl_smote": "Terapkan SMOTE (hanya data latih)",
        "pl_speed": "Ukuran model",
        "pl_fast": "Cepat (demo)",
        "pl_full": "Pengaturan paper (lambat)",
        "pl_maxrows": "Maks. baris (sampel berstrata)",
        "pl_rows_used": "Memakai {n} dari {total} baris.",
        "pl_run": "Jalankan pipeline",
        "pl_no_model": "Pilih minimal satu model.",
        "pl_no_eval": "Pilih minimal satu metode evaluasi.",
        "pl_done": "Selesai dalam {sec:.1f} dtk. Run #{id} disimpan untuk perbandingan.",
        "pl_res_title": "Hasil run terakhir",
        "pl_cfg": "Run #{id} · {ds} · {fs} · {lab} · {rows} baris · SMOTE {smote}",
        "pl_on": "aktif",
        "pl_off": "nonaktif",
        "pl_leak": "Semua fitur KronoDroid mencakup kolom turunan VirusTotal (Detection_Ratio, TimesSubmitted). Akurasi bisa menggelembung; bandingkan dengan set fitur lain.",
        "pl_res_holdout": "Hold-out (80:20)",
        "pl_res_cv": "Validasi silang k-fold berstrata",
        "pl_col_model": "Model",
        "pl_col_time": "Waktu (dtk)",
        "pl_col_std": "Std (%)",
        "pl_col_folds": "Akurasi per fold (%)",
        "pl_best": "Akurasi hold-out terbaik: {m} ({v:.2f}%).",
        "pl_cv_note": "Selisih yang lebih kecil dari simpangan baku antar fold tidak bermakna.",
        "pl_top_feats": "Fitur teratas (jumlah split LightGBM)",
        "pl_col_importance": "Kepentingan",
        "pl_hist": "Semua run pada sesi ini ({n})",
        "pl_col_run": "Run",
        "pl_col_best": "Hold-out terbaik (%)",
                # ---- Tahap 8: Model Comparison ----
        "mc_sub": "Bandingkan model, metode evaluasi, dan set fitur dari run di Pipeline Runner.",
        "mc_empty": "Belum ada run. Buka Pipeline Runner dan jalankan pipeline minimal sekali.",
        "mc_pick": "Run yang diperiksa",
        "mc_run_lbl": "Run #{id} · {ds} · {fs} · {lab}",
        "mc_tab_metrics": "Metrik",
        "mc_tab_cv": "Validasi silang",
        "mc_tab_cm": "Confusion matrix",
        "mc_tab_runs": "Antar run",
        "mc_metric": "Metrik",
        "mc_no_ho": "Run ini tidak punya hasil hold-out.",
        "mc_no_cv": "Run ini tidak punya hasil validasi silang.",
        "mc_axis_note": "Sumbu Y dimulai mendekati nilai terendah agar selisih kecil terlihat. Cek angka di tabel.",
        "mc_time_title": "Akurasi vs waktu latih",
        "mc_cv_title": "Rata-rata akurasi ± simpangan baku",
        "mc_fold_title": "Akurasi tiap fold",
        "mc_cm_model": "Model",
        "mc_cm_norm": "Tampilkan sebagai % dari tiap kelas sebenarnya",
        "mc_cm_true": "Kelas sebenarnya",
        "mc_cm_pred": "Kelas prediksi",
        "mc_pick_runs": "Run yang dibandingkan",
        "mc_source": "Evaluasi",
        "mc_runs_title": "Akurasi tiap model antar run",
        "mc_runs_note": "Bandingkan hanya run dengan jumlah baris, fold, dan pengaturan SMOTE yang sama; jika tidak, selisihnya tidak hanya disebabkan oleh set fitur.",
        "mc_download": "Unduh tabel (CSV)",
        "mc_download_all": "Unduh semua metrik run terpilih (CSV)",
                # ---- Tahap 9: Cross-Dataset ----
        "cd_sub": "Bandingkan CICMalDroid2020 dan KronoDroid: seberapa baik model yang dilatih di satu dataset mendeteksi malware di dataset lain?",
        "cd_tab_transfer": "Uji transfer",
        "cd_tab_runs": "Berdampingan",
        "cd_intro": "Kedua dataset dipersempit ke fitur syscall yang sama dan ke dua kelas (Benign vs Malware). Model dilatih di satu dataset lalu diuji di dataset lainnya.",
        "cd_shared": "{n} fitur syscall yang sama dipakai.",
        "cd_dir": "Arah",
        "cd_dir_ck": "Latih di CICMalDroid2020 → uji di KronoDroid",
        "cd_dir_kc": "Latih di KronoDroid → uji di CICMalDroid2020",
        "cd_baseline": "Sertakan baseline in-domain (latih dan uji di dataset target)",
        "cd_run": "Jalankan uji transfer",
        "cd_no_dir": "Pilih minimal satu arah.",
        "cd_done": "Selesai dalam {sec:.1f} dtk.",
        "cd_res_title": "Hasil transfer",
        "cd_in": "In-domain",
        "cd_cross": "Lintas dataset",
        "cd_col_dir": "Arah",
        "cd_col_setting": "Skenario",
        "cd_rec_ben": "Recall Benign (%)",
        "cd_rec_mal": "Recall Malware (%)",
        "cd_drop": "Penurunan (poin)",
        "cd_note": "Kedua dataset berbeda dalam cara pengumpulan dan keseimbangan kelas, jadi penurunan wajar terjadi. Besar penurunan, dan kelas mana yang kehilangan recall, itulah temuannya.",
        "cd_runs_none": "Perlu minimal satu run (dengan hold-out) di tiap dataset dari Pipeline Runner.",
        "cd_pick_cic": "Run CICMalDroid2020",
        "cd_pick_kr": "Run KronoDroid",
        "cd_incompat": "Kedua run ini tidak sepenuhnya sebanding. Pakai CICMalDroid2020 mode biner dan KronoDroid dengan set fitur hanya syscall.",
        "cd_runs_title": "Hasil hold-out berdampingan",
        "cd_download": "Unduh hasil transfer (CSV)",
                # ---- Tahap 10: Explainability ----
        "xp_sub": "Jelaskan fitur apa yang menggerakkan model terlatih, secara global dan per sampel (SHAP).",
        "xp_empty": "Perlu run dengan evaluasi hold-out. Jalankan pipeline dengan opsi hold-out dulu.",
        "xp_missing": "Paket shap belum terpasang. Jalankan: pip install shap",
        "xp_class": "Kelas yang dijelaskan",
        "xp_n": "Jumlah sampel yang dijelaskan",
        "xp_tab_global": "Kepentingan global",
        "xp_tab_dep": "Ketergantungan fitur",
        "xp_tab_local": "Satu sampel",
        "xp_global_title": "Rata-rata |nilai SHAP| (dampak rata-rata pada keluaran model)",
        "xp_vt_yes": "Kolom VirusTotal",
        "xp_vt_no": "Fitur lain",
        "xp_vt_warn": "Ada kolom VirusTotal di antara fitur teratas. Model mungkin bergantung pada sumber label, bukan perilaku aplikasi; bandingkan dengan run tanpa kolom VirusTotal.",
        "xp_dep_feat": "Fitur",
        "xp_dep_title": "Nilai SHAP vs nilai fitur",
        "xp_dep_x": "Nilai fitur (log1p)",
        "xp_dep_y": "Nilai SHAP",
        "xp_sample": "Nomor sampel",
        "xp_true": "Kelas sebenarnya",
        "xp_pred": "Kelas prediksi",
        "xp_conf": "Keyakinan",
        "xp_local_title": "Mengapa prediksi ini? Kontribusi terbesar",
        "xp_up": "Mendorong ke kelas yang dijelaskan",
        "xp_down": "Menjauhi kelas yang dijelaskan",
        "xp_note": "SHAP menjelaskan model yang sudah dilatih, bukan penyebab di dunia nyata. Nilai fitur ditampilkan setelah log1p; model juga memakai RobustScaler.",
        "xp_computing": "Menghitung nilai SHAP...",
                # ---- Tahap 11: Prediction + SUS ----
        "pr_sub": "Prediksi dengan model terlatih dari Pipeline Runner, dan kumpulkan umpan balik kegunaan (SUS).",
        "pr_tab_pred": "Prediksi",
        "pr_tab_sus": "Uji kegunaan (SUS)",
        "pr_empty": "Perlu run dengan evaluasi hold-out. Jalankan pipeline dengan opsi hold-out dulu.",
        "pr_src": "Masukan",
        "pr_src_csv": "Unggah CSV",
        "pr_src_sample": "Sampel uji dan skenario ubah",
        "pr_upload": "Unggah CSV yang memiliki kolom fitur dari run ini",
        "pr_missing": "{n} kolom fitur tidak ada, misalnya: {ex}.",
        "pr_nan": "{n} nilai kosong diganti dengan 0.",
        "pr_rows": "{n} baris diprediksi.",
        "pr_col_pred": "Prediksi",
        "pr_col_conf": "Keyakinan (%)",
        "pr_download": "Unduh prediksi (CSV)",
        "pr_label_acc": "File memiliki kolom Label: kecocokan dengan kolom itu {v:.2f}%.",
        "pr_dist": "Distribusi kelas hasil prediksi",
        "pr_sample": "Sampel uji",
        "pr_edit": "Skenario ubah: ubah fitur teratas (hitungan mentah)",
        "pr_orig": "Prediksi awal",
        "pr_new": "Prediksi setelah diubah",
        "pr_note": "Prediksi berasal dari model yang dilatih pada run terpilih. Ini prototipe penelitian, bukan produk keamanan.",
        "sus_intro": "Nilai tiap pernyataan dari 1 (sangat tidak setuju) sampai 5 (sangat setuju).",
        "sus_name": "ID responden (opsional)",
        "sus_submit": "Simpan jawaban",
        "sus_incomplete": "Jawab semua 10 pernyataan.",
        "sus_saved": "Jawaban disimpan. Skor SUS: {s:.1f}",
        "sus_results": "Hasil",
        "sus_n": "Jumlah responden",
        "sus_mean": "Rata-rata skor SUS",
        "sus_interp": "Interpretasi",
        "sus_below": "Di bawah rata-rata (< 68)",
        "sus_above": "Di atas rata-rata (68 sampai 80,3)",
        "sus_excel": "Sangat baik (80,3 atau lebih)",
        "sus_per_item": "Rata-rata skor per butir",
        "sus_per_resp": "Skor SUS per responden",
        "sus_download": "Unduh jawaban (CSV)",
        "sus_upload": "Muat jawaban tersimpan (CSV)",
        "sus_load": "Muat file",
        "sus_loaded": "{n} jawaban dimuat.",
        "sus_none": "Belum ada jawaban.",
        "sus_note": "Jawaban hanya tersimpan di sesi ini. Unduh CSV sebelum menutup aplikasi.",
    },
    "MS": {
        "nav_overview": "Ringkasan Data", "nav_pipeline": "Pipeline Runner",
        "nav_models": "Perbandingan Model", "nav_cross": "Merentas Set Data",
        "nav_explain": "Penjelasan Model", "nav_predict": "Ramalan",
        "ov_title": "Ringkasan Set Data",
        "ov_sub": "Ringkasan kualiti dan struktur set data sebelum pipeline dijalankan.",
        "dataset": "Set data",
        "err_file": "Fail tidak dijumpai: {f}. Letakkan CSV dalam folder `data/`.",
        "m_samples": "Sampel (selepas pembersihan)", "m_features": "Bilangan ciri",
        "m_classes": "Bilangan kelas", "m_ratio": "Kelas terbesar : terkecil",
        "q_dup": "Pendua dibuang", "q_na": "Baris hilang dibuang",
        "q_imb": "Ketidakseimbangan kelas",
        "prov": "Sumber: {src} · Baris mentah: {raw} · Pembersihan: dropna → drop_duplicates · Dimuatkan: {ts}",
        "tab_dist": "Taburan kelas", "tab_feat": "Penjelajah ciri",
        "tab_group": "Kumpulan ciri", "tab_raw": "Data mentah",
        "view": "Paparan", "bar": "Palang", "donut": "Donat", "as_pct": "Papar dalam peratus",
        "col_class": "Kelas", "col_count": "Bilangan", "col_pct": "Peratus",
        "col_feat": "Ciri", "col_score": "Skor", "col_group": "Kumpulan",
        "col_nfeat": "Bilangan ciri", "col_value": "Nilai",
        "feat_top_title": "Ciri yang paling membezakan Benign vs Malware",
        "top_n": "Bilangan ciri teratas",
        "grp_syscall": "Syscall", "grp_non": "Bukan syscall",
        "grp_sys_full": "Syscall (sepunya dengan CIC)",
        "grp_other_full": "Lain-lain (metadata / permission / VirusTotal)",
        "caption_krono": "Coklat = bukan syscall (metadata/VirusTotal). Jika banyak di bahagian atas, "
                         "kebocoran label berkemungkinan berlaku.",
        "feat_dist_title": "Taburan satu ciri bagi setiap kelas",
        "pick_feat": "Pilih ciri (taip untuk mencari)", "chart_kind": "Jenis carta",
        "box": "Kotak", "violin": "Biola", "hist": "Histogram", "log1p": "Guna log1p",
        "sep_caption": "Skor pemisahan `{f}`: {s:.3f} (sampel 20,000 baris)",
        "warn_vt": "Detection_Ratio dan TimesSubmitted datang daripada VirusTotal dan boleh "
                   "membocorkan label. Peringkat ablation menentukan ciri yang digunakan pipeline.",
        "expander_nonsys": "Papar senarai ciri bukan syscall",
        "info_cic": "CICMalDroid2020: semua ciri digunakan seperti asal sebagai ciri dinamik.",
        "filter_cls": "Kelas", "filter_cols": "Lajur", "n_rows": "Bilangan baris",
        "soon_sub": "Modul ini akan dibangunkan pada {stage}.",
        "soon_info": "Menunggu keputusan ablation dan Pipeline Runner.",
        "stage": "Peringkat {n}",
                # ---- Tahap 7: Pipeline Runner ----
        "pl_sub": "Tetapkan dan jalankan pipeline tanpa kebocoran: log1p → RobustScaler → SMOTE (data latihan sahaja) → model.",
        "pl_scenario": "Set ciri",
        "pl_sc_all": "Semua ciri ({n})",
        "pl_sc_syscall": "Syscall sahaja, sepunya dengan CIC ({n})",
        "pl_sc_no_vt": "Tanpa lajur VirusTotal ({n})",
        "pl_cic_na": "Pilihan set ciri hanya terpakai untuk KronoDroid.",
        "pl_labels": "Mod label",
        "pl_lab_orig": "Kelas asal ({n})",
        "pl_lab_binary": "Binari (Benign vs Malware)",
        "pl_models": "Model",
        "pl_eval": "Penilaian",
        "pl_holdout": "Hold-out 80:20",
        "pl_cv": "CV k-fold berstrata",
        "pl_folds": "Bilangan fold",
        "pl_smote": "Guna SMOTE (data latihan sahaja)",
        "pl_speed": "Saiz model",
        "pl_fast": "Pantas (demo)",
        "pl_full": "Tetapan kertas kerja (perlahan)",
        "pl_maxrows": "Maks. baris (sampel berstrata)",
        "pl_rows_used": "Menggunakan {n} daripada {total} baris.",
        "pl_run": "Jalankan pipeline",
        "pl_no_model": "Pilih sekurang-kurangnya satu model.",
        "pl_no_eval": "Pilih sekurang-kurangnya satu kaedah penilaian.",
        "pl_done": "Selesai dalam {sec:.1f} s. Run #{id} disimpan untuk perbandingan.",
        "pl_res_title": "Keputusan run terkini",
        "pl_cfg": "Run #{id} · {ds} · {fs} · {lab} · {rows} baris · SMOTE {smote}",
        "pl_on": "aktif",
        "pl_off": "tidak aktif",
        "pl_leak": "Semua ciri KronoDroid termasuk lajur terbitan VirusTotal (Detection_Ratio, TimesSubmitted). Ketepatan mungkin terlalu tinggi; bandingkan dengan set ciri lain.",
        "pl_res_holdout": "Hold-out (80:20)",
        "pl_res_cv": "Pengesahan silang k-fold berstrata",
        "pl_col_model": "Model",
        "pl_col_time": "Masa (s)",
        "pl_col_std": "Sisihan piawai (%)",
        "pl_col_folds": "Ketepatan setiap fold (%)",
        "pl_best": "Ketepatan hold-out terbaik: {m} ({v:.2f}%).",
        "pl_cv_note": "Perbezaan yang lebih kecil daripada sisihan piawai antara fold tidak bermakna.",
        "pl_top_feats": "Ciri teratas (bilangan split LightGBM)",
        "pl_col_importance": "Kepentingan",
        "pl_hist": "Semua run dalam sesi ini ({n})",
        "pl_col_run": "Run",
        "pl_col_best": "Hold-out terbaik (%)",
                # ---- Tahap 8: Model Comparison ----
        "mc_sub": "Bandingkan model, kaedah penilaian dan set ciri daripada run di Pipeline Runner.",
        "mc_empty": "Belum ada run. Buka Pipeline Runner dan jalankan pipeline sekurang-kurangnya sekali.",
        "mc_pick": "Run yang diperiksa",
        "mc_run_lbl": "Run #{id} · {ds} · {fs} · {lab}",
        "mc_tab_metrics": "Metrik",
        "mc_tab_cv": "Pengesahan silang",
        "mc_tab_cm": "Confusion matrix",
        "mc_tab_runs": "Antara run",
        "mc_metric": "Metrik",
        "mc_no_ho": "Run ini tiada keputusan hold-out.",
        "mc_no_cv": "Run ini tiada keputusan pengesahan silang.",
        "mc_axis_note": "Paksi Y bermula hampir nilai terendah supaya perbezaan kecil kelihatan. Semak angka dalam jadual.",
        "mc_time_title": "Ketepatan vs masa latihan",
        "mc_cv_title": "Purata ketepatan ± sisihan piawai",
        "mc_fold_title": "Ketepatan setiap fold",
        "mc_cm_model": "Model",
        "mc_cm_norm": "Papar sebagai % bagi setiap kelas sebenar",
        "mc_cm_true": "Kelas sebenar",
        "mc_cm_pred": "Kelas ramalan",
        "mc_pick_runs": "Run untuk dibandingkan",
        "mc_source": "Penilaian",
        "mc_runs_title": "Ketepatan setiap model antara run",
        "mc_runs_note": "Bandingkan run yang bilangan baris, fold dan tetapan SMOTE-nya sama sahaja; jika tidak, perbezaan bukan disebabkan oleh set ciri semata-mata.",
        "mc_download": "Muat turun jadual (CSV)",
        "mc_download_all": "Muat turun semua metrik run terpilih (CSV)",
                # ---- Tahap 9: Cross-Dataset ----
        "cd_sub": "Bandingkan CICMalDroid2020 dan KronoDroid: sejauh mana model yang dilatih pada satu set data mengesan malware pada set data lain?",
        "cd_tab_transfer": "Ujian pindahan",
        "cd_tab_runs": "Bersebelahan",
        "cd_intro": "Kedua-dua set data dikecilkan kepada ciri syscall yang sama dan dua kelas (Benign vs Malware). Model dilatih pada satu set data dan diuji pada set data yang lain.",
        "cd_shared": "{n} ciri syscall yang sama digunakan.",
        "cd_dir": "Arah",
        "cd_dir_ck": "Latih pada CICMalDroid2020 → uji pada KronoDroid",
        "cd_dir_kc": "Latih pada KronoDroid → uji pada CICMalDroid2020",
        "cd_baseline": "Sertakan garis asas in-domain (latih dan uji pada set data sasaran)",
        "cd_run": "Jalankan ujian pindahan",
        "cd_no_dir": "Pilih sekurang-kurangnya satu arah.",
        "cd_done": "Selesai dalam {sec:.1f} s.",
        "cd_res_title": "Keputusan pindahan",
        "cd_in": "In-domain",
        "cd_cross": "Merentas set data",
        "cd_col_dir": "Arah",
        "cd_col_setting": "Senario",
        "cd_rec_ben": "Recall Benign (%)",
        "cd_rec_mal": "Recall Malware (%)",
        "cd_drop": "Penurunan (mata)",
        "cd_note": "Kedua-dua set data berbeza dari segi kaedah pengumpulan dan keseimbangan kelas, jadi penurunan dijangka berlaku. Saiz penurunan, dan kelas mana yang kehilangan recall, ialah dapatannya.",
        "cd_runs_none": "Perlukan sekurang-kurangnya satu run (dengan hold-out) bagi setiap set data daripada Pipeline Runner.",
        "cd_pick_cic": "Run CICMalDroid2020",
        "cd_pick_kr": "Run KronoDroid",
        "cd_incompat": "Kedua-dua run ini tidak setanding sepenuhnya. Gunakan CICMalDroid2020 mod binari dan KronoDroid dengan set ciri syscall sahaja.",
        "cd_runs_title": "Keputusan hold-out bersebelahan",
        "cd_download": "Muat turun keputusan pindahan (CSV)",
                # ---- Tahap 10: Explainability ----
        "xp_sub": "Terangkan ciri yang menggerakkan model terlatih, secara global dan bagi satu sampel (SHAP).",
        "xp_empty": "Perlukan run dengan penilaian hold-out. Jalankan pipeline dengan pilihan hold-out dahulu.",
        "xp_missing": "Pakej shap belum dipasang. Jalankan: pip install shap",
        "xp_class": "Kelas yang diterangkan",
        "xp_n": "Bilangan sampel yang diterangkan",
        "xp_tab_global": "Kepentingan global",
        "xp_tab_dep": "Kebergantungan ciri",
        "xp_tab_local": "Satu sampel",
        "xp_global_title": "Purata |nilai SHAP| (kesan purata pada output model)",
        "xp_vt_yes": "Lajur VirusTotal",
        "xp_vt_no": "Ciri lain",
        "xp_vt_warn": "Terdapat lajur VirusTotal dalam kalangan ciri teratas. Model mungkin bergantung pada sumber label, bukan tingkah laku aplikasi; bandingkan dengan run tanpa lajur VirusTotal.",
        "xp_dep_feat": "Ciri",
        "xp_dep_title": "Nilai SHAP vs nilai ciri",
        "xp_dep_x": "Nilai ciri (log1p)",
        "xp_dep_y": "Nilai SHAP",
        "xp_sample": "Nombor sampel",
        "xp_true": "Kelas sebenar",
        "xp_pred": "Kelas ramalan",
        "xp_conf": "Keyakinan",
        "xp_local_title": "Mengapa ramalan ini? Sumbangan terbesar",
        "xp_up": "Menolak ke arah kelas yang diterangkan",
        "xp_down": "Menjauhi kelas yang diterangkan",
        "xp_note": "SHAP menerangkan model yang telah dilatih, bukan punca di dunia sebenar. Nilai ciri dipaparkan selepas log1p; model juga menggunakan RobustScaler.",
        "xp_computing": "Mengira nilai SHAP...",
                # ---- Tahap 11: Prediction + SUS ----
        "pr_sub": "Ramal dengan model terlatih daripada Pipeline Runner, dan kumpulkan maklum balas kebolehgunaan (SUS).",
        "pr_tab_pred": "Ramalan",
        "pr_tab_sus": "Ujian kebolehgunaan (SUS)",
        "pr_empty": "Perlukan run dengan penilaian hold-out. Jalankan pipeline dengan pilihan hold-out dahulu.",
        "pr_src": "Input",
        "pr_src_csv": "Muat naik CSV",
        "pr_src_sample": "Sampel ujian dan senario ubah",
        "pr_upload": "Muat naik CSV yang mempunyai lajur ciri bagi run ini",
        "pr_missing": "{n} lajur ciri tiada, contohnya: {ex}.",
        "pr_nan": "{n} nilai kosong diganti dengan 0.",
        "pr_rows": "{n} baris diramal.",
        "pr_col_pred": "Ramalan",
        "pr_col_conf": "Keyakinan (%)",
        "pr_download": "Muat turun ramalan (CSV)",
        "pr_label_acc": "Fail mempunyai lajur Label: persamaan dengannya {v:.2f}%.",
        "pr_dist": "Taburan kelas ramalan",
        "pr_sample": "Sampel ujian",
        "pr_edit": "Senario ubah: ubah ciri teratas (kiraan mentah)",
        "pr_orig": "Ramalan asal",
        "pr_new": "Ramalan selepas perubahan",
        "pr_note": "Ramalan datang daripada model yang dilatih pada run terpilih. Ini prototaip penyelidikan, bukan produk keselamatan.",
        "sus_intro": "Nilaikan setiap pernyataan daripada 1 (sangat tidak setuju) hingga 5 (sangat setuju).",
        "sus_name": "ID responden (pilihan)",
        "sus_submit": "Simpan jawapan",
        "sus_incomplete": "Jawab semua 10 pernyataan.",
        "sus_saved": "Jawapan disimpan. Skor SUS: {s:.1f}",
        "sus_results": "Keputusan",
        "sus_n": "Bilangan responden",
        "sus_mean": "Purata skor SUS",
        "sus_interp": "Tafsiran",
        "sus_below": "Di bawah purata (< 68)",
        "sus_above": "Di atas purata (68 hingga 80.3)",
        "sus_excel": "Cemerlang (80.3 atau lebih)",
        "sus_per_item": "Purata skor setiap item",
        "sus_per_resp": "Skor SUS setiap responden",
        "sus_download": "Muat turun jawapan (CSV)",
        "sus_upload": "Muat jawapan tersimpan (CSV)",
        "sus_load": "Muat fail",
        "sus_loaded": "{n} jawapan dimuatkan.",
        "sus_none": "Belum ada jawapan.",
        "sus_note": "Jawapan hanya tersimpan dalam sesi ini. Muat turun CSV sebelum menutup aplikasi.",
    },
    "ZH": {
        "nav_overview": "数据集概览", "nav_pipeline": "流程运行器",
        "nav_models": "模型对比", "nav_cross": "跨数据集对比",
        "nav_explain": "可解释性", "nav_predict": "预测",
        "ov_title": "数据集概览",
        "ov_sub": "在运行流程之前，概览数据集的质量与结构。",
        "dataset": "数据集",
        "err_file": "未找到文件：{f}。请将 CSV 放入 `data/` 文件夹。",
        "m_samples": "样本数（清洗后）", "m_features": "特征数量",
        "m_classes": "类别数量", "m_ratio": "最大类 : 最小类",
        "q_dup": "已删除重复项", "q_na": "已删除缺失行", "q_imb": "类别不平衡",
        "prov": "来源：{src} · 原始行数：{raw} · 清洗：dropna → drop_duplicates · 加载时间：{ts}",
        "tab_dist": "类别分布", "tab_feat": "特征探索",
        "tab_group": "特征分组", "tab_raw": "原始数据",
        "view": "视图", "bar": "柱状图", "donut": "环形图", "as_pct": "显示百分比",
        "col_class": "类别", "col_count": "数量", "col_pct": "百分比",
        "col_feat": "特征", "col_score": "得分", "col_group": "分组",
        "col_nfeat": "特征数量", "col_value": "数值",
        "feat_top_title": "最能区分 Benign 与 Malware 的特征",
        "top_n": "前 N 个特征",
        "grp_syscall": "系统调用", "grp_non": "非系统调用",
        "grp_sys_full": "系统调用（与 CIC 共有）",
        "grp_other_full": "其他（元数据 / 权限 / VirusTotal）",
        "caption_krono": "棕色 = 非系统调用（元数据/VirusTotal）。若大量出现在前列，则可能存在标签泄漏。",
        "feat_dist_title": "单个特征按类别的分布",
        "pick_feat": "选择特征（可输入搜索）", "chart_kind": "图表类型",
        "box": "箱线图", "violin": "小提琴图", "hist": "直方图", "log1p": "应用 log1p",
        "sep_caption": "`{f}` 的区分得分：{s:.3f}（抽样 20,000 行）",
        "warn_vt": "Detection_Ratio 和 TimesSubmitted 来自 VirusTotal，可能泄漏标签。"
                   "消融实验阶段将决定流程使用哪些特征。",
        "expander_nonsys": "查看非系统调用特征列表",
        "info_cic": "CICMalDroid2020：所有特征均按原样作为动态特征使用。",
        "filter_cls": "类别", "filter_cols": "列", "n_rows": "显示行数",
        "soon_sub": "该模块将在{stage}完成。",
        "soon_info": "等待消融实验结果与流程运行器。",
        "stage": "第 {n} 阶段",
                # ---- Tahap 7: Pipeline Runner ----
        "pl_sub": "配置并运行无泄漏流程：log1p → RobustScaler → SMOTE（仅训练数据）→ 模型。",
        "pl_scenario": "特征集",
        "pl_sc_all": "全部特征（{n}）",
        "pl_sc_syscall": "仅系统调用，与 CIC 共有（{n}）",
        "pl_sc_no_vt": "不含 VirusTotal 列（{n}）",
        "pl_cic_na": "特征集选项仅适用于 KronoDroid。",
        "pl_labels": "标签模式",
        "pl_lab_orig": "原始类别（{n}）",
        "pl_lab_binary": "二分类（Benign vs Malware）",
        "pl_models": "模型",
        "pl_eval": "评估方式",
        "pl_holdout": "留出法 80:20",
        "pl_cv": "分层 k 折交叉验证",
        "pl_folds": "折数",
        "pl_smote": "应用 SMOTE（仅训练数据）",
        "pl_speed": "模型规模",
        "pl_fast": "快速（演示）",
        "pl_full": "论文设置（较慢）",
        "pl_maxrows": "最大行数（分层抽样）",
        "pl_rows_used": "使用 {total} 行中的 {n} 行。",
        "pl_run": "运行流程",
        "pl_no_model": "请至少选择一个模型。",
        "pl_no_eval": "请至少选择一种评估方式。",
        "pl_done": "耗时 {sec:.1f} 秒。第 {id} 次运行已保存以供对比。",
        "pl_res_title": "最近一次运行结果",
        "pl_cfg": "第 {id} 次运行 · {ds} · {fs} · {lab} · {rows} 行 · SMOTE {smote}",
        "pl_on": "开",
        "pl_off": "关",
        "pl_leak": "KronoDroid 的全部特征包含源自 VirusTotal 的列（Detection_Ratio、TimesSubmitted）。准确率可能被高估，请与其他特征集对比。",
        "pl_res_holdout": "留出法（80:20）",
        "pl_res_cv": "分层 k 折交叉验证",
        "pl_col_model": "模型",
        "pl_col_time": "耗时（秒）",
        "pl_col_std": "标准差（%）",
        "pl_col_folds": "各折准确率（%）",
        "pl_best": "留出法最佳准确率：{m}（{v:.2f}%）。",
        "pl_cv_note": "小于折间标准差的差异没有实际意义。",
        "pl_top_feats": "重要特征（LightGBM 分裂次数）",
        "pl_col_importance": "重要性",
        "pl_hist": "本次会话的全部运行（{n}）",
        "pl_col_run": "运行",
        "pl_col_best": "最佳留出法（%）",
                # ---- Tahap 8: Model Comparison ----
        "mc_sub": "基于流程运行器的结果，对比模型、评估方式与特征集。",
        "mc_empty": "尚无运行记录。请先在流程运行器中至少运行一次。",
        "mc_pick": "查看的运行",
        "mc_run_lbl": "第 {id} 次运行 · {ds} · {fs} · {lab}",
        "mc_tab_metrics": "指标",
        "mc_tab_cv": "交叉验证",
        "mc_tab_cm": "混淆矩阵",
        "mc_tab_runs": "跨运行对比",
        "mc_metric": "指标",
        "mc_no_ho": "该运行没有留出法结果。",
        "mc_no_cv": "该运行没有交叉验证结果。",
        "mc_axis_note": "纵轴从接近最小值处开始，以便看清微小差异。请以表格中的数字为准。",
        "mc_time_title": "准确率与训练时间",
        "mc_cv_title": "平均准确率 ± 标准差",
        "mc_fold_title": "各折准确率",
        "mc_cm_model": "模型",
        "mc_cm_norm": "按真实类别显示百分比",
        "mc_cm_true": "真实类别",
        "mc_cm_pred": "预测类别",
        "mc_pick_runs": "要对比的运行",
        "mc_source": "评估方式",
        "mc_runs_title": "各模型在不同运行中的准确率",
        "mc_runs_note": "仅对比行数、折数和 SMOTE 设置相同的运行，否则差异不只是由特征集造成的。",
        "mc_download": "下载表格（CSV）",
        "mc_download_all": "下载所选运行的全部指标（CSV）",
                # ---- Tahap 9: Cross-Dataset ----
        "cd_sub": "对比 CICMalDroid2020 与 KronoDroid：在一个数据集上训练的模型，能否检测另一个数据集中的恶意软件？",
        "cd_tab_transfer": "迁移测试",
        "cd_tab_runs": "并排对比",
        "cd_intro": "两个数据集都缩减为共有的系统调用特征和两个类别（Benign 与 Malware）。模型在一个数据集上训练，在另一个数据集上测试。",
        "cd_shared": "使用 {n} 个共有的系统调用特征。",
        "cd_dir": "方向",
        "cd_dir_ck": "在 CICMalDroid2020 上训练 → 在 KronoDroid 上测试",
        "cd_dir_kc": "在 KronoDroid 上训练 → 在 CICMalDroid2020 上测试",
        "cd_baseline": "包含同域基线（在目标数据集上训练并测试）",
        "cd_run": "运行迁移测试",
        "cd_no_dir": "请至少选择一个方向。",
        "cd_done": "耗时 {sec:.1f} 秒。",
        "cd_res_title": "迁移结果",
        "cd_in": "同域",
        "cd_cross": "跨数据集",
        "cd_col_dir": "方向",
        "cd_col_setting": "设置",
        "cd_rec_ben": "Benign 召回率（%）",
        "cd_rec_mal": "Malware 召回率（%）",
        "cd_drop": "下降（百分点）",
        "cd_note": "两个数据集的采集方式和类别比例不同，出现下降是正常的。下降幅度以及哪一类的召回率损失最大，才是发现。",
        "cd_runs_none": "需要在流程运行器中为每个数据集至少运行一次（含留出法）。",
        "cd_pick_cic": "CICMalDroid2020 运行",
        "cd_pick_kr": "KronoDroid 运行",
        "cd_incompat": "这两次运行不能严格对比。请使用二分类模式的 CICMalDroid2020 和仅系统调用特征集的 KronoDroid。",
        "cd_runs_title": "留出法结果并排对比",
        "cd_download": "下载迁移结果（CSV）",
                # ---- Tahap 10: Explainability ----
        "xp_sub": "解释已训练模型依赖哪些特征，包括全局视角和单个样本（SHAP）。",
        "xp_empty": "需要包含留出法评估的运行。请先在运行流程时勾选留出法。",
        "xp_missing": "尚未安装 shap 包。请运行：pip install shap",
        "xp_class": "解释的类别",
        "xp_n": "解释的样本数",
        "xp_tab_global": "全局重要性",
        "xp_tab_dep": "特征依赖",
        "xp_tab_local": "单个样本",
        "xp_global_title": "平均 |SHAP 值|（对模型输出的平均影响）",
        "xp_vt_yes": "VirusTotal 列",
        "xp_vt_no": "其他特征",
        "xp_vt_warn": "VirusTotal 列位居重要特征之列。模型可能依赖标签来源而非应用行为；请与不含 VirusTotal 列的运行对比。",
        "xp_dep_feat": "特征",
        "xp_dep_title": "SHAP 值与特征值的关系",
        "xp_dep_x": "特征值（log1p）",
        "xp_dep_y": "SHAP 值",
        "xp_sample": "样本编号",
        "xp_true": "真实类别",
        "xp_pred": "预测类别",
        "xp_conf": "置信度",
        "xp_local_title": "为何得到该预测？主要贡献",
        "xp_up": "推向被解释的类别",
        "xp_down": "偏离被解释的类别",
        "xp_note": "SHAP 解释的是已训练的模型，而非现实中的因果。特征值显示的是 log1p 之后的结果；模型还会应用 RobustScaler。",
        "xp_computing": "正在计算 SHAP 值……",
                # ---- Tahap 11: Prediction + SUS ----
        "pr_sub": "使用流程运行器中已训练的模型进行预测，并收集可用性反馈（SUS）。",
        "pr_tab_pred": "预测",
        "pr_tab_sus": "可用性测试（SUS）",
        "pr_empty": "需要包含留出法评估的运行。请先在运行流程时勾选留出法。",
        "pr_src": "输入方式",
        "pr_src_csv": "上传 CSV",
        "pr_src_sample": "测试样本与假设分析",
        "pr_upload": "上传包含本次运行特征列的 CSV",
        "pr_missing": "缺少 {n} 个特征列，例如：{ex}。",
        "pr_nan": "{n} 个缺失值已替换为 0。",
        "pr_rows": "已预测 {n} 行。",
        "pr_col_pred": "预测",
        "pr_col_conf": "置信度（%）",
        "pr_download": "下载预测结果（CSV）",
        "pr_label_acc": "文件包含 Label 列：与其一致的比例为 {v:.2f}%。",
        "pr_dist": "预测类别分布",
        "pr_sample": "测试样本",
        "pr_edit": "假设分析：修改最重要的特征（原始计数）",
        "pr_orig": "原始预测",
        "pr_new": "修改后的预测",
        "pr_note": "预测来自所选运行中训练的模型。这是研究原型，不是安全产品。",
        "sus_intro": "请对每条陈述打分，1 表示非常不同意，5 表示非常同意。",
        "sus_name": "受访者编号（可选）",
        "sus_submit": "保存回答",
        "sus_incomplete": "请回答全部 10 条陈述。",
        "sus_saved": "回答已保存。SUS 得分：{s:.1f}",
        "sus_results": "结果",
        "sus_n": "受访者人数",
        "sus_mean": "SUS 平均得分",
        "sus_interp": "解读",
        "sus_below": "低于平均水平（< 68）",
        "sus_above": "高于平均水平（68 至 80.3）",
        "sus_excel": "优秀（80.3 及以上）",
        "sus_per_item": "各题平均分",
        "sus_per_resp": "各受访者 SUS 得分",
        "sus_download": "下载回答（CSV）",
        "sus_upload": "加载已保存的回答（CSV）",
        "sus_load": "加载文件",
        "sus_loaded": "已加载 {n} 份回答。",
        "sus_none": "暂无回答。",
        "sus_note": "回答仅保存在本次会话中。关闭应用前请下载 CSV。",
    },
}


def t(key, **kw):
    lang = st.session_state.get("lang", DEFAULT_LANG)
    s = TEXT.get(lang, TEXT["EN"]).get(key) or TEXT["EN"][key]
    return s.format(**kw) if kw else s


# ---------------- PALET (biru + krem + bata) ----------------
CLASS_COLORS = {
    "Benign": "#3A6EA5", "Malware": "#924A2F",
    "Adware": "#D2A35C", "Banking": "#B5603F", "SMS": "#233A6E", "Riskware": "#7A8DB3",
}
BLUE_SCALE = ["#DCE7F1", "#8FB4D1", "#3A6EA5", "#141F52"]

# ---------------- CSS ----------------
st.markdown("""


<style>
.steps{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:6px 0 14px;}
.step{background:#EDE3D0;color:#141F52;border-radius:999px;padding:4px 14px;
      font-size:.82rem;font-weight:600;}
.step.off{opacity:.4;text-decoration:line-through;}
.arrow{color:#7A8DB3;}

.block-container{max-width:1240px;padding-top:5.5rem;}

/* Navbar atas (bawaan Streamlit, diberi "penopang") */
[data-testid="stHeader"]{background:#FFFFFF;border-bottom:1px solid #E0D5BE;
      box-shadow:0 2px 10px rgba(20,31,82,.08);}
[data-testid="stHeader"]::before{content:"MALDROID INSIGHT";position:absolute;left:28px;
      top:50%;transform:translateY(-50%);font-weight:800;letter-spacing:.12em;
      font-size:.9rem;color:#141F52;white-space:nowrap;}
[data-testid="stToolbar"] > div > div:first-child{flex:0 0 240px;}
[data-testid="stToolbar"] > div > div:last-child{flex:0 0 240px;display:flex;justify-content:flex-end;}
[data-testid="stTopNav"]{flex:1 1 0;min-width:0;justify-content:center;}
[data-testid="stAppDeployButton"]{display:none;}
@media (max-width:1350px){[data-testid="stHeader"]::before{display:none;}
      [data-testid="stToolbar"] > div > div:first-child{flex-basis:0;}}

/* Dropdown bahasa di pojok kanan navbar */
.st-key-langbox{position:fixed;top:11px;right:64px;width:170px;z-index:1000000;}
.st-key-langbox [data-baseweb="select"] > div{min-height:36px;border-radius:999px;
      background:#F3EBDA;border:1px solid #E0D5BE;}

/* Hero & kartu */
.hero{background:linear-gradient(120deg,#141F52 0%,#254F73 58%,#5B94BD 100%);
      color:#fff;padding:22px 28px;border-radius:14px;margin-bottom:18px;
      box-shadow:0 4px 14px rgba(20,31,82,.22);}
.hero-title{font-size:1.9rem;font-weight:700;letter-spacing:-0.01em;}
.hero p{margin:.3rem 0 0;opacity:.92;}
[data-testid="stMetric"]{background:linear-gradient(180deg,#FFFFFF 0%,#F3EBDA 100%);
      border:1px solid #E0D5BE;border-left:5px solid #254F73;border-radius:10px;
      padding:14px 16px;box-shadow:0 1px 3px rgba(20,31,82,.10);}
[data-testid="stMetricValue"]{font-weight:700;color:#141F52;}
h2,h3,h5{color:#141F52;}
.chip{display:inline-block;padding:2px 11px;border-radius:999px;font-size:.78rem;
      font-weight:600;color:#fff;margin-left:6px;}
.chip.ok{background:#3C7A5A}.chip.warn{background:#C9983A}.chip.bad{background:#924A2F}
.prov{font-size:.82rem;color:#2B3566;background:linear-gradient(90deg,#EDE3D0,#F8F2E4);
      border-radius:8px;padding:8px 14px;margin:6px 0 14px;}
</style>
""", unsafe_allow_html=True)

DATASETS = {"CICMalDroid2020": D.load_cic, "KronoDroid (real-device)": D.load_krono}


# ---------------- HELPER ----------------
def hero(title, subtitle):
    st.markdown(f'<div class="hero"><div class="brand">MalDroid Insight</div>'
                f'<div class="hero-title">{title}</div><p>{subtitle}</p></div>',
                unsafe_allow_html=True)



def chip(text, lvl):
    return f'<span class="chip {lvl}">{text}</span>'


def level(value, warn, bad):
    return "ok" if value < warn else ("warn" if value < bad else "bad")


def style(fig, legend=False):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      font=dict(color="#141F52", size=14),
                      margin=dict(t=30, b=10, l=10, r=10))
    fig.update_yaxes(gridcolor="#E6DEC9")
    return fig


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                      # Streamlit versi lama
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def separation_scores(name):
    """Seberapa kuat tiap fitur memisahkan Benign vs non-Benign (log1p, standardized mean diff)."""
    df, feats, _ = DATASETS[name]()
    X = np.log1p(np.abs(df[feats].to_numpy(dtype=float)))
    is_ben = (df["Label"] == "Benign").to_numpy()
    a, b = X[is_ben], X[~is_ben]
    pooled = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2) + 1e-9
    score = np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled
    return pd.Series(score, index=feats).sort_values(ascending=False)


# ---------------- HALAMAN: DATASET OVERVIEW ----------------
def page_overview():
    hero(t("ov_title"), t("ov_sub"))
    name = st.radio(t("dataset"), list(DATASETS), horizontal=True, key="ds")

    try:
        df, feats, info = DATASETS[name]()
    except FileNotFoundError as e:
        st.error(t("err_file", f=e.filename))
        return

    is_krono = "KronoDroid" in name
    raw = len(df) + info["na_removed"] + info["dup_removed"]
    dup_pct = info["dup_removed"] / raw * 100
    na_pct = info["na_removed"] / raw * 100
    counts = df["Label"].value_counts()
    imb = counts.max() / counts.min()

    # --- Ringkasan ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("m_samples"), f"{len(df):,}")
    k2.metric(t("m_features"), len(feats))
    k3.metric(t("m_classes"), len(counts))
    k4.metric(t("m_ratio"), f"{imb:.2f} : 1")

    # --- Kualitas data ---
    q1, q2, q3 = st.columns(3)
    q1.markdown(f"**{t('q_dup')}** {chip(f'{dup_pct:.2f}%', level(dup_pct, 1, 5))}",
                unsafe_allow_html=True)
    q2.markdown(f"**{t('q_na')}** {chip(f'{na_pct:.2f}%', level(na_pct, 1, 5))}",
                unsafe_allow_html=True)
    q3.markdown(f"**{t('q_imb')}** {chip(f'{imb:.2f}:1', level(imb, 1.5, 4))}",
                unsafe_allow_html=True)

    src = ("real_malware_v1.csv + real_legitimate_v1.csv" if is_krono else "CICMALDROID2020.csv")
    st.markdown(f'<div class="prov">'
                f'{t("prov", src=src, raw=f"{raw:,}", ts=f"{pd.Timestamp.now():%d %b %Y %H:%M}")}'
                f'</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [t("tab_dist"), t("tab_feat"), t("tab_group"), t("tab_raw")])

    cc, cn, cp = t("col_class"), t("col_count"), t("col_pct")
    cf, cs, cg, cv = t("col_feat"), t("col_score"), t("col_group"), t("col_value")

    # --- Tab 1: distribusi kelas ---
    with tab1:
        c1, c2 = st.columns(2)
        ctype = c1.radio(t("view"), ["bar", "donut"], horizontal=True,
                         format_func=lambda k: t(k), key=f"ctype_{st.session_state.lang}")
        as_pct = c2.toggle(t("as_pct"), value=False, key="pct")
        dist = counts.rename_axis(cc).reset_index(name=cn)
        dist[cp] = (dist[cn] / dist[cn].sum() * 100).round(2)
        if ctype == "bar":
            ycol = cp if as_pct else cn
            fig = px.bar(dist, x=cc, y=ycol, color=cc,
                         color_discrete_map=CLASS_COLORS, text=ycol)
            show(style(fig))
        else:
            fig = px.pie(dist, names=cc, values=cn, hole=0.55,
                         color=cc, color_discrete_map=CLASS_COLORS)
            show(style(fig, legend=True))

    # --- Tab 2: penjelajah fitur ---
    with tab2:
        scores = separation_scores(name)
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown(f"##### {t('feat_top_title')}")
            top_n = st.slider(t("top_n"), 5, 30, 10, key="topn")
            top = scores.head(top_n).rename_axis(cf).reset_index(name=cs)
            if is_krono:
                _, cic_feats, _ = D.load_cic()
                groups = D.krono_groups(feats, cic_feats)
                sys_set = set(next(iter(groups.values())))
                top[cg] = np.where(top[cf].isin(sys_set), t("grp_syscall"), t("grp_non"))
                fig = px.bar(top, x=cs, y=cf, color=cg, orientation="h",
                             color_discrete_map={t("grp_syscall"): "#3A6EA5",
                                                 t("grp_non"): "#924A2F"})
                fig.update_yaxes(autorange="reversed")
                show(style(fig, legend=True))
                st.caption(t("caption_krono"))
            else:
                fig = px.bar(top, x=cs, y=cf, color=cs, orientation="h",
                             color_continuous_scale=BLUE_SCALE)
                fig.update_coloraxes(showscale=False)
                fig.update_yaxes(autorange="reversed")
                show(style(fig))

        with right:
            st.markdown(f"##### {t('feat_dist_title')}")
            feat = st.selectbox(t("pick_feat"), list(scores.index), key=f"feat_{name}")
            kind = st.radio(t("chart_kind"), ["box", "violin", "hist"], horizontal=True,
                            format_func=lambda k: t(k), key=f"gkind_{st.session_state.lang}")
            use_log = st.checkbox(t("log1p"), value=True, key="log")
            vals = df[feat].astype(float)
            if use_log:
                vals = np.log1p(np.abs(vals))
            pdf = pd.DataFrame({cc: df["Label"], cv: vals})
            pdf = pdf.sample(min(len(pdf), 20000), random_state=42)
            if kind == "box":
                fig = px.box(pdf, x=cc, y=cv, color=cc, color_discrete_map=CLASS_COLORS)
            elif kind == "violin":
                fig = px.violin(pdf, x=cc, y=cv, color=cc, box=True,
                                color_discrete_map=CLASS_COLORS)
            else:
                fig = px.histogram(pdf, x=cv, color=cc, barmode="overlay", opacity=0.65,
                                   nbins=40, color_discrete_map=CLASS_COLORS)
            show(style(fig, legend=(kind == "hist")))
            st.caption(t("sep_caption", f=feat, s=scores[feat]))

    # --- Tab 3: kelompok fitur ---
    with tab3:
        if is_krono:
            _, cic_feats, _ = D.load_cic()
            sys_feats, other_feats = D.krono_groups(feats, cic_feats).values()
            g = pd.DataFrame({cg: [t("grp_sys_full"), t("grp_other_full")],
                              t("col_nfeat"): [len(sys_feats), len(other_feats)]})
            fig = px.bar(g, x=t("col_nfeat"), y=cg, orientation="h", color=cg,
                         color_discrete_sequence=["#3A6EA5", "#924A2F"], text=t("col_nfeat"))
            show(style(fig))
            st.warning(t("warn_vt"))
            with st.expander(t("expander_nonsys")):
                st.dataframe(pd.DataFrame({cf: other_feats}), hide_index=True)
        else:
            st.info(t("info_cic"))

    # --- Tab 4: data mentah ---
    with tab4:
        f1, f2, f3 = st.columns([1, 2, 1])
        all_cls = sorted(df["Label"].unique())
        sel_cls = f1.multiselect(t("filter_cls"), all_cls, default=all_cls, key=f"cls_{name}")
        sel_cols = f2.multiselect(t("filter_cols"), feats, default=feats[:8], key=f"cols_{name}")
        n_rows = f3.slider(t("n_rows"), 10, 500, 50, key="rows")
        st.dataframe(df[df["Label"].isin(sel_cls)][["Label"] + sel_cols].head(n_rows))


# ---------------- PLACEHOLDER MODUL LAIN ----------------
def coming_soon(key, n):
    hero(t(key), t("soon_sub", stage=t("stage", n=n)))
    st.info(t("soon_info"))


# ---------------- MESIN PIPELINE (Tahap 7) ----------------
SEED = 42
MODEL_NAMES = ["LightGBM", "XGBoost", "RandomForest", "ExtraTrees"]
VT_COLUMNS = ["Detection_Ratio", "TimesSubmitted"]     # kolom turunan VirusTotal


def feature_scenarios(feats, cic_feats=None):
    """Skenario set fitur. 'syscall' dan 'no_vt' hanya untuk KronoDroid (cic_feats diberikan)."""
    out = {"all": list(feats)}
    if cic_feats is not None:
        cic = set(cic_feats)
        out["syscall"] = [c for c in feats if c in cic]
        out["no_vt"] = [c for c in feats if c not in VT_COLUMNS]
    return out


def prepare_xy(df, cols, mode="orig"):
    """log1p(|x|) (stateless, aman sebelum split) + encode label. mode: 'orig' | 'binary'."""
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


def get_models(fast=True, n_classes=2):
    n_big, n_tree = (200, 150) if fast else (1000, 700)
    lr = 0.1 if fast else 0.03
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=n_big, max_depth=8, learning_rate=lr, num_leaves=127,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.05, reg_lambda=0.5, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=-1),
        "XGBoost": XGBClassifier(
            n_estimators=n_big, max_depth=7, learning_rate=lr, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=5, reg_alpha=0.05, reg_lambda=0.5,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            random_state=SEED, n_jobs=-1),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_tree, max_features="sqrt", class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=n_tree, class_weight="balanced", random_state=SEED, n_jobs=-1),
    }


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


def run_pipeline(X, y, feats, classes, models, holdout=True, cv=True, folds=5,
                 smote=True, fast=True, progress=None):
    """Jalankan hold-out 80:20 dan/atau k-fold CV untuk tiap model.

    progress(done, total, text) dipanggil sebelum tiap langkah (untuk progress bar).
    """
    n_classes = len(classes)
    defs = get_models(fast, n_classes)
    total = len(models) * ((1 if holdout else 0) + (folds if cv else 0))
    done = 0
    res = {"holdout": {}, "cv": {}, "importance": {}, "fitted": {},
           "classes": classes, "features": list(feats)}

    if holdout:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)
        res["test"] = {"X": X_te[:2000], "y": y_te[:2000]}      # untuk SHAP / prediksi nanti
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
            m["cm"] = confusion_matrix(y_te, pred, labels=list(range(n_classes)))
            res["holdout"][name] = m
            res["fitted"][name] = pipe
            res["importance"][name] = pd.Series(
                pipe.named_steps["clf"].feature_importances_, index=feats
            ).sort_values(ascending=False)
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
                "acc": fm["acc"].mean(), "std": fm["acc"].std(ddof=0),
                "prec": fm["prec"].mean(), "rec": fm["rec"].mean(),
                "f1": fm["f1"].mean(), "f1_macro": fm["f1_macro"].mean(),
                "folds": fm["acc"].tolist(), "time": time.time() - t0,
            }
    if progress:
        progress(total, total, "done")
    return res


# ---------------- HALAMAN: PIPELINE RUNNER ----------------
def page_pipeline():
    hero(t("nav_pipeline"), t("pl_sub"))
    st.session_state.setdefault("runs", [])
    lang = st.session_state.lang

    name = st.radio(t("dataset"), list(DATASETS), horizontal=True, key="pl_ds")
    try:
        df, feats, info = DATASETS[name]()
    except FileNotFoundError as e:
        st.error(t("err_file", f=e.filename))
        return
    is_krono = "KronoDroid" in name
    cic_feats = D.load_cic()[1] if is_krono else None
    scen = feature_scenarios(feats, cic_feats)

    left, right = st.columns(2)
    with left:
        sc_key = st.selectbox(t("pl_scenario"), list(scen), key=f"pl_sc_{name}_{lang}",
                              format_func=lambda k: t("pl_sc_" + k, n=len(scen[k])))
        if not is_krono:
            st.caption(t("pl_cic_na"))
        if is_krono:
            mode = "binary"
        else:
            mode = st.radio(t("pl_labels"), ["orig", "binary"], horizontal=True,
                            key=f"pl_mode_{lang}",
                            format_func=lambda k: t("pl_lab_" + k, n=df["Label"].nunique()))
        models = st.multiselect(t("pl_models"), MODEL_NAMES, default=MODEL_NAMES,
                                key="pl_models")
    with right:
        e1, e2 = st.columns(2)
        do_ho = e1.checkbox(t("pl_holdout"), value=True, key="pl_ho")
        do_cv = e2.checkbox(t("pl_cv"), value=True, key="pl_cvc")
        folds = st.slider(t("pl_folds"), 3, 10, 5, key="pl_folds", disabled=not do_cv)
        use_smote = st.toggle(t("pl_smote"), value=True, key="pl_smote")
        speed = st.radio(t("pl_speed"), ["fast", "full"], horizontal=True,
                         key=f"pl_speed_{lang}", format_func=lambda k: t("pl_" + k))
        total = len(df)
        if total > 2000:
            n_used = int(st.number_input(t("pl_maxrows"), min_value=2000, max_value=total,
                                         value=min(20000, total), step=1000,
                                         key=f"pl_rows_{name}"))
        else:
            n_used = total
        st.caption(t("pl_rows_used", n=f"{n_used:,}", total=f"{total:,}"))

    steps = [("Dedup", True), ("log1p", True), ("RobustScaler", True),
             ("SMOTE", use_smote), ("Model", True), ("Evaluate", True)]
    chips = '<span class="arrow">→</span>'.join(
        f'<span class="step{"" if on else " off"}">{n}</span>' for n, on in steps)
    st.markdown(f'<div class="steps">{chips}</div>', unsafe_allow_html=True)

    if st.button(t("pl_run"), type="primary", key="pl_go"):
        if not models:
            st.warning(t("pl_no_model"))
        elif not (do_ho or do_cv):
            st.warning(t("pl_no_eval"))
        else:
            cols = scen[sc_key]
            X, y, classes = prepare_xy(df, cols, mode)
            X, y = stratified_sample(X, y, n_used)
            bar = st.progress(0.0)
            status = st.empty()

            def cb(done, tot, msg):
                bar.progress(done / tot if tot else 1.0)
                status.markdown(f"**{msg}**")

            t0 = time.time()
            res = run_pipeline(X, y, cols, classes, models, holdout=do_ho, cv=do_cv,
                               folds=folds, smote=use_smote, fast=(speed == "fast"),
                               progress=cb)
            secs = time.time() - t0
            bar.empty()
            status.empty()
            run_id = len(st.session_state.runs) + 1
            st.session_state.runs.append({
                "id": run_id, "dataset": name, "scenario": sc_key, "mode": mode,
                "n_feat": len(cols), "n_rows": len(y), "smote": use_smote,
                "fast": speed == "fast", "folds": folds, "seconds": secs, **res})
            st.success(t("pl_done", sec=secs, id=run_id))

    if st.session_state.runs:
        show_run(st.session_state.runs[-1])


def show_run(run):
    st.markdown(f"### {t('pl_res_title')}")
    fs = t("pl_sc_" + run["scenario"], n=run["n_feat"])
    lab = t("pl_lab_" + run["mode"], n=len(run["classes"]))
    smote = t("pl_on") if run["smote"] else t("pl_off")
    rows_txt = f"{run['n_rows']:,}"
    cfg = t("pl_cfg", id=run["id"], ds=run["dataset"], fs=fs, lab=lab, rows=rows_txt, smote=smote)
    st.markdown(f'<div class="prov">{cfg}</div>', unsafe_allow_html=True)
    is_krono = "KronoDroid" in run["dataset"]
    if is_krono and run["scenario"] == "all":
        st.warning(t("pl_leak"))

    cm_col = t("pl_col_model")
    if run["holdout"]:
        st.markdown(f"##### {t('pl_res_holdout')}")
        ho = run["holdout"]
        st.dataframe(pd.DataFrame({
            cm_col: list(ho),
            "Acc (%)": [round(v["acc"] * 100, 2) for v in ho.values()],
            "Prec (%)": [round(v["prec"] * 100, 2) for v in ho.values()],
            "Rec (%)": [round(v["rec"] * 100, 2) for v in ho.values()],
            "F1 (%)": [round(v["f1"] * 100, 2) for v in ho.values()],
            t("pl_col_time"): [round(v["time"], 1) for v in ho.values()],
        }), hide_index=True)
        best = max(ho, key=lambda k: ho[k]["acc"])
        st.caption(t("pl_best", m=best, v=ho[best]["acc"] * 100))

    if run["cv"]:
        st.markdown(f"##### {t('pl_res_cv')}")
        cv = run["cv"]
        st.dataframe(pd.DataFrame({
            cm_col: list(cv),
            "Acc (%)": [round(v["acc"] * 100, 2) for v in cv.values()],
            t("pl_col_std"): [round(v["std"] * 100, 3) for v in cv.values()],
            "Prec (%)": [round(v["prec"] * 100, 2) for v in cv.values()],
            "Rec (%)": [round(v["rec"] * 100, 2) for v in cv.values()],
            "F1 (%)": [round(v["f1"] * 100, 2) for v in cv.values()],
            t("pl_col_folds"): [", ".join(f"{a * 100:.2f}" for a in v["folds"])
                                for v in cv.values()],
        }), hide_index=True)
        st.caption(t("pl_cv_note"))

    imp = run["importance"].get("LightGBM")
    if imp is not None:
        st.markdown(f"##### {t('pl_top_feats')}")
        top = imp.head(8)
        tbl = {t("col_feat"): list(top.index),
               t("pl_col_importance"): [float(v) for v in top.values]}
        if is_krono:
            sys_set = set(D.load_cic()[1])
            tbl[t("col_group")] = [t("grp_syscall") if f in sys_set else t("grp_non")
                                   for f in top.index]
        st.dataframe(pd.DataFrame(tbl), hide_index=True)

    runs = st.session_state.runs
    with st.expander(t("pl_hist", n=len(runs))):
        hist = []
        for r in runs:
            ho = r["holdout"]
            best_acc = round(max(v["acc"] for v in ho.values()) * 100, 2) if ho else None
            hist.append({
                t("pl_col_run"): r["id"], t("dataset"): r["dataset"],
                t("pl_scenario"): t("pl_sc_" + r["scenario"], n=r["n_feat"]),
                t("pl_labels"): t("pl_lab_" + r["mode"], n=len(r["classes"])),
                t("pl_col_best"): best_acc,
            })
        st.dataframe(pd.DataFrame(hist), hide_index=True)

# ---------------- HALAMAN: MODEL COMPARISON (Tahap 8) ----------------
MODEL_COLORS = {"LightGBM": "#254F73", "XGBoost": "#3A6EA5",
                "RandomForest": "#D2A35C", "ExtraTrees": "#924A2F"}
METRICS = {"acc": "Accuracy", "prec": "Precision", "rec": "Recall",
           "f1": "F1 (weighted)", "f1_macro": "F1 (macro)"}
METRIC_COLORS = ["#141F52", "#254F73", "#3A6EA5", "#8FB4D1", "#924A2F"]


def run_label(r):
    fs = t("pl_sc_" + r["scenario"], n=r["n_feat"])
    lab = t("pl_lab_" + r["mode"], n=len(r["classes"]))
    return t("mc_run_lbl", id=r["id"], ds=r["dataset"], fs=fs, lab=lab)


def short_label(r):
    """Label pendek, netral bahasa, untuk sumbu grafik antar-run."""
    return f"#{r['id']} {r['dataset'].split()[0]}/{r['scenario']}/{r['mode']} · {r['n_rows']:,}"


def runs_table(selected):
    rows = []
    for r in selected:
        base = {"run": r["id"], "label": short_label(r), "dataset": r["dataset"],
                "scenario": r["scenario"], "mode": r["mode"], "n_feat": r["n_feat"],
                "n_rows": r["n_rows"], "smote": r["smote"], "fast": r["fast"]}
        for ev in ("holdout", "cv"):
            for m, v in r[ev].items():
                rows.append({**base, "eval": ev, "model": m,
                             "acc": round(v["acc"] * 100, 3), "prec": round(v["prec"] * 100, 3),
                             "rec": round(v["rec"] * 100, 3), "f1": round(v["f1"] * 100, 3),
                             "f1_macro": round(v["f1_macro"] * 100, 3),
                             "std": round(v["std"] * 100, 3) if "std" in v else None})
    return pd.DataFrame(rows)


def page_models():
    hero(t("nav_models"), t("mc_sub"))
    runs = st.session_state.get("runs", [])
    if not runs:
        st.info(t("mc_empty"))
        return

    by_id = {r["id"]: r for r in runs}
    ids = [r["id"] for r in runs][::-1]                    # run terbaru di atas
    rid = st.selectbox(t("mc_pick"), ids, key="mc_run", format_func=lambda i: run_label(by_id[i]))
    run = by_id[rid]
    mdl = t("pl_col_model")

    tab1, tab2, tab3, tab4 = st.tabs(
        [t("mc_tab_metrics"), t("mc_tab_cv"), t("mc_tab_cm"), t("mc_tab_runs")])

    # --- Tab 1: metrik hold-out ---
    with tab1:
        ho = run["holdout"]
        if not ho:
            st.info(t("mc_no_ho"))
        else:
            long = pd.DataFrame([{mdl: m, t("mc_metric"): lab, "%": v[k] * 100}
                                 for m, v in ho.items() for k, lab in METRICS.items()
                                 for v in [v]])
            fig = px.bar(long, x=mdl, y="%", color=t("mc_metric"), barmode="group",
                         color_discrete_sequence=METRIC_COLORS)
            fig.update_yaxes(range=[max(0, long["%"].min() - 5), 100])
            show(style(fig, legend=True))
            st.caption(t("mc_axis_note"))

            c1, c2 = st.columns([1.1, 1])
            with c1:
                tbl = pd.DataFrame({
                    mdl: list(ho),
                    **{lab: [round(v[k] * 100, 2) for v in ho.values()]
                       for k, lab in METRICS.items()},
                    t("pl_col_time"): [round(v["time"], 1) for v in ho.values()],
                })
                st.dataframe(tbl, hide_index=True)
                st.download_button(t("mc_download"), tbl.to_csv(index=False).encode(),
                                   file_name=f"run{run['id']}_holdout.csv", mime="text/csv",
                                   key="mc_dl1")
            with c2:
                st.markdown(f"##### {t('mc_time_title')}")
                sc = pd.DataFrame({mdl: list(ho),
                                   t("pl_col_time"): [v["time"] for v in ho.values()],
                                   "Acc (%)": [v["acc"] * 100 for v in ho.values()]})
                fig = px.scatter(sc, x=t("pl_col_time"), y="Acc (%)", color=mdl, text=mdl,
                                 color_discrete_map=MODEL_COLORS)
                fig.update_traces(marker=dict(size=14), textposition="top center")
                show(style(fig))

    # --- Tab 2: cross-validation ---
    with tab2:
        cv = run["cv"]
        if not cv:
            st.info(t("mc_no_cv"))
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"##### {t('mc_cv_title')}")
                dfc = pd.DataFrame({mdl: list(cv),
                                    "Acc (%)": [v["acc"] * 100 for v in cv.values()],
                                    t("pl_col_std"): [v["std"] * 100 for v in cv.values()]})
                fig = px.bar(dfc, x=mdl, y="Acc (%)", error_y=t("pl_col_std"), color=mdl,
                             color_discrete_map=MODEL_COLORS)
                fig.update_yaxes(range=[max(0, (dfc["Acc (%)"] - dfc[t("pl_col_std")]).min() - 3), 100])
                show(style(fig))
            with c2:
                st.markdown(f"##### {t('mc_fold_title')}")
                fl = pd.DataFrame([{mdl: m, "Fold": i + 1, "Acc (%)": a * 100}
                                   for m, v in cv.items() for i, a in enumerate(v["folds"])])
                fig = px.box(fl, x=mdl, y="Acc (%)", color=mdl, points="all",
                             color_discrete_map=MODEL_COLORS)
                show(style(fig))
            st.caption(t("pl_cv_note"))
            st.dataframe(pd.DataFrame({
                mdl: list(cv),
                "Acc (%)": [round(v["acc"] * 100, 2) for v in cv.values()],
                t("pl_col_std"): [round(v["std"] * 100, 3) for v in cv.values()],
                "F1 (%)": [round(v["f1"] * 100, 2) for v in cv.values()],
                "F1 macro (%)": [round(v["f1_macro"] * 100, 2) for v in cv.values()],
                t("pl_col_time"): [round(v["time"], 1) for v in cv.values()],
            }), hide_index=True)

    # --- Tab 3: confusion matrix ---
    with tab3:
        ho = run["holdout"]
        if not ho:
            st.info(t("mc_no_ho"))
        else:
            c1, c2 = st.columns([1, 1])
            model = c1.selectbox(t("mc_cm_model"), list(ho), key=f"mc_cm_{rid}")
            pct = c2.toggle(t("mc_cm_norm"), value=True, key="mc_norm")
            cm = np.asarray(ho[model]["cm"], dtype=float)
            if pct:
                cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
            fig = px.imshow(cm, x=run["classes"], y=run["classes"],
                            text_auto=".1f" if pct else ".0f",
                            color_continuous_scale=BLUE_SCALE, aspect="auto",
                            labels=dict(x=t("mc_cm_pred"), y=t("mc_cm_true"),
                                        color="%" if pct else "n"))
            fig.update_coloraxes(showscale=False)
            show(style(fig))

    # --- Tab 4: antar run (ablation) ---
    with tab4:
        sel_ids = st.multiselect(t("mc_pick_runs"), ids, default=ids[:6], key="mc_runs_sel",
                                 format_func=lambda i: run_label(by_id[i]))
        src = st.radio(t("mc_source"), ["holdout", "cv"], horizontal=True,
                       key=f"mc_src_{st.session_state.lang}",
                       format_func=lambda k: t("pl_res_holdout" if k == "holdout" else "pl_res_cv"))
        if sel_ids:
            tbl = runs_table([by_id[i] for i in sorted(sel_ids)])
            sub = tbl[tbl["eval"] == src] if len(tbl) else tbl
            if len(sub):
                st.markdown(f"##### {t('mc_runs_title')}")
                fig = px.bar(sub, x="label", y="acc", color="model", barmode="group",
                             color_discrete_map=MODEL_COLORS,
                             labels={"label": "", "acc": "Acc (%)", "model": mdl})
                fig.update_yaxes(range=[max(0, sub["acc"].min() - 5), 100])
                show(style(fig, legend=True))
                st.caption(t("mc_axis_note"))
                piv = sub.pivot_table(index="model", columns="label", values="acc",
                                      aggfunc="first").round(2)
                st.dataframe(piv)
                st.caption(t("mc_runs_note"))
            else:
                st.info(t("mc_no_ho" if src == "holdout" else "mc_no_cv"))
            st.download_button(t("mc_download_all"), tbl.to_csv(index=False).encode(),
                               file_name="maldroid_runs.csv", mime="text/csv", key="mc_dl2")
            
# ---------------- HALAMAN: CROSS-DATASET (Tahap 9) ----------------
DS_COLORS = {"CICMalDroid2020": "#3A6EA5", "KronoDroid": "#924A2F"}


def _cross_row(direction, name, setting, y_true, y_pred, secs):
    m = metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"direction": direction, "model": name, "setting": setting,
            "acc": m["acc"] * 100, "f1_macro": m["f1_macro"] * 100,
            "rec_ben": cm[0, 0] / max(cm[0].sum(), 1) * 100,
            "rec_mal": cm[1, 1] / max(cm[1].sum(), 1) * 100,
            "time": secs}


def transfer_eval(direction, models, n_rows, fast, smote, baseline, progress=None):
    """direction: 'ck' (latih CIC, uji Krono) atau 'kc'. Fitur: irisan syscall, label biner."""
    cic_df, cic_feats, _ = D.load_cic()
    kr_df, kr_feats, _ = D.load_krono()
    cols = feature_scenarios(kr_feats, cic_feats)["syscall"]
    src_df, tgt_df = (cic_df, kr_df) if direction == "ck" else (kr_df, cic_df)
    Xs, ys, cs = prepare_xy(src_df, cols, "binary")
    Xt, yt, ct = prepare_xy(tgt_df, cols, "binary")
    if cs != ct:
        raise ValueError(f"Kelas biner tidak sama: {cs} vs {ct}")
    Xs, ys = stratified_sample(Xs, ys, n_rows)
    Xt, yt = stratified_sample(Xt, yt, n_rows)
    if baseline:        # uji di 20% data target; baseline melatih di 80% sisanya
        Xtr, Xev, ytr, yev = train_test_split(Xt, yt, test_size=0.2, random_state=SEED, stratify=yt)
    else:
        Xev, yev = Xt, yt

    defs = get_models(fast, 2)
    total = len(models) * (2 if baseline else 1)
    done, rows = 0, []
    for name in models:
        if progress:
            progress(done, total, f"{name} · {direction} · cross")
        t0 = time.time()
        pipe = make_pipeline(clone(defs[name]), smote).fit(Xs, ys)
        rows.append(_cross_row(direction, name, "cross", yev, pipe.predict(Xev), time.time() - t0))
        done += 1
        if baseline:
            if progress:
                progress(done, total, f"{name} · {direction} · in-domain")
            t0 = time.time()
            pipe = make_pipeline(clone(defs[name]), smote).fit(Xtr, ytr)
            rows.append(_cross_row(direction, name, "in", yev, pipe.predict(Xev), time.time() - t0))
            done += 1
    if progress:
        progress(total, total, "done")
    return rows


def show_transfer(df):
    df = df.copy()
    df["dir_l"] = df["direction"].map(lambda d: t("cd_dir_" + d))
    df["set_l"] = df["setting"].map(lambda s: t("cd_in") if s == "in" else t("cd_cross"))
    mdl = t("pl_col_model")

    fig = px.bar(df, x="model", y="acc", color="set_l", barmode="group", facet_col="dir_l",
                 color_discrete_map={t("cd_in"): "#3A6EA5", t("cd_cross"): "#924A2F"},
                 labels={"model": mdl, "acc": "Acc (%)", "set_l": t("cd_col_setting")})
    fig.update_yaxes(range=[max(0, df["acc"].min() - 5), 100])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1]))
    show(style(fig, legend=True))
    st.caption(t("mc_axis_note"))

    tbl = pd.DataFrame({
        t("cd_col_dir"): df["dir_l"], mdl: df["model"], t("cd_col_setting"): df["set_l"],
        "Acc (%)": df["acc"].round(2), "F1 macro (%)": df["f1_macro"].round(2),
        t("cd_rec_ben"): df["rec_ben"].round(2), t("cd_rec_mal"): df["rec_mal"].round(2),
        t("pl_col_time"): df["time"].round(1)})
    st.dataframe(tbl, hide_index=True)

    pv = df.pivot_table(index=["direction", "model"], columns="setting", values="acc")
    if "in" in pv.columns and "cross" in pv.columns:
        d = pv.reset_index()
        st.dataframe(pd.DataFrame({
            t("cd_col_dir"): d["direction"].map(lambda k: t("cd_dir_" + k)),
            mdl: d["model"], t("cd_in") + " (%)": d["in"].round(2),
            t("cd_cross") + " (%)": d["cross"].round(2),
            t("cd_drop"): (d["in"] - d["cross"]).round(2)}), hide_index=True)
    st.caption(t("cd_note"))


def page_cross():
    hero(t("nav_cross"), t("cd_sub"))
    lang = st.session_state.lang
    st.session_state.setdefault("cross_runs", [])
    tab1, tab2 = st.tabs([t("cd_tab_transfer"), t("cd_tab_runs")])

    # --- Tab 1: uji transfer ---
    with tab1:
        try:
            cic_feats = D.load_cic()[1]
            kr_feats = D.load_krono()[1]
        except FileNotFoundError as e:
            st.error(t("err_file", f=e.filename))
            return
        st.markdown(t("cd_intro"))
        st.info(t("cd_shared", n=len(feature_scenarios(kr_feats, cic_feats)["syscall"])))

        c1, c2 = st.columns(2)
        dirs = c1.multiselect(t("cd_dir"), ["ck", "kc"], default=["ck", "kc"], key="cd_dirs",
                              format_func=lambda k: t("cd_dir_" + k))
        models = c1.multiselect(t("pl_models"), MODEL_NAMES, default=["LightGBM", "XGBoost"],
                                key="cd_models")
        n_rows = int(c2.number_input(t("pl_maxrows"), min_value=2000, max_value=100000,
                                     value=20000, step=1000, key="cd_rows"))
        speed = c2.radio(t("pl_speed"), ["fast", "full"], horizontal=True,
                         key=f"cd_speed_{lang}", format_func=lambda k: t("pl_" + k))
        smote = c2.toggle(t("pl_smote"), value=True, key="cd_smote")
        base = c2.checkbox(t("cd_baseline"), value=True, key="cd_base")

        if st.button(t("cd_run"), type="primary", key="cd_go"):
            if not dirs:
                st.warning(t("cd_no_dir"))
            elif not models:
                st.warning(t("pl_no_model"))
            else:
                grand = len(dirs) * len(models) * (2 if base else 1)
                bar, status = st.progress(0.0), st.empty()
                offset, all_rows, t0 = 0, [], time.time()
                run_no = len(st.session_state.cross_runs) + 1
                for d in dirs:
                    def cb(done, tot, msg, _off=offset):
                        bar.progress(min((_off + done) / grand, 1.0))
                        status.markdown(f"**{msg}**")
                    rows = transfer_eval(d, models, n_rows, speed == "fast", smote, base, cb)
                    offset += len(models) * (2 if base else 1)
                    all_rows += rows
                bar.empty()
                status.empty()
                for r in all_rows:
                    r.update({"run": run_no, "n_rows": n_rows, "smote": smote,
                              "fast": speed == "fast"})
                st.session_state.cross_runs.append(all_rows)
                st.success(t("cd_done", sec=time.time() - t0))

        if st.session_state.cross_runs:
            st.markdown(f"### {t('cd_res_title')}")
            show_transfer(pd.DataFrame(st.session_state.cross_runs[-1]))
            allr = pd.DataFrame([r for run in st.session_state.cross_runs for r in run])
            st.download_button(t("cd_download"), allr.round(3).to_csv(index=False).encode(),
                               file_name="maldroid_transfer.csv", mime="text/csv", key="cd_dl")

    # --- Tab 2: dua run berdampingan ---
    with tab2:
        runs = st.session_state.get("runs", [])
        cic_runs = [r for r in runs if r["dataset"].startswith("CIC") and r["holdout"]]
        kr_runs = [r for r in runs if r["dataset"].startswith("Krono") and r["holdout"]]
        if not cic_runs or not kr_runs:
            st.info(t("cd_runs_none"))
        else:
            by_id = {r["id"]: r for r in runs}
            a1, a2 = st.columns(2)
            ia = a1.selectbox(t("cd_pick_cic"), [r["id"] for r in cic_runs][::-1], key="cd_pa",
                              format_func=lambda i: run_label(by_id[i]))
            ib = a2.selectbox(t("cd_pick_kr"), [r["id"] for r in kr_runs][::-1], key="cd_pb",
                              format_func=lambda i: run_label(by_id[i]))
            ra, rb = by_id[ia], by_id[ib]
            if not (ra["mode"] == "binary" and rb["scenario"] == "syscall"):
                st.warning(t("cd_incompat"))
            metric = st.radio(t("mc_metric"), list(METRICS), horizontal=True, key="cd_metric",
                              format_func=lambda k: METRICS[k])
            rows = []
            for r, ds in ((ra, "CICMalDroid2020"), (rb, "KronoDroid")):
                for m, v in r["holdout"].items():
                    rows.append({"model": m, "dataset": ds, "value": v[metric] * 100})
            df = pd.DataFrame(rows)
            st.markdown(f"##### {t('cd_runs_title')}")
            fig = px.bar(df, x="model", y="value", color="dataset", barmode="group",
                         color_discrete_map=DS_COLORS,
                         labels={"model": t("pl_col_model"), "value": METRICS[metric] + " (%)",
                                 "dataset": t("dataset")})
            fig.update_yaxes(range=[max(0, df["value"].min() - 5), 100])
            show(style(fig, legend=True))
            st.caption(t("mc_axis_note"))
            st.dataframe(df.pivot_table(index="model", columns="dataset", values="value").round(2))

# ---------------- HALAMAN: EXPLAINABILITY (Tahap 10) ----------------
def get_shap(run, model, n):
    """Nilai SHAP bentuk (n_sampel, n_fitur, n_kelas). Disimpan di sesi agar tidak dihitung ulang."""
    cache = st.session_state.setdefault("shap_cache", {})
    key = (run["id"], model, n)
    if key not in cache:
        import shap
        pipe = run["fitted"][model]
        Xs = pipe.named_steps["scaler"].transform(run["test"]["X"][:n])
        sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xs, check_additivity=False)
        if isinstance(sv, list):                      # versi lama: satu array per kelas
            arr = np.stack(sv, axis=-1)
        else:
            arr = np.asarray(sv)
            if arr.ndim == 2:                         # biner: nilai untuk kelas 1
                arr = np.stack([-arr, arr], axis=-1)
        cache[key] = arr
    return cache[key]


def page_explain():
    hero(t("nav_explain"), t("xp_sub"))
    try:
        import shap  # noqa: F401
    except ImportError:
        st.error(t("xp_missing"))
        return
    runs = [r for r in st.session_state.get("runs", []) if r["holdout"] and r.get("fitted")]
    if not runs:
        st.info(t("xp_empty"))
        return

    by_id = {r["id"]: r for r in runs}
    ids = [r["id"] for r in runs][::-1]
    c1, c2, c3 = st.columns(3)
    rid = c1.selectbox(t("mc_pick"), ids, key="xp_run", format_func=lambda i: run_label(by_id[i]))
    run = by_id[rid]
    model = c2.selectbox(t("pl_col_model"), list(run["fitted"]), key=f"xp_model_{rid}")
    classes = run["classes"]
    cls = c3.selectbox(t("xp_class"), classes, key=f"xp_cls_{rid}",
                       index=classes.index("Malware") if "Malware" in classes else 0)
    ci = classes.index(cls)

    n_max = len(run["test"]["X"])
    n = st.slider(t("xp_n"), 100, n_max, min(500, n_max), 50, key=f"xp_n_{rid}")
    with st.spinner(t("xp_computing")):
        arr = get_shap(run, model, n)
    feats = run["features"]
    Xraw = run["test"]["X"][:n]
    y_true = run["test"]["y"][:n]
    pipe = run["fitted"][model]
    st.caption(t("xp_note"))

    tab1, tab2, tab3 = st.tabs([t("xp_tab_global"), t("xp_tab_dep"), t("xp_tab_local")])
    glob = pd.Series(np.abs(arr[:, :, ci]).mean(axis=0), index=feats).sort_values(ascending=False)

    # --- Tab 1: kepentingan global ---
    with tab1:
        top_n = st.slider(t("top_n"), 5, 30, 15, key="xp_topn")
        top = glob.head(top_n)
        df = pd.DataFrame({t("col_feat"): top.index, "SHAP": top.values})
        df[t("col_group")] = [t("xp_vt_yes") if f in VT_COLUMNS else t("xp_vt_no")
                              for f in top.index]
        st.markdown(f"##### {t('xp_global_title')}")
        fig = px.bar(df, x="SHAP", y=t("col_feat"), color=t("col_group"), orientation="h",
                     color_discrete_map={t("xp_vt_yes"): "#924A2F", t("xp_vt_no"): "#3A6EA5"})
        fig.update_yaxes(autorange="reversed")
        show(style(fig, legend=True))
        if any(f in VT_COLUMNS for f in top.index):
            st.warning(t("xp_vt_warn"))

    # --- Tab 2: dependence ---
    with tab2:
        feat = st.selectbox(t("xp_dep_feat"), list(glob.index[:30]), key=f"xp_feat_{rid}")
        j = feats.index(feat)
        names = np.array(classes)[y_true]
        dep = pd.DataFrame({t("xp_dep_x"): Xraw[:, j], t("xp_dep_y"): arr[:, j, ci],
                            t("col_class"): names})
        st.markdown(f"##### {t('xp_dep_title')}")
        fig = px.scatter(dep, x=t("xp_dep_x"), y=t("xp_dep_y"), color=t("col_class"),
                         color_discrete_map=CLASS_COLORS, opacity=0.7)
        show(style(fig, legend=True))

    # --- Tab 3: satu sampel ---
    with tab3:
        idx = int(st.number_input(t("xp_sample"), min_value=0, max_value=n - 1, value=0,
                                  step=1, key=f"xp_i_{rid}_{n}"))
        proba = pipe.predict_proba(Xraw[idx:idx + 1])[0]
        k1, k2, k3 = st.columns(3)
        k1.metric(t("xp_true"), classes[int(y_true[idx])])
        k2.metric(t("xp_pred"), classes[int(np.argmax(proba))])
        k3.metric(t("xp_conf"), f"{proba.max() * 100:.1f}%")

        contrib = arr[idx, :, ci]
        order = np.argsort(-np.abs(contrib))[:10]
        loc = pd.DataFrame({
            t("col_feat"): [f"{feats[j]} = {Xraw[idx, j]:.2f}" for j in order],
            "SHAP": contrib[order],
            t("col_group"): [t("xp_up") if contrib[j] > 0 else t("xp_down") for j in order]})
        st.markdown(f"##### {t('xp_local_title')}")
        fig = px.bar(loc, x="SHAP", y=t("col_feat"), color=t("col_group"), orientation="h",
                     color_discrete_map={t("xp_up"): "#924A2F", t("xp_down"): "#3A6EA5"})
        fig.update_yaxes(autorange="reversed")
        show(style(fig, legend=True))

# ---------------- HALAMAN: PREDICTION + SUS (Tahap 11) ----------------
SUS_ITEMS = {
    "EN": ["I think that I would like to use this system frequently.",
           "I found the system unnecessarily complex.",
           "I thought the system was easy to use.",
           "I think that I would need the support of a technical person to be able to use this system.",
           "I found the various functions in this system were well integrated.",
           "I thought there was too much inconsistency in this system.",
           "I would imagine that most people would learn to use this system very quickly.",
           "I found the system very cumbersome to use.",
           "I felt very confident using the system.",
           "I needed to learn a lot of things before I could get going with this system."],
    "ID": ["Saya pikir saya akan sering menggunakan sistem ini.",
           "Saya merasa sistem ini terlalu rumit tanpa perlu.",
           "Saya pikir sistem ini mudah digunakan.",
           "Saya pikir saya memerlukan bantuan orang teknis untuk dapat menggunakan sistem ini.",
           "Saya merasa berbagai fungsi dalam sistem ini terintegrasi dengan baik.",
           "Saya pikir ada terlalu banyak ketidakkonsistenan dalam sistem ini.",
           "Saya membayangkan kebanyakan orang akan belajar menggunakan sistem ini dengan sangat cepat.",
           "Saya merasa sistem ini sangat merepotkan untuk digunakan.",
           "Saya merasa sangat percaya diri menggunakan sistem ini.",
           "Saya perlu mempelajari banyak hal sebelum dapat mulai menggunakan sistem ini."],
    "MS": ["Saya fikir saya akan kerap menggunakan sistem ini.",
           "Saya dapati sistem ini terlalu rumit tanpa keperluan.",
           "Saya fikir sistem ini mudah digunakan.",
           "Saya fikir saya memerlukan bantuan orang teknikal untuk menggunakan sistem ini.",
           "Saya dapati pelbagai fungsi dalam sistem ini disepadukan dengan baik.",
           "Saya fikir terdapat terlalu banyak ketidakselarasan dalam sistem ini.",
           "Saya bayangkan kebanyakan orang akan belajar menggunakan sistem ini dengan sangat cepat.",
           "Saya dapati sistem ini sangat menyusahkan untuk digunakan.",
           "Saya berasa sangat yakin semasa menggunakan sistem ini.",
           "Saya perlu mempelajari banyak perkara sebelum dapat mula menggunakan sistem ini."],
    "ZH": ["我认为我会经常使用这个系统。",
           "我发现这个系统过于复杂，没有必要。",
           "我认为这个系统很容易使用。",
           "我认为我需要技术人员的帮助才能使用这个系统。",
           "我发现这个系统中的各项功能整合得很好。",
           "我认为这个系统中有太多不一致的地方。",
           "我想大多数人会很快学会使用这个系统。",
           "我发现这个系统使用起来非常繁琐。",
           "我使用这个系统时非常有信心。",
           "在开始使用这个系统之前，我需要学习很多东西。"],
}


def sus_score(ans):
    """Butir ganjil: (skor-1). Butir genap: (5-skor). Jumlah x 2,5 -> 0..100."""
    return sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(ans)) * 2.5


def predict_proba_df(run, model, df):
    feats = run["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        return None, missing, 0
    Xd = df[feats].apply(pd.to_numeric, errors="coerce")
    n_nan = int(Xd.isna().sum().sum())
    X = np.log1p(np.abs(Xd.fillna(0).to_numpy(dtype=float)))
    return run["fitted"][model].predict_proba(X), [], n_nan


def prediction_tab():
    runs = [r for r in st.session_state.get("runs", []) if r["holdout"] and r.get("fitted")]
    if not runs:
        st.info(t("pr_empty"))
        return
    by_id = {r["id"]: r for r in runs}
    ids = [r["id"] for r in runs][::-1]
    c1, c2 = st.columns(2)
    rid = c1.selectbox(t("mc_pick"), ids, key="pr_run", format_func=lambda i: run_label(by_id[i]))
    run = by_id[rid]
    model = c2.selectbox(t("pl_col_model"), list(run["fitted"]), key=f"pr_model_{rid}")
    pipe, classes, feats = run["fitted"][model], run["classes"], run["features"]
    st.caption(t("pr_note"))
    src = st.radio(t("pr_src"), ["csv", "sample"], horizontal=True,
                   key=f"pr_src_{st.session_state.lang}",
                   format_func=lambda k: t("pr_src_" + k))

    if src == "csv":
        up = st.file_uploader(t("pr_upload"), type="csv", key=f"pr_up_{rid}")
        if up is None:
            return
        df = pd.read_csv(up)
        proba, missing, n_nan = predict_proba_df(run, model, df)
        if proba is None:
            st.error(t("pr_missing", n=len(missing), ex=", ".join(missing[:5])))
            return
        if n_nan:
            st.warning(t("pr_nan", n=n_nan))
        pred = np.array(classes)[proba.argmax(axis=1)]
        out = df.copy()
        out[t("pr_col_pred")] = pred
        out[t("pr_col_conf")] = (proba.max(axis=1) * 100).round(2)
        st.success(t("pr_rows", n=f"{len(out):,}"))
        if "Label" in df.columns:
            lab = df["Label"].astype(str).to_numpy()
            if run["mode"] == "binary":
                lab = np.where(lab == "Benign", "Benign", "Malware")
            st.info(t("pr_label_acc", v=float((lab == pred).mean() * 100)))
        d1, d2 = st.columns([1.3, 1])
        show_cols = [t("pr_col_pred"), t("pr_col_conf")] + [c for c in out.columns[:6]
                                                           if c not in (t("pr_col_pred"), t("pr_col_conf"))]
        d1.dataframe(out[show_cols].head(200))
        cnt = pd.Series(pred).value_counts().rename_axis(t("col_class")).reset_index(name=t("col_count"))
        with d2:
            st.markdown(f"##### {t('pr_dist')}")
            show(style(px.bar(cnt, x=t("col_class"), y=t("col_count"), color=t("col_class"),
                              color_discrete_map=CLASS_COLORS)))
        st.download_button(t("pr_download"), out.to_csv(index=False).encode(),
                           file_name=f"predictions_run{rid}.csv", mime="text/csv", key="pr_dl")
        return

    # --- sampel uji + skenario ubah ---
    Xte, yte = run["test"]["X"], run["test"]["y"]
    idx = int(st.number_input(t("pr_sample"), min_value=0, max_value=len(Xte) - 1, value=0,
                              step=1, key=f"pr_i_{rid}"))
    x0 = Xte[idx].copy()
    x = x0.copy()
    st.markdown(f"##### {t('pr_edit')}")
    top = list(run["importance"][model].head(8).index)
    cols = st.columns(4)
    for k, f in enumerate(top):
        j = feats.index(f)
        val = cols[k % 4].number_input(f, min_value=0.0, value=float(round(np.expm1(x0[j]), 4)),
                                       step=1.0, format="%.4f", key=f"pr_w_{rid}_{idx}_{f}")
        x[j] = np.log1p(val)
    p0 = pipe.predict_proba(x0[None, :])[0]
    p1 = pipe.predict_proba(x[None, :])[0]
    m1, m2, m3 = st.columns(3)
    m1.metric(t("xp_true"), classes[int(yte[idx])])
    m2.metric(t("pr_orig"), classes[int(p0.argmax())], f"{p0.max() * 100:.1f}%", delta_color="off")
    m3.metric(t("pr_new"), classes[int(p1.argmax())], f"{p1.max() * 100:.1f}%", delta_color="off")
    pdf = pd.DataFrame({t("col_class"): classes, "%": p1 * 100})
    fig = px.bar(pdf, x=t("col_class"), y="%", color=t("col_class"), color_discrete_map=CLASS_COLORS)
    fig.update_yaxes(range=[0, 100])
    show(style(fig))


def sus_tab():
    lang = st.session_state.lang
    items = SUS_ITEMS.get(lang, SUS_ITEMS["EN"])
    st.session_state.setdefault("sus", [])
    ver = st.session_state.setdefault("sus_ver", 0)
    if "sus_flash" in st.session_state:
        st.success(t("sus_saved", s=st.session_state.pop("sus_flash")))
    st.info(t("sus_intro"))

    with st.form(f"sus_form_{ver}"):
        who = st.text_input(t("sus_name"), key=f"sus_name_{ver}")
        answers = [st.radio(f"{i}. {q}", [1, 2, 3, 4, 5], index=None, horizontal=True,
                            key=f"sus_q{i}_{ver}") for i, q in enumerate(items, 1)]
        ok = st.form_submit_button(t("sus_submit"), type="primary")
    if ok:
        if None in answers:
            st.warning(t("sus_incomplete"))
        else:
            score = sus_score(answers)
            row = {"respondent": who.strip() or f"R{len(st.session_state.sus) + 1}"}
            row.update({f"q{i}": a for i, a in enumerate(answers, 1)})
            row["score"] = score
            st.session_state.sus.append(row)
            st.session_state.sus_ver += 1
            st.session_state.sus_flash = score
            st.rerun()

    st.markdown(f"### {t('sus_results')}")
    up = st.file_uploader(t("sus_upload"), type="csv", key="sus_up")
    if up is not None and st.button(t("sus_load"), key="sus_loadbtn"):
        st.session_state.sus = pd.read_csv(up).to_dict("records")
        st.success(t("sus_loaded", n=len(st.session_state.sus)))
    if not st.session_state.sus:
        st.caption(t("sus_none"))
        return

    df = pd.DataFrame(st.session_state.sus)
    mean = float(df["score"].mean())
    band = t("sus_below") if mean < 68 else (t("sus_above") if mean < 80.3 else t("sus_excel"))
    k1, k2, k3 = st.columns(3)
    k1.metric(t("sus_n"), len(df))
    k2.metric(t("sus_mean"), f"{mean:.1f}")
    k3.metric(t("sus_interp"), band)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"##### {t('sus_per_resp')}")
        fig = px.bar(df, x="respondent", y="score", color_discrete_sequence=["#3A6EA5"])
        fig.update_yaxes(range=[0, 100])
        fig.add_hline(y=68, line_dash="dash", line_color="#924A2F")
        show(style(fig))
    with g2:
        st.markdown(f"##### {t('sus_per_item')}")
        qm = df[[f"q{i}" for i in range(1, 11)]].mean().rename_axis(t("sus_name")).reset_index(name="mean")
        fig = px.bar(qm, x=t("sus_name"), y="mean", color_discrete_sequence=["#254F73"])
        fig.update_yaxes(range=[0, 5])
        show(style(fig))
    st.dataframe(df, hide_index=True)
    st.download_button(t("sus_download"), df.to_csv(index=False).encode(),
                       file_name="maldroid_sus.csv", mime="text/csv", key="sus_dl")
    st.caption(t("sus_note"))


def page_predict():
    hero(t("nav_predict"), t("pr_sub"))
    tab1, tab2 = st.tabs([t("pr_tab_pred"), t("pr_tab_sus")])
    with tab1:
        prediction_tab()
    with tab2:
        sus_tab()


PAGES = [
    ("overview", page_overview, ":material/dataset:"),
    ("pipeline", page_pipeline, ":material/tune:"),
    ("models",   page_models,   ":material/leaderboard:"),
    ("cross",    page_cross,    ":material/compare_arrows:"),
    ("explain",  page_explain,  ":material/manage_search:"),
    ("predict",  page_predict,  ":material/online_prediction:"),
]

# ---------------- NAVBAR ATAS (bawaan Streamlit) ----------------
nav = st.navigation(
    [st.Page(fn, title=t("nav_" + key), icon=icon, url_path=key, default=(key == "overview"))
     for key, fn, icon in PAGES],
    position="top",
)
nav.run()

# ---------------- DROPDOWN BAHASA (dipasang di pojok kanan navbar lewat CSS) ----------------
with st.container(key="langbox"):
    st.selectbox("Language", list(LANGS), key="lang", format_func=lambda k: LANGS[k],
                 label_visibility="collapsed")