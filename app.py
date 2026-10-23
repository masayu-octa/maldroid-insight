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

.steps{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:6px 0 14px;}
.step{background:#EDE3D0;color:#141F52;border-radius:999px;padding:4px 14px;
      font-size:.82rem;font-weight:600;}
.step.off{opacity:.4;text-decoration:line-through;}
.arrow{color:#7A8DB3;}
<style>
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
        
def page_models():   coming_soon("nav_models", 8)
def page_cross():    coming_soon("nav_cross", 9)
def page_explain():  coming_soon("nav_explain", 10)
def page_predict():  coming_soon("nav_predict", 11)


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