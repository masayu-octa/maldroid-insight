import json
T = {
"en": "The model is almost completely sure about this sample, so each single feature changes the result by a negligible amount. Grey bars are such negligible effects (under 0.1 percentage points) and should not be read as findings. For more informative explanations, pick a sample the model got wrong or is less sure about.",
"id": "Model hampir sepenuhnya yakin pada sampel ini, sehingga pengaruh tiap fitur terhadap hasil sangat kecil. Batang abu-abu menandakan pengaruh yang dapat diabaikan (di bawah 0,1 poin persentase) dan tidak perlu dibaca sebagai temuan. Untuk penjelasan yang lebih informatif, pilih sampel yang diprediksi salah atau yang keyakinannya lebih rendah.",
"ms": "Model hampir sepenuhnya yakin tentang sampel ini, jadi pengaruh setiap ciri terhadap keputusan sangat kecil. Bar kelabu menandakan pengaruh yang boleh diabaikan (di bawah 0.1 mata peratusan) dan tidak perlu dibaca sebagai penemuan. Untuk penerangan yang lebih bermaklumat, pilih sampel yang diramal salah atau yang keyakinannya lebih rendah.",
"zh": "模型对该样本几乎完全确定，因此单个特征对结果的影响微乎其微。灰色条表示可忽略的影响（低于 0.1 个百分点），不应将其视为结论。若想得到更有价值的解释，请选择模型判断错误或把握较低的样本。",
}
for k, v in T.items():
    json.dump({"ex_sat_note": v}, open(f"i18n/{k}_explain_fix.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")