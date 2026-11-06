import json, pathlib
T = {
 "en": ("Dataset added. Next step: train models on it in Pipeline Runner.", "Continue to Pipeline Runner"),
 "id": ("Dataset berhasil ditambahkan. Langkah berikutnya: latih model di Pipeline Runner.", "Lanjut ke Pipeline Runner"),
 "ms": ("Set data berjaya ditambah. Langkah seterusnya: latih model dalam Pipeline Runner.", "Teruskan ke Pipeline Runner"),
 "zh": ("数据集已添加。下一步：在 Pipeline Runner 中训练模型。", "继续前往 Pipeline Runner"),
}
out = pathlib.Path(__file__).resolve().parent.parent / "i18n"
for k, (a, b) in T.items():
    (out / f"{k}_dsnext.json").write_text(
        json.dumps({"ds_next": a, "ds_next_btn": b}, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")