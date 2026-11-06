import json, pathlib
T = {
 "en": dict(lp_good="- This is a research and learning tool, not a replacement for antivirus software.\n- High accuracy depends on the dataset. A model that is excellent on one dataset may do poorly on another, so treat results as valid for the dataset you used.\n- Datasets, runs and models are stored in the storage/ folder of the computer that runs the app.\n- The first run on a large dataset can take a while; start with Quick and a limited number of rows."),
 "id": dict(lp_good="- Ini alat riset dan belajar, bukan pengganti antivirus.\n- Akurasi tinggi bergantung pada datasetnya. Model yang sangat baik di satu dataset bisa buruk di dataset lain, jadi anggap hasil hanya berlaku untuk dataset yang kamu pakai.\n- Dataset, run, dan model disimpan di folder storage/ pada komputer yang menjalankan aplikasi.\n- Run pertama pada dataset besar bisa lama; mulai dengan Cepat dan jumlah baris terbatas."),
 "ms": dict(lp_good="- Ini alat penyelidikan dan pembelajaran, bukan pengganti perisian antivirus.\n- Ketepatan tinggi bergantung pada set data. Model yang cemerlang pada satu set data boleh lemah pada yang lain, jadi anggap keputusan hanya sah untuk set data yang anda guna.\n- Set data, run dan model disimpan dalam folder storage/ pada komputer yang menjalankan aplikasi.\n- Run pertama pada set data besar boleh mengambil masa; mulakan dengan Pantas dan bilangan baris terhad."),
 "zh": dict(lp_good="- 这是研究和学习工具，不能替代杀毒软件。\n- 高准确率取决于数据集。在一个数据集上表现出色的模型，在另一个数据集上可能表现不佳，因此结果只适用于你所使用的数据集。\n- 数据集、运行记录和模型保存在运行本应用的电脑的 storage/ 文件夹中。\n- 在大型数据集上首次运行可能较慢；建议先用“快速”并限制行数。"),
}
out = pathlib.Path(__file__).resolve().parent.parent / "i18n"
for k, v in T.items():
    (out / f"{k}_nocross.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")