import json, pathlib
T = {
 "en": dict(pl_speed="Training thoroughness", pl_fast="Quick", pl_full="Thorough",
   pl_speed_help="Quick trains with fewer trees/iterations, so it finishes sooner. Thorough uses more, takes longer and is usually slightly more accurate. Use Quick to try things out and Thorough for your final results.",
   pl_hyper_title="What Quick and Thorough change for each model", pl_open_explain="Explain this run (Explainability)"),
 "id": dict(pl_speed="Ketelitian pelatihan", pl_fast="Cepat", pl_full="Teliti",
   pl_speed_help="Cepat melatih dengan jumlah pohon/iterasi lebih sedikit, jadi selesai lebih cepat. Teliti memakai lebih banyak, lebih lama, dan biasanya sedikit lebih akurat. Pakai Cepat untuk mencoba-coba dan Teliti untuk hasil akhir.",
   pl_hyper_title="Perbedaan Cepat dan Teliti pada tiap model", pl_open_explain="Jelaskan run ini (Explainability)"),
 "ms": dict(pl_speed="Ketelitian latihan", pl_fast="Pantas", pl_full="Teliti",
   pl_speed_help="Pantas melatih dengan bilangan pokok/lelaran lebih sedikit, jadi siap lebih cepat. Teliti menggunakan lebih banyak, lebih lama dan biasanya sedikit lebih tepat. Guna Pantas untuk mencuba dan Teliti untuk keputusan akhir.",
   pl_hyper_title="Perbezaan Pantas dan Teliti bagi setiap model", pl_open_explain="Terangkan larian ini (Explainability)"),
 "zh": dict(pl_speed="训练精细程度", pl_fast="快速", pl_full="精细",
   pl_speed_help="快速使用较少的树/迭代次数，完成得更快。精细使用更多，耗时更长，通常精度略高。试验时用快速，最终结果用精细。",
   pl_hyper_title="快速与精细对各模型的具体差异", pl_open_explain="解释此次运行（Explainability）"),
}
out = pathlib.Path(__file__).resolve().parent.parent / "i18n"
for k, v in T.items():
    (out / f"{k}_plfix.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")