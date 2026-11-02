import json
T = {
    "en": ("Your progress", "{n} of {total} steps ready", "Class distribution per dataset"),
    "id": ("Progresmu", "{n} dari {total} langkah siap", "Distribusi kelas per dataset"),
    "ms": ("Kemajuan anda", "{n} daripada {total} langkah sedia", "Taburan kelas bagi setiap set data"),
    "zh": ("你的进度", "{n} / {total} 步已就绪", "各数据集的类别分布"),
}
for k, (a, b, c) in T.items():
    json.dump({"lp_prog_title": a, "lp_prog_val": b, "lp_ws_chart": c},
              open(f"i18n/{k}_home_fx.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")