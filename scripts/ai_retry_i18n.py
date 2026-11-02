import json
T = {"en": "Try again", "id": "Coba lagi", "ms": "Cuba lagi", "zh": "重试"}
for k, v in T.items():
    json.dump({"ai_retry": v}, open(f"i18n/{k}_ai_retry.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")