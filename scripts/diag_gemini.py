"""Diagnosis kecepatan Gemini. Jalankan dari folder proyek: python scripts\\diag_gemini.py"""
import time
import tomllib
from pathlib import Path

from google import genai
from google.genai import types

cfg = tomllib.loads(Path(".streamlit/secrets.toml").read_text(encoding="utf-8"))
key = cfg["GEMINI_API_KEY"]

client = genai.Client(api_key=key, http_options=types.HttpOptions(
    timeout=40000, retry_options=types.HttpRetryOptions(attempts=1)))   # tanpa pengulangan otomatis

print("Model teks yang tersedia (nama mengandung 'flash'):")
names = []
try:
    for m in client.models.list():
        n = (m.name or "").replace("models/", "")
        if "flash" in n and "image" not in n and "tts" not in n and "live" not in n and "embed" not in n:
            names.append(n)
except Exception as e:
    print("  gagal membaca daftar model:", str(e)[:200])
for n in sorted(set(names)):
    print("  -", n)

candidates = [cfg.get("GEMINI_MODEL", "gemini-3.8-flash")]
candidates += [n for n in sorted(set(names)) if "lite" in n][:2]
print("\nUji kecepatan (batas 40 detik, tanpa pengulangan):")
for model in dict.fromkeys(candidates):
    t0 = time.time()
    try:
        r = client.models.generate_content(model=model, contents="Jawab satu kalimat: apa itu malware?")
        print(f"  {model}: {time.time() - t0:.1f} detik -> {(r.text or '').strip()[:60]}")
    except Exception as e:
        print(f"  {model}: GAGAL setelah {time.time() - t0:.1f} detik -> {str(e)[:160]}")