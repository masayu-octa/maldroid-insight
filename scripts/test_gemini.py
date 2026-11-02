"""Uji cepat kunci Gemini. Jalankan dari folder proyek: python scripts\\test_gemini.py"""
import tomllib
from pathlib import Path

from google import genai

cfg = tomllib.loads(Path(".streamlit/secrets.toml").read_text(encoding="utf-8"))
client = genai.Client(api_key=cfg["GEMINI_API_KEY"])
resp = client.models.generate_content(
    model=cfg.get("GEMINI_MODEL", "gemini-2.5-flash"),
    contents="Jawab satu kalimat dalam Bahasa Indonesia: apa itu malware?",
)
print(resp.text)