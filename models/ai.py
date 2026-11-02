"""Asisten AI (Gemini) untuk Explainability. Tanpa Streamlit: mudah diuji.
Hanya RINGKASAN hasil yang dikirim ke Gemini (bukan data mentah)."""

LANG_NAMES = {"EN": "English", "ID": "Bahasa Indonesia", "MS": "Bahasa Melayu", "ZH": "Simplified Chinese"}
TIMEOUT_MS = 90000
MAX_HISTORY = 8          # pesan terakhir yang ikut dikirim ke model

SYSTEM = (
    "You are an assistant inside MalDroid Insight, a tool that evaluates machine-learning models for Android "
    "malware detection. Explain results to a student in simple, friendly language. Answer ONLY from the RESULTS "
    "block below and general machine-learning knowledge. If the answer is not in the results, say you cannot "
    "tell from the results. Never invent numbers. Importance shows what a model uses, not what causes malware. "
    "The per-sample chart shows the features with the largest effect on the predicted class: GREEN bars support "
    "the predicted class, RED (brownish-orange) bars oppose it; a bar's length is the effect in percentage points. "
    "The global chart ranks features by permutation importance. "
    "Keep answers short (at most 6 sentences) unless asked for more. Reply in {lang}."
)


class AIError(Exception):
    """key: kunci terjemahan pesan untuk pengguna."""

    def __init__(self, key, detail=""):
        super().__init__(key)
        self.key, self.detail = key, detail


def build_context(run_info, importance, sample=None):
    """Teks ringkas hasil. run_info: dict dataset/model/classes/n_features/accuracy/f1.
    importance: [(fitur, persen)] teratas. sample: dict true/pred/conf/effects[(fitur, poin)] atau None."""
    lines = [f"Dataset: {run_info['dataset']}", f"Model: {run_info['model']}",
             f"Classes: {', '.join(run_info['classes'])}", f"Number of features: {run_info['n_features']}"]
    if run_info.get("accuracy") is not None:
        lines.append(f"Hold-out accuracy: {run_info['accuracy'] * 100:.2f}%")
    if run_info.get("f1") is not None:
        lines.append(f"Hold-out macro F1: {run_info['f1'] * 100:.2f}%")
    lines.append("Most important features (permutation importance, share of total):")
    lines += [f"- {f}: {p:.1f}%" for f, p in importance]
    if sample:
        lines.append(f"Selected test sample: true class {sample['true']}, predicted {sample['pred']}, "
                     f"confidence {sample['conf']}.")
        lines.append("Effect of each feature shown in the per-sample chart (percentage points; positive = green bar, "
                     "supports the predicted class; negative = red bar, opposes it):")
        lines += [f"- {f}: {v:+.4f}" for f, v in sample["effects"]]
    return "\n".join(lines)


def _client(key):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS)), types


def _call(key, model, contents, system):
    client, types = _client(key)
    try:
        resp = client.models.generate_content(
            model=model, contents=contents, config=types.GenerateContentConfig(system_instruction=system))
    except Exception as e:                                  # pesan ramah, detail teknis disimpan
        raise AIError(_classify(e), str(e)[:300]) from e
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise AIError("ai_err_empty")
    return text


def _classify(e):
    s = str(e)
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code == 429 or "429" in s or "RESOURCE_EXHAUSTED" in s:
        return "ai_err_quota"
    if code in (401, 403) or "API key" in s or "PERMISSION_DENIED" in s:
        return "ai_err_key"
    if code == 404 or "NOT_FOUND" in s:
        return "ai_err_model"
    if "timed out" in s.lower() or "timeout" in s.lower() or "Deadline" in s:
        return "ai_err_timeout"
    return "ai_err_net"


def explain(key, model, context, lang="ID"):
    """Ringkasan otomatis dari hasil."""
    system = SYSTEM.format(lang=LANG_NAMES.get(lang, "English")) + "\n\nRESULTS:\n" + context
    ask = ("Explain these results in 4 to 6 short sentences: which features the model relies on, whether anything "
           "looks suspicious (for example one feature dominating), and one thing to check next.")
    return _call(key, model, ask, system)


def chat(key, model, context, history, question, lang="ID"):
    """Tanya jawab. history: [(role, teks)] role 'user' | 'assistant'."""
    system = SYSTEM.format(lang=LANG_NAMES.get(lang, "English")) + "\n\nRESULTS:\n" + context
    contents = []
    for role, text in history[-MAX_HISTORY:]:
        contents.append({"role": "user" if role == "user" else "model", "parts": [{"text": text}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    return _call(key, model, contents, system)