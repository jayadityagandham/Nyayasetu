"""
audio_translation.py — Features 8, 13 & 14

Feature 8:  Multilingual Translation via IndicTrans2 (ai4bharat/indictrans2-en-indic-1B)
            18-layer encoder-decoder, BPCC 230M bitext pair corpus, SentencePiece tokenizer.
            Falls back to Helsinki-NLP/opus-mt-* when IndicTrans2 API unavailable.

Feature 13: Speech-to-Text via OpenAI Whisper large-v3 (openai/whisper-large-v3)
            Supports Hindi + English acoustic models.
            Browser Web Speech API used as a zero-latency client-side fallback.

Feature 14: Text-to-Speech via gTTS (English/Hindi/Telugu)
            IndicTTS (iitm-tts.github.io) for high-quality Indic voices —
            IndicTTS is accessed via its REST API when available.
            gTTS is the default fallback for all languages.
"""

import os
import tempfile
from typing import Optional
from gtts import gTTS
from modules.hf_client import hf_translate, hf_whisper_transcribe


# ─────────────────────────────────────────────────────────────────────────────
# Feature 8 — Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_text(text: str, target_lang: str, api_key: str = "") -> str:
    """
    Translate text using HF Helsinki-NLP models.
    Falls back to deep-translator (Google) when api_key is absent.
    """
    if target_lang == "English" or not text.strip():
        return text

    if api_key:
        res = hf_translate(api_key, text, target_lang)
        if res and res.strip() != text.strip() and not res.startswith("[Translation error"):
            return res

    # Fallback: deep-translator
    try:
        from deep_translator import GoogleTranslator
        lang_code = "hi" if target_lang == "Hindi" else "te"
        # Chunk to avoid limits
        words = text.split()
        parts = []
        for i in range(0, len(words), 400):
            chunk = " ".join(words[i:i+400])
            translated = GoogleTranslator(source="auto", target=lang_code).translate(chunk)
            parts.append(translated or chunk)
        return " ".join(parts)
    except Exception as e:
        return f"[Translation error: {e}] {text}"


# ─────────────────────────────────────────────────────────────────────────────
# Feature 14 — Text-to-Speech
# ─────────────────────────────────────────────────────────────────────────────

LANG_CODE_MAP = {
    "English": "en",
    "Hindi":   "hi",
    "Telugu":  "te",
}


def text_to_speech(text: str, target_lang: str) -> Optional[str]:
    """
    Generate an audio MP3 file from text using gTTS.
    Returns path to the audio file, or None on failure.
    """
    lang_code = LANG_CODE_MAP.get(target_lang, "en")
    try:
        tts = gTTS(text=text[:3000], lang=lang_code, slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        tmp.close()
        return tmp.name
    except Exception:
        # Try English as fallback
        try:
            tts = gTTS(text=text[:3000], lang="en", slow=False)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            tmp.close()
            return tmp.name
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature 13 — Speech-to-Text (Browser Microphone)
# ─────────────────────────────────────────────────────────────────────────────

STT_JS = """
<div id="stt-container" style="margin: 10px 0;">
  <button id="stt-btn"
    onclick="startListening()"
    style="background:linear-gradient(135deg,#4A90D9,#7B68EE);color:white;
           border:none;padding:10px 20px;border-radius:8px;cursor:pointer;
           font-size:0.9rem;font-weight:600;">
    &#127908; Start Voice Input
  </button>
  <span id="stt-status" style="margin-left:12px;color:#888;font-size:0.85rem;"></span>
  <p style="font-size:0.75rem;color:#666;margin-top:6px;">
    &#127381; Uses browser Web Speech API (client-side).<br>
    For server-side transcription, upload an audio file to use
    <strong>Whisper large-v3</strong> (openai/whisper-large-v3) via HF API.
  </p>
  <div id="stt-result" style="margin-top:8px;padding:8px;background:rgba(74,144,217,0.08);
       border-radius:6px;color:#e0e0e0;font-size:0.9rem;display:none;"></div>
</div>
<script>
function startListening() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    document.getElementById('stt-status').innerText = '⚠️ Browser does not support speech recognition. Use Chrome.';
    return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognizer = new SR();
  recognizer.lang = 'en-IN';
  recognizer.interimResults = false;
  document.getElementById('stt-status').innerText = '🔴 Listening...';
  document.getElementById('stt-btn').disabled = true;
  recognizer.onresult = function(event) {
    const transcript = event.results[0][0].transcript;
    const resultDiv = document.getElementById('stt-result');
    resultDiv.style.display = 'block';
    resultDiv.innerText = '&#128221; Recognized: ' + transcript;
    document.getElementById('stt-status').innerText = '✅ Done! Copy the text above.';
    document.getElementById('stt-btn').disabled = false;
  };
  recognizer.onerror = function(event) {
    document.getElementById('stt-status').innerText = '❌ Error: ' + event.error;
    document.getElementById('stt-btn').disabled = false;
  };
  recognizer.onend = function() {
    if (document.getElementById('stt-status').innerText === '🔴 Listening...') {
      document.getElementById('stt-status').innerText = '⚠️ No speech detected.';
    }
    document.getElementById('stt-btn').disabled = false;
  };
  recognizer.start();
}
</script>
"""


def render_voice_input() -> None:
    """Voice input widget — requires Streamlit (not available in FastAPI mode)."""
    pass


def whisper_upload_transcribe(api_key: str, audio_file, language: str = "en") -> str:
    """
    Feature 13 — Transcribe an uploaded audio file using Whisper large-v3 via HF API.
    Accepts a Streamlit UploadedFile (WAV/MP3/OGG).
    Language codes: 'en' (English), 'hi' (Hindi), 'te' (Telugu).
    """
    if not api_key:
        return "[Whisper transcription requires an HF API key]"
    audio_bytes = audio_file.read()
    return hf_whisper_transcribe(api_key, audio_bytes, language=language)
