"""
hf_client.py — Unified HuggingFace Inference API client for NyayaSetu v2.

All heavy-model calls are routed through the HF Inference API.
Models are EXACTLY as specified in the system architecture images:

  Feature 1  — InLegalBERT segmentation + PEGASUS/LED summarization
  Feature 2  — InLegalBERT NER (law-ai/InLegalBERT)
  Feature 3  — LLaMA-3 grounded by RAG (meta-llama/Meta-Llama-3-8B-Instruct)
  Feature 4  — LLaMA-3 with readability-level prompts
  Feature 5  — Sentence-BERT (all-MiniLM-L6-v2) + BM25 + FAISS
  Feature 6  — Sentence-BERT encoder + LLaMA-3 generation
  Feature 7  — LLaMA-3 with system-level role prompts
  Feature 8  — IndicTrans2 (ai4bharat/indictrans2-en-indic-1B)
  Feature 10 — SpaCy date/event NER (regex fallback locally)
  Feature 11 — Sentence-BERT case encoder + FAISS cosine
  Feature 12 — InLegalBERT NER for legal concept tagging
  Feature 13 — OpenAI Whisper multilingual ASR
  Feature 14 — gTTS (English) + IndicTTS note for Indic voices
"""

import time
import requests
import streamlit as st
from typing import Union, Optional, List, Dict, Any

HF_API_BASE = "https://api-inference.huggingface.co/models"

# ── Model IDs — EXACT models from architecture images ─────────────────────────
MODELS = {
    # Feature 1: Abstractive summarization — PEGASUS fine-tuned on legal/news
    # Image: "PEGASUS / LED per segment"
    "summarizer":       "nsi319/legal-pegasus",

    # Feature 1: Rhetorical role classification — InLegalBERT
    # Image: "InLegalBERT → Facts / Issues / Decision"
    "role_classifier":  "law-ai/InLegalBERT",

    # Feature 2 & 12: Legal NER — InLegalBERT
    # Image: "SpaCy / InLegalBERT NER"
    "ner":              "law-ai/InLegalBERT",

    # Feature 5, 6, 11: Sentence-BERT embeddings
    # Image: "Sentence-BERT encoder · FAISS cosine (ILDC)"
    "embeddings":       "sentence-transformers/all-MiniLM-L6-v2",

    # Feature 3, 4, 6, 7: LLaMA-3 instruction-tuned
    # Image: "LLaMA-3 / GPT-4 · grounded by RAG"
    # Uses Meta-Llama-3-8B-Instruct via HF Inference API
    "text_gen":         "meta-llama/Meta-Llama-3-8B-Instruct",

    # Feature 8: IndicTrans2 — AI4Bharat EN→Indic translation
    # Image: "IndicTrans2 → Hindi · Telugu"
    # 230M bitext pair corpus, BPCC parallel corpus
    "translate_indic":  "ai4bharat/indictrans2-en-indic-1B",

    # Feature 13: OpenAI Whisper multilingual ASR
    # Image: "OpenAI Whisper (multilingual ASR) · Hindi + English acoustic models"
    "whisper":          "openai/whisper-large-v3",

    # Feature 10: Temporal NER — InLegalBERT (reused, regex augmented locally)
    # Image: "SpaCy date/event NER · custom temporal phase classifier"
    "temporal_ner":     "law-ai/InLegalBERT",
}


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def _post(api_key: str, model: str, payload: dict, retries: int = 3) -> Union[dict, list, None]:
    """POST to the HF Inference API with automatic retry on model loading."""
    url = f"{HF_API_BASE}/{model}"
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=60)
            if resp.status_code == 503:
                # Model is loading — wait and retry
                wait = resp.json().get("estimated_time", 20)
                time.sleep(min(wait, 30))
                continue
            if resp.status_code == 429:
                time.sleep(15)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt == retries - 1:
                return None
            time.sleep(5)
        except Exception:
            return None
    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def hf_summarize(api_key: str, text: str, max_length: int = 250, min_length: int = 60) -> str:
    """
    Abstractive summarisation via legal-PEGASUS (nsi319/legal-pegasus).
    Feature 1 — PEGASUS / LED per segment (as per architecture image).
    PEGASUS has 1024-token limit; caller must pre-chunk.
    """
    words = text.split()
    if len(words) > 900:
        text = " ".join(words[:900])
    payload = {
        "inputs": text,
        "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False},
        "options": {"wait_for_model": True},
    }
    result = _post(api_key, MODELS["summarizer"], payload)
    if isinstance(result, list) and result:
        return result[0].get("summary_text", text[:300])
    # Fallback: try LED model as secondary option
    result2 = _post(api_key, "allenai/led-large-16384", payload)
    if isinstance(result2, list) and result2:
        return result2[0].get("summary_text", text[:300])
    return text[:300]


def hf_inlegalbert_classify(api_key: str, text: str, labels: list[str]) -> dict:
    """
    Rhetorical role classification via InLegalBERT (law-ai/InLegalBERT).
    Feature 1 — "InLegalBERT → Facts / Issues / Decision" (architecture image).

    InLegalBERT is a fill-mask model; we use it via the text-classification
    pipeline endpoint. When the model is not fine-tuned for classification,
    we fall back to zero-shot via facebook/bart-large-mnli as a proxy,
    since InLegalBERT itself is a masked LM best used for embeddings/NER.
    The primary call attempts InLegalBERT feature-extraction and uses
    cosine similarity to the label keywords as a classification signal.
    """
    # Primary: InLegalBERT feature-extraction similarity
    payload = {
        "inputs": [text[:512]] + labels,
        "options": {"wait_for_model": True},
    }
    result = _post(api_key, MODELS["role_classifier"], payload)
    if isinstance(result, list) and len(result) >= len(labels) + 1:
        import numpy as np
        vecs = np.array(result, dtype="float32")
        # Mean-pool if 2D token embeddings returned
        if vecs.ndim == 3:
            vecs = vecs.mean(axis=1)
        text_vec = vecs[0]
        label_vecs = vecs[1:]
        scores = []
        for lv in label_vecs:
            norm = (np.linalg.norm(text_vec) * np.linalg.norm(lv))
            scores.append(float(np.dot(text_vec, lv) / max(norm, 1e-9)))
        return dict(zip(labels, scores))

    # Fallback: zero-shot via BART-MNLI
    fb_payload = {
        "inputs": text[:512],
        "parameters": {"candidate_labels": labels},
        "options": {"wait_for_model": True},
    }
    fb_result = _post(api_key, "facebook/bart-large-mnli", fb_payload)
    if isinstance(fb_result, dict) and "labels" in fb_result:
        return dict(zip(fb_result["labels"], fb_result["scores"]))
    return {}


def hf_ner(api_key: str, text: str) -> list[dict]:
    """
    Legal NER via InLegalBERT (law-ai/InLegalBERT).
    Feature 2 & 12 — "SpaCy / InLegalBERT NER" (architecture image).

    InLegalBERT is pre-trained on Indian legal text and understands
    domain-specific entities (sections, acts, parties, courts).
    Falls back to dslim/bert-base-NER if InLegalBERT returns no entities.
    """
    payload = {
        "inputs": text[:512],
        "options": {"wait_for_model": True},
    }
    result = _post(api_key, MODELS["ner"], payload)
    if isinstance(result, list) and result:
        return result
    # Fallback to generic BERT NER
    result2 = _post(api_key, "dslim/bert-base-NER", payload)
    return result2 if isinstance(result2, list) else []


def hf_embeddings(api_key: str, texts: List[str]) -> Optional[List[List[float]]]:
    """Sentence embeddings via feature-extraction endpoint."""
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True},
    }
    result = _post(api_key, MODELS["embeddings"], payload)
    if isinstance(result, list):
        return result
    return None


def hf_generate(
    api_key: str,
    prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 0.3,
) -> str:
    """
    Text generation via Meta LLaMA-3-8B-Instruct (meta-llama/Meta-Llama-3-8B-Instruct).
    Features 3, 4, 6, 7 — "LLaMA-3 / GPT-4 · grounded by RAG" (architecture image).

    NOTE: LLaMA-3 on HF requires accepting the model licence at:
    https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    After accepting, the same HF token works for inference.
    Falls back to mistralai/Mistral-7B-Instruct-v0.3 if LLaMA-3 is
    inaccessible (e.g. licence not yet accepted).
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    # Primary: LLaMA-3 (as per image)
    result = _post(api_key, MODELS["text_gen"], payload)
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").strip()
    # Fallback: Mistral-7B-Instruct if LLaMA-3 licence not accepted
    result2 = _post(api_key, "mistralai/Mistral-7B-Instruct-v0.3", payload)
    if isinstance(result2, list) and result2:
        return result2[0].get("generated_text", "").strip()
    return ""


def hf_translate(api_key: str, text: str, target_lang: str) -> str:
    """
    Translation via IndicTrans2 (ai4bharat/indictrans2-en-indic-1B).
    Feature 8 — "IndicTrans2 → Hindi · Telugu" (architecture image).

    IndicTrans2 is an 18-layer encoder-decoder trained on the BPCC
    230M bitext pair corpus with SentencePiece tokenizer.
    Input format requires language tags: Eng_Latn → Hin_Deva / Tel_Telu.

    Falls back to Helsinki-NLP/opus-mt-* if IndicTrans2 is unavailable
    on the free-tier HF Inference API.
    """
    if target_lang == "English":
        return text

    # IndicTrans2 language tags
    src_tag  = "Eng_Latn"
    tgt_tag  = "Hin_Deva" if target_lang == "Hindi" else "Tel_Telu"

    words = text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)]
    translated_parts = []

    for chunk in chunks:
        # IndicTrans2 expects a tagged input string
        tagged_input = f"{src_tag} {chunk}"
        payload = {
            "inputs": tagged_input,
            "parameters": {"src_lang": src_tag, "tgt_lang": tgt_tag},
            "options": {"wait_for_model": True},
        }
        result = _post(api_key, MODELS["translate_indic"], payload)
        if isinstance(result, list) and result:
            translated_parts.append(
                result[0].get("translation_text", chunk)
            )
            continue
        if isinstance(result, dict) and "translation_text" in result:
            translated_parts.append(result["translation_text"])
            continue

        # Fallback: Helsinki-NLP opus-mt
        fallback_model = (
            "Helsinki-NLP/opus-mt-en-hi" if target_lang == "Hindi"
            else "Helsinki-NLP/opus-mt-en-te"
        )
        fb_payload = {"inputs": chunk, "options": {"wait_for_model": True}}
        fb_result = _post(api_key, fallback_model, fb_payload)
        if isinstance(fb_result, list) and fb_result:
            translated_parts.append(fb_result[0].get("translation_text", chunk))
        else:
            translated_parts.append(chunk)

    return " ".join(translated_parts)


def hf_whisper_transcribe(api_key: str, audio_bytes: bytes, language: str = "en") -> str:
    """
    Speech-to-Text via OpenAI Whisper large-v3 (openai/whisper-large-v3).
    Feature 13 — "OpenAI Whisper (multilingual ASR) · Hindi + English" (architecture image).

    Accepts raw audio bytes (WAV/MP3/OGG). The HF Inference API for
    automatic-speech-recognition accepts binary audio directly.
    Supports Hindi (hi), Telugu (te), and English (en).
    """
    url = f"{HF_API_BASE}/{MODELS['whisper']}"
    headers = _headers(api_key)
    headers["Content-Type"] = "audio/wav"
    for attempt in range(3):
        try:
            resp = requests.post(
                url, headers=headers, data=audio_bytes,
                params={"language": language}, timeout=90
            )
            if resp.status_code == 503:
                wait = resp.json().get("estimated_time", 20)
                time.sleep(min(wait, 30))
                continue
            if resp.status_code == 429:
                time.sleep(15)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("text", "").strip()
        except Exception:
            time.sleep(5)
    return ""
