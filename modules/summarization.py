"""
summarization.py — Feature 1 & 4: Case Summarization via HF BART API

Hierarchical summarization pipeline:
  1. Chunk each rhetorical role segment
  2. Summarize each chunk via BART-large-CNN (HF API)
  3. Merge chunk summaries and re-summarize for final output

Feature 4: Readability-level conditioning is applied by adjusting
max_length and appending a simplification post-pass via Mistral.
"""

import re
import streamlit as st
from typing import List, Dict, Tuple, Any, Optional, Union
from modules.hf_client import hf_summarize, hf_generate


def _chunk_text(text: str, max_words: int = 800, overlap: int = 50) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


ROLE_CONFIG = {
    "FACTS":     {"max_len": 280, "min_len": 100},
    "ISSUE":     {"max_len": 180, "min_len": 60},
    "ARGUMENT":  {"max_len": 260, "min_len": 90},
    "STATUTE":   {"max_len": 160, "min_len": 50},
    "PRECEDENT": {"max_len": 200, "min_len": 60},
    "RATIO":     {"max_len": 240, "min_len": 80},
    "DECISION":  {"max_len": 200, "min_len": 60},
}


def _summarize_one(api_key: str, text: str, max_len: int = 220, min_len: int = 60) -> str:
    """Summarize a single chunk, with graceful fallback."""
    if len(text.split()) < 30:
        return text
    result = hf_summarize(api_key, text, max_length=max_len, min_length=min_len)
    if not result or result == text[:300]:
        # Extractive fallback
        sents = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sents[:3])
    return result


def summarize_roles(segments: dict, api_key: str = "") -> dict:
    """
    Produce one summary string per rhetorical role using HF BART.
    Falls back to extractive summarization when api_key is absent.
    """
    summaries = {}
    for role, text in segments.items():
        text = text.strip()
        if not text or len(text.split()) < 20:
            summaries[role] = None
            continue

        cfg = ROLE_CONFIG.get(role, {"max_len": 220, "min_len": 60})
        chunks = _chunk_text(text, max_words=800)

        if not api_key:
            # Pure extractive fallback (no API needed)
            sents = re.split(r"(?<=[.!?])\s+", text)
            summaries[role] = " ".join(sents[:4])
            continue

        if len(chunks) == 1:
            summaries[role] = _summarize_one(api_key, chunks[0], cfg["max_len"], cfg["min_len"])
        else:
            chunk_sums = [_summarize_one(api_key, c, 160, 40) for c in chunks]
            merged = " ".join(chunk_sums)
            summaries[role] = _summarize_one(api_key, merged, cfg["max_len"], cfg["min_len"])

    return summaries


def apply_readability_rewrite(api_key: str, text: str, level: str) -> str:
    """
    Feature 4: Rewrite a summary at the given readability level using Mistral.
    Levels: Basic (citizen), Student (law student), Advanced (practitioner).
    """
    if not api_key or not text.strip():
        return text

    if level == "Basic":
        instruction = (
            "Rewrite the following legal text in simple English that any Indian citizen "
            "without legal background can understand. Use short sentences, avoid jargon, "
            "and explain technical terms in brackets.\n\nText:\n"
        )
    elif level == "Student":
        instruction = (
            "Rewrite the following legal text for a law student. Use correct legal "
            "terminology, cite relevant principles, and maintain academic precision.\n\nText:\n"
        )
    else:  # Advanced
        instruction = (
            "Rewrite the following for a legal practitioner: include actus reus/mens rea "
            "analysis, relevant doctrines, and statutory cross-references.\n\nText:\n"
        )

    prompt = f"[INST] {instruction}{text[:600]} [/INST]"
    result = hf_generate(api_key, prompt, max_new_tokens=350, temperature=0.2)
    return result if result else text


def generate_story_summary(api_key: str, text: str, readability: str = "Basic") -> str:
    """
    Generate a cohesive, narrative story of the case.
    """
    if not api_key:
        return "⚠️ Please provide an HF API key to generate a story summary."

    prompt = (
        f"[INST] You are an expert legal storyteller. Read the following case text and write a captivating, "
        f"easy-to-follow narrative story summarizing what happened, the main conflict, and the court's final decision. "
        f"Make it flow like a cohesive story. Readability level: {readability}.\n\n"
        f"Text:\n{text[:2000]} [/INST]"
    )
    result = hf_generate(api_key, prompt, max_new_tokens=600, temperature=0.4)
    return result if result else "Failed to generate story summary."


def generate_glossary(api_key: str, keywords: List[Tuple[str, str]], readability: str = "Basic") -> Dict:
    """
    Generate a short, simple definition for the extracted legal keywords.
    """
    if not api_key or not keywords:
        return {}

    # Extract unique legal terms and laws
    terms = list(set([kw for kw, t in keywords if t in ("LEGAL_TERM", "LAW")]))
    if not terms:
        return {}

    prompt = (
        f"[INST] You are a legal dictionary. Briefly define the following Indian legal terms "
        f"in simple language suitable for a {readability} audience. "
        f"Format the output strictly as 'Term: Definition' on each line.\n\n"
        f"Terms:\n{', '.join(terms[:15])} [/INST]"
    )
    result = hf_generate(api_key, prompt, max_new_tokens=500, temperature=0.1)
    
    glossary = {}
    if result:
        for line in result.split("\n"):
            if ":" in line:
                term, definition = line.split(":", 1)
                glossary[term.strip().replace("*", "")] = definition.strip()
    return glossary

