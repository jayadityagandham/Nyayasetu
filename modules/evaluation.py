"""
evaluation.py — Automated Evaluation Metrics

Computes:
  - ROUGE-L (summarization quality)
  - Flesch-Kincaid Grade Level & Reading Ease (readability)
  - RAGAS proxy (faithfulness, answer relevance, context precision)
  - Section extraction statistics
  - COMET-22 proxy via BLEU overlap for translation quality
"""

import re
import math
from rouge_score import rouge_scorer
import textstat


def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 3),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 3),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 3),
    }


def flesch_kincaid_grade(text: str) -> float:
    try:
        return round(textstat.flesch_kincaid_grade(text), 1)
    except Exception:
        return 0.0


def reading_ease(text: str) -> float:
    try:
        return round(textstat.flesch_reading_ease(text), 1)
    except Exception:
        return 0.0


def ease_label(score: float) -> tuple[str, str]:
    if score >= 70:
        return ("Very Easy (Citizen-friendly)", "#4caf50")
    elif score >= 50:
        return ("Fairly Easy (Student)", "#8bc34a")
    elif score >= 30:
        return ("Difficult (Professional)", "#ff9800")
    else:
        return ("Very Difficult (Expert only)", "#f44336")


def proxy_ragas(context_str: str, explanation: str) -> dict:
    ctx_w = set(context_str.lower().split())
    exp_w = set(explanation.lower().split())
    overlap = len(ctx_w & exp_w) / max(len(exp_w), 1)
    faithfulness    = min(1.0, round(overlap * 2.5, 3))
    relevance       = min(1.0, round(len(explanation.split()) / 120, 3))
    ctx_precision   = min(1.0, round(overlap * 1.8, 3))
    avg             = round((faithfulness + relevance + ctx_precision) / 3, 3)
    return {
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_precision": ctx_precision,
        "avg_score": avg,
    }


def section_extraction_stats(sections: dict) -> dict:
    total = len(sections)
    mapped = sum(1 for v in sections.values() if v.get("bns_equivalent", "").startswith("Section"))
    return {
        "sections_found": total,
        "mapped_to_bns": mapped,
        "unmapped": total - mapped,
        "coverage_pct": round(mapped / total * 100, 1) if total else 0.0,
    }


def summary_stats(text: str) -> dict:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return {
        "word_count": len(text.split()),
        "sentence_count": len([s for s in sentences if len(s) > 5]),
        "avg_words_per_sentence": round(len(text.split()) / max(len(sentences), 1), 1),
    }


def translation_quality_proxy(source: str, translated: str) -> float:
    """
    Very rough BLEU-1 proxy for translation quality when ground truth is unavailable.
    In a real evaluation, COMET-22 would be used with reference translations.
    """
    src_tokens = set(source.lower().split())
    tgt_tokens = set(translated.lower().split())
    # Penalise if output language is same as source (possible API failure)
    overlap_ratio = len(src_tokens & tgt_tokens) / max(len(src_tokens), 1)
    if overlap_ratio > 0.85:
        return 0.0  # Translation likely failed
    return min(1.0, round(1.0 - overlap_ratio, 3))
