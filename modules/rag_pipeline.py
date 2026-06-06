"""
rag_pipeline.py — Stage 2 & 3: RAG Retrieval + Explanation Generation
(Features 3, 4, 5, 6, 7, 11)

Architecture:
  - Stage 2: Hybrid BM25 (sparse) + FAISS via HF Embeddings API (dense)
             with Reciprocal Rank Fusion
  - Stage 3: LLM prompting via Mistral-7B-Instruct on HF (Features 3, 4, 6)
  - Feature 5: RAG-based Legal Knowledge Retrieval
  - Feature 7: Readability Level Control (Basic / Student / Advanced)
  - Feature 11: Similar Case Retrieval via cosine similarity
"""

import json
import os
import re
import numpy as np
import streamlit as st
from typing import Union, Optional, List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from modules.hf_client import hf_embeddings, hf_generate


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base & Index Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_knowledge_base() -> List[Dict]:
    fp = os.path.join("data", "mock_statutes.json")
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _build_search_texts(kb: List[Dict]) -> List[str]:
    return [
        f"{item.get('title','')} {item.get('short_title','')} {item.get('description','')} "
        f"{' '.join(item.get('ingredients',[]))} "
        f"{item.get('layperson_explanation','')} {item.get('real_example','')}"
        for item in kb
    ]


@st.cache_resource
def initialize_indices(_kb_tuple, api_key: str = ""):
    """
    Build FAISS dense index (via HF embeddings API) + BM25 sparse index.
    Falls back to BM25-only when api_key is absent or HF call fails.
    """
    kb = list(_kb_tuple)  # unhash the tuple
    if not kb:
        return None, None, []

    texts = _build_search_texts(kb)
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    faiss_index = None
    if FAISS_AVAILABLE and api_key:
        embs = hf_embeddings(api_key, texts)
        if embs:
            arr = np.array(embs, dtype="float32")
            # Normalise for cosine similarity
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-9)
            faiss_index = faiss.IndexFlatIP(arr.shape[1])
            faiss_index.add(arr)

    return faiss_index, bm25, texts


# ─────────────────────────────────────────────────────────────────────────────
# ILDC Case Index — 35k cases, pre-built offline (Features 5, 11)
# Run scripts/build_ildc_index.py ONCE to generate these files.
# ─────────────────────────────────────────────────────────────────────────────

_IDX_DIR        = os.path.join("data", "indices")
ILDC_META_FILE  = os.path.join(_IDX_DIR, "ildc_metadata.json")
ILDC_BM25_FILE  = os.path.join(_IDX_DIR, "ildc_bm25.pkl")
ILDC_FAISS_FILE = os.path.join(_IDX_DIR, "ildc_faiss.index")
ILDC_TEXTS_FILE = os.path.join(_IDX_DIR, "ildc_texts.pkl")
ILDC_STATS_FILE = os.path.join(_IDX_DIR, "build_stats.json")


def ildc_index_exists() -> bool:
    """True when the pre-built ILDC index files are present on disk."""
    return os.path.exists(ILDC_META_FILE) and os.path.exists(ILDC_BM25_FILE)


@st.cache_data(show_spinner=False)
def load_ildc_metadata() -> List[Dict]:
    """
    Load 35k ILDC case metadata from disk.
    Each entry: case_id, name, court, date, label, label_text,
                sections_cited, annotations, preview, context
    """
    if not os.path.exists(ILDC_META_FILE):
        return []
    with open(ILDC_META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_ildc_indices():
    """
    Load pre-built BM25 + FAISS from disk (built once by build_ildc_index.py).
    Returns (faiss_index | None, bm25 | None, search_texts list)
    """
    import pickle
    bm25, faiss_index, texts = None, None, []
    if os.path.exists(ILDC_BM25_FILE):
        with open(ILDC_BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
    if os.path.exists(ILDC_TEXTS_FILE):
        with open(ILDC_TEXTS_FILE, "rb") as f:
            texts = pickle.load(f)
    if FAISS_AVAILABLE and os.path.exists(ILDC_FAISS_FILE):
        faiss_index = faiss.read_index(ILDC_FAISS_FILE)
    return faiss_index, bm25, texts


def get_ildc_stats() -> dict:
    """Return build statistics for the ILDC index."""
    if os.path.exists(ILDC_STATS_FILE):
        with open(ILDC_STATS_FILE) as f:
            return json.load(f)
    meta = load_ildc_metadata()
    return {
        "total_indexed": len(meta),
        "faiss_built": os.path.exists(ILDC_FAISS_FILE),
        "bm25_built": os.path.exists(ILDC_BM25_FILE),
    }


def retrieve_ildc_cases(
    query: str,
    ildc_meta: List[Dict],
    ildc_faiss,
    ildc_bm25,
    ildc_texts: List[str],
    api_key: str = "",
    top_k: int = 5,
    label_filter: str = "all",
    section_filter: str = "",
) -> List[Dict]:
    """
    Feature 5 & 11 — Retrieve similar ILDC cases using BM25 + FAISS RRF.

    Supports:
      - label_filter:   filter results by judgment outcome
      - section_filter: filter by IPC section cited in the case
    """
    if not ildc_meta or (ildc_faiss is None and ildc_bm25 is None):
        return []

    rrf: dict = {}

    # Sparse BM25
    if ildc_bm25:
        bm25_scores = ildc_bm25.get_scores(query.lower().split())
        sparse_hits = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_k * 3]
        for rank, (i, _) in enumerate(sparse_hits):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (rank + 1)

    # Dense FAISS
    if ildc_faiss is not None and FAISS_AVAILABLE and api_key:
        q_emb = hf_embeddings(api_key, [query[:500]])
        if q_emb:
            q_arr = np.array(q_emb, dtype="float32")
            q_arr /= np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-9)
            n_search = min(top_k * 3, len(ildc_meta))
            scores, idxs = ildc_faiss.search(q_arr, n_search)
            for rank, idx in enumerate(idxs[0]):
                if idx >= 0:
                    rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rank + 1)
                    # Store similarity score in metadata temporarily
                    if idx < len(ildc_meta):
                        ildc_meta[idx]["_similarity"] = round(float(scores[0][rank]), 3)

    ranked = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)
    results = []
    for idx in ranked:
        if idx >= len(ildc_meta):
            continue
        case = ildc_meta[idx]

        # Apply filters
        if label_filter == "accepted" and case.get("label") != 1:
            continue
        if label_filter == "rejected" and case.get("label") != 0:
            continue
        if section_filter and section_filter not in case.get("sections_cited", []):
            continue

        entry = dict(case)
        entry["_rrf_score"] = round(rrf[idx], 4)
        results.append(entry)

        if len(results) >= top_k:
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Feature 11 — Similar Case Retrieval (over ILDC, via embedding)
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_cases(
    case_summary: str,
    ildc_meta: List[Dict],
    ildc_faiss,
    ildc_texts: List[str],
    api_key: str = "",
    top_k: int = 5,
) -> List[Dict]:
    """
    Feature 11 — Sentence-BERT case encoder + FAISS cosine similarity.
    Embeds the input case summary and finds the most semantically
    similar cases in the 35k ILDC index.
    Falls back to BM25 keyword search via retrieve_ildc_cases.
    """
    if not ildc_meta:
        return []

    if api_key and ildc_faiss is not None and FAISS_AVAILABLE:
        q_emb = hf_embeddings(api_key, [case_summary[:500]])
        if q_emb:
            q_arr = np.array(q_emb, dtype="float32")
            q_arr /= np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-9)
            n_search = min(top_k, len(ildc_meta))
            scores, idxs = ildc_faiss.search(q_arr, n_search)
            results = []
            for i, idx in enumerate(idxs[0]):
                if idx >= 0 and idx < len(ildc_meta):
                    entry = dict(ildc_meta[idx])
                    entry["_similarity"] = round(float(scores[0][i]), 3)
                    results.append(entry)
            return results

    # BM25 fallback — no FAISS or no api_key
    if ildc_meta:
        from rank_bm25 import BM25Okapi
        _, ildc_bm25, _ = load_ildc_indices()
        if ildc_bm25:
            return retrieve_ildc_cases(
                case_summary, ildc_meta, None, ildc_bm25, [], api_key="", top_k=top_k
            )
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — Hybrid Retrieval (BM25 + FAISS + RRF)

# ─────────────────────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    kb: List[Dict],
    faiss_index,
    bm25,
    texts: List[str],
    api_key: str = "",
    top_k: int = 3,
) -> Tuple[List[Dict], str]:
    """Retrieve top-k KB entries using Reciprocal Rank Fusion of BM25 + FAISS."""
    if not kb or (faiss_index is None and bm25 is None):
        return [], "No knowledge base available."

    rrf: dict = {}

    # Sparse BM25
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_hits = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[: top_k * 2]
    for rank, (i, _) in enumerate(sparse_hits):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (rank + 1)

    # Dense FAISS
    if faiss_index is not None and FAISS_AVAILABLE and api_key:
        q_emb = hf_embeddings(api_key, [query])
        if q_emb:
            q_arr = np.array(q_emb, dtype="float32")
            q_arr /= np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-9)
            scores, idxs = faiss_index.search(q_arr, min(top_k * 2, len(kb)))
            for rank, idx in enumerate(idxs[0]):
                if idx >= 0:
                    rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rank + 1)

    ranked = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)
    matched = [kb[i] for i in ranked[:top_k]]
    ctx_str = " | ".join(e.get("description", "") for e in matched)
    return matched, ctx_str


# ─────────────────────────────────────────────────────────────────────────────
# Feature 11 — Similar Case Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_cases_kb(
    case_summary: str,
    kb: List[Dict],
    faiss_index,
    texts: List[str],
    api_key: str = "",
    top_k: int = 3,
) -> List[Dict]:
    """
    Embed the current case summary and find semantically similar KB entries.
    Returns top-k most similar entries as surrogate 'similar cases'.
    """
    if not api_key or faiss_index is None or not FAISS_AVAILABLE:
        return []
    q_emb = hf_embeddings(api_key, [case_summary[:400]])
    if not q_emb:
        return []
    q_arr = np.array(q_emb, dtype="float32")
    q_arr /= np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-9)
    scores, idxs = faiss_index.search(q_arr, min(top_k, len(kb)))
    results = []
    for i, idx in enumerate(idxs[0]):
        if idx >= 0:
            entry = dict(kb[idx])
            entry["_similarity"] = round(float(scores[0][i]), 3)
            results.append(entry)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builders (Features 3, 4, 7)
# ─────────────────────────────────────────────────────────────────────────────

def _readability_system(level: str) -> str:
    if level == "Basic":
        return (
            "You are a legal educator explaining Indian law to ordinary citizens. "
            "Use simple language (Grade 8 reading level), real-life analogies, "
            "avoid jargon, and keep sentences short."
        )
    elif level == "Student":
        return (
            "You are a law professor explaining Indian law to undergraduate law students. "
            "Use correct legal terminology, cite statutory provisions precisely, "
            "and explain the doctrinal basis."
        )
    else:
        return (
            "You are a senior advocate writing a legal opinion. "
            "Provide technical statutory analysis, cite actus reus and mens rea requirements, "
            "note transitional implications of the IPC→BNS shift, and cite landmark precedents."
        )


def _build_why_applied_prompt(
    section_key: str, title: str, bns: str,
    description: str, ingredients: List[str],
    case_context: str, level: str,
) -> str:
    sys = _readability_system(level)
    ing_text = "\n".join(f"- {i}" for i in ingredients) if ingredients else "See statutory text."
    return (
        f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n"
        f"Explain WHY {section_key} ({title}) was applied in this case.\n\n"
        f"Statutory provision: {description}\n"
        f"Ingredients:\n{ing_text}\n\n"
        f"Case facts (excerpt): {case_context[:500]}\n\n"
        f"Also mention that under BNS 2023 this is now {bns}. "
        f"Generate a clear, structured explanation. [/INST]"
    )


def _build_plain_lang_prompt(
    section_key: str, title: str, bns: str,
    description: str, penalty: str, layperson: str,
    level: str,
) -> str:
    sys = _readability_system(level)
    return (
        f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n"
        f"Explain {section_key} ({title}) in {level.lower()} language.\n\n"
        f"Statutory text: {description}\n"
        f"Penalty: {penalty}\n"
        f"Simple explanation: {layperson}\n"
        f"BNS 2023 equivalent: {bns}\n\n"
        f"Write a clear explanation with the penalty and a note about the BNS update. [/INST]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS Proxy Scorer
# ─────────────────────────────────────────────────────────────────────────────

def _ragas_score(context_str: str, explanation: str) -> dict:
    ctx_w = set(context_str.lower().split())
    exp_w = set(explanation.lower().split())
    overlap = len(ctx_w & exp_w) / max(len(exp_w), 1)
    faithfulness    = min(1.0, round(overlap * 2.5, 3))
    relevance       = min(1.0, round(len(explanation.split()) / 120, 3))
    ctx_precision   = min(1.0, round(overlap * 1.8, 3))
    avg             = round((faithfulness + relevance + ctx_precision) / 3, 3)
    if avg >= 0.65:
        badge = ("🟢 High Confidence", "green")
    elif avg >= 0.35:
        badge = ("🟡 Medium Confidence", "orange")
    else:
        badge = ("🔴 Low Confidence", "red")
    return {
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_precision": ctx_precision,
        "avg": avg,
        "badge": badge,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — "Why Law Applied" + Plain Language Explanation (LLM-grounded)
# ─────────────────────────────────────────────────────────────────────────────

def _rag_compose_fallback(section_key, title, bns, kb_entry, case_context, level):
    """Pure RAG composition when no API key is available."""
    desc       = kb_entry.get("description", "") if kb_entry else ""
    layperson  = kb_entry.get("layperson_explanation", desc[:200]) if kb_entry else ""
    ingredients = kb_entry.get("ingredients", []) if kb_entry else []
    penalty    = kb_entry.get("penalty", "N/A") if kb_entry else "N/A"
    example    = kb_entry.get("real_example", "") if kb_entry else ""

    why_lines = [f"**Why was {section_key} applied?**\n"]
    why_lines.append(f"Statutory text: *{desc}*")
    if ingredients:
        why_lines.append("\n**Ingredients:**")
        for ing in ingredients:
            why_lines.append(f"- {ing}")
    why_lines.append(f"\n📌 Under BNS 2023, {section_key} is renumbered as **{bns}**.")

    plain_lines = [f"**{section_key} — {title}**\n"]
    plain_lines.append(layperson or desc[:300])
    plain_lines.append(f"\n**Penalty:** {penalty}")
    plain_lines.append(f"\n**BNS equivalent:** {bns}")

    ex = f"**Example:** {example}" if example else f"Any conduct satisfying all ingredients of {title} attracts this section."

    return "\n".join(why_lines), "\n".join(plain_lines), ex


def generate_explanation(
    section_key: str,
    section_info: dict,
    kb_entry: Optional[Dict],
    case_context: str,
    readability: str = "Basic",
    api_key: str = "",
) -> dict:
    """
    Generate structured explanations for a legal section.
    Uses Mistral via HF API when api_key is provided; pure RAG fallback otherwise.
    """
    title  = section_info.get("title", section_key)
    bns    = section_info.get("bns_equivalent", "Not yet mapped")
    desc   = kb_entry.get("description", "") if kb_entry else ""
    ing    = kb_entry.get("ingredients", []) if kb_entry else []
    pen    = kb_entry.get("penalty", "N/A") if kb_entry else "N/A"
    layp   = kb_entry.get("layperson_explanation", desc[:200]) if kb_entry else desc[:200]
    ex     = kb_entry.get("real_example", "") if kb_entry else ""

    used_llm = False
    if api_key:
        # Feature 3 — Why Law Applied
        why_prompt = _build_why_applied_prompt(section_key, title, bns, desc, ing, case_context, readability)
        why = hf_generate(api_key, why_prompt, max_new_tokens=350, temperature=0.2)

        # Feature 4 — Plain Language
        plain_prompt = _build_plain_lang_prompt(section_key, title, bns, desc, pen, layp, readability)
        plain = hf_generate(api_key, plain_prompt, max_new_tokens=300, temperature=0.2)

        # Example from KB
        example_text = (
            f"**Real-life example (from knowledge base):** {ex}"
            if ex else
            f"Any conduct satisfying all ingredients of **{title}** ({section_key}) attracts this provision."
        )

        if why and plain:
            used_llm = True
        else:
            why, plain, example_text = _rag_compose_fallback(section_key, title, bns, kb_entry, case_context, readability)
    else:
        why, plain, example_text = _rag_compose_fallback(section_key, title, bns, kb_entry, case_context, readability)

    ctx_str = (desc + " " + layp) if kb_entry else case_context[:300]
    ragas = _ragas_score(ctx_str, why + plain + example_text)

    return {
        "🔎 Why This Law Was Applied": why,
        "📖 Plain Language Explanation": plain,
        "🌍 Real-Life Example": example_text,
        "_ragas": ragas,
        "_used_llm": used_llm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Legal Learning QA (Standalone)
# ─────────────────────────────────────────────────────────────────────────────

def answer_legal_question(
    query: str,
    readability: str,
    kb: List[Dict],
    faiss_index,
    bm25,
    texts: List[str],
    api_key: str = "",
) -> Tuple[str, dict]:
    """
    Answer a free-form legal question using the hybrid RAG pipeline.
    Uses Mistral-7B-Instruct when api_key is provided.
    """
    matched, ctx_str = retrieve_context(query, kb, faiss_index, bm25, texts, api_key, top_k=3)

    if not matched:
        msg = (
            "> ⚠️ No relevant statutory provisions found. "
            "Try rephrasing with specific legal terms (e.g., 'murder', 'bail', 'FIR', 'conspiracy')."
        )
        return msg, _ragas_score("", msg)

    # Build context from retrieved entries
    context_block = "\n\n".join(
        f"**{e['section']} IPC / {e['bns_equivalent']} BNS** — {e['title']}\n"
        f"{e.get('description', '')}\n"
        f"Penalty: {e.get('penalty', 'N/A')}"
        for e in matched
    )

    if api_key:
        sys = _readability_system(readability)
        prompt = (
            f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n"
            f"Use ONLY the following retrieved legal provisions to answer the question. "
            f"Do not add information not present in the context. "
            f"Always cite the section numbers.\n\n"
            f"Retrieved Statutory Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Provide a structured answer with: Direct Answer, Relevant Provisions, "
            f"and a Disclaimer that this is for educational purposes only. [/INST]"
        )
        llm_answer = hf_generate(api_key, prompt, max_new_tokens=450, temperature=0.2)
        if llm_answer:
            answer = (
                f"### Legal Answer — *{readability} level*\n\n"
                f"**Question:** *{query}*\n\n---\n\n"
                f"{llm_answer}\n\n---\n\n"
                f"> ⚠️ **Disclaimer:** For educational purposes only. "
                f"Consult a qualified advocate for specific legal matters."
            )
            return answer, _ragas_score(ctx_str, answer)

    # Pure RAG fallback (no API key or LLM failed)
    lines = [
        f"### Legal Answer — *{readability} level*\n",
        f"**Your question:** *{query}*\n\n---\n",
        "## 📌 Retrieved Statutory Provisions\n",
    ]
    for e in matched:
        lines.append(f"### {e['section']} IPC → {e['bns_equivalent']}")
        lines.append(f"**{e['title']}**\n")
        lines.append(f"{e.get('description', '')}\n")
        if e.get("ingredients"):
            lines.append("**Ingredients:** " + " | ".join(e["ingredients"]))
        if e.get("penalty"):
            lines.append(f"**Penalty:** `{e['penalty']}`\n")
        if e.get("layperson_explanation") and readability == "Basic":
            lines.append(f"**In plain words:** {e['layperson_explanation']}\n")
        lines.append("---")
    lines.append(
        "\n## 📌 IPC → BNS Update\n"
        "The Bharatiya Nyaya Sanhita (BNS) 2023, effective 1 July 2024, replaced the IPC. "
        "All provisions above now have BNS equivalents.\n"
    )
    lines.append(
        "> ⚠️ **Disclaimer:** For educational purposes only. "
        "Consult a qualified advocate for specific legal matters."
    )
    answer = "\n".join(lines)
    return answer, _ragas_score(ctx_str, answer)


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up Question Generation
# ─────────────────────────────────────────────────────────────────────────────

_FOLLOWUP_BANK = {
    "murder":          ["What is the difference between murder and culpable homicide?","Can a murder accused get bail?","How does the court decide between death penalty and life imprisonment?"],
    "culpable homicide":["When does culpable homicide become murder?","What is the penalty for culpable homicide not amounting to murder?"],
    "conspiracy":      ["How is criminal conspiracy proven without direct evidence?","Can one be convicted of conspiracy if the main crime was not completed?"],
    "rape":            ["What protections does the law give to rape victims during trial?","What is the difference between rape and sexual assault under BNS?"],
    "bail":            ["What is the difference between regular bail and anticipatory bail?","On what grounds can bail be refused in a murder case?"],
    "fir":             ["What can you do if the police refuse to register an FIR?","Can an FIR be quashed by the High Court?"],
    "theft":           ["What is the difference between theft, robbery, and dacoity?","What is the punishment for repeat theft offenders?"],
    "cheating":        ["What must the prosecution prove to establish cheating?","Is online fraud covered under Section 318 BNS?"],
    "evidence":        ["Can a statement made to police be used in court?","What is the difference between direct and circumstantial evidence?"],
    "appeal":          ["What is the time limit to file a criminal appeal?","Can the High Court enhance a sentence on appeal by the accused?"],
    "supreme court":   ["How is an SLP different from a regular criminal appeal?","What is the time limit for filing a Special Leave Petition?"],
    "domestic":        ["What is the difference between Section 498A IPC and the Domestic Violence Act?","Is dowry harassment a cognizable offence?"],
}

_GENERIC_FOLLOWUPS = [
    "What rights does an accused person have during a criminal trial?",
    "How does the Indian court hierarchy work?",
    "What is the difference between cognizable and non-cognizable offences?",
    "What happens if a key witness turns hostile during trial?",
    "What is the doctrine of circumstantial evidence in Indian criminal law?",
    "Can a Sessions Court acquittal be appealed by the State?",
]


def generate_followup_questions(retrieved: List[Dict], current_query: str = "", top_n: int = 4) -> List[str]:
    questions, seen = [], set()
    for entry in retrieved:
        combined = (entry.get("short_title","") + " " + entry.get("title","")).lower()
        for keyword, qs in _FOLLOWUP_BANK.items():
            if keyword in combined:
                for q in qs:
                    if q not in seen and q.lower() != current_query.lower():
                        questions.append(q)
                        seen.add(q)
                        if len(questions) >= top_n:
                            return questions
    for q in _GENERIC_FOLLOWUPS:
        if len(questions) >= top_n:
            break
        if q not in seen and q.lower() != current_query.lower():
            questions.append(q)
            seen.add(q)
    return questions[:top_n]


def generate_summary_followups(sections: dict, segments: dict) -> List[str]:
    questions, seen = [], set()
    for sec_key, info in list(sections.items())[:2]:
        title = info.get("title", sec_key)
        q = f"What does {sec_key} ({title}) mean in simple language?"
        if q not in seen:
            questions.append(q); seen.add(q)
        bns = info.get("bns_equivalent", "")
        if bns and bns != "Not yet mapped":
            q2 = f"How does {sec_key} IPC differ from its BNS equivalent?"
            if q2 not in seen:
                questions.append(q2); seen.add(q2)
    procedural = [
        "Can the convicted accused appeal against this judgment?",
        "What is the role of the prosecution in proving guilt beyond reasonable doubt?",
        "What type of evidence is most important in cases like this?",
        "How does the court decide between life imprisonment and the death penalty?",
        "What constitutional rights does the accused have during and after trial?",
    ]
    for q in procedural:
        if len(questions) >= 6:
            break
        if q not in seen:
            questions.append(q); seen.add(q)
    return questions[:6]
