"""
document_processing.py — Stage 1: Document Processing Pipeline (Features 1, 2, 10, 12)

Uses HuggingFace Inference API for:
  - Zero-shot rhetorical role labelling (Feature 1) 
  - Named entity recognition (Feature 2, 12)
  - Keyword & concept highlighting (Feature 12)
  - Timeline extraction (Feature 10)

Lightweight regex-based operations run locally (no GPU needed).
"""

import re
import pdfplumber
from typing import List, Dict, Tuple, Any, Optional, Union
from modules.hf_client import hf_inlegalbert_classify, hf_ner, hf_whisper_transcribe

# ── Role display config ───────────────────────────────────────────────────────
ROLE_DISPLAY = {
    "FACTS":     ("📁 Facts of the Case",            "#1a7abf"),
    "ISSUE":     ("❓ Issues / Questions of Law",    "#9c27b0"),
    "ARGUMENT":  ("⚖️ Arguments Advanced",           "#2e7d32"),
    "STATUTE":   ("📜 Statutes & Provisions Cited",  "#e65100"),
    "PRECEDENT": ("📚 Precedents & Case Laws Cited", "#5c35cc"),
    "RATIO":     ("🔍 Ratio Decidendi (Reasoning)",  "#00838f"),
    "DECISION":  ("🔔 Decision / Order",             "#c62828"),
}

ROLE_LABELS = list(ROLE_DISPLAY.keys())

# Keyword fallback (used when HF API is unavailable)
ROLE_KEYWORDS = {
    "FACTS":     ["alleged","fir","complainant","incident","occurred","victim","deceased","accused","arrested","forensic","witness"],
    "ISSUE":     ["whether","question of law","issue before","arises for consideration","point for determination"],
    "ARGUMENT":  ["submitted","urged","contended","argued","counsel","on behalf of","it is submitted","it was argued"],
    "STATUTE":   ["section","ipc","crpc","bns","indian penal code","act,","provision","punishable under"],
    "PRECEDENT": [" v. "," vs. "," versus ","air ","scc ","(sc)","held in","relying upon","this court in"],
    "RATIO":     ["we hold","we are of the view","we find","in our opinion","it is settled","after careful consideration"],
    "DECISION":  ["set aside","is allowed","is dismissed","convicted","acquitted","sentenced to","appeal is allowed","we direct"],
}


# ─────────────────────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(file) -> str:
    """Extract raw text from PDF or TXT upload."""
    if file.name.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    return file.getvalue().decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"S\.\s*(\d+)", r"Section \1", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — Rhetorical Role Segmentation (HF zero-shot)
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_role(sentence: str) -> str:
    """Fallback keyword-based role classifier."""
    sl = sentence.lower()
    scores = {role: sum(1 for kw in kws if kw in sl) for role, kws in ROLE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "FACTS"


def rhetorical_role_segmentation(text: str, api_key: str = "", use_hf: bool = True) -> dict:
    """
    Segment judgment text into 7 rhetorical roles.
    Uses HF zero-shot classification when api_key is provided,
    falls back to keyword matching otherwise.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
    segments = {role: [] for role in ROLE_DISPLAY}

    for sent in sentences:
        if api_key and use_hf and len(sent.split()) > 5:
            # Feature 1: InLegalBERT → Facts / Issues / Decision
            # law-ai/InLegalBERT with cosine similarity role classification
            scores = hf_inlegalbert_classify(api_key, sent, ROLE_LABELS)
            if scores:
                role = max(scores, key=scores.get)
            else:
                role = _keyword_role(sent)
        else:
            role = _keyword_role(sent)
        segments[role].append(sent)

    return {role: " ".join(sents) for role, sents in segments.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Legal Section Extraction
# ─────────────────────────────────────────────────────────────────────────────

# Extended IPC → BNS cross-reference map
IPC_BNS_MAP = {
    "4":    {"bns": "1(4)",   "title": "Extra-territorial offences"},
    "5":    {"bns": "1(5)",   "title": "Certain laws not to be affected"},
    "6":    {"bns": "1(6)",   "title": "Definitions subject to exceptions"},
    "7":    {"bns": "1(7)",   "title": "Sense of expression once explained"},
    "8":    {"bns": "2",      "title": "Gender"},
    "9":    {"bns": "2",      "title": "Number"},
    "10":   {"bns": "2",      "title": "Man, Woman"},
    "120B": {"bns": "61(2)",  "title": "Criminal Conspiracy"},
    "299":  {"bns": "99",     "title": "Culpable Homicide"},
    "300":  {"bns": "100",    "title": "Murder"},
    "302":  {"bns": "103",    "title": "Punishment for Murder"},
    "304":  {"bns": "105",    "title": "Culpable Homicide not amounting to Murder"},
    "304A": {"bns": "106",    "title": "Causing Death by Negligence"},
    "307":  {"bns": "109",    "title": "Attempt to Murder"},
    "323":  {"bns": "115",    "title": "Voluntarily Causing Hurt"},
    "324":  {"bns": "117",    "title": "Voluntarily Causing Hurt by Dangerous Weapon"},
    "325":  {"bns": "116",    "title": "Grievous Hurt"},
    "326":  {"bns": "118",    "title": "Grievous Hurt by Dangerous Weapon"},
    "341":  {"bns": "126",    "title": "Wrongful Restraint"},
    "354":  {"bns": "74",     "title": "Assault / Outrage Modesty"},
    "363":  {"bns": "137",    "title": "Kidnapping"},
    "365":  {"bns": "139",    "title": "Kidnapping for Secret Confinement"},
    "376":  {"bns": "64",     "title": "Punishment for Rape"},
    "377":  {"bns": "---",    "title": "Unnatural Offences (repealed / modified)"},
    "378":  {"bns": "303",    "title": "Theft"},
    "379":  {"bns": "303(2)", "title": "Punishment for Theft"},
    "392":  {"bns": "309",    "title": "Robbery"},
    "395":  {"bns": "310",    "title": "Dacoity"},
    "396":  {"bns": "311",    "title": "Dacoity with Murder"},
    "406":  {"bns": "316",    "title": "Criminal Breach of Trust"},
    "420":  {"bns": "318",    "title": "Cheating and Dishonest Inducement"},
    "498A": {"bns": "85",     "title": "Cruelty by Husband or Relatives"},
    "499":  {"bns": "356",    "title": "Defamation"},
    "500":  {"bns": "356(2)", "title": "Punishment for Defamation"},
}

def extract_legal_sections(text: str) -> dict:
    """Regex + NER-based legal section extractor with IPC→BNS cross-reference."""
    found = {}

    # Pattern 1: "Section 302 of the Indian Penal Code"
    p1 = re.finditer(
        r"Section\s+(\d+[A-Z]?)\s*(?:of\s+)?(?:the\s+)?(IPC|Indian Penal Code|BNS|Bharatiya Nyaya Sanhita)",
        text, re.IGNORECASE
    )
    for m in p1:
        sec_num = m.group(1).upper()
        _add_section(found, sec_num)

    # Pattern 2: standalone "s.302 IPC" or "302 IPC"
    p2 = re.finditer(r"\b[Ss]\.?\s*(\d+[A-Z]?)\s+IPC\b", text)
    for m in p2:
        _add_section(found, m.group(1).upper())

    # Pattern 3: "u/s 302" or "under Section 302"
    p3 = re.finditer(r"(?:u/s|under [Ss]ection)\s+(\d+[A-Z]?)", text)
    for m in p3:
        _add_section(found, m.group(1).upper())

    return found


def _add_section(found: dict, sec_num: str):
    key = f"Section {sec_num} IPC"
    if key not in found:
        if sec_num in IPC_BNS_MAP:
            found[key] = {
                "bns_equivalent": f"Section {IPC_BNS_MAP[sec_num]['bns']} BNS",
                "title": IPC_BNS_MAP[sec_num]["title"],
            }
        else:
            found[key] = {"bns_equivalent": "Not yet mapped", "title": "Refer to BNS gazette"}


# ─────────────────────────────────────────────────────────────────────────────
# Feature 10 — Case Timeline Extraction
# ─────────────────────────────────────────────────────────────────────────────

EVENT_KEYWORDS = {
    "Incident / Offence":            ["incident","occurred","took place","found dead","murdered","assault","offence","crime","committed","deceased","victim"],
    "FIR / Complaint Filed":         ["fir","first information report","complaint","lodged","registered","filed","police station","informant"],
    "Legal Notice / Pre-Trial":      ["legal notice","notice","demand","served","issued notice","application filed","charge sheet"],
    "Trial / Sessions Court":        ["trial","sessions court","trial court","convicted","acquitted","sentence","sentenced","trial judge"],
    "High Court":                    ["high court","division bench","appellate court","high court judgment","high court order"],
    "Supreme Court / Final Judgment":["supreme court","this court","present appeal","set aside","restored","dismissed","allowed","final judgment","date of judgment"],
}

DATE_PATTERNS = [
    r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
    r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b",
]
MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]


def _is_proper_date(d: str) -> bool:
    dl = d.lower().strip()
    if any(m in dl for m in MONTHS):
        return True
    if re.match(r"^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}$", dl):
        return True
    return False


def _classify_event(ctx: str) -> str:
    scores = {ev: sum(1 for kw in kws if kw in ctx) for ev, kws in EVENT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Reference"


def extract_timeline(text: str) -> List[Dict]:
    """Extract dated events and classify them into canonical timeline phases."""
    results = {}
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            ds = m.group(0)
            if _is_proper_date(ds) and ds not in results:
                s = max(0, m.start() - 200)
                e = min(len(text), m.end() + 200)
                results[ds] = text[s:e].lower()

    timeline = []
    for date_str, ctx in results.items():
        event_label = _classify_event(ctx)
        m2 = re.search(re.escape(date_str), text, re.IGNORECASE)
        if m2:
            s = max(0, m2.start() - 80)
            e = min(len(text), m2.end() + 120)
            snippet = "..." + text[s:e].strip() + "..."
        else:
            snippet = ctx[:180] + "..."
        timeline.append({"date": date_str, "event": event_label, "snippet": snippet})

    order = list(EVENT_KEYWORDS.keys()) + ["General Reference"]
    timeline.sort(key=lambda x: order.index(x["event"]) if x["event"] in order else 99)
    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# Feature 12 — Keyword & Concept Highlighting
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_TERMS_REGEX = (
    r"\b(Section\s+\d+[A-Z]?\s*(?:of\s+)?(?:IPC|BNS|CrPC|BNSS)?"
    r"|murder|rape|theft|conspiracy|abetment|culpable homicide"
    r"|mens rea|actus reus|burden of proof|prima facie|suo motu"
    r"|ratio decidendi|obiter dicta|stare decisis|habeas corpus"
    r"|res judicata|locus standi|FIR|charge sheet|cognizable"
    r"|bail|anticipatory bail|SLP|writ petition|mandamus"
    r"|certiorari|prohibition|quo warranto)\b"
)


def extract_keywords(text: str, api_key: str = "") -> List[Tuple]:
    """
    Extract named entities and legal terms. HF NER is used when api_key
    is provided; regex-only extraction is the fallback.
    """
    keywords = {}

    # Legal terms via regex (always available)
    for term in re.findall(LEGAL_TERMS_REGEX, text[:60000], re.IGNORECASE):
        keywords[term.strip()] = "LEGAL_TERM"

    # HF NER for persons, orgs, locations
    if api_key:
        for chunk_start in range(0, min(len(text), 10000), 400):
            chunk = text[chunk_start:chunk_start + 400]
            entities = hf_ner(api_key, chunk)
            for ent in entities:
                word = ent.get("word", "").replace("##", "").strip()
                label = ent.get("entity_group", ent.get("entity", "")).upper()
                if len(word) > 2 and label in ("PER", "ORG", "LOC", "MISC"):
                    label_map = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE", "MISC": "LAW"}
                    keywords[word] = label_map.get(label, label)
    else:
        # Regex fallback for person-like tokens (capitalized sequences)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text[:30000]):
            w = m.group(1)
            if len(w.split()) >= 2 and w not in keywords:
                keywords[w] = "PERSON"

    return [(k, v) for k, v in keywords.items() if len(k) > 2]


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7 — Parties & Metadata Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_parties(text: str) -> dict:
    info = {}
    m = re.search(r"(Criminal Appeal|Civil Appeal|Writ Petition|SLP)[^\n]{0,80}(\d{4})", text, re.IGNORECASE)
    if m:
        info["Case Number"] = m.group(0).strip()
    m = re.search(r"([A-Z][^\n]{5,60})\s*\.{3,}APPELLANT", text)
    if m:
        info["Appellant"] = m.group(1).strip()
    m = re.search(r"([A-Z][^\n]{5,60})\s*\.{3,}RESPONDENT", text)
    if m:
        info["Respondent"] = m.group(1).strip()
    m = re.search(r"(IN THE (?:SUPREME COURT|HIGH COURT)[^\n]{0,60})", text, re.IGNORECASE)
    if m:
        info["Court"] = m.group(1).strip()
    m = re.search(r"DATE OF JUDGMENT[:\s]+([^\n]{5,40})", text, re.IGNORECASE)
    if m:
        info["Date of Judgment"] = m.group(1).strip()
    judges = re.findall(r"HON'BLE (?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+([A-Z][A-Z\.\s]+)", text)
    if judges:
        info["Bench"] = " & ".join(j.strip() for j in judges)
    return info
