"""
build_ildc_index.py — Offline ILDC Dataset Ingestion & Index Building
=====================================================================

Run this ONCE to process your 35k ILDC dataset and build pre-saved indices.
Streamlit then loads these pre-built files at startup (fast).

ILDC Dataset Format (auto-detected):
  CSV columns : case_id | text (or description) | label | annotations (optional)
                name | court | date_of_decision | split
  JSON/JSONL  : same fields as keys
  Hugging Face: loaded directly via datasets library

Usage:
  py -3 scripts/build_ildc_index.py --dataset path/to/ildc.csv
  py -3 scripts/build_ildc_index.py --dataset path/to/ildc/ --format dir
  py -3 scripts/build_ildc_index.py --hf_dataset anuragiiser/ILDC_expert

Output (saved to data/indices/):
  ildc_metadata.json     — list of case dicts (case_id, name, label, preview, sections...)
  ildc_bm25.pkl          — BM25Okapi index (pickle)
  ildc_faiss.index       — FAISS IndexFlatIP (binary)
  ildc_texts.pkl         — list of search strings aligned to FAISS rows
  build_stats.json       — ingestion statistics
"""

import os
import re
import sys
import json
import pickle
import argparse
import time
from pathlib import Path

import numpy as np

# ── Dependency checks ──────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss-cpu not installed. Run: pip install faiss-cpu")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("❌ rank_bm25 not installed. Run: pip install rank_bm25")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data/indices")
METADATA_FILE   = DATA_DIR / "ildc_metadata.json"
BM25_FILE       = DATA_DIR / "ildc_bm25.pkl"
FAISS_FILE      = DATA_DIR / "ildc_faiss.index"
TEXTS_FILE      = DATA_DIR / "ildc_texts.pkl"
STATS_FILE      = DATA_DIR / "build_stats.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Feature 5, 11
BATCH_SIZE      = 64          # embedding batch size
TEXT_PREVIEW_LEN = 600        # chars stored for UI display
MAX_DOCS        = None        # set to e.g. 1000 for a test run

# IPC section extractor (reused from document_processing)
IPC_BNS_MAP = {
    "120B":"61(2)","302":"103","304":"105","304A":"106","307":"109",
    "323":"115","354":"74","363":"137","376":"64","378":"303",
    "379":"303(2)","392":"309","395":"310","420":"318","498A":"85",
}

SECTION_PATTERN = re.compile(
    r"\b(?:Section|S\.)\s*(\d+[A-Z]?)\s*(?:of\s+)?(?:the\s+)?(?:IPC|Indian Penal Code|BNS)\b",
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_ildc_csv(path: str) -> list[dict]:
    """Load ILDC from a single CSV file."""
    if not PANDAS_AVAILABLE:
        print("❌ pandas required for CSV loading. Run: pip install pandas")
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    print(f"   Loaded {len(df)} rows from {path}")
    print(f"   Columns: {list(df.columns)}")
    return df.to_dict(orient="records")


def load_ildc_jsonl(path: str) -> list[dict]:
    """Load ILDC from JSONL (one JSON object per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"   Loaded {len(records)} records from {path}")
    return records


def load_ildc_json(path: str) -> list[dict]:
    """Load ILDC from a JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Handle dict-of-dicts keyed by case_id
    if isinstance(data, dict):
        return [{"case_id": k, **v} for k, v in data.items()]
    return []


def load_ildc_dir(path: str) -> list[dict]:
    """Load ILDC from a directory of .txt files (original ILDC format)."""
    records = []
    base = Path(path)
    # Look for label files (common ILDC structure)
    label_file = base / "labels.json"
    labels = {}
    if label_file.exists():
        with open(label_file) as f:
            labels = json.load(f)

    for txt_file in sorted(base.glob("**/*.txt")):
        case_id = txt_file.stem
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        records.append({
            "case_id": case_id,
            "text": text,
            "label": labels.get(case_id, labels.get(str(txt_file.relative_to(base)), -1)),
            "name": case_id.replace("_", " ").title(),
        })
    print(f"   Loaded {len(records)} .txt files from {path}")
    return records


def load_ildc_hf(dataset_name: str) -> list[dict]:
    """Load directly from HuggingFace datasets hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets library not installed. Run: pip install datasets")
        sys.exit(1)
    print(f"   Downloading from HuggingFace: {dataset_name}")
    ds = load_dataset(dataset_name)
    # Combine all splits
    records = []
    for split_name, split_data in ds.items():
        for row in split_data:
            row["_split"] = split_name
            records.append(dict(row))
    print(f"   Loaded {len(records)} records across all splits")
    return records


def detect_and_load(dataset_path: str, fmt: str = "auto") -> list[dict]:
    """Auto-detect file format and load accordingly."""
    if fmt == "hf":
        return load_ildc_hf(dataset_path)
    if fmt == "dir" or (fmt == "auto" and os.path.isdir(dataset_path)):
        return load_ildc_dir(dataset_path)
    ext = Path(dataset_path).suffix.lower()
    if fmt == "csv" or ext == ".csv":
        return load_ildc_csv(dataset_path)
    if fmt == "jsonl" or ext in (".jsonl", ".ndjson"):
        return load_ildc_jsonl(dataset_path)
    if fmt == "json" or ext == ".json":
        return load_ildc_json(dataset_path)
    # Try CSV as default
    return load_ildc_csv(dataset_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Normalize records to canonical schema
# ═══════════════════════════════════════════════════════════════════════════════

def _get_field(record: dict, *candidates, default="") -> str:
    """Try multiple field name candidates and return first non-empty."""
    for c in candidates:
        v = record.get(c, "")
        if v and str(v).strip() not in ("", "nan", "None"):
            return str(v).strip()
    return default


def _extract_sections(text: str) -> list[str]:
    """Extract IPC section numbers from text."""
    found = []
    for m in SECTION_PATTERN.finditer(text[:5000]):
        sec = m.group(1).upper()
        label = f"Section {sec} IPC"
        if label not in found:
            found.append(label)
    return found


def _extract_annotations(raw_annotations) -> list[str]:
    """
    Normalise the annotations field.
    ILDC annotations can be:
      - A Python list literal string: "['sent1', 'sent2']"
      - Already a list
      - A JSON string
      - A pipe-separated string
    """
    if not raw_annotations or str(raw_annotations).strip() in ("", "nan", "[]"):
        return []
    if isinstance(raw_annotations, list):
        return [str(a).strip() for a in raw_annotations if str(a).strip()]
    s = str(raw_annotations).strip()
    # Try JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(a).strip() for a in parsed]
    except Exception:
        pass
    # Try Python literal eval
    try:
        import ast
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(a).strip() for a in parsed]
    except Exception:
        pass
    # Pipe-separated
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    return [s[:500]]


def normalize_record(record: dict, idx: int) -> dict | None:
    """
    Convert a raw ILDC row into canonical NyayaSetu case format.
    Returns None if the record is invalid/empty.
    """
    # ── Text ──
    text = _get_field(record, "text", "description", "Case Description", "judgment_text", "content", "doc")
    if not text or len(text.split()) < 20:
        return None

    # ── Case ID ──
    case_id = _get_field(record, "case_id", "Case ID", "doc_id", "id", "case_no", default=f"ILDC_{idx:05d}")

    # ── Name ──
    name = _get_field(record, "name", "case_name", "title", default=case_id)

    # ── Label ──
    raw_label = _get_field(record, "label", "decision", "outcome", "Official Decision", default="-1")
    try:
        label_int = int(float(str(raw_label)))
    except Exception:
        raw_lower = str(raw_label).lower()
        label_int = 1 if any(w in raw_lower for w in ("accept", "allow", "granted", "1")) else 0
    label_text = "Accepted / Allowed" if label_int == 1 else ("Dismissed / Rejected" if label_int == 0 else "Unknown")

    # ── Court & Date ──
    court = _get_field(record, "court", "court_name", default="Supreme Court of India")
    date  = _get_field(record, "date", "date_of_decision", "judgment_date", default="")

    # ── Annotations (explanation sentences) ──
    raw_ann = record.get("annotations", record.get("explanation", record.get("rationale", record.get("Official Reasoning", ""))))
    annotations = _extract_annotations(raw_ann)

    # ── Sections cited ──
    sections = _extract_sections(text)

    # ── Split ──
    split = _get_field(record, "split", "_split", default="train")

    # ── Preview ──
    preview = text[:TEXT_PREVIEW_LEN].replace("\n", " ")

    # ── Search text (used for BM25 + embedding) ──
    ann_text = " ".join(annotations[:3])
    search_text = f"{name} {court} {label_text} {' '.join(sections)} {ann_text} {text[:1000]}"

    return {
        "case_id":      case_id,
        "name":         name,
        "court":        court,
        "date":         date,
        "label":        label_int,
        "label_text":   label_text,
        "split":        split,
        "sections_cited": sections,
        "annotations":  annotations[:5],         # top 5 annotated explanation sentences
        "preview":      preview,
        "text_length":  len(text.split()),
        "search_text":  search_text,
        # Keep full text only for RAG retrieval context
        "context":      text[:2000],             # first 2000 words for LLM context
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Build indices
# ═══════════════════════════════════════════════════════════════════════════════

def build_bm25(search_texts: list[str]) -> BM25Okapi:
    tokenized = [t.lower().split() for t in search_texts]
    return BM25Okapi(tokenized)


def build_faiss(search_texts: list[str], model_name: str = EMBEDDING_MODEL):
    """Build a FAISS IndexFlatIP using local sentence-transformers."""
    print(f"\nBuilding FAISS index with {model_name}...")
    print(f"   This is a ONE-TIME operation for {len(search_texts):,} documents.")
    print(f"   Using CPU — estimated time: ~{max(1, len(search_texts)//2000)} min(s)\n")

    model = SentenceTransformer(model_name)

    all_embeddings = []
    total_batches = (len(search_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_i in range(total_batches):
        start = batch_i * BATCH_SIZE
        batch = search_texts[start:start + BATCH_SIZE]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embs)

        if (batch_i + 1) % 50 == 0 or (batch_i + 1) == total_batches:
            done = min(start + BATCH_SIZE, len(search_texts))
            pct = done / len(search_texts) * 100
            print(f"   [{done:>6,}/{len(search_texts):,}] {pct:.1f}% complete", flush=True)

    arr = np.vstack(all_embeddings).astype("float32")
    # Already normalized by normalize_embeddings=True
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)
    print(f"   [DONE] FAISS index built: {index.ntotal:,} vectors, dim={dim}")
    return index


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Save to disk
# ═══════════════════════════════════════════════════════════════════════════════

def save_all(metadata: list[dict], bm25, faiss_index, search_texts: list[str], stats: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving indices to disk...")

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   [OK] {METADATA_FILE}  ({len(metadata):,} records, {METADATA_FILE.stat().st_size/1e6:.1f} MB)")

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"   [OK] {BM25_FILE}  ({BM25_FILE.stat().st_size/1e6:.1f} MB)")

    if faiss_index is not None:
        faiss.write_index(faiss_index, str(FAISS_FILE))
        print(f"   [OK] {FAISS_FILE}  ({FAISS_FILE.stat().st_size/1e6:.1f} MB)")

    with open(TEXTS_FILE, "wb") as f:
        pickle.dump(search_texts, f)
    print(f"   [OK] {TEXTS_FILE}")

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   [OK] {STATS_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build ILDC search indices for NyayaSetu v2")
    parser.add_argument("--dataset",  required=False, help="Path to ILDC dataset file or directory")
    parser.add_argument("--hf_dataset", required=False, help="HuggingFace dataset name (e.g. anuragiiser/ILDC_expert)")
    parser.add_argument("--format",   default="auto", choices=["auto","csv","json","jsonl","dir","hf"])
    parser.add_argument("--max_docs", type=int, default=None, help="Limit number of docs (for testing)")
    parser.add_argument("--no_faiss", action="store_true", help="Skip FAISS (BM25 only)")
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    args = parser.parse_args()

    if not args.dataset and not args.hf_dataset:
        parser.error("Provide --dataset <path> or --hf_dataset <name>")

    t0 = time.time()
    print("=" * 60)
    print("  NyayaSetu v2 — ILDC Index Builder")
    print("=" * 60)

    # ── Load ──
    print("\nStep 1/4 - Loading dataset...")
    if args.hf_dataset:
        raw_records = load_ildc_hf(args.hf_dataset)
    else:
        raw_records = detect_and_load(args.dataset, args.format)

    if args.max_docs:
        raw_records = raw_records[:args.max_docs]
        print(f"   [WARN] Capped at {args.max_docs} docs (--max_docs)")

    # ── Normalize ──
    print(f"\nStep 2/4 - Normalizing {len(raw_records):,} records...")
    metadata, search_texts = [], []
    skipped = 0

    for i, raw in enumerate(raw_records):
        norm = normalize_record(raw, i)
        if norm is None:
            skipped += 1
            continue
        metadata.append(norm)
        search_texts.append(norm["search_text"])

    print(f"   [OK] {len(metadata):,} valid records  |  Skipped {skipped} (too short)")

    label_counts = {}
    for m in metadata:
        label_counts[m["label_text"]] = label_counts.get(m["label_text"], 0) + 1
    print(f"   Label distribution: {label_counts}")

    # ── BM25 ──
    print(f"\nStep 3/4 - Building BM25 index...")
    bm25 = build_bm25(search_texts)
    print(f"   [OK] BM25 built on {len(search_texts):,} documents")

    # ── FAISS ──
    faiss_index = None
    if not args.no_faiss:
        if not ST_AVAILABLE:
            print("\n[WARN] sentence-transformers not available - skipping FAISS.")
            print("   Install with: pip install sentence-transformers")
            print("   Then re-run without --no_faiss")
        elif not FAISS_AVAILABLE:
            print("\n[WARN] faiss-cpu not available - skipping FAISS.")
        else:
            faiss_index = build_faiss(search_texts, args.embedding_model)
    else:
        print("\nSkipping FAISS (--no_faiss flag)")

    # ── Save ──
    print(f"\nStep 4/4 - Saving to {DATA_DIR}/...")
    elapsed = time.time() - t0
    stats = {
        "total_raw":     len(raw_records),
        "total_indexed": len(metadata),
        "skipped":       skipped,
        "label_counts":  label_counts,
        "embedding_model": args.embedding_model if not args.no_faiss else "none",
        "faiss_built":   faiss_index is not None,
        "build_time_sec": round(elapsed, 1),
        "dataset_source": args.hf_dataset or args.dataset,
    }
    save_all(metadata, bm25, faiss_index, search_texts, stats)

    print(f"\n{'='*60}")
    print(f"  [DONE] in {elapsed/60:.1f} min")
    print(f"  {len(metadata):,} cases indexed")
    print(f"  Output: {DATA_DIR.resolve()}/")
    print(f"  -> Now restart NyayaSetu v2 - indices load automatically.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
