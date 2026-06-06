from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io

os.makedirs("data", exist_ok=True)

from modules.document_processing import (
    extract_text_from_file, clean_text, rhetorical_role_segmentation,
    extract_timeline, extract_keywords, extract_parties, extract_legal_sections
)
from modules.summarization import summarize_roles, apply_readability_rewrite, generate_story_summary, generate_glossary
from modules.rag_pipeline import (
    load_knowledge_base, initialize_indices, retrieve_context,
    generate_explanation, answer_legal_question,
    find_similar_cases, retrieve_ildc_cases,
    load_ildc_metadata, load_ildc_indices,
    ildc_index_exists, get_ildc_stats,
    generate_followup_questions, generate_summary_followups
)
from modules.audio_translation import translate_text
from modules.evaluation import (
    compute_rouge, flesch_kincaid_grade, reading_ease, ease_label,
    section_extraction_stats, proxy_ragas, summary_stats
)

app = FastAPI(title="NyayaSetu API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

kb = load_knowledge_base()
kb_tuple = tuple(d.get("section","") for d in kb)
faiss_idx, bm25_idx, texts = initialize_indices(tuple(kb), "")

class QARequest(BaseModel):
    query: str
    hf_api_key: str = ""
    readability_level: str = "Student"
    target_language: str = "English"

class SearchRequest(BaseModel):
    query: str
    hf_api_key: str = ""
    top_k: int = 5
    label_filter: str = "all"
    section_filter: str = ""
    search_mode: str = "Hybrid"

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    hf_api_key: str = Form(""),
    use_hf_roles: bool = Form(False),
    readability_level: str = Form("Student"),
    target_language: str = Form("English")
):
    try:
        content = await file.read()
        
        class DummyFile:
            name = file.filename
            def getvalue(self): return content
            def read(self): return content
            
        dummy = DummyFile()
        
        raw = extract_text_from_file(dummy)
        cleaned = clean_text(raw)
        parties = extract_parties(cleaned)
        segments = rhetorical_role_segmentation(cleaned, hf_api_key, use_hf_roles)
        summaries = summarize_roles(segments, hf_api_key)
        sections = extract_legal_sections(cleaned)
        timeline = extract_timeline(cleaned)
        keywords = extract_keywords(cleaned, hf_api_key)
        
        # Apply translation and readability
        processed_summaries = {}
        for role, summary in summaries.items():
            if not summary: continue
            display_text = summary
            if readability_level != "Student":
                display_text = apply_readability_rewrite(hf_api_key, display_text, readability_level) or display_text
            display_text = translate_text(display_text, target_language, hf_api_key)
            processed_summaries[role] = display_text
        
        return {
            "success": True,
            "parties": parties,
            "summaries": processed_summaries,
            "segments": segments,
            "sections": sections,
            "timeline": timeline,
            "keywords": keywords,
            "cleaned": cleaned
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/qa")
async def qa(req: QARequest):
    try:
        answer, ragas = answer_legal_question(req.query, req.readability_level, kb, faiss_idx, bm25_idx, texts, req.hf_api_key)
        translated = translate_text(answer, req.target_language, req.hf_api_key)
        matched, _ = retrieve_context(req.query, kb, faiss_idx, bm25_idx, texts, req.hf_api_key)
        followups = generate_followup_questions(matched, req.query)
        return {
            "success": True,
            "answer": translated,
            "ragas": ragas,
            "followups": followups
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/search")
async def search(req: SearchRequest):
    if not ildc_index_exists():
        return {"success": False, "error": "ILDC index not built yet."}
    try:
        ildc_meta = load_ildc_metadata()
        ildc_faiss, ildc_bm25, ildc_texts = load_ildc_indices()
        
        use_faiss = "FAISS" in req.search_mode and ildc_faiss is not None and req.hf_api_key
        use_bm25  = "BM25"  in req.search_mode or not use_faiss
        
        if req.search_mode.startswith("FAISS"):
            results = find_similar_cases(req.query, ildc_meta, ildc_faiss, ildc_texts, req.hf_api_key, top_k=req.top_k)
        else:
            results = retrieve_ildc_cases(
                req.query, ildc_meta,
                ildc_faiss if use_faiss else None,
                ildc_bm25  if use_bm25  else None,
                ildc_texts,
                req.hf_api_key, top_k=req.top_k,
                label_filter=req.label_filter,
                section_filter=req.section_filter,
            )
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
