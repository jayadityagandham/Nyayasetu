# NyayaSetu v2 — Comprehensive Technical Documentation

## Executive Summary
**NyayaSetu v2** is a sophisticated, AI-powered legal education and analysis platform specifically designed for the Indian legal context (IPC and the newer BNS 2023). It processes raw legal documents (judgments, FIRs) and provides structured summaries, legal explanations, timelines, and multilingual support.

Originally built as a monolithic Streamlit application (`app.py`), the project has been successfully migrated to a modern, decoupled architecture featuring a **Next.js frontend** and a **FastAPI backend**.

## 1. Migration from Streamlit to Next.js + FastAPI
The project began as a single-file Streamlit app which handled UI, file processing, and model inference simultaneously. To improve scalability, UI aesthetics, and separation of concerns, it was refactored into:
- **Backend (`api.py`)**: A FastAPI server that exposes REST endpoints (`/api/upload`, `/api/qa`, `/api/search`) which encapsulate the core Python logic and HuggingFace integrations.
- **Frontend (`frontend/`)**: A Next.js 16 (React 19) application styled with modern CSS (e.g., glassmorphism, responsive layouts) for a highly polished, premium user experience.

## 2. Technology Stack
### Frontend
- **Framework**: Next.js (TypeScript)
- **UI/Styling**: Vanilla CSS with modern aesthetics, responsive design, and animations.
- **State Management**: React Hooks (useState, useEffect)

### Backend
- **Framework**: FastAPI (Python)
- **Document Parsing**: `pdfplumber`, regex.
- **Search & Retrieval**: `faiss-cpu` (Dense Search), `rank_bm25` (Sparse Search).
- **Text Processing**: `gTTS` (Text-to-Speech), `deep-translator` (Translation fallback), `textstat` (Readability), `rouge-score` (Evaluation).

### AI & ML Models (via HuggingFace Inference API)
- **Summarization**: `nsi319/legal-pegasus` (with LED fallback).
- **Rhetorical Role Classification**: `law-ai/InLegalBERT` (with `facebook/bart-large-mnli` fallback).
- **Named Entity Recognition (NER)**: `law-ai/InLegalBERT` (with `dslim/bert-base-NER` fallback).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Text Generation (LLM)**: `meta-llama/Meta-Llama-3-8B-Instruct` (with `mistralai/Mistral-7B-Instruct-v0.3` fallback).
- **Translation**: `ai4bharat/indictrans2-en-indic-1B` (with `Helsinki-NLP/opus-mt-*` fallback).
- **Speech-to-Text**: `openai/whisper-large-v3`.

## 3. Core Logic & Feature Breakdown

### Feature 1: Rhetorical Role Segmentation & Summarization
- **Logic**: Sentences are segmented and classified into 7 rhetorical roles: FACTS, ISSUE, ARGUMENT, STATUTE, PRECEDENT, RATIO, and DECISION.
- **Implementation**: `hf_inlegalbert_classify` performs zero-shot or similarity-based classification. `hf_summarize` (PEGASUS) chunks the text and summarizes each role independently.

### Feature 2 & 12: Legal Section & Entity Extraction
- **Logic**: Uses a combination of regex patterns (mapping IPC to BNS sections) and InLegalBERT NER to extract Persons, Organizations, Locations, and specific Legal Terms.
- **Implementation**: `extract_legal_sections` regex maps legacy IPC to BNS 2023 equivalents. `hf_ner` extracts entities.

### Feature 3 & 4: RAG-Grounded Explanations & Plain Language
- **Logic**: Generates explanations of *why* a law was applied, adjusting the readability level (Basic, Student, Advanced) based on user preference.
- **Implementation**: Uses the `meta-llama/Meta-Llama-3-8B-Instruct` model to generate context-aware, structured explanations.

### Feature 5 & 11: Hybrid RAG & Similar Case Retrieval (ILDC)
- **Logic**: Retrieves similar cases from a 35k case ILDC (Indian Legal Documents Corpus) index.
- **Implementation**: Uses a Reciprocal Rank Fusion (RRF) of Sparse Search (BM25) and Dense Search (FAISS + Sentence-BERT embeddings).

### Feature 6: Legal Learning QA
- **Logic**: A conversational interface allowing users to ask free-form legal questions.
- **Implementation**: Embeds the query, searches the knowledge base/statutes using the Hybrid RAG pipeline, and uses LLaMA-3 to generate the answer strictly grounded in the retrieved statutes. Also generates relevant follow-up questions.

### Feature 7: Readability & Role-based Prompting
- **Logic**: Output can be tuned to "Basic" (citizens), "Student" (law students), or "Advanced" (practitioners).
- **Implementation**: Alters the `system_prompt` injected into the LLaMA-3 text generation calls.

### Feature 8: Multilingual Translation
- **Logic**: Translates summaries and answers into Hindi and Telugu.
- **Implementation**: Uses IndicTrans2 via HuggingFace with `deep-translator` as a local fallback.

### Feature 10: Case Timeline Extraction
- **Logic**: Extracts dates from the document and categorizes them into timeline events (e.g., Incident, FIR, Trial, Supreme Court).
- **Implementation**: Uses regex for date extraction and keyword matching for event categorization.

### Feature 13 & 14: Voice Input & Text-to-Speech
- **Logic**: Allows voice querying and reads out text responses.
- **Implementation**: Browser's Web Speech API for client-side STT. `openai/whisper-large-v3` for uploaded audio. `gTTS` for Text-to-Speech output.

## 4. Evaluation Metrics
The system incorporates automated evaluation to ensure quality:
- **ROUGE Scores**: Computes ROUGE-1, ROUGE-2, and ROUGE-L for abstractive summaries compared against the source segments.
- **Readability**: Calculates Flesch-Kincaid Grade and Reading Ease.
- **RAGAS Proxy**: Evaluates Faithfulness, Answer Relevance, and Context Precision of the RAG responses via overlap metrics.
- **Section Extraction**: Measures mapping coverage of IPC to BNS.

## 5. API Endpoints (`api.py`)
- `POST /api/upload`: Accepts a document (PDF/TXT), parses text, extracts roles, generates summaries, translates text, and returns a comprehensive JSON payload of the document metadata and analysis.
- `POST /api/qa`: Accepts a legal query and returns an LLM-generated answer strictly grounded in the RAG pipeline, translated if requested, along with follow-up questions.
- `POST /api/search`: Accepts case descriptions or parameters and queries the ILDC index using Hybrid RRF (BM25+FAISS) to return similar past judgments.
