import streamlit as st, os, pandas as pd
os.makedirs("data", exist_ok=True)

from modules.document_processing import (
    extract_text_from_file, clean_text, rhetorical_role_segmentation,
    extract_timeline, extract_keywords, extract_parties, ROLE_DISPLAY, extract_legal_sections
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
from modules.audio_translation import translate_text, text_to_speech, render_voice_input, whisper_upload_transcribe
from modules.evaluation import (
    compute_rouge, flesch_kincaid_grade, reading_ease, ease_label,
    section_extraction_stats, proxy_ragas, summary_stats
)

st.set_page_config(page_title="NyayaSetu v2", layout="wide", page_icon="⚖️", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-title{font-size:2.2rem;font-weight:700;background:linear-gradient(135deg,#4A90D9,#7B68EE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.subtitle{color:#888;font-size:1rem;margin-bottom:1.2rem;}
.role-card{border-left:4px solid;padding:1rem 1.2rem;border-radius:0 8px 8px 0;margin-bottom:1rem;background:rgba(255,255,255,0.03);}
.role-title{font-size:1.05rem;font-weight:600;margin-bottom:0.4rem;}
.role-text{font-size:0.92rem;line-height:1.7;color:#ccc;}
.section-card{border:1px solid rgba(123,104,238,0.3);border-radius:10px;padding:1rem;margin-bottom:0.8rem;background:rgba(123,104,238,0.05);}
.meta-card{background:rgba(74,144,217,0.08);border:1px solid rgba(74,144,217,0.2);border-radius:10px;padding:1rem;margin-bottom:1rem;}
.kw-badge{display:inline-block;background:rgba(74,144,217,0.15);color:#7ab8f5;border:1px solid rgba(74,144,217,0.3);border-radius:6px;padding:3px 9px;font-size:0.78rem;margin:3px;}
.kw-legal{background:rgba(255,165,0,0.12);color:#ffc87a;border-color:rgba(255,165,0,0.3);}
.kw-person{background:rgba(76,175,80,0.12);color:#88d48a;border-color:rgba(76,175,80,0.3);}
.bns-badge{display:inline-block;background:rgba(123,104,238,0.2);color:#a992ff;border-radius:20px;padding:2px 10px;font-size:0.8rem;font-weight:600;margin-left:8px;}
.timeline-date{font-weight:700;color:#4A90D9;font-size:1rem;}
.timeline-event{font-weight:600;color:#e0e0e0;font-size:0.95rem;}
.timeline-snip{font-size:0.82rem;color:#999;font-style:italic;margin-top:4px;}
.hf-badge{background:rgba(255,212,0,0.15);color:#ffd400;border:1px solid rgba(255,212,0,0.3);border-radius:20px;padding:3px 12px;font-size:0.8rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ NyayaSetu v2")
    st.caption("HuggingFace-Powered Legal Education System")
    st.divider()
    page = st.radio("📌 Navigate", ["Case Analysis", "Legal Learning QA", "Similar Case Retrieval"])
    st.divider()
    st.markdown("### 🔑 HuggingFace API Key")
    hf_api_key = st.text_input("HF Token", type="password", placeholder="hf_xxxxxxxxxx",
        help="Get free token at huggingface.co/settings/tokens")
    if hf_api_key:
        st.success("✅ API key set — full LLM pipeline active")
        st.markdown('<span class="hf-badge">🤗 HF Inference API</span>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No key — pure RAG fallback mode")
    st.divider()
    st.markdown("### ⚙️ Settings")
    readability_level = st.selectbox("Readability Level", ["Basic", "Student", "Advanced"],
        help="Basic=citizen | Student=law student | Advanced=practitioner")
    target_language = st.selectbox("Output Language", ["English", "Hindi", "Telugu"])
    use_hf_roles = st.checkbox("Use InLegalBERT for Role Labelling", value=False,
        help="InLegalBERT (law-ai/InLegalBERT) — BERT pre-trained on Indian legal corpus for Facts/Issues/Decision segmentation. Slower but domain-accurate.")
    st.divider()
    st.info("**Models (via HF API — as per architecture):**\n"
            "- **Roles**: InLegalBERT (law-ai/InLegalBERT)\n"
            "- **Summarization**: PEGASUS (nsi319/legal-pegasus) + LED fallback\n"
            "- **NER**: InLegalBERT (legal entities)\n"
            "- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)\n"
            "- **LLM**: LLaMA-3-8B-Instruct (meta-llama)\n"
            "  *(Mistral-7B fallback if licence not accepted)*\n"
            "- **Translation**: IndicTrans2 (ai4bharat 1B)\n"
            "  *(Helsinki-NLP opus-mt fallback)*\n"
            "- **STT**: Whisper large-v3 (openai)\n"
            "- **TTS**: gTTS + IndicTTS note\n"
            "- **Retrieval**: BM25 + FAISS (RRF)")

# Load shared resources
@st.cache_data
def get_kb():
    return load_knowledge_base()

kb = get_kb()
kb_tuple = tuple(d.get("section","") for d in kb)

@st.cache_resource
def get_indices(_key):
    return initialize_indices(tuple(kb), _key)

faiss_idx, bm25_idx, texts = get_indices(hf_api_key)

ROLE_COLORS = {"FACTS":"#1a7abf","ISSUE":"#9c27b0","ARGUMENT":"#2e7d32",
               "STATUTE":"#e65100","PRECEDENT":"#5c35cc","RATIO":"#00838f","DECISION":"#c62828"}

EVENT_ICONS = {
    "Incident / Offence":("🔴","#c62828"), "FIR / Complaint Filed":("📋","#1565c0"),
    "Legal Notice / Pre-Trial":("📜","#e65100"), "Trial / Sessions Court":("🏛️","#2e7d32"),
    "High Court":("⚖️","#5c35cc"), "Supreme Court / Final Judgment":("🔔","#c79000"),
    "General Reference":("📅","#555"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CASE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Case Analysis":
    st.markdown('<div class="main-title">NyayaSetu — Legal Case Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload an Indian Supreme Court judgment for structured summaries, legal explanations, timeline, and multilingual output.</div>', unsafe_allow_html=True)

    col_up, col_voice = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader("📂 Upload Judgment (PDF or TXT)", type=["pdf","txt"])
    with col_voice:
        st.markdown("**🎤 Voice Input (Feature 13)**")
        render_voice_input()

    if uploaded_file and st.button("⚡ Process Document", type="primary"):
        with st.spinner("Running full pipeline via HuggingFace Inference API…"):
            raw = extract_text_from_file(uploaded_file)
            cleaned = clean_text(raw)
            parties = extract_parties(cleaned)
            segments = rhetorical_role_segmentation(cleaned, hf_api_key, use_hf_roles)
            summaries = summarize_roles(segments, hf_api_key)
            sections = extract_legal_sections(cleaned)
            timeline = extract_timeline(cleaned)
            keywords = extract_keywords(cleaned, hf_api_key)
            st.session_state.update({
                'parties':parties,'summaries':summaries,'segments':segments,
                'sections':sections,'timeline':timeline,'keywords':keywords,
                'cleaned':cleaned,'processed':True
            })
        st.success("✅ Document processed!")

    if st.session_state.get('processed'):
        parties   = st.session_state['parties']
        summaries = st.session_state['summaries']
        segments  = st.session_state['segments']
        sections  = st.session_state['sections']
        timeline  = st.session_state['timeline']
        keywords  = st.session_state['keywords']
        cleaned   = st.session_state['cleaned']

        # Metadata bar
        if parties:
            st.markdown('<div class="meta-card">', unsafe_allow_html=True)
            cols = st.columns(3)
            for i,(k,v) in enumerate(parties.items()):
                with cols[i%3]:
                    st.markdown(f"**{k}:** {v}")
            st.markdown('</div>', unsafe_allow_html=True)

        tabs = st.tabs(["📋 Structured Summary", "📖 Story Summary", "⚖️ Legal Sections", "🕐 Timeline", "🔑 Keywords & Glossary", "📊 Evaluation"])

        # ── Tab 1: Structured Summary ───────────────────────────────────────────
        with tabs[0]:
            st.subheader("Structured Rhetorical Role Summary")
            st.caption("Each segment classified by rhetorical role and summarized via BART-large-CNN (HF API)")
            any_shown = False
            for role,(label,_) in ROLE_DISPLAY.items():
                summary = summaries.get(role)
                if not summary: continue
                any_shown = True
                color = ROLE_COLORS.get(role,"#555")
                display_text = summary
                # 1. Apply Readability (Basic/Advanced) first in English
                if readability_level != "Student":
                    display_text = apply_readability_rewrite(hf_api_key, display_text, readability_level) or display_text
                
                # 2. Then Translate to Target Language (e.g. Telugu)
                display_text = translate_text(display_text, target_language, hf_api_key)
                display_label = translate_text(label, target_language, hf_api_key) if target_language != "English" else label
                
                st.markdown(
                    f'<div class="role-card" style="border-left:5px solid {color}; padding-left:1.5rem; background:rgba(255,255,255,0.02);">'
                    f'<div class="role-title" style="color:{color}; font-size:1.15rem;">{display_label}</div>'
                    f'<div class="role-text" style="font-size:1rem;">{display_text}</div></div>',
                    unsafe_allow_html=True)
                with st.expander(f"📄 Raw segment — {display_label}"):
                    st.write(segments.get(role,"")[:2000])
            if not any_shown:
                st.info("No role segments extracted.")
            if any_shown:
                st.divider()
                st.markdown("💡 **Explore Further:**")
                fqs = generate_summary_followups(sections, segments)
                fq_cols = st.columns(min(3, max(len(fqs),1)))
                for i,fq in enumerate(fqs):
                    with fq_cols[i%len(fq_cols)]:
                        if st.button(fq, key=f"sfq_{i}", use_container_width=True):
                            st.session_state['sum_active_q'] = fq
                c1,c2 = st.columns([5,1])
                with c1:
                    cq = st.text_input("Ask about this judgment…", key="sum_cq", label_visibility="collapsed")
                with c2:
                    if st.button("➤ Ask", key="sum_ask"):
                        if cq.strip(): st.session_state['sum_active_q'] = cq.strip()
                if st.session_state.get('sum_active_q'):
                    q = st.session_state['sum_active_q']
                    with st.spinner(f"Answering: *{q}*…"):
                        ans, ragas = answer_legal_question(q, readability_level, kb, faiss_idx, bm25_idx, texts, hf_api_key)
                    st.markdown(f"---\n#### 💬 *{q}*\n{ans}")
                    bt,bc = ragas.get('badge',('N/A','grey'))
                    st.markdown(f"**Confidence:** <span style='color:{bc}'>{bt}</span>", unsafe_allow_html=True)
                    if st.button("❌ Clear", key="sum_clr"): st.session_state['sum_active_q']=None; st.rerun()

        # ── Tab 2: Story Summary ───────────────────────────────────────────────
        with tabs[1]:
            st.subheader("📖 Case Narrative (Story Mode)")
            st.caption(f"A cohesive narrative summary generated at {readability_level} level.")
            
            if st.button("Generate Story Summary", type="primary"):
                with st.spinner("Writing narrative..."):
                    story = generate_story_summary(hf_api_key, cleaned, readability_level)
                    st.session_state["story_summary"] = story
            
            if st.session_state.get("story_summary"):
                story_text = st.session_state["story_summary"]
                story_translated = translate_text(story_text, target_language, hf_api_key)
                
                st.markdown(
                    f'<div style="background:rgba(74,144,217,0.05); border:1px solid rgba(74,144,217,0.2); '
                    f'padding:2rem; border-radius:12px; font-size:1.05rem; line-height:1.8;">{story_translated}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔊 Read Story Aloud"):
                    af = text_to_speech(story_translated, target_language)
                    if af: st.audio(af)

        # ── Tab 3: Legal Sections ───────────────────────────────────────────────
        with tabs[2]:
            st.subheader("Legal Sections & RAG-Grounded Explanations")
            st.caption("BM25 + FAISS hybrid retrieval • Mistral-7B explanation via HF API")
            if not sections:
                st.warning("No IPC/BNS sections detected. Try a judgment with explicit 'Section X IPC' references.")
            else:
                for sec_key, info in sections.items():
                    bns   = info.get('bns_equivalent','N/A')
                    title = info.get('title','Unknown')
                    matched, ctx = retrieve_context(sec_key+" "+title, kb, faiss_idx, bm25_idx, texts, hf_api_key, top_k=3)
                    kb_entry = matched[0] if matched else None
                    penalty = kb_entry.get('penalty','') if kb_entry else ''
                    with st.expander(f"📌 {sec_key}  →  {bns}   |   {title}", expanded=False):
                        st.markdown(
                            f'<div class="section-header" style="font-size:1.2rem;font-weight:600;">{sec_key}'
                            f'<span class="bns-badge">Now: {bns}</span></div>',
                            unsafe_allow_html=True)
                        if penalty: st.markdown(f"⚡ **Penalty:** {penalty}")
                        if kb_entry:
                            st.markdown(f"**📚 Source:** `{kb_entry.get('section','')} IPC / {kb_entry.get('bns_equivalent','')}`")
                        expls = generate_explanation(sec_key, info, kb_entry, ctx, readability_level, hf_api_key)
                        ragas = expls.pop("_ragas",{})
                        used_llm = expls.pop("_used_llm", False)
                        badge_str = "🟢 **Mistral-7B via HF API**" if used_llm else "🔵 **Pure RAG** (no API key)"
                        st.markdown(badge_str)
                        for lbl, body in expls.items():
                            translated_lbl = translate_text(lbl, target_language, hf_api_key) if target_language != "English" else lbl
                            st.markdown(f"#### {translated_lbl}")
                            st.markdown(translate_text(body, target_language, hf_api_key))
                        bt,bc = ragas.get('badge',('N/A','grey'))
                        st.markdown(
                            f"**Reliability:** <span style='color:{bc}'>{bt}</span> &nbsp;|&nbsp; "
                            f"Faithfulness: `{ragas.get('faithfulness','N/A')}` &nbsp;|&nbsp; "
                            f"Relevance: `{ragas.get('answer_relevance','N/A')}`",
                            unsafe_allow_html=True)
                        if st.button("🔊 Audio", key=f"aud_{sec_key}"):
                            body_all = " ".join(v for v in expls.values())
                            af = text_to_speech(translate_text(body_all, target_language, hf_api_key), target_language)
                            if af: st.audio(af)

        # ── Tab 4: Timeline ─────────────────────────────────────────────────────
        with tabs[3]:
            st.subheader("Case Timeline")
            
            # Legend
            with st.expander("📖 Timeline Legend", expanded=True):
                st.markdown("""
                * 🔴 **Incident / Offence:** The date the crime/event occurred.
                * 📋 **FIR / Complaint Filed:** The date the police registered the case.
                * 📜 **Legal Notice / Pre-Trial:** Notices, warrants, or pre-trial events.
                * 🏛️ **Trial / Sessions Court:** Events or judgments at the lower/trial court level.
                * ⚖️ **High Court:** Appeals or proceedings at the High Court.
                * 🔔 **Supreme Court / Final:** The final judgment or SLP hearings.
                * 📅 **General Reference:** Other dates mentioned in the text.
                """)
            st.divider()

            if not timeline:
                st.warning("No proper calendar dates found.")
            else:
                for i,entry in enumerate(timeline):
                    icon,color = EVENT_ICONS.get(entry['event'],("📅","#555"))
                    c1,c2 = st.columns([1,5])
                    with c1:
                        st.markdown(f"<div style='text-align:center;font-size:2rem'>{icon}</div>"
                            f"<div style='text-align:center;color:{color};font-weight:700;font-size:0.85rem'>{entry['date']}</div>",
                            unsafe_allow_html=True)
                    with c2:
                        st.markdown(
                            f"<div class='timeline-event' style='color:{color}'>{icon}&nbsp;{entry['event']}</div>"
                            f"<div class='timeline-snip' style='font-size:0.9rem;'>{entry['snippet']}</div>",
                            unsafe_allow_html=True)
                    if i < len(timeline)-1:
                        st.markdown("<hr style='border:none;border-top:1px dashed rgba(255,255,255,0.1);margin:16px 0'>", unsafe_allow_html=True)

        # ── Tab 5: Keywords & Glossary ───────────────────────────────────────────
        with tabs[4]:
            col_k, col_g = st.columns([1, 1])
            
            with col_k:
                st.subheader("Keywords & Entities")
                TYPE_CSS = {"LEGAL_TERM":"kw-badge kw-legal","PERSON":"kw-badge kw-person","ORG":"kw-badge","GPE":"kw-badge","LAW":"kw-badge kw-legal"}
                TYPE_LABEL = {"LEGAL_TERM":"Legal Terms","PERSON":"Persons","ORG":"Organisations","GPE":"Places","LAW":"Laws"}
                if not keywords:
                    st.info("No keywords found.")
                else:
                    groups = {}
                    for kw,kt in keywords:
                        groups.setdefault(kt,[]).append(kw)
                    for kt,kws in groups.items():
                        st.markdown(f"**{TYPE_LABEL.get(kt,kt)}**")
                        html = " ".join(f'<span class="{TYPE_CSS.get(kt,"kw-badge")}">{kw}</span>' for kw in sorted(set(kws)))
                        st.markdown(html, unsafe_allow_html=True)
                        st.markdown("")
            
            with col_g:
                st.subheader("📖 Legal Glossary")
                st.caption(f"Generate simple definitions for extracted legal terms at {readability_level} level.")
                if st.button("Generate Glossary", type="primary"):
                    with st.spinner("Looking up legal definitions..."):
                        glossary = generate_glossary(hf_api_key, keywords, readability_level)
                        st.session_state["glossary"] = glossary
                
                if st.session_state.get("glossary"):
                    for term, defi in st.session_state["glossary"].items():
                        translated_defi = translate_text(defi, target_language, hf_api_key)
                        st.markdown(f"**{term}**: {translated_defi}")
                        st.markdown("---")

        # ── Tab 6: Evaluation ───────────────────────────────────────────────────
        with tabs[5]:
            st.subheader("📊 Evaluation Scores")
            st.markdown("### 🔍 Section Extraction")
            ss = section_extraction_stats(sections)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Sections Found", ss['sections_found'])
            c2.metric("Mapped to BNS", ss['mapped_to_bns'])
            c3.metric("Unmapped", ss['unmapped'])
            c4.metric("BNS Coverage", f"{ss['coverage_pct']}%")
            st.divider()
            st.markdown("### 📝 ROUGE Scores")
            rouge_rows = []
            for role,summary in summaries.items():
                if not summary: continue
                ref = segments.get(role,"")
                if not ref.strip(): continue
                r = compute_rouge(summary,ref)
                s2 = summary_stats(summary)
                rouge_rows.append({"Role":role,"ROUGE-1":r['ROUGE-1'],"ROUGE-2":r['ROUGE-2'],"ROUGE-L":r['ROUGE-L'],"Words":s2['word_count']})
            if rouge_rows:
                st.dataframe(pd.DataFrame(rouge_rows).set_index("Role"), use_container_width=True)
            st.divider()
            st.markdown("### 📖 Readability")
            read_rows = []
            for role,summary in summaries.items():
                if not summary: continue
                fk = flesch_kincaid_grade(summary); re_ = reading_ease(summary)
                lbl,_ = ease_label(re_)
                read_rows.append({"Role":role,"FK Grade":fk,"Reading Ease":re_,"Audience":lbl})
            if read_rows:
                st.dataframe(pd.DataFrame(read_rows).set_index("Role"), use_container_width=True)
            st.divider()
            st.markdown("### 🎯 RAGAS Proxy")
            ragas_rows = []
            for sec_key,info in sections.items():
                title = info.get('title',sec_key)
                m2,ctx2 = retrieve_context(sec_key+" "+title, kb, faiss_idx, bm25_idx, texts, hf_api_key)
                kb_e = m2[0] if m2 else None
                ex2 = generate_explanation(sec_key,info,kb_e,ctx2,readability_level,hf_api_key)
                ex2.pop("_ragas",None); ex2.pop("_used_llm",None)
                full_exp = " ".join(v for v in ex2.values() if isinstance(v,str))
                pr = proxy_ragas(ctx2, full_exp)
                ragas_rows.append({"Section":sec_key,"Faithfulness":pr['faithfulness'],"Relevance":pr['answer_relevance'],"Precision":pr['context_precision'],"Avg":pr['avg_score']})
            if ragas_rows:
                st.dataframe(pd.DataFrame(ragas_rows).set_index("Section"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LEGAL LEARNING QA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Legal Learning QA":
    st.markdown('<div class="main-title">Legal Learning QA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask any Indian legal question — retrieved from statutes, answered by Mistral-7B via HuggingFace.</div>', unsafe_allow_html=True)

    if 'qa_history' not in st.session_state: st.session_state['qa_history'] = []
    if 'qa_query' not in st.session_state: st.session_state['qa_query'] = ''

    st.markdown("**💡 Try:**")
    examples = ["What is the punishment for murder in India?","What does criminal conspiracy mean?",
                "What is the difference between murder and culpable homicide?","What are the ingredients of theft?"]
    ec = st.columns(2)
    for i,ex in enumerate(examples):
        if ec[i%2].button(ex, key=f"eg_{i}"):
            st.session_state['qa_query'] = ex; st.session_state['auto_submit'] = True; st.rerun()

    st.divider()
    col_q, col_v = st.columns([4,1])
    with col_q:
        query = st.text_area("Your legal question:", value=st.session_state.get('qa_query',''), height=90, key="qa_input")
    with col_v:
        st.markdown("**🎤 Voice**")
        render_voice_input()

    auto = st.session_state.pop('auto_submit', False)
    if st.button("🔍 Get Legal Answer", type="primary") or auto:
        effective_q = (st.session_state.get('qa_query','') if auto else query).strip()
        if effective_q:
            with st.spinner("Searching statutes via BM25+FAISS, generating answer with Mistral-7B…"):
                answer, ragas = answer_legal_question(effective_q, readability_level, kb, faiss_idx, bm25_idx, texts, hf_api_key)
                translated = translate_text(answer, target_language, hf_api_key)
                matched, _ = retrieve_context(effective_q, kb, faiss_idx, bm25_idx, texts, hf_api_key)
                followups = generate_followup_questions(matched, effective_q)
            st.session_state['qa_history'].insert(0, {'query':effective_q,'answer':translated,'ragas':ragas,'followups':followups})
            st.session_state['qa_query'] = ''
        else:
            st.warning("Please enter a question.")

    history = st.session_state.get('qa_history',[])
    if history and st.button("🗑️ Clear History"): st.session_state['qa_history']=[]; st.rerun()

    for h_idx, h in enumerate(history):
        st.markdown(f"---\n#### 💬 Q{len(history)-h_idx}: *{h['query']}*")
        st.markdown(h['answer'])
        bt,bc = h['ragas'].get('badge',('N/A','grey'))
        st.markdown(f"**Confidence:** <span style='color:{bc}'>{bt}</span>", unsafe_allow_html=True)
        with st.expander("🔊 Listen"):
            if st.button("Generate Audio", key=f"aqa_{h_idx}"):
                af = text_to_speech(h['answer'], target_language)
                if af: st.audio(af)
        if h.get('followups'):
            st.markdown("💡 **Suggested follow-ups:**")
            fqc = st.columns(2)
            for fi,fq in enumerate(h['followups']):
                if fqc[fi%2].button(fq, key=f"fq_{h_idx}_{fi}", use_container_width=True):
                    st.session_state['qa_query']=fq; st.session_state['auto_submit']=True; st.rerun()
        cq2 = st.text_input("✏️ Your follow-up:", key=f"cfq_{h_idx}", label_visibility="collapsed", placeholder="Type a follow-up…")
        if st.button("➤ Ask", key=f"cfqs_{h_idx}") and cq2.strip():
            st.session_state['qa_query']=cq2.strip(); st.session_state['auto_submit']=True; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIMILAR CASE RETRIEVAL (Feature 11 — ILDC 35k cases)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="main-title">Similar Case Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Feature 11 — Search 35,000 Indian Supreme Court cases (ILDC dataset) using Sentence-BERT + FAISS cosine similarity + BM25 keyword retrieval.</div>', unsafe_allow_html=True)

    # ── Index status banner ──────────────────────────────────────────────────
    if ildc_index_exists():
        stats = get_ildc_stats()
        total = stats.get("total_indexed", "?")
        faiss_ok = stats.get("faiss_built", False)
        bm25_ok  = stats.get("bm25_built",  False)
        build_t  = stats.get("build_time_sec", "?")
        st.success(
            f"✅ **ILDC Index Loaded** — {total:,} cases indexed  |  "
            f"FAISS: {'✅' if faiss_ok else '❌ (BM25 only)'}  |  "
            f"BM25: {'✅' if bm25_ok else '❌'}  |  "
            f"Built in {build_t}s"
        )
        ildc_meta = load_ildc_metadata()
        ildc_faiss, ildc_bm25, ildc_texts = load_ildc_indices()
    else:
        st.warning(
            "⚠️ **ILDC index not built yet.** Run this command once to index your 35k dataset:\n\n"
            "```\npy -3 scripts/build_ildc_index.py --dataset path/to/your/ildc.csv\n```\n\n"
            "Supported formats: **CSV · JSON · JSONL · directory of .txt files · HuggingFace datasets**"
        )
        with st.expander("📖 Full indexing instructions"):
            st.markdown("""
**Step 1 — Install sentence-transformers** (needed only for the one-time build):
```bash
py -3 -m pip install sentence-transformers datasets
```

**Step 2 — Run the indexer on your ILDC dataset:**
```bash
# CSV file:
py -3 scripts/build_ildc_index.py --dataset C:/path/to/ildc.csv

# JSONL file:
py -3 scripts/build_ildc_index.py --dataset C:/path/to/ildc.jsonl

# Directory of .txt files (original ILDC format):
py -3 scripts/build_ildc_index.py --dataset C:/path/to/ildc/ --format dir

# Directly from HuggingFace hub:
py -3 scripts/build_ildc_index.py --hf_dataset anuragiiser/ILDC_expert

# Test run with first 500 docs:
py -3 scripts/build_ildc_index.py --dataset ildc.csv --max_docs 500
```

**Step 3 — Restart NyayaSetu.** The indices load automatically.

**Expected columns in your CSV:** `case_id`, `text` (or `description`), `label`, `annotations`, `name`, `court`, `date`
            """)
        st.stop()

    # ── Search controls ──────────────────────────────────────────────────────
    st.divider()
    search_col, filter_col = st.columns([3, 1])

    with search_col:
        case_query = st.text_area(
            "🔍 Describe your case or paste a case summary / legal question:",
            height=120,
            placeholder=(
                "e.g. Accused convicted for murder under Section 302 IPC. "
                "High Court upheld conviction. Defence argued private self-defence..."
            )
        )

    with filter_col:
        st.markdown("**Filters**")
        label_filter = st.selectbox(
            "Judgment Outcome",
            ["all", "accepted", "rejected"],
            format_func=lambda x: {"all":"All outcomes","accepted":"✅ Accepted/Allowed","rejected":"❌ Dismissed/Rejected"}[x]
        )
        top_k = st.slider("Results to show", 3, 20, 5)

        # Section filter from found sections
        section_filter = ""
        if st.session_state.get("processed"):
            secs = list(st.session_state.get("sections", {}).keys())
            if secs:
                section_filter = st.selectbox("Filter by cited section", [""] + secs, format_func=lambda x: x or "Any section")

    search_mode = st.radio(
        "Search Mode",
        ["Hybrid (BM25 + FAISS — best)", "BM25 keyword only", "FAISS semantic only"],
        horizontal=True,
        help="Hybrid uses Reciprocal Rank Fusion of both indices (recommended)"
    )

    # Also search from processed case
    if st.session_state.get("processed") and st.session_state.get("summaries"):
        facts_summary = st.session_state["summaries"].get("FACTS", "") or ""
        if facts_summary and st.button("🔗 Find cases similar to the currently uploaded judgment"):
            st.session_state["ildc_query"] = facts_summary[:600]
            st.session_state["ildc_auto"] = True

    if st.session_state.pop("ildc_auto", False):
        case_query = st.session_state.get("ildc_query", case_query)

    if st.button("🔍 Search Similar Cases", type="primary") and case_query.strip():
        with st.spinner(f"Searching {len(ildc_meta):,} ILDC cases…"):

            use_faiss = "FAISS" in search_mode and ildc_faiss is not None and hf_api_key
            use_bm25  = "BM25"  in search_mode or not use_faiss

            if search_mode.startswith("FAISS"):
                results = find_similar_cases(case_query, ildc_meta, ildc_faiss, ildc_texts, hf_api_key, top_k=top_k)
            else:
                results = retrieve_ildc_cases(
                    case_query, ildc_meta,
                    ildc_faiss if use_faiss else None,
                    ildc_bm25  if use_bm25  else None,
                    ildc_texts,
                    hf_api_key, top_k=top_k,
                    label_filter=label_filter,
                    section_filter=section_filter,
                )

        if not results:
            st.warning("No matching cases found. Try broader keywords or remove filters.")
        else:
            st.success(f"Found **{len(results)}** similar cases:")
            for i, case in enumerate(results):
                sim  = case.get("_similarity", case.get("_rrf_score", "–"))
                sim_str = f"{sim:.3f}" if isinstance(sim, float) else str(sim)
                outcome_icon = "✅" if case.get("label") == 1 else "❌"
                label_text   = case.get("label_text", "Unknown")
                sections_str = ", ".join(case.get("sections_cited", [])) or "—"
                date_str     = case.get("date", "")

                with st.expander(
                    f"**#{i+1}** {case.get('name','Case')}  |  {outcome_icon} {label_text}  |  Score: {sim_str}",
                    expanded=(i == 0)
                ):
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.markdown(f"**Court:** {case.get('court','—')}")
                    mc2.markdown(f"**Date:** {date_str or '—'}")
                    mc3.markdown(f"**Case ID:** `{case.get('case_id','—')}`")

                    st.markdown(f"**IPC/BNS Sections Cited:** `{sections_str}`")
                    st.markdown(f"**Judgment:** {outcome_icon} **{label_text}**")

                    st.markdown("**📄 Case Preview:**")
                    st.markdown(f"> {case.get('preview','')[:500]}…")

                    annotations = case.get("annotations", [])
                    if annotations:
                        st.markdown("**🔍 Key Explanation Sentences (Annotations):**")
                        for ann in annotations[:3]:
                            st.markdown(f"> _{ann}_")

                    # Use as RAG context
                    if st.button(f"📚 Use this case as RAG context", key=f"use_rag_{i}"):
                        ctx = case.get("context", case.get("preview",""))
                        st.session_state["extra_rag_context"] = ctx
                        st.success("✅ Case context added to RAG — it will be included in the next QA answer.")

                    if st.button(f"🔊 Read preview aloud", key=f"tts_{i}"):
                        af = text_to_speech(case.get("preview","")[:800], target_language)
                        if af: st.audio(af)

