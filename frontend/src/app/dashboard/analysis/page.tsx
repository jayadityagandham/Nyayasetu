"use client";
import { useState, useRef, useCallback, useContext } from 'react';
import { SettingsCtx } from '../layout';
import { uploadDocument } from '@/lib/api';

const TABS = ['Structured Summary', 'Legal Sections', 'Original Text'];

export default function AnalysisPage() {
  const { language, useHfRoles } = useContext(SettingsCtx);
  const [readability, setReadability] = useState('Student');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0); // 0: init, 1: doc processing, 2: RAG, 3: Explanation, 4: Personalisation
  const [estimatedTime, setEstimatedTime] = useState(0); // in seconds
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState('');
  const [mainTab, setMainTab] = useState(0);
  const [filterRole, setFilterRole] = useState<string>('All');
  const fileRef = useRef<HTMLInputElement>(null);

  const simulateLoadingSteps = (fileSize: number) => {
    // Approx 6 bytes per word. Assume ~50 words processed per second on average backend hardware.
    const wordCount = Math.max(100, Math.ceil(fileSize / 6));
    const estSec = Math.max(5, Math.ceil(wordCount / 50));
    setEstimatedTime(estSec);
    setTimeElapsed(0);

    const estMs = estSec * 1000;
    setLoadingStep(0); // Initializing
    setTimeout(() => setLoadingStep(1), 500); // Doc processing
    setTimeout(() => setLoadingStep(2), estMs * 0.35); // RAG Retrieval
    setTimeout(() => setLoadingStep(3), estMs * 0.65); // Generation
    setTimeout(() => setLoadingStep(4), estMs * 0.9); // Formatting
  };

  const handleFile = useCallback(async (file: File, currentReadability: string = readability) => {
    setLoading(true); setError(''); setData(null); setSelectedFile(file);
    simulateLoadingSteps(file.size);
    
    // Timer interval for UI update
    const timer = setInterval(() => {
      setTimeElapsed(prev => prev + 1);
    }, 1000);

    try {
      const result = await uploadDocument(file, currentReadability, language, useHfRoles);
      if (!result.success) throw new Error(result.error || 'Processing failed');
      setData(result);
    } catch (e: any) { setError(e.message); }
    finally { 
      clearInterval(timer);
      setLoading(false); 
    }
  }, [language, useHfRoles, readability]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f, readability);
  };

  const regenerate = () => {
    if (selectedFile) handleFile(selectedFile, readability);
  };

  const formatTime = (totalSeconds: number) => {
    if (totalSeconds < 60) return `${totalSeconds}s`;
    const m = Math.floor(totalSeconds / 60);
    const s = totalSeconds % 60;
    return `${m}m ${s}s`;
  };

  return (
    <div className="analysis-shell">
      {/* ── Main Content Area ── */}
      <div className="analysis-main">
        {loading && (
          <div className="loader-overlay">
            <div className="loader-box" style={{ width: '500px' }}>
              <h3>Analyzing Document</h3>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-4)', textTransform: 'uppercase', fontWeight: 700 }}>Estimated Time</span>
                  <span style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--blue)' }}>~{formatTime(estimatedTime)}</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', textAlign: 'right' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-4)', textTransform: 'uppercase', fontWeight: 700 }}>Elapsed Time</span>
                  <span style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-2)' }}>{formatTime(timeElapsed)}</span>
                </div>
              </div>

              <div className="step-list">
                <div className={`step-item ${loadingStep >= 1 ? 'active' : ''} ${loadingStep > 1 ? 'done' : ''}`}>
                  <div className="step-icon">{loadingStep > 1 ? '✓' : '1'}</div>
                  <span className="step-text"><strong>Stage 1: Document Processing</strong><br/><span style={{fontSize: '0.8rem', color: 'var(--text-4)', fontWeight: 'normal'}}>Rhetorical Roles, Sections, Timeline</span></span>
                </div>
                <div className={`step-item ${loadingStep >= 2 ? 'active' : ''} ${loadingStep > 2 ? 'done' : ''}`}>
                  <div className="step-icon">{loadingStep > 2 ? '✓' : '2'}</div>
                  <span className="step-text"><strong>Stage 2: RAG Retrieval</strong><br/><span style={{fontSize: '0.8rem', color: 'var(--text-4)', fontWeight: 'normal'}}>Statutory Rules & Similar Cases</span></span>
                </div>
                <div className={`step-item ${loadingStep >= 3 ? 'active' : ''} ${loadingStep > 3 ? 'done' : ''}`}>
                  <div className="step-icon">{loadingStep > 3 ? '✓' : '3'}</div>
                  <span className="step-text"><strong>Stage 3: Explanation Generation</strong><br/><span style={{fontSize: '0.8rem', color: 'var(--text-4)', fontWeight: 'normal'}}>Role-Aware Summarization & "Why Law Applied"</span></span>
                </div>
                <div className={`step-item ${loadingStep >= 4 ? 'active' : ''} ${loadingStep > 4 ? 'done' : ''}`}>
                  <div className="step-icon">{loadingStep > 4 ? '✓' : '4'}</div>
                  <span className="step-text"><strong>Stage 4: Personalisation & Output</strong><br/><span style={{fontSize: '0.8rem', color: 'var(--text-4)', fontWeight: 'normal'}}>Readability ({readability}) & Multilingual ({language})</span></span>
                </div>
              </div>
            </div>
          </div>
        )}

        {!data && !loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '2rem' }}>
            
            <div style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '1rem', background: 'var(--surface)', padding: '1rem 2rem', borderRadius: '9999px', boxShadow: 'var(--shadow-sm)', border: '1px solid var(--border)' }}>
              <span style={{ fontWeight: 600, color: 'var(--text-1)' }}>Target Readability:</span>
              <select value={readability} onChange={e => setReadability(e.target.value)} style={{ padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--border)', outline: 'none', background: 'var(--bg)', fontWeight: 500 }}>
                <option>Basic</option>
                <option>Student</option>
                <option>Advanced</option>
              </select>
            </div>

            <div
              className="upload-zone"
              onClick={() => fileRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={onDrop}
              style={{ width: '100%', maxWidth: '600px' }}
            >
              <div className="upload-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </div>
              <h4>Upload a Document</h4>
              <p>Drag and drop a PDF or TXT judgment, or click to browse</p>
              <input ref={fileRef} type="file" accept=".pdf,.txt" style={{ display: 'none' }} onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f, readability); }} />
            </div>
          </div>
        )}

        {error && <div style={{ padding: '2rem', color: 'red' }}>Error: {error}</div>}

        {data && !loading && (
          <>
            <div className="tab-header">
              {TABS.map((t, i) => (
                <button key={t} className={`tab-btn ${mainTab === i ? 'active' : ''}`} onClick={() => setMainTab(i)}>{t}</button>
              ))}
            </div>
            
            <div className="content-area">
              {mainTab === 0 && (
                <div>
                  {Object.entries(data.summaries || {})
                    .filter(([r]) => filterRole === 'All' || filterRole === r)
                    .map(([role, text]) => (
                      <div key={role} className="role-card">
                        <div className="role-header">
                          <div className="role-dot" style={{ background: 'var(--blue)' }} />
                          <span className="role-title">{role}</span>
                        </div>
                        <p className="role-body">{String(text)}</p>
                      </div>
                    ))}
                </div>
              )}

              {mainTab === 1 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {Object.entries(data.sections || {}).map(([key, info]: [string, any]) => (
                    <div key={key} className="role-card">
                      <div className="role-header" style={{ justifyContent: 'space-between', width: '100%' }}>
                        <span className="role-title" style={{ fontSize: '1rem' }}>{key} → {info.bns_equivalent || 'N/A'}</span>
                        <span className="result-badge badge-green" style={{ margin: 0 }}>{info.title || ''}</span>
                      </div>
                      {info.penalty && <p className="role-body" style={{ marginTop: '0.5rem' }}><strong>Penalty:</strong> {info.penalty}</p>}
                    </div>
                  ))}
                  {Object.keys(data.sections || {}).length === 0 && <p className="role-body">No IPC/BNS sections detected.</p>}
                </div>
              )}

              {mainTab === 2 && (
                <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6', color: 'var(--text-2)', fontSize: '0.95rem' }}>
                  {data?.cleaned || 'Original text not available.'}
                </div>
              )}
            </div>

            <div className="chat-footer">
              <div className="chat-input-wrap">
                <input placeholder="Ask a question about this judgment…" />
                <button className="send-btn">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── Context Sidebar ── */}
      <div className="analysis-side">
        {data && !loading ? (
          <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto' }}>
            
            <div className="filter-group" style={{ background: 'var(--surface)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', marginBottom: '2rem' }}>
              <p className="filter-label">Regenerate Analysis</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <select value={readability} onChange={e => setReadability(e.target.value)} style={{ padding: '0.6rem', borderRadius: '8px', border: '1px solid var(--border)', background: 'var(--bg)', width: '100%', outline: 'none' }}>
                  <option>Basic</option>
                  <option>Student</option>
                  <option>Advanced</option>
                </select>
                <button onClick={regenerate} style={{ background: 'var(--text-1)', color: 'white', padding: '0.75rem', borderRadius: '8px', border: 'none', fontWeight: 600 }}>
                  Regenerate
                </button>
              </div>
            </div>

            <div className="filter-group">
              <p className="filter-label">Filter Roles</p>
              <div>
                <button className={`filter-pill ${filterRole === 'All' ? 'active' : ''}`} onClick={() => setFilterRole('All')}>All</button>
                {Object.keys(data.summaries || {}).map(r => (
                  <button key={r} className={`filter-pill ${filterRole === r ? 'active' : ''}`} onClick={() => setFilterRole(r)}>{r}</button>
                ))}
              </div>
            </div>

            <div className="filter-group" style={{ flex: 1 }}>
              <p className="filter-label">Timeline of Events</p>
              {data.timeline?.length > 0 ? (
                <div>
                  {data.timeline.map((e: any, i: number) => (
                    <div key={i} className="t-item">
                      <div className="t-dot" />
                      <div className="t-content">
                        <p className="t-date">{e.date}</p>
                        <p className="t-event">{e.event}</p>
                        <p className="t-desc">{e.snippet}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: '0.85rem', color: 'var(--text-4)' }}>No timeline extracted.</p>
              )}
            </div>
          </div>
        ) : (
          <div style={{ padding: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <p style={{ color: 'var(--text-4)', textAlign: 'center', fontSize: '0.9rem' }}>Upload a document to view context timeline and filters here.</p>
          </div>
        )}
      </div>
    </div>
  );
}
