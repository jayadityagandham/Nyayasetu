"use client";
import { useState, useContext } from 'react';
import { SettingsCtx } from '../layout';
import { searchCases } from '@/lib/api';

type Result = { name?: string; court?: string; date?: string; case_id?: string; label?: number; label_text?: string; _similarity?: number; _rrf_score?: number; preview?: string; sections_cited?: string[]; annotations?: string[]; };

export default function SearchPage() {
  const { hfKey } = useContext(SettingsCtx);
  const [query, setQuery] = useState('');
  const [outcome, setOutcome] = useState('all');
  const [mode, setMode] = useState('Hybrid (BM25 + FAISS)');
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searched, setSearched] = useState(false);

  const doSearch = async () => {
    if (!query.trim()) return;
    setLoading(true); setError(''); setSearched(true);
    try {
      const res = await searchCases(query, hfKey, topK, outcome, '', mode);
      if (!res.success) throw new Error(res.error);
      setResults(res.results || []);
    } catch (e: any) { setError(e.message); setResults([]); }
    setLoading(false);
  };

  const OUTCOMES = [
    { value: 'all', label: 'All Outcomes' },
    { value: 'accepted', label: 'Accepted / Allowed' },
    { value: 'rejected', label: 'Dismissed / Rejected' },
  ];
  const MODES = ['Hybrid (BM25 + FAISS)', 'BM25 Keyword Only', 'FAISS Semantic Only'];

  return (
    <div className="analysis-shell">
      {/* ── Main Column ── */}
      <div className="analysis-main">
        <div className="search-hero">
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1.5rem', textAlign: 'center', color: 'var(--text-1)' }}>Search Legal Database</h2>
          <div className="search-bar-large">
            <input
              type="text" value={query} onChange={e => setQuery(e.target.value)}
              placeholder="Describe your case or paste a summary..."
              onKeyDown={e => e.key === 'Enter' && doSearch()}
            />
            <button onClick={doSearch} disabled={loading}>
              {loading ? <div className="spinner" /> : 'Search Cases'}
            </button>
          </div>
        </div>

        <div className="content-area">
          {error && <div style={{ padding: '1.5rem', background: '#fee2e2', color: '#dc2626', borderRadius: 'var(--radius-md)' }}>Error: {error}</div>}

          {!searched && !loading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-3)' }}>
              <div className="upload-icon" style={{ background: 'var(--bg)', marginBottom: '1.5rem', width: '80px', height: '80px' }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              </div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-1)', marginBottom: '0.5rem' }}>Find Precedents Instantly</h3>
              <p>Search through 35,000+ ILDC cases using hybrid AI search.</p>
            </div>
          )}

          {searched && !loading && results.length === 0 && !error && (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-3)' }}>
              No matching cases found. Try broader keywords or adjusting your filters.
            </div>
          )}

          {results.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <p style={{ fontSize: '0.9rem', color: 'var(--text-3)', fontWeight: 500 }}>
                Found <strong style={{ color: 'var(--text-1)' }}>{results.length}</strong> highly relevant cases
              </p>
              
              {results.map((c, i) => {
                const score = c._similarity ?? c._rrf_score;
                const accepted = c.label === 1;
                return (
                  <div key={i} className="result-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                      <div>
                        <h4 className="result-title">{c.name || `Case Record #${i + 1}`}</h4>
                        <p className="result-meta">
                          {c.court || 'Court Unspecified'} {c.date ? ` • ${c.date}` : ''} {typeof score === 'number' ? ` • Match Score: ${(score * 100).toFixed(1)}%` : ''}
                        </p>
                      </div>
                      <span className={`result-badge ${accepted ? 'badge-green' : 'badge-red'}`}>
                        {c.label_text || (accepted ? 'Allowed' : 'Dismissed')}
                      </span>
                    </div>
                    
                    <p className="result-preview">
                      {c.preview ? `"...${c.preview.slice(0, 350)}..."` : 'No preview text available.'}
                    </p>
                    
                    {c.sections_cited && c.sections_cited.length > 0 && (
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginTop: '1rem' }}>
                        {c.sections_cited.map((s, si) => (
                          <span key={si} style={{ background: 'var(--bg)', border: '1px solid var(--border)', padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.8rem', color: 'var(--text-2)' }}>
                            {s}
                          </span>
                        ))}
                      </div>
                    )}
                    
                    {c.annotations && c.annotations.length > 0 && (
                      <div style={{ marginTop: '1.5rem', background: 'var(--bg)', padding: '1rem', borderRadius: 'var(--radius-md)' }}>
                        <p style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>AI Extracted Findings</p>
                        {c.annotations.slice(0, 2).map((a, ai) => (
                          <p key={ai} style={{ fontSize: '0.9rem', color: 'var(--text-2)', fontStyle: 'italic', marginBottom: '0.5rem' }}>• {a}</p>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Side Column: Filters ── */}
      <div className="analysis-side">
        <div style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '2rem', color: 'var(--text-1)' }}>Search Filters</h3>

          <div className="filter-group">
            <p className="filter-label">Judgment Outcome</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {OUTCOMES.map(o => (
                <button key={o.value} className={`filter-pill ${outcome === o.value ? 'active' : ''}`} style={{ width: '100%', justifyContent: 'space-between', margin: 0 }} onClick={() => setOutcome(o.value)}>
                  {o.label}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <p className="filter-label">Retrieval Engine</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {MODES.map(m => (
                <button key={m} className={`filter-pill ${mode === m ? 'active' : ''}`} style={{ width: '100%', justifyContent: 'flex-start', margin: 0 }} onClick={() => setMode(m)}>
                  {m}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <p className="filter-label">Results Limit: {topK}</p>
            <input type="range" min={3} max={20} value={topK} onChange={e => setTopK(Number(e.target.value))} style={{ width: '100%', accentColor: 'var(--blue)', marginTop: '0.5rem' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-4)', marginTop: '0.5rem' }}>
              <span>3</span>
              <span>20</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
