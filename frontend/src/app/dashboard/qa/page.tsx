"use client";
import { useState, useContext, useRef, useEffect } from 'react';
import { SettingsCtx } from '../layout';
import { askLegalQuestion } from '@/lib/api';

const EXAMPLES = [
  'What is the punishment for murder in India?',
  'What does criminal conspiracy mean?',
  'Difference between murder and culpable homicide?',
  'What are the ingredients of theft?',
];

interface Msg { role: 'user' | 'ai'; text: string; ragas?: any; followups?: string[]; }

export default function QAPage() {
  const { language } = useContext(SettingsCtx);
  const [readability, setReadability] = useState('Student');
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const send = async (q: string) => {
    if (!q.trim()) return;
    setMsgs(m => [...m, { role: 'user', text: q }]);
    setInput('');
    setLoading(true);
    try {
      const res = await askLegalQuestion(q, readability, language);
      if (res.success) {
        setMsgs(m => [...m, { role: 'ai', text: res.answer, ragas: res.ragas, followups: res.followups }]);
      } else {
        setMsgs(m => [...m, { role: 'ai', text: `Error: ${res.error}` }]);
      }
    } catch {
      setMsgs(m => [...m, { role: 'ai', text: 'Could not connect to backend. Make sure the FastAPI server is running on port 8000.' }]);
    }
    setLoading(false);
  };

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs]);

  return (
    <div className="analysis-shell" style={{ maxWidth: '900px', margin: '0 auto', height: 'calc(100vh - 8rem)' }}>
      <div className="analysis-main" style={{ borderRight: 'none' }}>
        <div className="tab-header" style={{ justifyContent: 'space-between', alignItems: 'center', padding: '1rem 2rem' }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-1)' }}>Legal Learning Assistant</h2>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', background: 'var(--bg)', padding: '0.4rem 1rem', borderRadius: '9999px', border: '1px solid var(--border)' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-4)' }}>Readability:</span>
            <select value={readability} onChange={e => setReadability(e.target.value)} style={{ border: 'none', background: 'transparent', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-1)', outline: 'none', cursor: 'pointer' }}>
              <option>Basic</option>
              <option>Student</option>
              <option>Advanced</option>
            </select>
          </div>
        </div>

        <div className="content-area" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', padding: '2rem 3rem' }}>
          {msgs.length === 0 && (
            <div style={{ textAlign: 'center', marginTop: 'auto', marginBottom: 'auto' }}>
              <div className="upload-icon" style={{ margin: '0 auto 1.5rem', width: '80px', height: '80px', background: 'var(--blue-light)' }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              </div>
              <h3 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--text-1)' }}>How can I help you understand the law?</h3>
              <p style={{ color: 'var(--text-3)', marginBottom: '2rem' }}>Ask me about statutes, cases, or legal concepts.</p>
              
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', justifyContent: 'center', maxWidth: '600px', margin: '0 auto' }}>
                {EXAMPLES.map(ex => (
                  <button key={ex} className="filter-pill" onClick={() => send(ex)} style={{ background: 'var(--bg)', borderRadius: '12px', padding: '0.75rem 1.25rem' }}>
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}

          {msgs.map((m, i) => (
            m.role === 'user' ? (
              <div key={i} style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{ background: 'var(--text-1)', color: 'white', padding: '1rem 1.5rem', borderRadius: '24px 24px 0 24px', maxWidth: '80%', fontSize: '0.95rem', lineHeight: '1.5', boxShadow: 'var(--shadow-sm)' }}>
                  {m.text}
                </div>
              </div>
            ) : (
              <div key={i} style={{ display: 'flex', gap: '1rem' }}>
                <div style={{ width: '2.5rem', height: '2.5rem', borderRadius: '50%', background: 'var(--blue)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: '0.25rem' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2a10 10 0 1 0 10 10H12V2z"/></svg>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxWidth: '85%' }}>
                  <div style={{ background: 'var(--bg)', color: 'var(--text-1)', padding: '1.25rem 1.5rem', borderRadius: '0 24px 24px 24px', fontSize: '0.95rem', lineHeight: '1.65' }} dangerouslySetInnerHTML={{ __html: m.text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                  
                  {m.ragas?.badge && (
                    <div>
                      <span className="result-badge badge-green" style={{ marginBottom: 0, background: m.ragas.badge[1] === 'green' ? '#dcfce7' : '#fef9c3', color: m.ragas.badge[1] === 'green' ? '#16a34a' : '#854d0e' }}>
                        Confidence: {m.ragas.badge[0]}
                      </span>
                    </div>
                  )}

                  {m.followups?.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-4)', alignSelf: 'center', marginRight: '0.5rem' }}>Follow-ups:</span>
                      {m.followups.map((fq, fi) => (
                        <button key={fi} className="filter-pill" style={{ margin: 0 }} onClick={() => send(fq)}>{fq}</button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )
          ))}

          {loading && (
            <div style={{ display: 'flex', gap: '1rem' }}>
              <div style={{ width: '2.5rem', height: '2.5rem', borderRadius: '50%', background: 'var(--blue)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
              </div>
              <div style={{ background: 'var(--bg)', color: 'var(--text-3)', padding: '1rem 1.5rem', borderRadius: '0 24px 24px 24px', fontSize: '0.95rem', fontStyle: 'italic' }}>
                Analyzing legal statutes...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="chat-footer">
          <div className="chat-input-wrap">
            <input
              value={input} onChange={e => setInput(e.target.value)}
              placeholder="Ask your legal question here…"
              onKeyDown={e => e.key === 'Enter' && !loading && send(input)}
            />
            <button className="send-btn" onClick={() => !loading && send(input)}>
              {loading ? <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} /> : <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
