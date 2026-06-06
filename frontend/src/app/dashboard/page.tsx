"use client";
import { useState } from 'react';
import Link from 'next/link';

const ANALYSES = [
  { title: "Kesavananda Bharati v. Kerala", site: "Supreme Court", icon: "🏛", color: "#2563eb", before: "145K", after: "3.2K", red: "97%" },
  { title: "Section 302 IPC — Murder", site: "India Code", icon: "⚖", color: "#7c3aed", before: "840", after: "120", red: "85%" },
  { title: "Maneka Gandhi v. Union of India", site: "Supreme Court", icon: "🏛", color: "#16a34a", before: "45K", after: "1.5K", red: "96%" },
];

export default function DashboardHome() {
  const [tab, setTab] = useState('All');

  return (
    <div className="page-container">
      <div className="greeting">
        <h1>Shape Your Future<br/>with the Right Knowledge</h1>
        <p>Discover a world of legal intelligence with our cutting-edge AI.</p>
      </div>

      <div className="dash-grid">
        {/* Left Column (Main Stats & Banner) */}
        <div className="dash-col">
          <div className="stat-grid">
            <Link href="/dashboard/analysis">
              <div className="stat-card c-blue">
                <div className="stat-icon">🏛</div>
                <h3>55</h3>
                <p>Judgments Analyzed</p>
              </div>
            </Link>
            <div className="stat-card c-purple">
              <div className="stat-icon">⚖</div>
              <h3>12</h3>
              <p>Statutes Parsed</p>
            </div>
            <Link href="/dashboard/search">
              <div className="stat-card c-green">
                <div className="stat-icon">🔍</div>
                <h3>24</h3>
                <p>Cases Found</p>
              </div>
            </Link>
          </div>

          <div className="panel panel-pad" style={{ padding: '0', overflow: 'hidden' }}>
            <div style={{ padding: '2rem 2rem 1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.25rem' }}>Latest Analyses</h3>
                <p style={{ color: 'var(--text-3)', fontSize: '0.9rem' }}>Recent documents processed by NyayaSetu</p>
              </div>
              <div className="pill-tabs">
                {['All', 'Judgments', 'Statutes'].map(t => (
                  <button key={t} className={`pill-tab${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>{t}</button>
                ))}
              </div>
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ paddingLeft: '2rem' }}>Document</th>
                  <th style={{ textAlign: 'center' }}>Reduced by</th>
                  <th style={{ textAlign: 'center' }}>Before (Words)</th>
                  <th style={{ textAlign: 'center' }}>After (Words)</th>
                </tr>
              </thead>
              <tbody>
                {ANALYSES.map((item, i) => (
                  <tr key={i} style={{ transition: 'background 0.2s', cursor: 'pointer' }} onMouseEnter={e => e.currentTarget.style.background = 'var(--bg)'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <td style={{ paddingLeft: '2rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <div style={{ width: '2.5rem', height: '2.5rem', borderRadius: '10px', background: `${item.color}15`, color: item.color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem' }}>
                          {item.icon}
                        </div>
                        <div>
                          <p className="doc-title">{item.title}</p>
                          <p className="doc-site">{item.site}</p>
                        </div>
                      </div>
                    </td>
                    <td style={{ textAlign: 'center' }}><span className="badge-green result-badge" style={{ marginBottom: 0 }}>{item.red}</span></td>
                    <td style={{ textAlign: 'center', fontWeight: 600 }}>{item.before}</td>
                    <td style={{ textAlign: 'center', fontWeight: 600, color: 'var(--blue)' }}>{item.after}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Column (Banner & Usage) */}
        <div className="dash-col">
          <div className="banner">
            <span className="badge">50% OFF</span>
            <h2>Premium+<br/>$160 / Year To Get All Access</h2>
            <button className="cta">Upgrade Now</button>
          </div>

          <div className="panel panel-pad" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1.5rem' }}>Usage Statistics</h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', flex: 1, justifyContent: 'center' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>API Quota</span>
                  <span style={{ fontSize: '0.9rem', color: 'var(--blue)', fontWeight: 700 }}>68%</span>
                </div>
                <div style={{ height: '8px', background: 'var(--bg)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: '68%', background: 'var(--blue)', borderRadius: '4px' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Storage</span>
                  <span style={{ fontSize: '0.9rem', color: 'var(--purple)', fontWeight: 700 }}>45%</span>
                </div>
                <div style={{ height: '8px', background: 'var(--bg)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: '45%', background: 'var(--purple)', borderRadius: '4px' }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
