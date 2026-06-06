"use client";
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, createContext, useContext } from 'react';

/* ── Global settings context shared across all pages ── */
interface Settings { language: string; useHfRoles: boolean; }
export const SettingsCtx = createContext<Settings>({ language: 'English', useHfRoles: false });
export const useSettings = () => useContext(SettingsCtx);

const NAV = [
  { href: '/dashboard', icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>, label: 'Overview' },
  { href: '/dashboard/analysis', icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>, label: 'Case Analysis' },
  { href: '/dashboard/qa', icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>, label: 'Legal QA' },
  { href: '/dashboard/search', icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>, label: 'Case Search' },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [language, setLanguage] = useState('English');
  const [useHfRoles, setUseHfRoles] = useState(false); // Can be moved or kept internal if not needed globally

  return (
    <SettingsCtx.Provider value={{ language, useHfRoles }}>
      <div className="shell">
        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="sidebar-logo">
            <div className="dot"><span /></div>
            <span className="name">NyayaSetu</span>
          </div>

          <Link href="/dashboard/analysis" style={{ textDecoration: 'none' }}>
            <button className="new-btn">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 5v14M5 12h14"/></svg>
              <span>New Analysis</span>
            </button>
          </Link>

          <div className="nav-section" style={{ flex: 1 }}>
            <span className="filter-label ws-label" style={{ padding: '0 1rem', display: 'block', marginBottom: '0.75rem' }}>Workspace</span>
            {NAV.map(n => (
              <Link key={n.href} href={n.href} className={`nav-link${pathname === n.href ? ' active' : ''}`}>
                {n.icon} <span>{n.label}</span>
              </Link>
            ))}
          </div>

          {/* Simple footer area in sidebar instead of large settings panel */}
          <div style={{ marginTop: 'auto', paddingTop: '1.5rem', borderTop: '1px solid var(--border)', overflow: 'hidden' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div className="avatar" style={{ width: '2.5rem', height: '2.5rem', flexShrink: 0 }}>
                <div style={{width: '100%', height: '100%', background: 'var(--blue)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '1rem'}}>
                  J
                </div>
              </div>
              <div className="user-info" style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-1)' }}>John Doe</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-3)' }}>Premium Plan</span>
              </div>
            </div>
          </div>
        </aside>

        {/* ── Main ── */}
        <div className="main">
          <header className="topbar">
            <div className="search-wrap">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              <input type="text" placeholder="Search across documents..." />
            </div>
            
            <div className="topbar-right">
              {/* Language Selector in Topbar */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '9999px', padding: '0.25rem 0.5rem 0.25rem 1rem', boxShadow: 'var(--shadow-sm)' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-3)' }}>Language:</span>
                <select 
                  value={language} 
                  onChange={e => setLanguage(e.target.value)}
                  style={{ border: 'none', background: 'transparent', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-1)', padding: '0.25rem', cursor: 'pointer', outline: 'none' }}
                >
                  <option>English</option>
                  <option>Hindi</option>
                  <option>Telugu</option>
                </select>
              </div>

              <button className="icon-btn">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
              </button>
            </div>
          </header>
          <div className="page-scroll">{children}</div>
        </div>
      </div>
    </SettingsCtx.Provider>
  );
}
