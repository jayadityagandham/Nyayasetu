const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const HF_KEY = process.env.NEXT_PUBLIC_HF_TOKEN || '';

export async function uploadDocument(
  file: File,
  readabilityLevel: string,
  targetLanguage: string,
  useHfRoles: boolean
) {
  const form = new FormData();
  form.append('file', file);
  form.append('hf_api_key', HF_KEY);
  form.append('readability_level', readabilityLevel);
  form.append('target_language', targetLanguage);
  form.append('use_hf_roles', String(useHfRoles));

  const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form });
  return res.json();
}

export async function askLegalQuestion(
  query: string,
  readabilityLevel: string,
  targetLanguage: string
) {
  const res = await fetch(`${API_BASE}/api/qa`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, hf_api_key: HF_KEY, readability_level: readabilityLevel, target_language: targetLanguage }),
  });
  return res.json();
}

export async function searchCases(
  query: string,
  topK: number,
  labelFilter: string,
  sectionFilter: string,
  searchMode: string
) {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      hf_api_key: HF_KEY,
      top_k: topK,
      label_filter: labelFilter,
      section_filter: sectionFilter,
      search_mode: searchMode,
    }),
  });
  return res.json();
}
