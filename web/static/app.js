const file = document.querySelector('#file');
const fileEmpty = document.querySelector('#file-empty');
const empty = document.querySelector('#empty');
const app = document.querySelector('#app');
const loading = document.querySelector('#loading');
const loadingTitle = document.querySelector('#loading-title');
const loadingDetail = document.querySelector('#loading-detail');
const progressBar = document.querySelector('#progress-bar');
const summary = document.querySelector('#summary');
const pagesEl = document.querySelector('#pages');
const pageCount = document.querySelector('#page-count');
const content = document.querySelector('#content');

const STATEMENTS = {
  balance_sheet: { label: 'Balance Sheet', short: 'BS', icon: 'B' },
  income_statement: { label: 'Income Statement', short: 'P&L', icon: 'I' },
  cash_flow: { label: 'Cash Flow', short: 'CF', icon: 'C' },
};

let data = null;
let view = 'overview';
let selectedPage = 0;

function setLoading(title, detail, progress = 0) {
  loadingTitle.textContent = title;
  loadingDetail.textContent = detail;
  progressBar.style.width = `${Math.max(3, Math.min(100, progress))}%`;
}

async function handleFile(f) {
  if (!f) return;
  loading.hidden = false;
  setLoading('Uploading PDF', f.name, 3);
  const body = new FormData();
  body.append('file', f);
  try {
    const r = await fetch('/api/extract', { method: 'POST', body });
    const job = await r.json();
    if (!r.ok) throw new Error(job.detail || 'Upload failed');
    data = await waitForRun(job.run_id);
    selectedPage = firstUsefulPage();
    view = 'overview';
    empty.hidden = true;
    app.hidden = false;
    render();
  } catch (e) {
    alert(e.message);
  } finally {
    loading.hidden = true;
  }
}

file.addEventListener('change', () => handleFile(file.files[0]));
fileEmpty.addEventListener('change', () => handleFile(fileEmpty.files[0]));

async function waitForRun(runId) {
  while (true) {
    const r = await fetch(`/api/runs/${runId}/status`);
    const s = await r.json();
    if (!r.ok) throw new Error(s.detail || 'Status check failed');
    setLoading('Analyzing document', s.message || 'Working…', s.progress || 0);
    if (s.status === 'failed') throw new Error(s.message || 'Extraction failed');
    if (s.status === 'complete') {
      const rr = await fetch(`/api/runs/${runId}/result`);
      const result = await rr.json();
      if (!rr.ok || !result.ready) throw new Error('Result was not ready');
      return result;
    }
    await new Promise(resolve => setTimeout(resolve, 700));
  }
}

function statementTables(type) {
  return data?.statement_tables?.[type]?.tables || [];
}

function firstUsefulPage() {
  for (const type of Object.keys(STATEMENTS)) {
    const tables = statementTables(type);
    if (tables.length) return tables[0].page_number;
  }
  const fallback = data?.tables?.tables?.find(t => t.validated);
  return fallback ? fallback.page_number : 0;
}

function pageData(n) {
  return data.document.pages.find(p => p.page_number === n) || {
    page_number: n,
    page_number_human: n + 1,
    raw_text: '',
    extraction_method: 'unknown'
  };
}

function pageVisual(n) {
  return data.visuals.pages.find(p => p.page_number_human === n + 1);
}

function pageTables(n) {
  return data.tables.tables.filter(t => t.page_number === n);
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function numericCell(value) {
  const text = String(value || '').replace(/\s/g, '');
  return /^[-(₹$€£]?\d[\d,]*(\.\d+)?%?\)?$/.test(text);
}

function normalizeRows(rows) {
  const clean = rows
    .map(r => Array.isArray(r) ? r.map(v => String(v ?? '').trim()) : [String(r ?? '').trim()])
    .filter(r => r.some(Boolean));
  const cols = Math.max(1, ...clean.map(r => r.length));
  return clean.map(r => [...r, ...Array(cols - r.length).fill('')]);
}

function tableHtml(t) {
  const rows = normalizeRows(t.table || []);
  if (!rows.length) return '<div class="no-table">No cells extracted.</div>';
  const cols = rows[0].length;
  const title = rows.length > 1 && rows[0].filter(Boolean).length <= 2 && rows[0].join(' ').length > 20;
  const visibleRows = title ? rows.slice(1) : rows;
  const titleHtml = title ? `<div class="table-title-row">${esc(rows[0].filter(Boolean).join(' · '))}</div>` : '';
  return `${titleHtml}<div class="table-scroll"><table class="financial-table"><colgroup><col class="label-col">${Array(Math.max(0, cols - 1)).fill('<col>').join('')}</colgroup><tbody>${visibleRows.map((row, i) => `<tr class="${i === 0 ? 'first-row' : ''}">${row.map((c, j) => `<td class="${j > 0 || numericCell(c) ? 'numeric-col' : 'label-cell'}">${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function tableMeta(t) {
  const statement = t.statement_type ? STATEMENTS[t.statement_type]?.label : 'Unassigned';
  const assignment = t.statement_assignment || (t.validated ? 'validated' : 'rejected');
  const badgeClass = assignment === 'provisional' ? 'provisional' : assignment === 'validated' ? 'good' : 'muted';
  return `<div class="table-meta"><span>${esc(statement)}</span><span>Page ${t.page_number_human}</span><span>${esc(t.source || 'unknown source')}</span><span>Table score ${esc(t.score)}</span><span class="${badgeClass}">${esc(assignment)}</span></div>`;
}

function statementCard(type) {
  const info = STATEMENTS[type];
  const bucket = data?.statement_tables?.[type] || { tables: [], pages: [], status: 'empty' };
  const tables = bucket.tables || [];
  const provisional = tables.filter(t => t.statement_assignment === 'provisional').length;
  const sub = tables.length
    ? `${tables.length} table${tables.length === 1 ? '' : 's'} · pages ${bucket.pages.join(', ')}${provisional ? ` · ${provisional} provisional` : ''}`
    : 'No assigned table yet';
  return `<button class="statement-card" onclick="setView('${type}')"><div class="statement-icon">${info.icon}</div><div class="statement-body"><div class="statement-name">${info.label}</div><div class="statement-sub">${sub}</div></div><div class="statement-arrow">→</div></button>`;
}

function renderSummary() {
  const s = data.summary;
  const m = s.metadata || {};
  const validated = s.table_summary?.validated_count ?? 0;
  const provisional = Object.values(data.statement_tables || {}).reduce((n, b) => n + (b.tables || []).filter(t => t.statement_assignment === 'provisional').length, 0);
  summary.innerHTML = `
    <div class="doc-card">
      <div class="doc-kicker">DOCUMENT</div>
      <div class="doc-name">${esc(s.source_name)}</div>
      <div class="doc-meta">${s.total_pages} pages · ${s.elapsed_seconds}s</div>
      <div class="doc-grid">
        <div><span>Company</span><b>${esc(m.company_name || '—')}</b></div>
        <div><span>Financial year</span><b>${esc(m.financial_year || '—')}</b></div>
        <div><span>Currency</span><b>${esc(m.currency || '—')}</b></div>
        <div><span>Validated</span><b>${validated}</b></div>
      </div>
      ${provisional ? `<div class="provisional-note">${provisional} provisional statement table${provisional === 1 ? '' : 's'} available</div>` : ''}
    </div>`;
}

function renderPages() {
  const marked = new Map();
  Object.keys(STATEMENTS).forEach(type => statementTables(type).forEach(t => marked.set(t.page_number, type)));
  pageCount.textContent = `${data.summary.total_pages}`;
  pagesEl.innerHTML = '';
  for (let i = 0; i < data.summary.total_pages; i++) {
    const type = marked.get(i);
    const b = document.createElement('button');
    b.className = i === selectedPage && view === 'page' ? 'active' : '';
    b.innerHTML = `<span>Page ${i + 1}</span>${type ? `<em>${STATEMENTS[type].short}</em>` : ''}`;
    b.onclick = () => { selectedPage = i; view = 'page'; render(); };
    pagesEl.appendChild(b);
  }
}

function renderNav() {
  document.querySelectorAll('#main-nav button').forEach(b => {
    b.classList.toggle('active', b.dataset.view === view);
    b.onclick = () => { view = b.dataset.view; render(); };
  });
}

function render() {
  renderSummary();
  renderPages();
  renderNav();
  if (view === 'overview') renderOverview();
  else if (STATEMENTS[view]) renderStatement(view);
  else if (view === 'tables') renderAllTables();
  else renderRawText();
}

function renderOverview() {
  const buckets = Object.keys(STATEMENTS).map(t => data.statement_tables?.[t] || { tables: [] });
  const isolated = buckets.reduce((n, b) => n + b.tables.length, 0);
  const validated = data.summary.table_summary?.validated_count || 0;
  const rejected = Math.max(0, (data.tables.tables || []).length - validated);
  content.innerHTML = `
    <div class="hero-row">
      <div><div class="eyebrow">EXTRACTION RESULT</div><h1>Financial statements</h1><p>The evidence layer is ready. Start with the three core statements or inspect everything.</p></div>
      <div class="hero-actions"><button class="secondary" onclick="setView('tables')">Review candidates</button><label class="upload-button compact"><input id="file-inline" type="file" accept="application/pdf" />Upload another</label></div>
    </div>
    <div class="metric-strip">
      <div><span>Pages</span><strong>${data.summary.total_pages}</strong></div>
      <div><span>Validated tables</span><strong>${validated}</strong></div>
      <div><span>Core statements</span><strong>${isolated}</strong></div>
      <div><span>Other / rejected</span><strong>${rejected}</strong></div>
    </div>
    <div class="section-head"><h2>Core statements</h2><span>Agent-ready streams</span></div>
    <div class="statement-grid">${Object.keys(STATEMENTS).map(statementCard).join('')}</div>
    <div class="section-head"><h2>Evidence layers</h2><span>Always traceable</span></div>
    <div class="evidence-grid">
      <div class="info-card"><b>01</b><h3>Structured tables</h3><p>Validated and provisional candidates retain source page, extractor and score.</p></div>
      <div class="info-card"><b>02</b><h3>Original page image</h3><p>The visual source stays attached so agents can verify layout and numbers.</p></div>
      <div class="info-card"><b>03</b><h3>Raw text / OCR</h3><p>Every page remains available as the underlying textual evidence.</p></div>
    </div>
    <div class="section-head"><h2>Run quality</h2><span>What the engine actually found</span></div>
    <div class="coverage-row">${Object.keys(STATEMENTS).map(t => { const b=data.statement_tables?.[t]||{tables:[],status:'empty'}; return `<div class="coverage"><span>${STATEMENTS[t].label}</span><strong>${b.tables.length}</strong><small>${b.status === 'validated' ? 'validated' : b.status === 'provisional' ? 'provisional' : 'not isolated'}</small></div>`; }).join('')}</div>`;

  const inline = document.querySelector('#file-inline');
  if (inline) inline.addEventListener('change', () => handleFile(inline.files[0]));
}

function renderStatement(type) {
  const info = STATEMENTS[type];
  const bucket = data.statement_tables?.[type] || { tables: [], status: 'empty' };
  const tables = bucket.tables || [];
  const validated = tables.filter(t => t.statement_assignment === 'validated');
  const provisional = tables.filter(t => t.statement_assignment === 'provisional');

  if (!tables.length) {
    content.innerHTML = `<div class="empty-panel"><div class="big-icon">${info.icon}</div><h2>${info.label}</h2><p>No table was assigned yet. That is a classification issue, not missing evidence.</p><div class="empty-actions"><button class="secondary" onclick="setView('tables')">Inspect all candidates</button><button class="secondary" onclick="setView('text')">Inspect raw text</button></div></div>`;
    return;
  }

  content.innerHTML = `
    <div class="statement-header"><div><div class="eyebrow">ISOLATED OUTPUT</div><h1>${info.label}</h1><p>${validated.length} validated · ${provisional.length} provisional · source-linked</p></div><div class="statement-count">${tables.length}</div></div>
    <div class="statement-stack">
      ${validated.map(t => statementTableCard(t, 'VALIDATED')).join('')}
      ${provisional.map(t => statementTableCard(t, 'PROVISIONAL')).join('')}
    </div>
    <div class="bottom-hint">Numbers shown here remain traceable to the original page. Nothing has been normalized or interpreted yet.</div>`;
}

function statementTableCard(t, status) {
  return `<article class="table-card"><div class="table-card-head"><div><strong>Page ${t.page_number_human}</strong><span>${status}</span></div><button onclick="jumpToPage(${t.page_number})">View source →</button></div>${tableMeta(t)}${tableHtml(t)}</article>`;
}

function renderAllTables() {
  const tables = data.tables.tables || [];
  content.innerHTML = `<div class="statement-header"><div><div class="eyebrow">TABLE REVIEW</div><h1>All table candidates</h1><p>${tables.filter(t=>t.validated).length} validated · ${tables.filter(t=>!t.validated).length} rejected or uncertain. Useful evidence is never hidden.</p></div><div class="statement-count">${tables.length}</div></div><div class="statement-stack">${tables.map((t, i) => `<article class="table-card ${t.validated ? '' : 'rejected'}"><div class="table-card-head"><div><strong>Candidate ${i + 1}</strong><span>${t.validated ? 'validated' : 'rejected'}</span></div><button onclick="jumpToPage(${t.page_number})">Page ${t.page_number_human} →</button></div>${tableMeta(t)}${tableHtml(t)}</article>`).join('')}</div>`;
}

function renderPage(n) {
  const p = pageData(n);
  const v = pageVisual(n);
  const ts = pageTables(n);
  const best = ts.find(t => t.validated) || ts[0];
  content.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">SOURCE EVIDENCE</div><h1>Page ${p.page_number_human}</h1><p>${esc(best?.statement_type ? STATEMENTS[best.statement_type].label : 'Not assigned')} · ${esc(p.extraction_method)}</p></div><a class="pdf-link" href="${data.run_id ? `/api/runs/${data.run_id}/source` : '#'}" target="_blank">Open PDF ↗</a></div>
    <div class="page-grid">
      <div class="source-card"><div class="card-title">Original page</div>${v ? `<img class="page-image" src="${v.url}" alt="Page ${p.page_number_human}">` : '<div class="no-table">No rendered image.</div>'}</div>
      <div class="source-card"><div class="card-title">Best table candidate</div>${best ? `${tableMeta(best)}${tableHtml(best)}` : '<div class="no-table">No table candidate on this page.</div>'}<div class="mini-label">Extraction method</div><div class="method-pill">${esc(p.extraction_method)}</div></div>
    </div>`;
}

function renderRawText() {
  const p = pageData(selectedPage);
  content.innerHTML = `<div class="page-head"><div><div class="eyebrow">RAW EVIDENCE</div><h1>Page ${p.page_number_human}</h1><p>${esc(p.extraction_method)}</p></div><button class="secondary" onclick="jumpToPage(${selectedPage})">View page →</button></div><div class="source-card"><pre>${esc(p.raw_text || 'No text extracted.')}</pre></div>`;
}

window.setView = type => { view = type; render(); };
window.jumpToPage = n => { selectedPage = n; view = 'page'; render(); };
