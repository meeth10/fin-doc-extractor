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
  return 0;
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
  return /^\s*[-(₹$€£]?\d[\d,]*(\.\d+)?%?\)?\s*$/.test(String(value || ''));
}

function normalizeRows(rows) {
  const clean = rows.map(r => Array.isArray(r) ? r.map(v => String(v ?? '')) : [String(r ?? '')]);
  const cols = Math.max(1, ...clean.map(r => r.length));
  return clean.map(r => [...r, ...Array(cols - r.length).fill('')]);
}

function tableHtml(t) {
  const rows = normalizeRows(t.table || []);
  if (!rows.length) return '<div class="no-table">No cells extracted.</div>';
  const cols = rows[0].length;
  const title = rows.length && rows[0].filter(Boolean).length <= 2 && rows[0].join(' ').length > 20;
  const visibleRows = title ? rows.slice(1) : rows;
  const titleHtml = title ? `<div class="table-title-row">${esc(rows[0].filter(Boolean).join(' · '))}</div>` : '';
  return `${titleHtml}<div class="table-scroll"><table class="financial-table"><colgroup><col class="label-col">${Array(Math.max(0, cols - 1)).fill('<col>').join('')}</colgroup><tbody>${visibleRows.map((row, i) => `<tr class="${i === 0 ? 'first-row' : ''}">${row.map((c, j) => `<td class="${j > 0 || numericCell(c) ? 'numeric-col' : 'label-cell'}">${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function tableMeta(t) {
  const statement = t.statement_type ? STATEMENTS[t.statement_type]?.label : 'Unassigned';
  return `<div class="table-meta"><span>${esc(statement)}</span><span>Page ${t.page_number_human}</span><span>${esc(t.source)}</span><span>Table ${t.score}</span></div>`;
}

function statementCard(type) {
  const info = STATEMENTS[type];
  const tables = statementTables(type);
  const pages = data.statement_tables?.[type]?.pages || [];
  const best = tables[0];
  return `<button class="statement-card" onclick="setView('${type}')"><div class="statement-icon">${info.icon}</div><div class="statement-body"><div class="statement-name">${info.label}</div><div class="statement-sub">${tables.length ? `${tables.length} validated table${tables.length === 1 ? '' : 's'} · pages ${pages.join(', ')}` : 'No confidently isolated table yet'}</div></div><div class="statement-arrow">→</div></button>`;
}

function renderSummary() {
  const s = data.summary;
  const m = s.metadata || {};
  const validated = s.table_summary?.validated_count ?? 0;
  summary.innerHTML = `
    <div class="doc-card">
      <div class="doc-kicker">DOCUMENT</div>
      <div class="doc-name">${esc(s.source_name)}</div>
      <div class="doc-meta">${s.total_pages} pages · ${s.elapsed_seconds}s</div>
      <div class="doc-grid">
        <div><span>Company</span><b>${esc(m.company_name || '—')}</b></div>
        <div><span>Financial year</span><b>${esc(m.financial_year || '—')}</b></div>
        <div><span>Currency</span><b>${esc(m.currency || '—')}</b></div>
        <div><span>Tables</span><b>${validated}</b></div>
      </div>
    </div>`;
}

function renderPages() {
  const marked = new Map();
  Object.keys(STATEMENTS).forEach(type => {
    statementTables(type).forEach(t => marked.set(t.page_number, type));
  });
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
  else renderPage(selectedPage);
}

function renderOverview() {
  const counts = Object.fromEntries(Object.keys(STATEMENTS).map(t => [t, statementTables(t).length]));
  content.innerHTML = `
    <div class="hero-row">
      <div><div class="eyebrow">EXTRACTION RESULT</div><h1>Financial statements</h1><p>Three isolated evidence streams for downstream analyst agents.</p></div>
      <div class="health-pill"><span class="dot"></span>${data.summary.table_summary.validated_count} validated tables</div>
    </div>
    <div class="statement-grid">
      ${Object.keys(STATEMENTS).map(statementCard).join('')}
    </div>
    <div class="section-head"><h2>What the agent receives</h2><span>Evidence-first</span></div>
    <div class="evidence-grid">
      <div class="info-card"><b>1</b><h3>Structured tables</h3><p>Ranked candidates with page, source and confidence preserved.</p></div>
      <div class="info-card"><b>2</b><h3>Original page image</h3><p>The visual source stays attached to every extracted page.</p></div>
      <div class="info-card"><b>3</b><h3>Raw page text</h3><p>Digital text or OCR text remains available for verification.</p></div>
    </div>
    <div class="section-head"><h2>Coverage</h2><span>Current run</span></div>
    <div class="coverage-row">
      ${Object.keys(STATEMENTS).map(t => `<div class="coverage"><span>${STATEMENTS[t].label}</span><strong>${counts[t]}</strong><small>${counts[t] ? 'isolated' : 'not confidently isolated'}</small></div>`).join('')}
    </div>`;
}

function renderStatement(type) {
  const info = STATEMENTS[type];
  const tables = statementTables(type);
  if (!tables.length) {
    content.innerHTML = `<div class="empty-panel"><div class="big-icon">${info.icon}</div><h2>${info.label}</h2><p>No confidently isolated table was produced. The evidence remains available under All Tables and Raw Text.</p><button class="secondary" onclick="setView('tables')">Inspect all candidates</button></div>`;
    return;
  }
  content.innerHTML = `<div class="statement-header"><div><div class="eyebrow">ISOLATED OUTPUT</div><h1>${info.label}</h1><p>${tables.length} validated table${tables.length === 1 ? '' : 's'} · ready for agent consumption</p></div><div class="statement-count">${tables.length}</div></div><div class="statement-stack">${tables.map(t => `<article class="table-card"><div class="table-card-head"><div><strong>Page ${t.page_number_human}</strong><span>statement confidence ${t.statement_confidence}</span></div><button onclick="jumpToPage(${t.page_number})">View source →</button></div>${tableMeta(t)}${tableHtml(t)}</article>`).join('')}</div>`;
}

function renderAllTables() {
  const tables = data.tables.tables;
  content.innerHTML = `<div class="statement-header"><div><div class="eyebrow">TABLE REVIEW</div><h1>All table candidates</h1><p>Validated and rejected candidates, with the source page available for inspection.</p></div><div class="statement-count">${tables.length}</div></div><div class="statement-stack">${tables.map((t, i) => `<article class="table-card ${t.validated ? '' : 'rejected'}"><div class="table-card-head"><div><strong>Candidate ${i + 1}</strong><span>${t.validated ? 'validated' : 'rejected'}</span></div><button onclick="jumpToPage(${t.page_number})">Page ${t.page_number_human} →</button></div>${tableMeta(t)}${tableHtml(t)}</article>`).join('')}</div>`;
}

function renderPage(n) {
  const p = pageData(n);
  const v = pageVisual(n);
  const ts = pageTables(n);
  const best = ts.find(t => t.validated) || ts[0];
  content.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">SOURCE EVIDENCE</div><h1>Page ${p.page_number_human}</h1><p>${esc(best?.statement_type ? STATEMENTS[best.statement_type].label : 'Not confidently assigned')} · ${esc(p.extraction_method)}</p></div><a class="pdf-link" href="${data.run_id ? `/api/runs/${data.run_id}/source#page=${p.page_number_human}` : '#'}" target="_blank">Open PDF ↗</a></div>
    <div class="page-grid">
      <div class="source-card"><div class="card-title">Original page</div>${v ? `<img class="page-image" src="${v.url}" alt="Page ${p.page_number_human}">` : '<div class="no-table">No rendered image.</div>'}</div>
      <div class="source-card"><div class="card-title">Best table candidate</div>${best ? `${tableMeta(best)}${tableHtml(best)}` : '<div class="no-table">No table candidate on this page.</div>'}<div class="mini-label">Extraction method</div><div class="method-pill">${esc(p.extraction_method)}</div></div>
    </div>`;
}

window.setView = type => { view = type; render(); };
window.jumpToPage = n => { selectedPage = n; view = 'page'; render(); };
