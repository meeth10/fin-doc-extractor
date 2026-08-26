(() => {
  const style = document.createElement('style');
  style.textContent = `
    .ask-shell{max-width:980px;margin:0 auto}.ask-hero{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;margin-bottom:22px}.ask-hero h1{margin:6px 0 4px;font-size:34px;letter-spacing:-.035em}.ask-hero p{margin:0;color:#7f8a9b;font-size:13px;line-height:1.5}.ollama-pill{border:1px solid #2a3340;background:#0d1218;border-radius:999px;padding:7px 10px;color:#aeb9c8;font-size:10px;white-space:nowrap}.ask-box{border:1px solid #293341;background:#0d1117;border-radius:14px;padding:14px;position:relative;z-index:1}.ask-input{display:block;width:100%;min-height:140px;resize:vertical;border:1px solid #252e3a;background:#0a0e13;color:#eef2f7;border-radius:10px;padding:13px;font-size:14px;line-height:1.5;outline:none;pointer-events:auto;user-select:text;cursor:text}.ask-input:focus{border-color:#4b586a;box-shadow:0 0 0 2px rgba(96,120,150,.12)}.ask-actions{display:flex;justify-content:space-between;align-items:center;margin-top:10px}.ask-hints{display:flex;gap:7px;flex-wrap:wrap}.ask-hints button{border:1px solid #252f3c;background:#111720;color:#9ba7b8;border-radius:999px;padding:6px 9px;font-size:10px}.ask-hints button:hover{color:#fff;border-color:#3a4656}.ask-submit{border:0;background:#eef3fa;color:#090c10;border-radius:8px;padding:9px 14px;font-weight:800;font-size:11px}.ask-submit:disabled{opacity:.5;cursor:not-allowed}.ask-result{margin-top:18px}.answer-card{border:1px solid #293341;background:#0d1117;border-radius:14px;padding:18px}.answer-top{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}.answer-label{font-size:10px;letter-spacing:.14em;color:#677489;font-weight:800}.answer-value{font-size:32px;font-weight:850;letter-spacing:-.04em;margin-top:5px}.answer-meta{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}.answer-badge{font-size:9px;border:1px solid #27313d;border-radius:999px;padding:5px 7px;color:#a7b2c2}.answer-badge.good{border-color:#355d45;color:#9ee0b0}.answer-badge.derived{border-color:#5c5131;color:#e7d391}.answer-badge.warn{border-color:#634532;color:#e8ad84}.answer-badge.ai{border-color:#394b63;color:#b8d4ff}.answer-explanation{margin-top:14px;color:#909bad;font-size:12px;line-height:1.6}.formula{margin-top:14px;padding:11px;border:1px dashed #2a3441;background:#0a0e13;border-radius:9px;color:#c3ccd9;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}.answer-inputs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:12px}.answer-input{border:1px solid #232c38;background:#10151c;border-radius:9px;padding:10px}.answer-input span{display:block;color:#657185;font-size:9px;text-transform:uppercase;letter-spacing:.08em}.answer-input strong{display:block;margin-top:4px;font-size:12px}.sources-head{margin:22px 0 9px;font-size:11px;font-weight:800}.source-row{display:flex;gap:8px;flex-wrap:wrap}.source-chip{border:1px solid #252f3b;background:#0d1218;color:#aab5c4;border-radius:8px;padding:7px 9px;font-size:10px}.ask-error{border:1px solid #5b3a31;background:#1a110e;color:#e2ad97;border-radius:12px;padding:14px;font-size:12px}.ask-empty{border:1px dashed #2a3442;background:#0c1016;border-radius:14px;padding:50px 25px;text-align:center;margin-top:18px}.ask-empty h2{margin:10px 0 6px;font-size:18px}.ask-empty p{margin:0;color:#788496;font-size:12px}.ask-foot{margin-top:12px;color:#5f6d80;font-size:10px;line-height:1.5}
  `;
  document.head.appendChild(style);

  const baseSetView = window.setView;
  window.setView = type => {
    if (type === 'ask') renderAsk();
    else if (baseSetView) baseSetView(type);
  };

  function renderAsk() {
    const content = document.querySelector('#content');
    if (!content) return;
    content.innerHTML = `
      <div class="ask-shell">
        <div class="ask-hero">
          <div><div class="eyebrow">FINANCIAL Q&A</div><h1>Ask Financials</h1><p>Evidence is resolved first, then Ollama interprets and explains the answer.</p></div>
          <div class="ollama-pill">Ollama · qwen3:8b · 16K context</div>
        </div>
        <div class="ask-box">
          <textarea id="ask-input" class="ask-input" autocomplete="off" spellcheck="false" aria-label="Financial question" placeholder="What was cash at FY2025?\n\nOr: What was EBITDA? Was it reported or derived?"></textarea>
          <div class="ask-actions"><div class="ask-hints">
            <button type="button" data-q="What was cash at the latest reported year?">Cash</button>
            <button type="button" data-q="What was revenue at the latest reported year?">Revenue</button>
            <button type="button" data-q="What was EBITDA? Was it reported or derived?">EBITDA</button>
            <button type="button" data-q="What was enterprise value?">EV</button>
            <button type="button" data-q="What was total debt at the latest reported year?">Debt</button>
          </div><button type="button" id="ask-submit" class="ask-submit">Ask</button></div>
        </div>
        <div id="ask-result" class="ask-result"></div>
        <div class="ask-foot">The resolver owns source truth and deterministic finance math. Ollama explains the evidence but cannot override verified values.</div>
      </div>`;
    const input = document.querySelector('#ask-input');
    document.querySelectorAll('.ask-hints button').forEach(btn => btn.onclick = () => { input.value = btn.dataset.q; input.focus(); });
    document.querySelector('#ask-submit').onclick = submit;
    input.addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') submit(); });
    setTimeout(() => input.focus(), 0);
  }

  async function submit() {
    const input = document.querySelector('#ask-input'); const button = document.querySelector('#ask-submit'); const result = document.querySelector('#ask-result');
    const question = input?.value.trim(); if (!question) return;
    button.disabled = true; result.innerHTML = '<div class="ask-empty"><h2>Thinking…</h2><p>Resolving evidence and asking Ollama.</p></div>';
    try {
      const response = await fetch('/api/ask', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
      const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || 'Question failed'); renderAnswer(payload);
    } catch(error) { result.innerHTML = `<div class="ask-error">${esc(error.message)}</div>`; }
    finally { button.disabled = false; }
  }

  function renderAnswer(a) {
    const result = document.querySelector('#ask-result'); if (!result) return;
    if (!a.metric && a.status === 'ambiguous') { result.innerHTML = `<div class="ask-error">${esc(a.message || 'I could not map that question to a supported financial metric.')}</div>`; return; }
    const statusClass = a.status === 'reported' ? 'good' : a.status === 'derived' ? 'derived' : 'warn';
    const value = a.answer === null ? 'Not available' : formatAnswer(a.answer, a.currency, a.unit);
    const sources = (a.sources || []).map(s => `<span class="source-chip">${esc(s.statement || 'Source')} · p.${esc(s.page ?? '—')}${s.table_title ? ` · ${esc(s.table_title)}` : ''}</span>`).join('');
    const inputs = (a.inputs || []).map(i => `<div class="answer-input"><span>${esc(i.name)}</span><strong>${esc(formatAnswer(i.value, a.currency, a.unit))}${i.page ? ` · p.${esc(i.page)}` : ''}</strong></div>`).join('');
    result.innerHTML = `<article class="answer-card"><div class="answer-top"><div><div class="answer-label">${esc(a.metric || 'FINANCIAL METRIC')}</div><div class="answer-value">${esc(value)}</div></div><div class="answer-meta"><span class="answer-badge ${statusClass}">${esc(String(a.status || '').toUpperCase())}</span><span class="answer-badge">${esc(String(a.confidence || '').toUpperCase())} CONFIDENCE</span>${a.llm_used ? `<span class="answer-badge ai">LLM · ${esc(a.llm_model || 'Ollama')}</span>` : ''}${a.period ? `<span class="answer-badge">${esc(a.period)}</span>` : ''}</div></div>${a.explanation ? `<div class="answer-explanation">${esc(a.explanation)}</div>` : ''}${a.formula ? `<div class="formula">${esc(a.formula)}</div>` : ''}${inputs ? `<div class="answer-inputs">${inputs}</div>` : ''}<div class="sources-head">Evidence</div><div class="source-row">${sources || '<span class="source-chip">No direct source recorded</span>'}</div></article>`;
  }

  function formatAnswer(value, currency, unit) { if (value === null || value === undefined) return '—'; if (typeof value === 'number') { const number = Number.isInteger(value) ? value.toLocaleString('en-IN') : value.toLocaleString('en-IN',{maximumFractionDigits:2}); return `${currency || ''}${currency ? ' ' : ''}${number}${unit ? ` ${unit}` : ''}`; } return String(value); }
  function esc(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  document.addEventListener('click', event => { const button = event.target.closest?.('#main-nav button[data-view="ask"]'); if (!button) return; event.preventDefault(); event.stopImmediatePropagation(); document.querySelectorAll('#main-nav button').forEach(b => b.classList.toggle('active', b === button)); window.renderAsk(); }, true);
  window.renderAsk = renderAsk;
})();
