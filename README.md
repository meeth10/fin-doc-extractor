# fin-doc-extractor

Local, free, deterministic agent that pulls the financial-statement pages
out of a financial document — a 2-page summary or a 300-page annual
report — and dumps raw text + tables. No interpretation, no field mapping,
no LLM calls.

## Install

```bash
pip install -r requirements.txt
# system dep: tesseract binary (OCR fallback)
#   macOS:  brew install tesseract
#   Ubuntu: apt-get install tesseract-ocr
```

## Usage

```bash
python cli.py path/to/report.pdf
python cli.py path/to/report.pdf --out my_output_dir
python cli.py path/to/report.pdf --include-notes      # also pull Notes to F/S
python cli.py path/to/report.pdf --config my_keywords.json
python cli.py path/to/report.pdf --summary-only        # just print sections, no JSON file
```

Output: `<out_dir>/<filename>_extracted.json`

```json
{
  "source_file": "...",
  "total_pages": 312,
  "sections_found": [
    {"start_page": 44, "end_page": 51, "categories": ["balance_sheet", "income_statement"]}
  ],
  "pages": [
    {
      "page_number": 44,
      "page_number_human": 45,
      "extraction_method": "digital",
      "raw_text": "...",
      "tables": [[["Line Item", "FY2025", "FY2024"], ["Cash", "1,245", "980"]]]
    }
  ]
}
```

# fin-doc-extractor

Local, free, mostly-deterministic pipeline that pulls the financial
statements out of a financial document — a 2-page summary or a 300-page
annual report — and locates them reliably even though a document mentions
"Balance Sheet" in a dozen places that aren't the actual balance sheet
(table of contents, auditor's report, cross-references in the MD&A).

**Current status: Phase 1 of the v3 pipeline** — statement-page detection
with confidence scoring, document metadata, raw text+table extraction.
Field normalization, reconciliation, and LLM-assisted resolution are
later phases, not yet built (see Roadmap below).

## Install

```bash
pip install -r requirements.txt
# system dep: tesseract binary (OCR fallback)
#   macOS:  brew install tesseract
#   Ubuntu: apt-get install tesseract-ocr
```

## Usage

```bash
python cli.py path/to/report.pdf
python cli.py path/to/report.pdf --out my_output_dir
python cli.py path/to/report.pdf --include-notes      # also pull Notes to F/S
python cli.py path/to/report.pdf --debug               # per-page confidence + signal breakdown
python cli.py path/to/report.pdf --summary-only        # just print sections, no JSON file
```

Output: `<out_dir>/<filename>_extracted.json`

```json
{
  "source_file": "...",
  "total_pages": 312,
  "document_metadata": {
    "company_name": "Acme Global Holdings Limited",
    "financial_year": "FY2026",
    "currency": "INR",
    "unit_scale": "lakhs",
    "standalone_or_consolidated": "standalone"
  },
  "sections_found": [
    {"start_page": 44, "end_page": 51, "categories": ["balance_sheet", "income_statement"], "confidence": 0.81}
  ],
  "ambiguous_pages": [52],
  "pages": [
    {
      "page_number": 44,
      "page_number_human": 45,
      "extraction_method": "digital",
      "raw_text": "...",
      "tables": [[["Line Item", "FY2025", "FY2024"], ["Cash", "1,245", "980"]]]
    }
  ]
}
```

## How it works

1. **Classify each page** — digital text layer vs. scanned (`extractor/classify.py`).
   Rule: if PyMuPDF's text layer for a page is under 20 chars, treat as scanned.
2. **OCR fallback** — only for scanned pages. Rasterizes with PyMuPDF itself
   (no poppler dependency) and runs Tesseract (`extractor/ocr.py`).
3. **Document metadata** — company name, financial year, currency/unit
   scale ("₹ in Lakhs", "USD in millions"), standalone-vs-consolidated,
   detected from the front matter (`extractor/metadata.py`). Numbers are
   never converted — only the source scale is recorded, so nothing gets
   silently rescaled later.
4. **Locate the financial statements** — `extractor/locator.py` scores
   every page on five independent signals (heading text, statement-specific
   line items like "Trade receivables" or "Finance costs", tabular-shape
   lines, monetary density, year-column headers) and applies penalties for
   things that mention a statement without being one: table-of-contents
   dot-leaders, auditor's-report language, heavy "refer note X"
   cross-referencing. A page only gets classified above a confidence
   threshold; anything in between is marked **ambiguous** rather than
   forced into a category — a contents-page line like *"IND AS Balance
   Sheet — page 69"* scores 0, not a false positive.
   Patterns are whitespace-tolerant (`\s*` instead of literal spaces)
   because OCR routinely drops or merges spacing between words.
5. **Segment into sections** — only *confident* pages anchor a section;
   ambiguous pages never start one on their own, but do get swept in by
   padding if they sit right next to a confident run.
6. **Extract tables only on flagged pages** — running table detection over
   an entire 300-page doc is wasted work. Tries pdfplumber's default
   line/ruling-based strategy first, falls back to a text-alignment
   strategy if that finds nothing (`extractor/table_extract.py`) — needed
   for borderless tables common in Word/Excel-exported statements.
7. **Emit raw JSON** — still no normalization, no line-item mapping, no
   reconciliation. That's Phases 2–9 (see Roadmap).

## Known limitations

- **Tables on scanned pages aren't extracted** — pdfplumber needs a vector
  text layer; a pure-image page only gets OCR'd raw text, no structured
  table.
- **Text-strategy table fallback is noisy** — no gridlines to anchor on,
  so it can pull in blank rows or a heading as a "cell". Left unfiltered
  on purpose — filter downstream once table reconstruction (Phase 3) lands.
- **Keyword/line-item signatures are English-only**, tuned toward
  US-GAAP and Indian/IFRS phrasing (per the current spec's real-world test
  set). Other-language filings will under-score until signatures are added.
- **Year detection is presence-only, not value-alignment** —
  `extract_year_labels()` tells you which fiscal years appear on a page,
  correctly (doesn't assume the first column is the latest year), but
  doesn't yet map a specific numeric column to a specific year's value.
  That's table reconstruction (Phase 3).
- **No LLM fallback**, by design, for page detection — the fix for a missed
  heading style is extending the signatures, not adding a model call.

## Roadmap (not yet built)

The current spec calls for a much larger pipeline than raw extraction:
field normalization to a canonical schema, accounting reconciliation
(assets = liabilities + equity, etc.), optional LLM-assisted ambiguity
resolution for genuinely unclear mappings, and derived metrics (margins,
ROE, FCF...). In implementation order:

- **Phase 2** — canonical financial-fact schema (`canonical_name`,
  `source_label`, `value`, `year`, `currency`, `unit`, `statement_type`,
  `page_number`, `confidence`).
- **Phase 3** — table reconstruction: align extracted labels to numeric
  columns using row/x-coordinate/y-coordinate order (raw `pdfplumber`
  table extraction is geometrically unreliable — a label and its value
  can land in different cells/rows).
- **Phase 4** — accounting reconciliation checks (assets ≈ equity +
  liabilities, cash flow opening + movement ≈ closing, etc.), with
  discrepancies flagged, never silently "fixed."
- **Phase 5** — deterministic label normalization ("Net sales" / "Revenue
  from operations" → `revenue_from_operations`), number normalization
  (parentheses as negative, comma/decimal handling, OCR digit errors).
- **Phase 6** — optional LLM ambiguity resolver, invoked only when
  confidence is low or a table's geometry is ambiguous — never as the
  primary source of a number, and every returned value must be traceable
  to the supplied source text/table or rejected. Needs a provider decision
  first (the spec says `OPENAI_API_KEY`; this project runs on Claude, so
  that should probably be `ANTHROPIC_API_KEY` against the Messages API —
  flagging rather than assuming).
- **Phase 7** — derived metrics (margins, ROE, ROCE, FCF, debt/EBITDA),
  computed in Python, never by the LLM.
- **Phase 8** — validate against real annual reports (clean digital,
  scanned, OCR-corrupted, bank/NBFC, manufacturing, services — the spec's
  minimum real-world test set). The synthetic sample below proves the
  pipeline logic is correct; it can't prove the signatures generalize to
  every real filing's phrasing.
- **Phase 9** — company-intelligence/strategy-analysis layer, explicitly
  gated on Phase 8 accuracy being acceptable first.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

36 tests, four layers:

- `tests/test_classify.py` — page classifier against fake page objects
  (no PDF needed), including the exact 20-char boundary.
- `tests/test_metadata.py` — company name, financial year (including the
  "2025-26 → FY2026" ending-year convention), currency/unit scale
  ("₹ in Lakhs", "USD in millions"), standalone/consolidated detection,
  and the case where both are mentioned (correctly left ambiguous rather
  than guessed).
- `tests/test_locator.py` — the confidence-scoring engine in isolation,
  using realistic multi-line-item page fixtures (a single heading mention
  alone is no longer enough to be "confident" — that's the whole point of
  this phase). Covers: correct category detection for all four statement
  types, a contents-page fixture that must score near-zero despite
  mentioning "Balance Sheet" three times, an auditor's-report page that
  must not be misclassified, heavy cross-referencing penalized, OCR-mangled
  ("Statementof") text still matching, gap-tolerance section merging,
  padding clamped to document bounds, and an ambiguous-only page correctly
  not anchoring a section by itself.
- `tests/test_pipeline.py` — full end-to-end regression test against a
  synthetic ~20-page sample doc built by `samples/make_sample.py`
  (10 pages MD&A filler + digital balance sheet/income statement/cash
  flow tables + 5 pages notes + one page rendered as a flat PNG with no
  text layer, to genuinely exercise the OCR path — the equity statement
  there has enough real line items to be confidently detected even
  through OCR noise). Regenerates the sample automatically if missing.

To eyeball it directly instead of running pytest:

```bash
python cli.py samples/sample_annual_report.pdf --out samples/output --debug
cat samples/output/sample_annual_report_extracted.json | python3 -m json.tool | less
```

Regenerate the sample doc (e.g. after editing `samples/make_sample.py`):

```bash
python samples/make_sample.py
```

### Testing against a real filing

The synthetic sample proves the pipeline logic is correct; it can't prove
the signatures cover every real-world heading/line-item phrasing. To
validate against an actual filing:

```bash
python cli.py path/to/real_10k.pdf --debug --summary-only
```

Check the printed page ranges and per-page scores against the actual PDF.
If a statement gets missed or lands in "ambiguous", it's almost always a
missing line-item phrase or heading variant — add it to
`LINE_ITEM_SIGNATURES` / `HEADING_PATTERNS` in `extractor/locator.py`
rather than lowering the confidence thresholds (that trades false
negatives for false positives on contents/auditor pages, which is the
exact failure mode this phase exists to prevent).

## Project structure

```
fin-doc-extractor/
  cli.py
  extractor/
    classify.py       # digital vs scanned page detection
    ocr.py             # Tesseract fallback
    metadata.py         # company/FY/currency/unit/standalone-consolidated
    locator.py           # confidence-scored statement-page detection
    table_extract.py      # pdfplumber wrapper (lines -> text-strategy fallback)
    pipeline.py             # orchestration
  samples/
    make_sample.py           # synthetic test PDF generator
  tests/
  requirements.txt
  requirements-dev.txt
```

