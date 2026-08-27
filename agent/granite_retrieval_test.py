from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from extractor.financial_resolver import build_evidence
from extractor.pipeline import extract
from .retrieval_agent import _candidate_index, retrieve


def run_granite_retrieval_test(pdf_path: str, out_dir: str = "output", question: str = "What was total debt?") -> Dict[str, Any]:
    data = extract(pdf_path, out_dir=out_dir, render_images=False)
    evidence = build_evidence(question, data)
    retrieval = retrieve(question, evidence)
    candidate_index = _candidate_index(evidence)

    diagnostics = {
        "question": question,
        "retrieval_model": retrieval.get("model"),
        "raw_granite_json": retrieval.get("raw_model_output"),
        "granite_response": retrieval.get("granite_response"),
        "granite_selected_sources_raw": retrieval.get("granite_selected_sources_raw"),
        "validated_selected_sources": retrieval.get("selected_sources", []),
        "rejected_selections": retrieval.get("rejected_selections", []),
        "fallback": bool(retrieval.get("fallback")),
        "warnings": retrieval.get("warnings", []),
        "metric": evidence.get("metric"),
        "candidate_count": len(candidate_index),
        "candidate_index": candidate_index,
        "selected_page_numbers": [
            item.get("page") for item in retrieval.get("selected_sources", []) if item.get("page") is not None
        ],
        "expected_sources": {
            "balance_sheet_page": 34,
            "required_rows": [
                "commercial paper",
                "current term debt",
                "non-current term debt",
            ],
        },
    }

    output_path = Path(out_dir) / "granite_retrieval_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"\nSaved diagnostic: {output_path}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Granite-only financial retrieval diagnostic")
    parser.add_argument("pdf_path")
    parser.add_argument("--out", default="output")
    parser.add_argument("--question", default="What was total debt?")
    args = parser.parse_args()
    run_granite_retrieval_test(args.pdf_path, args.out, args.question)


if __name__ == "__main__":
    main()
