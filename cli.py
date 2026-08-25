#!/usr/bin/env python3
import argparse
import sys

from extractor.pipeline import extract


def main():
    parser = argparse.ArgumentParser(
        description="Extract financial PDF evidence: text, validated table candidates, and page images."
    )
    parser.add_argument("pdf_path", help="Path to the input PDF")
    parser.add_argument("--out", default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--include-notes", action="store_true",
                        help="Also flag Notes to financial statements pages")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print located sections only; skip output files")
    parser.add_argument("--debug", action="store_true",
                        help="Include page scores and table diagnostics")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip rendering financial pages to PNG")
    args = parser.parse_args()

    result = extract(
        args.pdf_path,
        out_dir=None if args.summary_only else args.out,
        include_notes=args.include_notes,
        debug=args.debug,
        render_images=not args.no_images,
    )

    meta = result["document_metadata"]
    print(f"Source: {result['source_file']}")
    print(f"Total pages: {result['total_pages']}")
    print(f"Company: {meta['company_name'] or '(not detected)'}")
    print(f"Financial year: {meta['financial_year'] or '(not detected)'}")
    unit = f"{meta['currency']} {meta['unit_scale']}" if meta['currency'] else "(not detected)"
    print(f"Currency/unit: {unit}")
    print(f"Standalone/Consolidated: {meta['standalone_or_consolidated'] or '(not detected)'}")
    print(f"Flagged financial-statement pages: {result['flagged_page_count']}")
    print(f"Table candidates: {result['table_summary']['candidate_count']}; validated: {result['table_summary']['validated_count']}")
    print("Sections found:")
    for s in result["sections_found"]:
        print(f"  pages {s['start_page']+1}-{s['end_page']+1} ({', '.join(s['categories'])}) confidence={s['confidence']}")
    if result["ambiguous_pages"]:
        print(f"Ambiguous pages: {result['ambiguous_pages']}")

    if args.debug:
        print("\n--- validated tables ---")
        for p in result["pages"]:
            bt = p.get("best_table")
            if bt:
                print(f"  page {p['page_number_human']}: score={bt['score']} source={bt['source']} rows={bt['rows']} cols={bt['columns']}")
            elif p["table_candidate_count"]:
                print(f"  page {p['page_number_human']}: candidates rejected")

    if not args.summary_only:
        print("\nArtifacts:")
        for name, path in result.get("artifacts", {}).items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    sys.exit(main())
