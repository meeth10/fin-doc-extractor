from __future__ import annotations

import argparse
import json

from scraper import scrape_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape financial-statement evidence from a PDF")
    parser.add_argument("pdf", help="Path to PDF filing")
    parser.add_argument("--output", "-o", help="Write JSON evidence to this path")
    args = parser.parse_args()

    result = scrape_pdf(args.pdf).to_dict()
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
