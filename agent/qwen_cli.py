from __future__ import annotations

import argparse
import json

from extractor.pipeline import extract
from .financial_agent import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Qwen financial analysis stack.")
    parser.add_argument("pdf_path")
    parser.add_argument("--question", default="What was total debt?")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    data = extract(args.pdf_path, out_dir=args.out, render_images=False)
    result = answer_question(args.question, data)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
