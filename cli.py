"""Command-line ResearchCrew.

    python cli.py "How do vector databases differ from traditional databases?"
    python cli.py "What is LoRA fine-tuning and when should I use it?" --out lora.md
"""
import argparse
from pathlib import Path

from researchcrew import research
from researchcrew.report import to_markdown


def main():
    p = argparse.ArgumentParser(description="Research a question with a team of AI agents.")
    p.add_argument("question")
    p.add_argument("--out", help="save the Markdown report here")
    p.add_argument("--max-sub", type=int, default=4, help="max sub-questions the planner may create")
    args = p.parse_args()

    report = to_markdown(research(args.question, max_sub=args.max_sub))
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
