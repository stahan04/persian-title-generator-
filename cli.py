#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FW-TitleGen Command-Line Interface

Examples:
    python cli.py --model all
    python cli.py --model gpt-4o-mini
    python cli.py --model claude-sonnet-5
    python cli.py --model grok-4
    python cli.py --model all --limit 10
"""

import argparse

from evaluation.benchmark import MODELS
from evaluation.benchmark import run_all
from evaluation.benchmark import run_benchmark


def main():

    parser = argparse.ArgumentParser(
        prog="FW-TitleGen",
        description=(
            "Provider-independent benchmark for "
            "Persian SEO-oriented title generation."
        ),
    )

    parser.add_argument(
        "--model",
        default="all",
        choices=[
            "all",
            *MODELS.keys(),
        ],
        help=(
            "Select a benchmark model "
            "or run all supported models."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N dataset samples."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 55)
    print("FW-TitleGen")
    print(
        "Persian SEO-Oriented LLM Benchmark"
    )
    print("=" * 55)
    print()

    if args.model == "all":

        run_all(
            limit=args.limit
        )

    else:

        run_benchmark(
            model_name=args.model,
            limit=args.limit
        )


if __name__ == "__main__":
    main()
