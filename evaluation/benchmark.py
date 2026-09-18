#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FW-TitleGen
Provider-independent benchmark runner for Persian SEO title generation.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress

# ============================================================
# Configuration
# ============================================================

console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "persian_seo_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("AVALAI_API_KEY", "")
BASE_URL = os.getenv(
    "AVALAI_BASE_URL",
    "https://api.avalai.ir/v1"
)

# Public benchmark names -> provider model identifiers.
# Provider identifiers can be overridden through environment variables.
MODELS = {
    "gpt-4o-mini": os.getenv(
        "FW_MODEL_GPT",
        "gpt-4o-mini"
    ),
    "claude-sonnet-5": os.getenv(
        "FW_MODEL_CLAUDE",
        "claude-sonnet-5"
    ),
    "grok-4": os.getenv(
        "FW_MODEL_GROK",
        "grok-4"
    ),
}

MIN_TITLE_LENGTH = 50
MAX_TITLE_LENGTH = 70

MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
SAVE_EVERY = 10


# ============================================================
# Dataset
# ============================================================

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    rows = []

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# Prompt
# ============================================================

def build_prompt(keyword, topic=""):
    return (
        "یک عنوان فارسی مناسب سئو تولید کن.\n"
        f"کلیدواژه: {keyword}\n"
        f"موضوع: {topic}\n\n"
        "قوانین:\n"
        "1. فقط یک عنوان فارسی تولید کن.\n"
        "2. کلیدواژه باید در عنوان وجود داشته باشد.\n"
        "3. طول عنوان باید بین ۵۰ تا ۷۰ کاراکتر باشد.\n"
        "4. هیچ توضیح، Markdown یا متن اضافی ننویس.\n\n"
        "عنوان:"
    )


# ============================================================
# Model request
# ============================================================

def call_model(model_id, prompt):
    if not API_KEY:
        raise RuntimeError(
            "AVALAI_API_KEY is not configured. "
            "Set it as an environment variable."
        )

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            start = time.perf_counter()

            with httpx.Client(
                timeout=REQUEST_TIMEOUT
            ) as client:

                response = client.post(
                    f"{BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            latency = time.perf_counter() - start

            response.raise_for_status()

            data = response.json()

            title = (
                data["choices"][0]["message"]["content"]
                .strip()
            )

            usage = data.get("usage", {})

            return {
                "title": title,
                "latency": round(latency, 4),
                "input_tokens": usage.get(
                    "prompt_tokens", 0
                ),
                "output_tokens": usage.get(
                    "completion_tokens", 0
                ),
                "total_tokens": usage.get(
                    "total_tokens", 0
                ),
            }

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                wait_time = 2 ** (attempt - 1)
                time.sleep(wait_time)

    raise RuntimeError(
        f"Request failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# Validation
# ============================================================

def evaluate_constraints(title, keyword):

    title_length = len(title)

    return {
        "title_length": title_length,
        "has_keyword": keyword in title,
        "length_ok": (
            MIN_TITLE_LENGTH
            <= title_length
            <= MAX_TITLE_LENGTH
        ),
    }


# ============================================================
# Checkpoint / Resume
# ============================================================

def load_completed_ids(output_path):

    if not output_path.exists():
        return set()

    completed = set()

    with open(
        output_path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row.get("success") == "True":
                completed.add(str(row["id"]))

    return completed


def save_rows(rows, output_path):

    if not rows:
        return

    fields = [
        "id",
        "domain",
        "keyword",
        "topic",
        "model",
        "title",
        "title_length",
        "has_keyword",
        "length_ok",
        "latency",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "success",
        "error",
    ]

    file_exists = output_path.exists()

    with open(
        output_path,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


# ============================================================
# Benchmark
# ============================================================

def run_benchmark(
    model_name,
    limit=None
):

    if model_name not in MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    dataset = load_dataset()

    if limit:
        dataset = dataset[:limit]

    model_id = MODELS[model_name]

    safe_model_name = (
        model_name
        .replace("/", "_")
        .replace("-", "_")
    )

    output_path = (
        RESULTS_DIR
        / f"results_{safe_model_name}.csv"
    )

    completed_ids = load_completed_ids(
        output_path
    )

    pending = [
        row
        for row in dataset
        if str(row["id"]) not in completed_ids
    ]

    console.print(
        f"\n[bold cyan]FW-TitleGen[/bold cyan]"
    )

    console.print(
        f"Model: {model_name}"
    )

    console.print(
        f"Total: {len(dataset)} | "
        f"Completed: {len(completed_ids)} | "
        f"Pending: {len(pending)}"
    )

    buffer = []

    with Progress() as progress:

        task = progress.add_task(
            f"Running {model_name}",
            total=len(pending)
        )

        for sample in pending:

            sample_id = sample["id"]
            domain = sample.get("domain", "")
            keyword = sample["keyword"].strip()
            topic = sample.get("topic", "").strip()

            result = {
                "id": sample_id,
                "domain": domain,
                "keyword": keyword,
                "topic": topic,
                "model": model_name,
                "title": "",
                "title_length": 0,
                "has_keyword": False,
                "length_ok": False,
                "latency": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "success": False,
                "error": "",
            }

            try:

                prompt = build_prompt(
                    keyword,
                    topic
                )

                response = call_model(
                    model_id,
                    prompt
                )

                title = response["title"]

                constraints = (
                    evaluate_constraints(
                        title,
                        keyword
                    )
                )

                result.update(
                    {
                        "title": title,
                        **constraints,
                        "latency":
                            response["latency"],
                        "input_tokens":
                            response["input_tokens"],
                        "output_tokens":
                            response["output_tokens"],
                        "total_tokens":
                            response["total_tokens"],
                        "success": True,
                    }
                )

            except Exception as exc:

                result["error"] = str(exc)[:300]

            buffer.append(result)

            if len(buffer) >= SAVE_EVERY:
                save_rows(
                    buffer,
                    output_path
                )
                buffer = []

            progress.advance(task)

    if buffer:
        save_rows(
            buffer,
            output_path
        )

    console.print(
        f"[green]Results saved to: "
        f"{output_path}[/green]"
    )


# ============================================================
# Run all models
# ============================================================

def run_all(limit=None):

    for model_name in MODELS:
        run_benchmark(
            model_name=model_name,
            limit=limit
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "FW-TitleGen Persian LLM Benchmark"
        )
    )

    parser.add_argument(
        "--model",
        default="all",
        choices=[
            "all",
            *MODELS.keys()
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N samples.",
    )

    args = parser.parse_args()

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
