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
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress


# ============================================================
# Configuration
# ============================================================

console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DATASET_PATH = BASE_DIR / "persian_seo_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("AVALAI_API_KEY", "")

BASE_URL = os.getenv(
    "AVALAI_BASE_URL",
    "https://api.avalai.ir/v1",
)

MODELS = {
    "gpt-4o-mini": os.getenv(
        "FW_MODEL_GPT",
        "gpt-4o-mini",
    ),
    "claude-sonnet-5": os.getenv(
        "FW_MODEL_CLAUDE",
        "claude-sonnet-5",
    ),
    "grok-4": os.getenv(
        "FW_MODEL_GROK",
        "grok-4",
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
    """
    Load the Persian SEO benchmark dataset.

    reference_title is used ONLY for evaluation.
    It must never be included in the generation prompt.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "id",
            "domain",
            "keyword",
            "reference_title",
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        rows = list(reader)

    return rows


# ============================================================
# Prompt
# ============================================================

def build_prompt(domain, keyword):
    """
    Build the standardized Persian SEO title-generation prompt.

    IMPORTANT:
    reference_title is intentionally excluded from the prompt
    to prevent reference leakage during evaluation.
    """

    return (
        "یک عنوان فارسی مناسب سئو تولید کن.\n"
        f"حوزه: {domain}\n"
        f"کلیدواژه: {keyword}\n\n"
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
    """
    Send one request to an OpenAI-compatible provider endpoint.
    """

    if not API_KEY:
        raise RuntimeError(
            "AVALAI_API_KEY is not configured. "
            "Add it to your local .env file."
        )

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt,
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

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

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

            latency = (
                time.perf_counter() - start
            )

            response.raise_for_status()

            data = response.json()

            title = (
                data["choices"][0]
                ["message"]["content"]
                .strip()
            )

            usage = data.get(
                "usage",
                {},
            )

            return {
                "title": title,

                "latency": round(
                    latency,
                    4,
                ),

                "input_tokens": usage.get(
                    "prompt_tokens",
                    0,
                ),

                "output_tokens": usage.get(
                    "completion_tokens",
                    0,
                ),

                "total_tokens": usage.get(
                    "total_tokens",
                    0,
                ),
            }

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                wait_time = 2 ** (
                    attempt - 1
                )

                time.sleep(
                    wait_time
                )

    raise RuntimeError(
        f"Request failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# Constraint evaluation
# ============================================================

def evaluate_constraints(
    title,
    keyword,
):
    """
    Evaluate prompt-level constraints.
    """

    title_length = len(title)

    return {
        "title_length": title_length,

        "has_keyword": (
            keyword in title
        ),

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
    """
    Read successfully completed sample IDs.
    """

    if not output_path.exists():
        return set()

    completed = set()

    with open(
        output_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row.get("success")
                == "True"
            ):

                completed.add(
                    str(row["id"])
                )

    return completed


def save_rows(
    rows,
    output_path,
):
    """
    Append benchmark results to CSV.
    """

    if not rows:
        return

    fields = [
        "id",
        "domain",
        "keyword",
        "reference_title",
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

    file_exists = (
        output_path.exists()
    )

    with open(
        output_path,
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Benchmark
# ============================================================

def run_benchmark(
    model_name,
    limit=None,
):
    """
    Run the benchmark for one model.
    """

    if model_name not in MODELS:
        raise ValueError(
            f"Unsupported model: "
            f"{model_name}"
        )

    dataset = load_dataset()

    if limit is not None:

        if limit <= 0:
            raise ValueError(
                "--limit must be greater than 0."
            )

        dataset = dataset[:limit]

    model_id = MODELS[
        model_name
    ]

    safe_model_name = (
        model_name
        .replace("/", "_")
        .replace("-", "_")
    )

    output_path = (
        RESULTS_DIR
        / f"results_{safe_model_name}.csv"
    )

    completed_ids = (
        load_completed_ids(
            output_path
        )
    )

    pending = [
        row
        for row in dataset
        if str(row["id"])
        not in completed_ids
    ]

    console.print(
        "\n[bold cyan]"
        "FW-TitleGen"
        "[/bold cyan]"
    )

    console.print(
        f"Model: {model_name}"
    )

    console.print(
        f"Total: {len(dataset)} | "
        f"Completed: "
        f"{len(completed_ids)} | "
        f"Pending: {len(pending)}"
    )

    buffer = []

    with Progress() as progress:

        task = progress.add_task(
            f"Running {model_name}",
            total=len(pending),
        )

        for sample in pending:

            sample_id = (
                sample["id"].strip()
            )

            domain = (
                sample["domain"].strip()
            )

            keyword = (
                sample["keyword"].strip()
            )

            # IMPORTANT:
            # Used only later for evaluation.
            reference_title = (
                sample[
                    "reference_title"
                ].strip()
            )

            result = {
                "id": sample_id,
                "domain": domain,
                "keyword": keyword,

                "reference_title":
                    reference_title,

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

                # reference_title is NOT
                # passed to the model.
                prompt = build_prompt(
                    domain,
                    keyword,
                )

                response = call_model(
                    model_id,
                    prompt,
                )

                title = response[
                    "title"
                ]

                constraints = (
                    evaluate_constraints(
                        title,
                        keyword,
                    )
                )

                result.update(
                    {
                        "title":
                            title,

                        **constraints,

                        "latency":
                            response[
                                "latency"
                            ],

                        "input_tokens":
                            response[
                                "input_tokens"
                            ],

                        "output_tokens":
                            response[
                                "output_tokens"
                            ],

                        "total_tokens":
                            response[
                                "total_tokens"
                            ],

                        "success":
                            True,
                    }
                )

            except Exception as exc:

                result["error"] = (
                    str(exc)[:300]
                )

            buffer.append(
                result
            )

            if (
                len(buffer)
                >= SAVE_EVERY
            ):

                save_rows(
                    buffer,
                    output_path,
                )

                buffer = []

            progress.advance(
                task
            )

    if buffer:

        save_rows(
            buffer,
            output_path,
        )

    console.print(
        "[green]"
        "Results saved to: "
        f"{output_path}"
        "[/green]"
    )


# ============================================================
# Run all models
# ============================================================

def run_all(limit=None):
    """
    Run the benchmark for all configured models.
    """

    for model_name in MODELS:

        run_benchmark(
            model_name=model_name,
            limit=limit,
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "FW-TitleGen Persian "
            "LLM Benchmark"
        )
    )

    parser.add_argument(
        "--model",
        default="all",

        choices=[
            "all",
            *MODELS.keys(),
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,

        help=(
            "Run only the first "
            "N samples."
        ),
    )

    args = parser.parse_args()

    if args.model == "all":

        run_all(
            limit=args.limit
        )

    else:

        run_benchmark(
            model_name=args.model,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
