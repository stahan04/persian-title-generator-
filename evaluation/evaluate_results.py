#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FW-TitleGen
Automatic quality evaluation for Persian SEO title generation.

Metrics:
- BLEU
- ROUGE-L
- BERTScore
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from bert_score import score as bert_score
from hazm import word_tokenize
from nltk.translate.bleu_score import (
    SmoothingFunction,
    sentence_bleu,
)
from rouge_score import rouge_scorer


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

BERT_MODEL = "bert-base-multilingual-cased"

smoothing = SmoothingFunction().method1

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=False,
)


# ============================================================
# Persian tokenization
# ============================================================

def tokenize(text):
    """
    Tokenize Persian text using Hazm.
    """

    if pd.isna(text):
        return []

    return word_tokenize(
        str(text).strip()
    )


# ============================================================
# BLEU
# ============================================================

def calculate_bleu(
    reference,
    prediction,
):
    """
    Calculate sentence-level BLEU.
    """

    reference_tokens = tokenize(
        reference
    )

    prediction_tokens = tokenize(
        prediction
    )

    if (
        not reference_tokens
        or not prediction_tokens
    ):
        return 0.0

    value = sentence_bleu(
        [reference_tokens],
        prediction_tokens,
        smoothing_function=smoothing,
    )

    return float(value)


# ============================================================
# ROUGE-L
# ============================================================

def calculate_rouge_l(
    reference,
    prediction,
):
    """
    Calculate ROUGE-L F1 score.
    """

    if (
        pd.isna(reference)
        or pd.isna(prediction)
    ):
        return 0.0

    scores = rouge.score(
        str(reference),
        str(prediction),
    )

    return float(
        scores["rougeL"].fmeasure
    )


# ============================================================
# BERTScore
# ============================================================

def calculate_bertscore(
    references,
    predictions,
):
    """
    Calculate multilingual BERTScore F1.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    _, _, f1 = bert_score(
        predictions,
        references,
        model_type=BERT_MODEL,
        lang="fa",
        device=device,
        verbose=False,
    )

    return f1.cpu().tolist()


# ============================================================
# Evaluation
# ============================================================

def evaluate_file(csv_path):
    """
    Evaluate one benchmark result CSV.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Result file not found: "
            f"{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "reference_title",
        "title",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Evaluate only successful generations.
    if "success" in df.columns:
        valid_mask = (
            df["success"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
    else:
        valid_mask = (
            df["title"].notna()
        )

    valid_indices = df.index[
        valid_mask
    ].tolist()

    if not valid_indices:
        print(
            f"No valid generations in "
            f"{csv_path.name}"
        )
        return

    references = (
        df.loc[
            valid_indices,
            "reference_title",
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    predictions = (
        df.loc[
            valid_indices,
            "title",
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # BLEU
    # --------------------------------------------------------

    bleu_values = [
        calculate_bleu(
            reference,
            prediction,
        )
        for reference, prediction
        in zip(
            references,
            predictions,
        )
    ]

    # --------------------------------------------------------
    # ROUGE-L
    # --------------------------------------------------------

    rouge_values = [
        calculate_rouge_l(
            reference,
            prediction,
        )
        for reference, prediction
        in zip(
            references,
            predictions,
        )
    ]

    # --------------------------------------------------------
    # BERTScore
    # --------------------------------------------------------

    print(
        f"Calculating BERTScore for "
        f"{csv_path.name}..."
    )

    bert_values = (
        calculate_bertscore(
            references,
            predictions,
        )
    )

    # Initialize columns.
    df["bleu"] = float("nan")
    df["rouge_l"] = float("nan")
    df["bert_score"] = float("nan")

    # Store metrics.
    df.loc[
        valid_indices,
        "bleu",
    ] = bleu_values

    df.loc[
        valid_indices,
        "rouge_l",
    ] = rouge_values

    df.loc[
        valid_indices,
        "bert_score",
    ] = bert_values

    # Save back to the same CSV.
    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Summary.
    print()
    print(
        f"Evaluated: {csv_path.name}"
    )

    print(
        f"Valid samples: "
        f"{len(valid_indices)}"
    )

    print(
        f"BLEU: "
        f"{sum(bleu_values) / len(bleu_values):.4f}"
    )

    print(
        f"ROUGE-L: "
        f"{sum(rouge_values) / len(rouge_values):.4f}"
    )

    print(
        f"BERTScore: "
        f"{sum(bert_values) / len(bert_values):.4f}"
    )


# ============================================================
# Evaluate all result files
# ============================================================

def evaluate_all():
    """
    Evaluate every benchmark result CSV.
    """

    files = sorted(
        RESULTS_DIR.glob(
            "results_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No result files found in "
            f"{RESULTS_DIR}"
        )

    for csv_path in files:
        evaluate_file(
            csv_path
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FW-TitleGen "
            "benchmark results."
        )
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help=(
            "Evaluate one CSV file. "
            "If omitted, all result "
            "files are evaluated."
        ),
    )

    args = parser.parse_args()

    if args.file:
        evaluate_file(
            args.file
        )
    else:
        evaluate_all()


if __name__ == "__main__":
    main()
