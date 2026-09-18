#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FW-TitleGen visualization module.

Generates comparative figures for:
- BLEU
- ROUGE-L
- BERTScore
- Latency
- Latency distribution
- Token usage
- Quality-latency trade-off
- Multi-dimensional radar analysis

Cost is intentionally excluded because reliable
historical pricing information is not available.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = Path(__file__).resolve().parent / "charts"

CHARTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Model configuration
# ============================================================

MODEL_FILES = {
    "GPT-4o-mini":
        "evaluated_results_gpt_4o_mini.csv",

    "Claude Sonnet 5":
        "evaluated_results_claude_sonnet_5.csv",

    "Grok-4":
        "evaluated_results_grok_4.csv",

    "Gemma 4 26B A4B IT":
        "evaluated_results_gemma_4_26b_a4b_it.csv",

    "Rule-Based":
        "evaluated_results_rule_based.csv",
}


# ============================================================
# Load results
# ============================================================

def load_results():

    results = {}

    for model_name, filename in MODEL_FILES.items():

        path = RESULTS_DIR / filename

        if not path.exists():
            print(
                f"Missing: {path}"
            )
            continue

        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        if "success" in df.columns:

            success = (
                df["success"]
                .astype(str)
                .str.lower()
                .eq("true")
            )

            df = df[
                success
            ].copy()

        if not df.empty:

            results[
                model_name
            ] = df

            print(
                f"Loaded {model_name}: "
                f"{len(df)} samples"
            )

    return results


# ============================================================
# Helpers
# ============================================================

def metric_means(
    results,
    metric,
):

    models = []
    values = []

    for model, df in results.items():

        if metric not in df.columns:
            continue

        numeric = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

        if numeric.notna().any():

            models.append(
                model
            )

            values.append(
                numeric.mean()
            )

    return models, values


def save_bar_chart(
    models,
    values,
    ylabel,
    title,
    filename,
):

    if not models:

        print(
            f"Skipped {filename}: "
            "metric unavailable."
        )

        return

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        models,
        values,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.4f}",

            ha="center",
            va="bottom",
        )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {filename}"
    )


# ============================================================
# BLEU
# ============================================================

def plot_bleu(results):

    models, values = metric_means(
        results,
        "bleu",
    )

    save_bar_chart(
        models,
        values,
        "BLEU",
        "BLEU by Model",
        "bleu.png",
    )


# ============================================================
# ROUGE-L
# ============================================================

def find_rouge_metric(results):

    candidates = [
        "rouge_l",
        "rouge",
        "rougeL",
        "rouge-l",
    ]

    for candidate in candidates:

        if any(
            candidate in df.columns
            for df in results.values()
        ):
            return candidate

    return None


def plot_rouge(results):

    metric = find_rouge_metric(
        results
    )

    if metric is None:

        print(
            "Skipped rouge.png: "
            "ROUGE-L unavailable."
        )

        return

    models, values = metric_means(
        results,
        metric,
    )

    save_bar_chart(
        models,
        values,
        "ROUGE-L",
        "ROUGE-L by Model",
        "rouge.png",
    )


# ============================================================
# BERTScore
# ============================================================

def find_bert_metric(results):

    candidates = [
        "bert_score",
        "bertscore",
        "bert_f1",
    ]

    for candidate in candidates:

        if any(
            candidate in df.columns
            for df in results.values()
        ):
            return candidate

    return None


def plot_bertscore(results):

    metric = find_bert_metric(
        results
    )

    if metric is None:

        print(
            "Skipped bert_score.png: "
            "BERTScore unavailable."
        )

        return

    models, values = metric_means(
        results,
        metric,
    )

    save_bar_chart(
        models,
        values,
        "BERTScore",
        "BERTScore by Model",
        "bert_score.png",
    )


# ============================================================
# Latency
# ============================================================

def plot_latency(results):

    models, values = metric_means(
        results,
        "latency",
    )

    save_bar_chart(
        models,
        values,
        "Latency (seconds)",
        "Average End-to-End Latency",
        "latency.png",
    )


def plot_latency_boxplot(results):

    data = []
    labels = []

    for model, df in results.items():

        if "latency" not in df.columns:
            continue

        values = pd.to_numeric(
            df["latency"],
            errors="coerce",
        ).dropna()

        if not values.empty:

            data.append(
                values.values
            )

            labels.append(
                model
            )

    if not data:

        print(
            "Skipped latency_boxplot.png."
        )

        return

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.boxplot(
        data,
        labels=labels,
    )

    ax.set_ylabel(
        "Latency (seconds)"
    )

    ax.set_title(
        "Latency Distribution by Model"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR
        / "latency_boxplot.png",

        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved: latency_boxplot.png"
    )


# ============================================================
# Token usage
# ============================================================

def plot_tokens(results):

    models, values = metric_means(
        results,
        "total_tokens",
    )

    save_bar_chart(
        models,
        values,
        "Total Tokens",
        "Average Token Usage",
        "token_usage.png",
    )


# ============================================================
# Quality vs Latency
# ============================================================

def plot_scatter(results):

    bert_metric = find_bert_metric(
        results
    )

    if bert_metric is None:

        print(
            "Skipped scatter.png: "
            "BERTScore unavailable."
        )

        return

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    plotted = False

    for model, df in results.items():

        if (
            "latency" not in df.columns
            or bert_metric not in df.columns
        ):
            continue

        latency = pd.to_numeric(
            df["latency"],
            errors="coerce",
        ).mean()

        quality = pd.to_numeric(
            df[bert_metric],
            errors="coerce",
        ).mean()

        if (
            pd.isna(latency)
            or pd.isna(quality)
        ):
            continue

        ax.scatter(
            latency,
            quality,
            s=100,
        )

        ax.annotate(
            model,
            (
                latency,
                quality,
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

        plotted = True

    if not plotted:

        plt.close()

        print(
            "Skipped scatter.png."
        )

        return

    ax.set_xlabel(
        "Average Latency (seconds)"
    )

    ax.set_ylabel(
        "Average BERTScore"
    )

    ax.set_title(
        "Quality-Latency Trade-off"
    )

    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR
        / "scatter.png",

        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved: scatter.png"
    )


# ============================================================
# Radar
# ============================================================

def normalize(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = np.nanmin(
        values
    )

    maximum = np.nanmax(
        values
    )

    if maximum == minimum:

        return np.ones_like(
            values
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def plot_radar(results):

    bert_metric = find_bert_metric(
        results
    )

    if bert_metric is None:

        print(
            "Skipped radar.png: "
            "BERTScore unavailable."
        )

        return

    required = [
        bert_metric,
        "latency",
        "total_tokens",
    ]

    usable = {}

    for model, df in results.items():

        if not all(
            column in df.columns
            for column in required
        ):
            continue

        usable[model] = {
            "Quality":
                pd.to_numeric(
                    df[bert_metric],
                    errors="coerce",
                ).mean(),

            "Latency":
                pd.to_numeric(
                    df["latency"],
                    errors="coerce",
                ).mean(),

            "Tokens":
                pd.to_numeric(
                    df["total_tokens"],
                    errors="coerce",
                ).mean(),
        }

    if len(usable) < 2:

        print(
            "Skipped radar.png: "
            "insufficient metrics."
        )

        return

    models = list(
        usable.keys()
    )

    metrics = [
        "Quality",
        "Latency",
        "Tokens",
    ]

    matrix = np.array(
        [
            [
                usable[model][metric]
                for metric in metrics
            ]
            for model in models
        ],
        dtype=float,
    )

    normalized = np.zeros_like(
        matrix,
        dtype=float,
    )

    for index, metric in enumerate(
        metrics
    ):

        column = normalize(
            matrix[:, index]
        )

        # Lower latency and token usage
        # are better.
        if metric in [
            "Latency",
            "Tokens",
        ]:

            column = (
                1 - column
            )

        normalized[
            :, index
        ] = column

    angles = np.linspace(
        0,
        2 * np.pi,
        len(metrics),
        endpoint=False,
    ).tolist()

    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={
            "polar": True
        },
    )

    for model_index, model in enumerate(
        models
    ):

        values = (
            normalized[
                model_index
            ]
            .tolist()
        )

        values += values[:1]

        ax.plot(
            angles,
            values,
            marker="o",
            label=model,
        )

        ax.fill(
            angles,
            values,
            alpha=0.1,
        )

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        metrics
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        "Multi-Dimensional Model Comparison"
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(
            1.35,
            1.1,
        ),
    )

    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR
        / "radar.png",

        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved: radar.png"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\nFW-TitleGen Visualization\n"
    )

    results = load_results()

    if not results:

        print(
            "No benchmark results found."
        )

        return

    plot_bleu(
        results
    )

    plot_rouge(
        results
    )

    plot_bertscore(
        results
    )

    plot_latency(
        results
    )

    plot_latency_boxplot(
        results
    )

    plot_tokens(
        results
    )

    plot_scatter(
        results
    )

    plot_radar(
        results
    )

    print(
        f"\nCharts directory: "
        f"{CHARTS_DIR}"
    )


if __name__ == "__main__":
    main()
