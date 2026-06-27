#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/plot_results.py
رسم نمودارهای مقایسه‌ای برای همه مدل‌ها
اجرا: python evaluation/plot_results.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# تنظیمات ظاهری
# ============================================================
plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 150
sns.set_theme(style="whitegrid")

COLORS    = {
    "qwen": "#2196F3",
    "llama": "#FF5722",
    "gemma": "#4CAF50",
    "gpt_4o_mini": "#9C27B0"
}
MODEL_MAP = {
    "qwen": "Qwen2.5",
    "llama": "Llama3.2",
    "gemma": "Gemma2",
    "gpt_4o_mini": "GPT-4o-mini"
}

EVAL_DIR    = Path(__file__).parent
# ✅ مسیر درست: پوشه results در سطح پروژه
RESULTS_DIR = EVAL_DIR.parent / "results"
CHARTS_DIR  = EVAL_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ============================================================
# لود داده‌ها
# ============================================================

def load_results() -> dict:
    """لود نتایج همه مدل‌ها (شامل GPT-4o-mini)"""
    results = {}
    
    for model in ["qwen", "llama", "gemma", "gpt_4o_mini"]:
        path = RESULTS_DIR / f"results_{model}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            df = df[df["success"] == True].copy()
            results[model] = df
            print(f"✅ {model}: {len(df)} نمونه لود شد")
        else:
            print(f"⚠️  {model}: فایل پیدا نشد ({path})")

    return results


# ============================================================
# نمودارها
# ============================================================

def plot_length_histogram(results: dict):
    """نمودار ۱: توزیع طول عنوان — همه مدل‌ها"""
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (model, df) in zip(axes, results.items()):
        ax.hist(df["title_length"], bins=20, color=COLORS.get(model, "#888"),
                alpha=0.85, edgecolor="white")
        ax.axvline(50, color="red",    linestyle="--", linewidth=2, label="SEO Min (50)")
        ax.axvline(60, color="orange", linestyle="--", linewidth=2, label="SEO Max (60)")
        ax.axvline(df["title_length"].mean(), color="blue", linestyle="-.",
                   linewidth=2, label=f"Mean: {df['title_length'].mean():.1f}")
        ax.set_title(f"{MODEL_MAP.get(model, model)} — Title Length", fontsize=12, fontweight="bold")
        ax.set_xlabel("Title Length (characters)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of Generated Title Lengths", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = CHARTS_DIR / "length_histogram.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 ذخیره شد: {path}")


def plot_latency_histogram(results: dict):
    """نمودار ۲: توزیع زمان پاسخ"""
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (model, df) in zip(axes, results.items()):
        sns.histplot(df["latency"], bins=15, kde=True,
                     color=COLORS.get(model, "#888"), ax=ax, edgecolor="white")
        ax.axvline(df["latency"].mean(), color="red", linestyle="--",
                   linewidth=2, label=f"Mean: {df['latency'].mean():.2f}s")
        ax.set_title(f"{MODEL_MAP.get(model, model)} — Response Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Response Time (seconds)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of System Response Time by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = CHARTS_DIR / "latency_histogram.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 ذخیره شد: {path}")


def plot_success_rate(results: dict):
    """نمودار ۳: نرخ موفقیت — نمودار میله‌ای"""
    model_names, rates = [], []
    for model in results:
        path = RESULTS_DIR / f"results_{model}.csv"
        if path.exists():
            df_all = pd.read_csv(path, encoding="utf-8-sig")
            rate = df_all["success"].mean() * 100
            model_names.append(MODEL_MAP.get(model, model))
            rates.append(rate)

    if not model_names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(model_names, rates,
                  color=[COLORS.get(m, "#888") for m in results],
                  edgecolor="white", width=0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("System Success Rate by Model", fontsize=14, fontweight="bold")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = CHARTS_DIR / "success_rate.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 ذخیره شد: {path}")


def plot_keyword_inclusion(results: dict):
    """نمودار ۴: Keyword Inclusion Rate به تفکیک حوزه"""
    if not any("domain" in df.columns for df in results.values()):
        print("⚠️  ستون 'domain' در داده‌ها وجود ندارد، نمودار ۴ رسم نشد.")
        return

    all_dfs = []
    for model, df in results.items():
        tmp = df[["domain", "has_keyword"]].copy()
        tmp["model"] = MODEL_MAP.get(model, model)
        all_dfs.append(tmp)

    combined = pd.concat(all_dfs, ignore_index=True)
    pivot = combined.groupby(["domain", "model"])["has_keyword"].mean().unstack() * 100

    if pivot.empty:
        print("⚠️  داده‌ای برای نمودار Keyword Inclusion وجود ندارد.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax,
               color=[COLORS.get(m.lower().replace("2.5","").replace("3.2","").replace("2","").replace("_mini",""), "#888")
                      for m in pivot.columns],
               width=0.7, edgecolor="white")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Keyword Inclusion Rate (%)", fontsize=12)
    ax.set_title("Keyword Inclusion Rate by Domain and Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Domain", fontsize=10)
    ax.legend(title="Model", fontsize=9)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    path = CHARTS_DIR / "keyword_inclusion_by_domain.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 ذخیره شد: {path}")


def plot_length_boxplot(results: dict):
    """نمودار ۵: باکس‌پلات طول عنوان — مقایسه مدل‌ها"""
    all_dfs = []
    for model, df in results.items():
        tmp = df[["title_length"]].copy()
        tmp["Model"] = MODEL_MAP.get(model, model)
        all_dfs.append(tmp)

    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=combined, x="Model", y="title_length",
                hue="Model", legend=False,
                palette={MODEL_MAP.get(m, m): COLORS.get(m, "#888") for m in results},
                ax=ax)
    ax.axhline(50, color="red",    linestyle="--", linewidth=2, label="SEO Min (50)")
    ax.axhline(60, color="orange", linestyle="--", linewidth=2, label="SEO Max (60)")
    ax.set_ylabel("Title Length (characters)", fontsize=12)
    ax.set_title("Title Length Distribution by Model", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = CHARTS_DIR / "length_boxplot.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 ذخیره شد: {path}")


def plot_radar(results: dict):
    """نمودار ۶: رادار — مقایسه کلی معیارها"""
    if not results:
        return

    metrics = {
        "Keyword\nInclusion": lambda df: df["has_keyword"].mean(),
        "Length\nCompliance": lambda df: df["length_ok"].mean() if "length_ok" in df.columns
                              else ((df["title_length"] >= 50) & (df["title_length"] <= 60)).mean(),
        "Success\nRate":      lambda df: 1.0,
        "Avg Length\nScore":  lambda df: 1 - abs(df["title_length"].mean() - 55) / 55,
        "Coverage":           lambda df: (df["title_length"] > 0).mean(),
    }

    labels = list(metrics.keys())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, df in results.items():
        values = [fn(df) for fn in metrics.values()]
        values += values[:1]
        color = COLORS.get(model, "#888")
        ax.plot(angles, values, "o-", linewidth=2,
                label=MODEL_MAP.get(model, model), color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison — Persian SEO Title Generation",
                 size=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    plt.tight_layout()
    path = CHARTS_DIR / "radar_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 ذخیره شد: {path}")


# ============================================================
# نقطه ورود
# ============================================================

def main():
    print("\n🎨 شروع رسم نمودارها...\n")

    results = load_results()

    if not results:
        print("❌ هیچ نتیجه‌ای پیدا نشد.")
        print("ابتدا benchmark.py را اجرا کنید:")
        print("  python evaluation/benchmark.py --mock --model all")
        return

    print(f"\n📂 {len(results)} مدل لود شد: {list(results.keys())}\n")

    plot_length_histogram(results)
    plot_latency_histogram(results)
    plot_success_rate(results)
    plot_keyword_inclusion(results)
    plot_length_boxplot(results)
    plot_radar(results)

    print(f"\n✅ همه نمودارها در پوشه '{CHARTS_DIR}' ذخیره شدند.")
    print("\nلیست فایل‌های تولیدشده:")
    for f in sorted(CHARTS_DIR.glob("*.png")):
        print(f"  📈 {f.name}")


if __name__ == "__main__":
    main()
