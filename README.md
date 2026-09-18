# FW-TitleGen

## A Unified Benchmark Framework for Evaluating Large Language Models in Persian SEO-Oriented Title Generation

FW-TitleGen is a provider-independent benchmarking framework designed for the systematic and reproducible evaluation of Large Language Models (LLMs) in Persian SEO-oriented title generation.

The framework provides a standardized pipeline for prompt execution, multi-model evaluation, metric computation, efficiency analysis, checkpointing, and automated reporting.

---

## Overview

FW-TitleGen focuses on evaluating LLMs rather than proposing a new language model.

The framework enables multiple models to be evaluated under the same dataset, prompt constraints, and evaluation procedure, providing a consistent basis for comparing generation quality, latency, token usage, and monetary cost.

### Main Benchmark Models

* GPT-4o-mini
* Claude Sonnet 5
* Grok-4

The main benchmark contains 100 Persian SEO title-generation samples.

---

## Generation Task

For each sample, the model receives a Persian SEO keyword and is instructed to generate a single Persian title satisfying the following constraints:

* Include the target keyword
* Produce one Persian title
* Keep the title between 50 and 70 characters
* Return no explanations or Markdown formatting

---

## Evaluation Metrics

FW-TitleGen uses multiple complementary metrics.

### Generation Quality

* BLEU
* ROUGE-L
* BERTScore

### Efficiency

* End-to-end latency
* Input tokens
* Output tokens
* Total tokens
* Monetary cost

Persian-aware tokenization is used where appropriate during evaluation.

---

## Framework Pipeline

Persian SEO Dataset
↓
Standardized Prompt
↓
Model Execution
├── GPT-4o-mini
├── Claude Sonnet 5
└── Grok-4
↓
Generated Titles
↓
Evaluation
├── BLEU
├── ROUGE-L
├── BERTScore
├── Latency
├── Token Usage
└── Cost
↓
Reports & Visualizations

---

## Framework Features

* Provider-independent architecture
* Standardized benchmarking lifecycle
* Multi-model evaluation
* Persian SEO-oriented generation
* Checkpoint and resume support
* Retry handling
* Execution logging
* Multi-dimensional evaluation
* Automated CSV and JSON outputs
* HTML reporting
* Automated visualization

---

## Visualizations

The framework can generate visual comparisons including:

* BLEU
* ROUGE-L
* BERTScore
* Latency
* Latency distribution
* Cost
* Quality-latency analysis
* Radar comparison

---

## Security

API credentials must never be hard-coded in source files.

Store credentials in environment variables or a local `.env` file.

Example:

`AVALAI_API_KEY=your_api_key_here`

The `.env` file must remain excluded from version control through `.gitignore`.

---

## Research Purpose

FW-TitleGen was developed as a research framework for studying the behavior and performance of LLMs under a standardized Persian title-generation task.

Its primary contribution is the reusable benchmarking pipeline and standardized evaluation lifecycle rather than a new language model.

---

## Author

**Sara Tahan**

PhD Researcher in Artificial Intelligence
Islamic Azad University of Qom

---

## Repository Status

This repository is being updated from an earlier single-model prototype to the current FW-TitleGen multi-model benchmarking framework.

Additional experimental components and comparative baselines may be added as the research develops.
