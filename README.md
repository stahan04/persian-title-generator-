# FW-TitleGen

## A Unified Benchmark Framework for Evaluating Language Models in Persian SEO-Oriented Title Generation

FW-TitleGen is a provider-independent benchmarking framework designed for the systematic and reproducible evaluation of language models in Persian SEO-oriented title generation.

The framework provides a standardized experimental pipeline for title generation, quality evaluation, efficiency analysis, and comparative visualization. Its main contribution is the benchmarking framework and evaluation lifecycle rather than the development of a new language model.

---

## Benchmark Task

Given a Persian SEO keyword and its domain, each model is instructed to generate a single Persian SEO-oriented title under the same generation constraints:

- The target keyword must appear in the generated title.
- Exactly one Persian title must be generated.
- The title length must be between 50 and 70 characters.
- No explanation, Markdown, or additional text is allowed.

The reference title is used only for evaluation and is never provided to the model during generation.

---

## Models and Comparative Methods

The main benchmark evaluates:

- GPT-4o-mini
- Claude Sonnet 5
- Grok-4

The extended comparative evaluation also supports:

- Gemma 4 26B A4B IT
- Rule-Based baseline

All methods are evaluated using the same Persian SEO title-generation task and evaluation procedure.

---

## Dataset

The benchmark uses a Persian SEO title-generation dataset containing 100 samples.

Each sample contains:

- `id`
- `domain`
- `keyword`
- `reference_title`

During generation, only the domain and keyword are included in the model prompt. The reference title is retained exclusively for post-generation quality evaluation.

---

## Evaluation Metrics

FW-TitleGen performs multi-dimensional evaluation across generation quality and computational efficiency.

### Generation Quality

- **BLEU**
- **ROUGE-L**
- **BERTScore**

Hazm-based Persian tokenization is used for BLEU and ROUGE-L computation. BERTScore provides a semantic similarity measure between generated and reference titles.

### Efficiency

- **End-to-end latency**
- **Input tokens**
- **Output tokens**
- **Total tokens**

This combination allows models to be compared beyond a single quality metric.

---

## Framework Pipeline

```text
Persian SEO Dataset
        |
        v
Standardized Prompt Construction
        |
        v
Model Execution
        |
        v
Raw Benchmark Results
(results_*.csv)
        |
        v
Quality Evaluation
   |       |        |
 BLEU   ROUGE-L  BERTScore
        |
        v
Evaluated Results
(evaluated_results_*.csv)
        |
        v
Comparative Analysis
   |          |          |
Quality    Latency    Token Usage
        |
        v
Visualizations
```

---

## Framework Features

- Provider-independent benchmarking design
- Standardized prompt construction
- Multi-model evaluation
- Persian SEO-oriented title generation
- Checkpoint and resume support
- Automatic retry handling
- Execution logging
- Separation of raw and evaluated results
- Persian-aware quality evaluation
- Multi-dimensional performance analysis
- Automated CSV output
- Automated comparative visualization

---

## Project Structure

```text
FW-TitleGen/
|
|-- cli.py
|-- persian_seo_dataset.csv
|-- requirements.txt
|-- README.md
|
|-- evaluation/
|   |-- benchmark.py
|   |-- evaluate_results.py
|   |-- plot_results.py
|   |
|   `-- charts/
|
`-- results/
```

### `benchmark.py`

Executes the selected models using the standardized prompt and stores raw benchmark results.

### `evaluate_results.py`

Computes BLEU, Persian-aware ROUGE-L, and BERTScore.

Raw benchmark results are preserved, while evaluated results are written to separate `evaluated_results_*.csv` files.

### `plot_results.py`

Reads the evaluated result files and generates comparative figures.

### `cli.py`

Provides a simple command-line interface for running the benchmark.

---

## Visualizations

FW-TitleGen supports automatic generation of:

- BLEU comparison
- ROUGE-L comparison
- BERTScore comparison
- Average latency comparison
- Latency distribution
- Token usage comparison
- Quality-latency trade-off
- Multi-dimensional radar comparison

Generated figures are stored in:

```text
evaluation/charts/
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## API Configuration

API credentials must not be hard-coded in the source code.

Create a local `.env` file in the project root:

```text
AVALAI_API_KEY=your_api_key_here
```

The `.env` file is excluded from version control through `.gitignore`.

---

## Usage

### Run all main benchmark models

```bash
python cli.py --model all
```

### Run a single model

```bash
python cli.py --model gpt-4o-mini
```

```bash
python cli.py --model claude-sonnet-5
```

```bash
python cli.py --model grok-4
```

A limited test run can be performed using:

```bash
python cli.py --model all --limit 10
```

---

## Evaluate Results

After generation, calculate the quality metrics with:

```bash
python evaluation/evaluate_results.py
```

This produces separate evaluated result files containing:

```text
bleu
rouge_l
bert_score
```

---

## Generate Visualizations

After evaluation, run:

```bash
python evaluation/plot_results.py
```

The generated figures are saved in:

```text
evaluation/charts/
```

---

## Reproducibility and Security

FW-TitleGen separates model execution, evaluation, and visualization into independent stages.

Raw generation results are preserved separately from evaluated results, allowing metrics and figures to be regenerated without repeating API calls.

API credentials are loaded from environment variables and must never be committed to the repository.

---

## Research Purpose

FW-TitleGen was developed as a research framework for reproducible and multi-dimensional evaluation of language models in Persian SEO-oriented title generation.

Rather than introducing a new language model, the framework provides a reusable experimental pipeline for standardized model execution, quality assessment, efficiency analysis, and comparative evaluation.

---

## Author

**Sara Tahan**  
PhD Researcher in Artificial Intelligence  
Islamic Azad University of Qom
