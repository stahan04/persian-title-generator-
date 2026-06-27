 
# Evaluating GPT-4o-mini for Persian SEO-Friendly Title Generation: An Empirical Study on 500 Samples

This repository contains the complete source code, dataset, and experimental results for the paper:

> **"Evaluating GPT-4o-mini for Persian SEO-Friendly Title Generation: An Empirical Study on 500 Samples"**

📌 **Status:** Currently under review at **Data Mining and Knowledge Discovery** (Springer Nature).

---

## 📁 Project Structure
.
├── evaluation/
│ ├── benchmark.py # Main benchmarking script (500 samples)
│ ├── plot_results.py # Plot generation script (for charts)
│ └── charts/ # Output directory for generated plots
├── results/ # CSV files with experimental results
├── persian_seo_dataset.csv # Dataset of 500 samples from 5 domains
├── requirements.txt # Python dependencies
├── cli.py # CLI tool for title generation
└── README.md # This file

text

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/stahan04/persian-title-generator.git
cd persian-title-generator
2. Install Dependencies
bash
pip install -r requirements.txt
3. Run Benchmark on a Subset
bash
# Test on 10 samples from all domains
python evaluation/benchmark.py --model gpt-4o-mini --limit 10

# Test on a specific domain (e.g., Technology & Digital)
python evaluation/benchmark.py --model gpt-4o-mini --domain "فناوری و دیجیتال" --limit 10

# Run benchmark on all 500 samples
python evaluation/benchmark.py --model gpt-4o-mini
4. Generate Evaluation Plots
bash
python evaluation/plot_results.py
Plots will be saved in evaluation/charts/.

5. Use the CLI for Single Title Generation
bash
python cli.py generate "هوش مصنوعی" "کاربردهای هوش مصنوعی در زندگی روزمره" --model qwen
🗝️ API Key Configuration
To run the benchmark with the actual GPT-4o-mini model, you need an API key from Avalai (or OpenAI).

Obtain your API key from your provider.

Open evaluation/benchmark.py.

Replace the placeholder API key in the file:

python
AVALAI_API_KEY = "your-api-key-here"
⚠️ Important: Ensure that your API key is never committed to the repository. Use .env files for production.

📊 Results Summary
Dataset: 500 samples from 5 domains (Technology, Health, Business, Education, Lifestyle).

Model: GPT-4o-mini (OpenAI).

Success Rate: 98.8% (500 out of 506 requests).

Keyword Inclusion Accuracy: 95.2%.

Length Compliance (50–60 chars): 100%.

Average Title Length: 55.9 ± 3.53 characters.

Mean Response Time: 1.32 ± 0.67 seconds.

Domain-Specific Performance
Domain	Keyword Inclusion	Avg Length	Response Time (s)
Technology & Digital	93.0%	55.9 ± 3.27	1.24 ± 0.83
Health & Medicine	100.0%	55.8 ± 3.63	1.30 ± 0.51
Business & Startups	88.0%	55.7 ± 3.68	1.31 ± 0.55
Education & Learning	97.0%	56.1 ± 3.51	1.41 ± 0.91
Lifestyle	98.0%	55.9 ± 3.59	1.32 ± 0.41
📄 Paper
The corresponding paper is currently under review at Data Mining and Knowledge Discovery (Springer Nature).

🔗 GitHub Repository: https://github.com/stahan04/persian-title-generator

📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

✍️ Author
Sara Tahan
Department of Computer Engineering, Qom Branch
Islamic Azad University, Qom, Iran
📧 s.tahan1365@gmail.com
🔗 GitHub

text

---
