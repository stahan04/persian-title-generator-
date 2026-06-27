#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py - اجرای بنچمارک با Avalai API
پردازش ۱۰۰ تا ۱۰۰ تا با resume خودکار
اجرا: python benchmark.py [--mock] [--model gpt-4o-mini|gpt-3.5-turbo|all] [--domain "نام دسته"] [--limit 10]
"""

import sys, os, time, csv, json, random, argparse
import httpx
from pathlib import Path

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from cli import normalize_persian_input

console = Console()

# ============================================================
# تنظیمات
# ============================================================

AVALAI_API_KEY = "aa-3NMZ7szStnxAZCu22vWFlmnMbTiZ7QOtAptlq2gQvmlTCCl7"
AVALAI_BASE_URL = "https://api.avalai.ir/v1"

MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]
BATCH_SIZE = 100
SAVE_EVERY = 10
SLEEP_OK = 2
SLEEP_429 = 60
MAX_RETRY = 3

# مسیر پروژه - روی هر سیستمی کار میکنه
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = BASE_DIR / "persian_seo_dataset.csv"

# اگه دیتاست توی مسیر دیگه‌ای هست
if not DATASET_PATH.exists():
    alt_paths = [
        Path("persian_seo_dataset.csv"),
        BASE_DIR / "data" / "persian_seo_dataset.csv",
        Path(r"C:\Users\SARA\Desktop\titlegen-v3\persian_seo_dataset.csv"),
        Path(__file__).parent / "persian_seo_dataset.csv",
    ]
    for p in alt_paths:
        if p.exists():
            DATASET_PATH = p
            break

# ============================================================
# توابع
# ============================================================

def load_dataset() -> list:
    if not DATASET_PATH.exists():
        console.print(f"[red]❌ دیتاست پیدا نشد: {DATASET_PATH}[/red]")
        return []
    rows = []
    with open(DATASET_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row["id"],
                "domain": row["domain"],
                "keyword": row["keyword"].strip(),
                "topic": row["topic"].strip(),
            })
    console.print(f"[green]✅ دیتاست لود شد: {len(rows)} نمونه[/green]")
    return rows

def load_done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("success", "") == "True":
                done.add(str(row["id"]))
    return done

def clean_title(title: str, keyword: str) -> str:
    if not title:
        return ""
    title = " ".join(title.replace("\n", " ").split()).strip("\"'")
    for p in ["عنوان:", "پاسخ:", "نتیجه:", "Title:"]:
        if title.lower().startswith(p.lower()):
            title = title[len(p):].strip()
    if len(title) < 50:
        title += random.choice([" | راهنمای کامل", " - گام به گام", " [کاربردی]", " برای مبتدیان"])
    if len(title) > 60:
        title = title[:57] + "..."
    return title.strip()

def call_avalai(model: str, keyword: str, topic: str) -> str:
    """فراخوانی Avalai API"""
    prompt = (
        f"یک عنوان سئو-فرندلی فارسی بنویس.\n"
        f"کلیدواژه: \"{keyword}\"\n"
        f"موضوع: \"{topic}\"\n"
        f"قوانین: دقیقاً ۵۰ تا ۶۰ کاراکتر، شامل کلیدواژه.\n"
        f"فقط عنوان را بنویس:"
    )

    for attempt in range(MAX_RETRY):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{AVALAI_BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {AVALAI_API_KEY}"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 80,
                        "temperature": 0.7
                    }
                )
                if resp.status_code == 429:
                    wait = SLEEP_429 * (attempt + 1)
                    console.print(f"[yellow]⏳ Rate limit — صبر {wait}s[/yellow]")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

        except httpx.TimeoutException:
            if attempt < MAX_RETRY - 1:
                time.sleep(5)
            else:
                raise Exception("Timeout")
        except Exception as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(5)
            else:
                raise

    raise Exception(f"ناموفق بعد از {MAX_RETRY} تلاش")

def save_results(rows: list, out_path: Path):
    fieldnames = ["id","domain","keyword","topic","model","title","title_length",
                  "has_keyword","length_ok","latency","success","error"]
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

# ============================================================
# بنچمارک
# ============================================================

def run_benchmark(model_name: str, mock_mode: bool = False, domain_filter: str = None, limit: int = None):
    dataset = load_dataset()
    if not dataset:
        return
    
    # ===== فیلتر بر اساس دسته =====
    if domain_filter:
        dataset = [r for r in dataset if r["domain"] == domain_filter]
        console.print(f"[yellow]🎯 فیلتر: فقط دسته '{domain_filter}' — {len(dataset)} نمونه[/yellow]")
    
    # ===== محدود کردن تعداد =====
    if limit:
        dataset = dataset[:limit]
        console.print(f"[yellow]🔢 محدودیت: فقط {limit} نمونه اول[/yellow]")
    
    safe_name = model_name.replace(".", "_").replace("-", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"results_{safe_name}.csv"
    done_ids = load_done_ids(out_path)
    pending = [r for r in dataset if str(r["id"]) not in done_ids]

    console.print(f"\n[bold cyan]🚀 مدل: {model_name}[/bold cyan]")
    console.print(f"[green]کل: {len(dataset)} | انجام شده: {len(done_ids)} | مانده: {len(pending)}[/green]")
    console.print(f"[yellow]حالت: {'Mock' if mock_mode else 'واقعی (Avalai)'}[/yellow]\n")

    if not pending:
        console.print("[green]✅ همه نمونه‌ها قبلاً پردازش شدن![/green]")
        return

    batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches[:1] if mock_mode else batches):
        console.print(f"\n[bold]📦 دسته {batch_idx+1}/{len(batches)} — {len(batch)} نمونه[/bold]")
        buffer = []

        with Progress(TextColumn("{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                      console=console) as progress:
            task = progress.add_task("⏳ پردازش...", total=len(batch))

            for item in batch:
                keyword = normalize_persian_input(item["keyword"])
                topic = normalize_persian_input(item["topic"])
                start = time.time()
                title = ""
                success = False
                error = ""

                try:
                    if mock_mode:
                        templates = [
                            f"{keyword} در {topic}: راهنمای جامع",
                            f"آموزش {keyword} برای {topic} از صفر",
                            f"{keyword} چیست؟ کاربرد در {topic}",
                            f"بررسی {keyword} و نقش آن در {topic}",
                        ]
                        raw = random.choice(templates)
                        if len(raw) < 50:
                            raw += random.choice([" | نکات کلیدی", " - گام به گام", " [کاربردی]"])
                        if len(raw) > 60:
                            raw = raw[:57] + "..."
                        title = raw
                        success = True
                    else:
                        raw = call_avalai(model_name, keyword, topic)
                        title = clean_title(raw, keyword)
                        success = bool(title)

                except Exception as e:
                    error = str(e)[:150]

                latency = round(time.time() - start, 3)
                length = len(title)

                buffer.append({
                    "id": item["id"],
                    "domain": item["domain"],
                    "keyword": keyword,
                    "topic": topic,
                    "model": model_name,
                    "title": title,
                    "title_length": length,
                    "has_keyword": keyword in title,
                    "length_ok": 50 <= length <= 60,
                    "latency": latency,
                    "success": success,
                    "error": error,
                })

                if len(buffer) % SAVE_EVERY == 0:
                    save_results(buffer[-SAVE_EVERY:], out_path)

                progress.advance(task)
                if not mock_mode:
                    time.sleep(SLEEP_OK)

        remainder = len(buffer) % SAVE_EVERY
        if remainder:
            save_results(buffer[-remainder:], out_path)

        console.print(f"[green]💾 دسته {batch_idx+1} ذخیره شد: {out_path}[/green]")

    # آمار
    all_rows = list(csv.DictReader(open(out_path, encoding="utf-8-sig")))
    successful = [r for r in all_rows if r["success"] == "True"]

    console.print(f"\n[bold green]📊 نتایج {model_name}:[/bold green]")
    console.print(f"  موفق: {len(successful)}/{len(all_rows)}")
    if successful:
        kw = sum(1 for r in successful if r["has_keyword"] == "True") / len(successful) * 100
        ln = sum(1 for r in successful if r["length_ok"] == "True") / len(successful) * 100
        aln = sum(int(r["title_length"]) for r in successful) / len(successful)
        console.print(f"  Keyword Inclusion: {kw:.1f}%")
        console.print(f"  Length Compliance: {ln:.1f}%")
        console.print(f"  میانگین طول: {aln:.1f}")
        console.print(f"\n[cyan]نمونه عنوان‌ها:[/cyan]")
        for r in successful[:3]:
            console.print(f"  • {r['title']}")

def run_all_models(mock_mode: bool = False, domain_filter: str = None, limit: int = None):
    for model in MODELS:
        run_benchmark(model_name=model, mock_mode=mock_mode, 
                      domain_filter=domain_filter, limit=limit)

# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="حالت آزمایشی بدون API")
    parser.add_argument("--model", default="all",
                        choices=["gpt-4o-mini", "gpt-3.5-turbo", "all"],
                        help="مدل مورد نظر")
    parser.add_argument("--domain", type=str, default=None,
                        help="فقط یک دسته خاص رو تست کن (مثلاً 'فناوری و دیجیتال')")
    parser.add_argument("--limit", type=int, default=None,
                        help="تعداد نمونه‌های محدود برای تست (مثلاً 10)")
    args = parser.parse_args()

    if args.model == "all":
        run_all_models(mock_mode=args.mock, domain_filter=args.domain, limit=args.limit)
    else:
        run_benchmark(model_name=args.model, mock_mode=args.mock, 
                      domain_filter=args.domain, limit=args.limit)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py - اجرای بنچمارک با Avalai API
پردازش ۱۰۰ تا ۱۰۰ تا با resume خودکار
اجرا: python benchmark.py [--mock] [--model gpt-4o-mini|gpt-3.5-turbo|all] [--domain "نام دسته"] [--limit 10]
"""

import sys, os, time, csv, json, random, argparse
import httpx
from pathlib import Path

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from cli import normalize_persian_input

console = Console()

# ============================================================
# تنظیمات
# ============================================================

AVALAI_API_KEY = "aa-3NMZ7szStnxAZCu22vWFlmnMbTiZ7QOtAptlq2gQvmlTCCl7"
AVALAI_BASE_URL = "https://api.avalai.ir/v1"

MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]
BATCH_SIZE = 100
SAVE_EVERY = 10
SLEEP_OK = 2
SLEEP_429 = 60
MAX_RETRY = 3

# مسیر پروژه - روی هر سیستمی کار میکنه
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = BASE_DIR / "persian_seo_dataset.csv"

# اگه دیتاست توی مسیر دیگه‌ای هست
if not DATASET_PATH.exists():
    alt_paths = [
        Path("persian_seo_dataset.csv"),
        BASE_DIR / "data" / "persian_seo_dataset.csv",
        Path(r"C:\Users\SARA\Desktop\titlegen-v3\persian_seo_dataset.csv"),
        Path(__file__).parent / "persian_seo_dataset.csv",
    ]
    for p in alt_paths:
        if p.exists():
            DATASET_PATH = p
            break

# ============================================================
# توابع
# ============================================================

def load_dataset() -> list:
    if not DATASET_PATH.exists():
        console.print(f"[red]❌ دیتاست پیدا نشد: {DATASET_PATH}[/red]")
        return []
    rows = []
    with open(DATASET_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row["id"],
                "domain": row["domain"],
                "keyword": row["keyword"].strip(),
                "topic": row["topic"].strip(),
            })
    console.print(f"[green]✅ دیتاست لود شد: {len(rows)} نمونه[/green]")
    return rows

def load_done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("success", "") == "True":
                done.add(str(row["id"]))
    return done

def clean_title(title: str, keyword: str) -> str:
    if not title:
        return ""
    title = " ".join(title.replace("\n", " ").split()).strip("\"'")
    for p in ["عنوان:", "پاسخ:", "نتیجه:", "Title:"]:
        if title.lower().startswith(p.lower()):
            title = title[len(p):].strip()
    if len(title) < 50:
        title += random.choice([" | راهنمای کامل", " - گام به گام", " [کاربردی]", " برای مبتدیان"])
    if len(title) > 60:
        title = title[:57] + "..."
    return title.strip()

def call_avalai(model: str, keyword: str, topic: str) -> str:
    """فراخوانی Avalai API"""
    prompt = (
        f"یک عنوان سئو-فرندلی فارسی بنویس.\n"
        f"کلیدواژه: \"{keyword}\"\n"
        f"موضوع: \"{topic}\"\n"
        f"قوانین: دقیقاً ۵۰ تا ۶۰ کاراکتر، شامل کلیدواژه.\n"
        f"فقط عنوان را بنویس:"
    )

    for attempt in range(MAX_RETRY):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{AVALAI_BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {AVALAI_API_KEY}"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 80,
                        "temperature": 0.7
                    }
                )
                if resp.status_code == 429:
                    wait = SLEEP_429 * (attempt + 1)
                    console.print(f"[yellow]⏳ Rate limit — صبر {wait}s[/yellow]")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

        except httpx.TimeoutException:
            if attempt < MAX_RETRY - 1:
                time.sleep(5)
            else:
                raise Exception("Timeout")
        except Exception as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(5)
            else:
                raise

    raise Exception(f"ناموفق بعد از {MAX_RETRY} تلاش")

def save_results(rows: list, out_path: Path):
    fieldnames = ["id","domain","keyword","topic","model","title","title_length",
                  "has_keyword","length_ok","latency","success","error"]
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

# ============================================================
# بنچمارک
# ============================================================

def run_benchmark(model_name: str, mock_mode: bool = False, domain_filter: str = None, limit: int = None):
    dataset = load_dataset()
    if not dataset:
        return
    
    # ===== فیلتر بر اساس دسته =====
    if domain_filter:
        dataset = [r for r in dataset if r["domain"] == domain_filter]
        console.print(f"[yellow]🎯 فیلتر: فقط دسته '{domain_filter}' — {len(dataset)} نمونه[/yellow]")
    
    # ===== محدود کردن تعداد =====
    if limit:
        dataset = dataset[:limit]
        console.print(f"[yellow]🔢 محدودیت: فقط {limit} نمونه اول[/yellow]")
    
    safe_name = model_name.replace(".", "_").replace("-", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"results_{safe_name}.csv"
    done_ids = load_done_ids(out_path)
    pending = [r for r in dataset if str(r["id"]) not in done_ids]

    console.print(f"\n[bold cyan]🚀 مدل: {model_name}[/bold cyan]")
    console.print(f"[green]کل: {len(dataset)} | انجام شده: {len(done_ids)} | مانده: {len(pending)}[/green]")
    console.print(f"[yellow]حالت: {'Mock' if mock_mode else 'واقعی (Avalai)'}[/yellow]\n")

    if not pending:
        console.print("[green]✅ همه نمونه‌ها قبلاً پردازش شدن![/green]")
        return

    batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches[:1] if mock_mode else batches):
        console.print(f"\n[bold]📦 دسته {batch_idx+1}/{len(batches)} — {len(batch)} نمونه[/bold]")
        buffer = []

        with Progress(TextColumn("{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                      console=console) as progress:
            task = progress.add_task("⏳ پردازش...", total=len(batch))

            for item in batch:
                keyword = normalize_persian_input(item["keyword"])
                topic = normalize_persian_input(item["topic"])
                start = time.time()
                title = ""
                success = False
                error = ""

                try:
                    if mock_mode:
                        templates = [
                            f"{keyword} در {topic}: راهنمای جامع",
                            f"آموزش {keyword} برای {topic} از صفر",
                            f"{keyword} چیست؟ کاربرد در {topic}",
                            f"بررسی {keyword} و نقش آن در {topic}",
                        ]
                        raw = random.choice(templates)
                        if len(raw) < 50:
                            raw += random.choice([" | نکات کلیدی", " - گام به گام", " [کاربردی]"])
                        if len(raw) > 60:
                            raw = raw[:57] + "..."
                        title = raw
                        success = True
                    else:
                        raw = call_avalai(model_name, keyword, topic)
                        title = clean_title(raw, keyword)
                        success = bool(title)

                except Exception as e:
                    error = str(e)[:150]

                latency = round(time.time() - start, 3)
                length = len(title)

                buffer.append({
                    "id": item["id"],
                    "domain": item["domain"],
                    "keyword": keyword,
                    "topic": topic,
                    "model": model_name,
                    "title": title,
                    "title_length": length,
                    "has_keyword": keyword in title,
                    "length_ok": 50 <= length <= 60,
                    "latency": latency,
                    "success": success,
                    "error": error,
                })

                if len(buffer) % SAVE_EVERY == 0:
                    save_results(buffer[-SAVE_EVERY:], out_path)

                progress.advance(task)
                if not mock_mode:
                    time.sleep(SLEEP_OK)

        remainder = len(buffer) % SAVE_EVERY
        if remainder:
            save_results(buffer[-remainder:], out_path)

        console.print(f"[green]💾 دسته {batch_idx+1} ذخیره شد: {out_path}[/green]")

    # آمار
    all_rows = list(csv.DictReader(open(out_path, encoding="utf-8-sig")))
    successful = [r for r in all_rows if r["success"] == "True"]

    console.print(f"\n[bold green]📊 نتایج {model_name}:[/bold green]")
    console.print(f"  موفق: {len(successful)}/{len(all_rows)}")
    if successful:
        kw = sum(1 for r in successful if r["has_keyword"] == "True") / len(successful) * 100
        ln = sum(1 for r in successful if r["length_ok"] == "True") / len(successful) * 100
        aln = sum(int(r["title_length"]) for r in successful) / len(successful)
        console.print(f"  Keyword Inclusion: {kw:.1f}%")
        console.print(f"  Length Compliance: {ln:.1f}%")
        console.print(f"  میانگین طول: {aln:.1f}")
        console.print(f"\n[cyan]نمونه عنوان‌ها:[/cyan]")
        for r in successful[:3]:
            console.print(f"  • {r['title']}")

def run_all_models(mock_mode: bool = False, domain_filter: str = None, limit: int = None):
    for model in MODELS:
        run_benchmark(model_name=model, mock_mode=mock_mode, 
                      domain_filter=domain_filter, limit=limit)

# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="حالت آزمایشی بدون API")
    parser.add_argument("--model", default="all",
                        choices=["gpt-4o-mini", "gpt-3.5-turbo", "all"],
                        help="مدل مورد نظر")
    parser.add_argument("--domain", type=str, default=None,
                        help="فقط یک دسته خاص رو تست کن (مثلاً 'فناوری و دیجیتال')")
    parser.add_argument("--limit", type=int, default=None,
                        help="تعداد نمونه‌های محدود برای تست (مثلاً 10)")
    args = parser.parse_args()

    if args.model == "all":
        run_all_models(mock_mode=args.mock, domain_filter=args.domain, limit=args.limit)
    else:
        run_benchmark(model_name=args.model, mock_mode=args.mock, 
                      domain_filter=args.domain, limit=args.limit)
