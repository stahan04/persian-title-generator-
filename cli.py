#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سیستم تولید خودکار عنوان سئو-فرندلی فارسی
با استفاده از Python، Typer، Pydantic، Rich و Hugging Face API (رایگان)
"""

import os
import random
import httpx
import typer
from enum import Enum
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from pydantic import BaseModel, Field, field_validator
from bidi.algorithm import get_display
import arabic_reshaper

# ============================================================
# تنظیمات اولیه
# ============================================================

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

app = typer.Typer(
    name="titlegen",
    help="تولید خودکار عنوان سئو-فرندلی فارسی با استفاده از هوش مصنوعی (رایگان)",
    add_completion=False,
)
console = Console()

# مدل‌های پشتیبانی‌شده روی Hugging Face (رایگان)
SUPPORTED_MODELS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen":  "Qwen/Qwen2.5-3B-Instruct",
    "gemma": "google/gemma-2-2b-it",
}

# ============================================================
# تعریف Enums و Models
# ============================================================

class Tone(str, Enum):
    """لحن‌های مختلف برای تولید عنوان"""
    FORMAL       = "formal"
    INFORMAL     = "informal"
    PROFESSIONAL = "professional"
    FRIENDLY     = "friendly"
    MODERN       = "modern"

    def get_persian_name(self) -> str:
        return {
            "formal":       "رسمی",
            "informal":     "خودمانی",
            "professional": "حرفه‌ای",
            "friendly":     "دوستانه",
            "modern":       "مدرن",
        }.get(self.value, "حرفه‌ای")


class ModelChoice(str, Enum):
    """مدل‌های زبانی قابل انتخاب"""
    LLAMA = "llama"
    QWEN  = "qwen"
    GEMMA = "gemma"


class TitleRequest(BaseModel):
    """مدل اعتبارسنجی ورودی‌های کاربر"""
    keyword:  str  = Field(..., min_length=2, max_length=100, description="کلیدواژه اصلی")
    topic:    str  = Field(..., min_length=2, max_length=200, description="موضوع محتوا")
    tone:     Tone = Field(Tone.PROFESSIONAL, description="لحن عنوان")
    language: str  = Field("fa", description="زبان خروجی")

    @field_validator("keyword")
    @classmethod
    def validate_keyword(cls, v: str) -> str:
        return v.strip()

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        return v.strip()


# ============================================================
# سرویس Hugging Face (رایگان)
# ============================================================

class HuggingFaceService:
    """سرویس ارتباط با Hugging Face Inference API (رایگان)"""

    BASE_URL = "https://api-inference.huggingface.co/models"

    def __init__(self, model_choice: ModelChoice = ModelChoice.QWEN):
        self.api_key = os.getenv("HF_API_KEY", "")
        self.model_id = SUPPORTED_MODELS[model_choice.value]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.client = httpx.Client(
            headers=headers,
            timeout=60.0,
        )

    def _build_prompt(self, request: TitleRequest) -> str:
        tone_persian = request.tone.get_persian_name()
        return (
            f"یک عنوان سئو-فرندلی و جذاب به زبان فارسی بنویس.\n\n"
            f"مشخصات:\n"
            f"- کلیدواژه: \"{request.keyword}\"\n"
            f"- موضوع: \"{request.topic}\"\n"
            f"- لحن: {tone_persian}\n\n"
            f"قوانین:\n"
            f"1. طول عنوان بین ۵۰ تا ۶۰ کاراکتر باشد\n"
            f"2. حتماً شامل کلیدواژه \"{request.keyword}\" باشد\n"
            f"3. جذاب و دارای CTR بالا باشد\n"
            f"4. فقط خود عنوان را بنویس، هیچ توضیح اضافه‌ای ندهید\n"
        )

    def _clean_title(self, title: str) -> str:
        """پردازش و تمیز کردن عنوان نهایی"""
        if not title:
            return ""

        title = title.replace("\n", " ").replace("\r", " ")
        title = " ".join(title.split())

        for prefix in ["عنوان:", "تولید شده:", "مثال:", "پاسخ:", "نتیجه:", "خروجی:"]:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()

        title = title.strip("\"'")
        title = title.replace(" :", ":").replace(" ،", "،").replace("  ", " ")

        if len(title) < 50:
            suffixes = [
                " [بررسی جامع و تخصصی]",
                " - از مبانی تا کاربردهای پیشرفته",
                " | راهنمای کامل و کاربردی",
                "؛ هر آنچه باید بدانید",
                " - آموزش گام به گام",
                " [به همراه مثال‌های عملی]",
            ]
            title = title + random.choice(suffixes)

        if len(title) > 60:
            for sep in [" - ", " | ", " : "]:
                if sep in title and len(title.split(sep)[0]) > 30:
                    title = title.split(sep)[0]
                    break
            if len(title) > 60:
                title = title[:57] + "..."

        return title.strip()

    def generate_title(self, request: TitleRequest) -> str:
        """تولید عنوان با Hugging Face Inference API"""
        prompt = self._build_prompt(request)

        try:
            url = f"{self.BASE_URL}/{self.model_id}"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 120,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False,
                },
            }
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and result:
                raw = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                raw = result.get("generated_text", "")
            else:
                raw = str(result)

            return self._clean_title(raw)

        except httpx.TimeoutException:
            raise Exception("زمان درخواست به پایان رسید. لطفاً مجدد تلاش کنید.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                raise Exception("مدل در حال بارگذاری است. چند ثانیه صبر کنید و دوباره تلاش کنید.")
            raise Exception(f"خطای سرویس: {e.response.status_code}")
        except Exception as e:
            raise Exception(f"خطا در تولید عنوان: {str(e)}")


# ============================================================
# توابع کمکی نمایش فارسی
# ============================================================

def print_persian(text: str, style: str = ""):
    if not text:
        return
    try:
        bidi_text = get_display(arabic_reshaper.reshape(text))
    except Exception:
        bidi_text = text
    if style:
        console.print(f"[{style}]{bidi_text}[/]")
    else:
        console.print(bidi_text)


def normalize_persian_input(text: str) -> str:
    if not text:
        return ""
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ").strip()
    return text


# ============================================================
# دستورات CLI
# ============================================================

@app.command()
def generate(
    keyword: str = typer.Argument(..., help="کلیدواژه اصلی"),
    topic:   str = typer.Argument(..., help="حوزه موضوعی"),
    tone:    Tone        = typer.Option(Tone.PROFESSIONAL, "--tone", "-t", help="لحن عنوان"),
    model:   ModelChoice = typer.Option(ModelChoice.QWEN,  "--model", "-m", help="مدل زبانی"),
    mock:    bool        = typer.Option(False, "--mock", help="حالت آزمایشی بدون API"),
):
    """
    تولید عنوان سئو-فرندلی فارسی با مدل‌های رایگان Hugging Face

    مثال:
        python cli.py generate "هوش مصنوعی" "تحول کسب‌وکار" --model qwen --tone professional
    """
    try:
        keyword = normalize_persian_input(keyword)
        topic   = normalize_persian_input(topic)

        console.print("\n[bold cyan]🎯 سیستم تولید عنوان سئو-فرندلی فارسی[/bold cyan]")
        console.print("[dim]" + "─" * 50 + "[/dim]\n")
        print_persian(f"📌 کلیدواژه: {keyword}", "yellow")
        print_persian(f"📖 موضوع: {topic}", "yellow")
        console.print(f"🎨 لحن: {tone.get_persian_name()}  |  🤖 مدل: {SUPPORTED_MODELS[model.value]}")

        request = TitleRequest(keyword=keyword, topic=topic, tone=tone)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("⏳ در حال تولید عنوان...", total=None)

            if mock:
                samples = [
                    f"نقش کلیدی {keyword} در {topic}: از چالش‌ها تا راهکارهای عملی",
                    f"بررسی جامع {keyword} و تأثیر آن بر {topic} در عصر دیجیتال",
                    f"{keyword} چیست؟ راهنمای کامل برای موفقیت در {topic}",
                ]
                title = random.choice(samples)
                title = HuggingFaceService()._clean_title(title)
            else:
                service = HuggingFaceService(model_choice=model)
                title = service.generate_title(request)

            progress.update(task, completed=True)

        console.print("\n[bold green]✨ عنوان پیشنهادی:[/bold green]")
        console.print("[cyan]" + "═" * 60 + "[/cyan]")
        print_persian(title, "bold cyan")
        console.print("[cyan]" + "═" * 60 + "[/cyan]")
        console.print(f"\n[dim]📏 طول عنوان: {len(title)} کاراکتر[/dim]")

        if len(title) < 50:
            console.print("[yellow]⚠️  کمتر از حد استاندارد (۵۰ کاراکتر)[/yellow]")
        elif len(title) <= 60:
            console.print("[green]✅ در محدوده استاندارد سئو (۵۰–۶۰ کاراکتر)[/green]")
        else:
            console.print("[yellow]⚠️  کمی بیشتر از حد استاندارد[/yellow]")

        console.print("\n[dim]" + "─" * 50 + "[/dim]")
        console.print("[green]✅ فرآیند تولید عنوان با موفقیت به پایان رسید.[/green]")

    except ValueError as e:
        console.print(f"[red]❌ خطا در اعتبارسنجی: {str(e)}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ خطا: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """نمایش اطلاعات و راهنمای استفاده"""
    console.print("\n[bold cyan]🤖 TitleGen - تولید عنوان سئو-فرندلی فارسی (رایگان)[/bold cyan]\n")
    console.print("این ابزار از مدل‌های رایگان Hugging Face استفاده می‌کند — بدون نیاز به پرداخت.\n")
    console.print("[bold]🤖 مدل‌های پشتیبانی‌شده:[/bold]")
    for k, v in SUPPORTED_MODELS.items():
        console.print(f"  • {k}: {v}")
    console.print("\n[bold]📝 مثال:[/bold]")
    console.print('  [dim]python cli.py generate "هوش مصنوعی" "تحول کسب‌وکار" --model qwen[/dim]')
    console.print('  [dim]python cli.py generate "دورکاری" "بهره‌وری" --mock[/dim]')


def main():
    app()


if __name__ == "__main__":
    main()
