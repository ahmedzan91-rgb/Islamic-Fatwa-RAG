# -*- coding: utf-8 -*-
"""
02_preprocessing.py
===================
المرحلة الثانية: تنظيف البيانات وتجهيز النصوص الشرعية.

فلسفة التنظيف في السياق الشرعي:
    النصوص الشرعية حسّاسة؛ الحذف المفرط يغيّر المعنى. لذلك نفصل بين:
      (أ) نص العرض (display_text): يُعرض للمستخدم ويُرسل للنموذج — تنظيف محافظ جداً
          (إزالة HTML، توحيد المسافات، تصحيح علامات الترقيم) مع الإبقاء على التشكيل والهمزات.
      (ب) نص البحث (search_text): يُستخدم لكشف التكرار والفهرسة النصية — تطبيع قوي
          (إزالة التشكيل، توحيد الألف/الهمزة/التاء المربوطة، إزالة التطويل).

المسؤوليات:
    1. إزالة وسوم HTML والكيانات ورموز التحكم.
    2. تطبيع المحارف العربية والأرقام.
    3. إزالة البسملة/الحمدلة المكرّرة في بداية كل فتوى (اختياري، مع الحفاظ عليها في العرض).
    4. حذف التكرارات (near-duplicates) عبر بصمة النص المطبّع.
    5. تصفية الفتاوى القصيرة جداً أو التالفة.
    6. بناء الحقل النهائي المُركّب (composite text) الذي سيدخل مرحلة التقطيع.

التشغيل:
    python 02_preprocessing.py --input artifacts/01_documents.parquet \
                               --output artifacts/02_clean.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import html
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

LOGGER = logging.getLogger("islamic_rag.preprocessing")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "artifacts", "01_documents.parquet")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "artifacts", "02_clean.parquet")


# ----------------------------------------------------------------------------- #
#                            التعبيرات النمطية                                    #
# ----------------------------------------------------------------------------- #

RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"[\w\.\-]+@[\w\.\-]+\.\w+")
RE_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
RE_TATWEEL = re.compile(r"\u0640+")                       # الكشيدة ـــ
RE_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0653-\u0655\u0640]")  # التشكيل
RE_MULTI_SPACE = re.compile(r"[ \t\u00A0\u200f\u200e]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_NON_ARABIC_NOISE = re.compile(r"[^\u0600-\u06FF0-9a-zA-Z\s\.\,\:\;\!\?\(\)\[\]\-«»\"'/%]")
RE_REPEATED_PUNCT = re.compile(r"([\.\،\؟\!\-])\1{2,}")

# عبارات افتتاحية متكرّرة في مواقع الفتاوى (تُحذف من نص البحث فقط)
OPENING_FORMULAS = [
    "بسم الله الرحمن الرحيم",
    "الحمد لله والصلاة والسلام على رسول الله وعلى آله وصحبه أما بعد",
    "الحمد لله والصلاة والسلام على رسول الله وعلى آله وصحبه، أما بعد",
    "الحمد لله رب العالمين والصلاة والسلام على نبينا محمد وعلى آله وصحبه أجمعين",
    "الحمد لله وحده والصلاة والسلام على من لا نبي بعده",
    "وبالله التوفيق",
    "والله أعلم",
]

# خرائط التطبيع
ARABIC_NORMALIZATION_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
    "ک": "ك", "ﻙ": "ك",
    "ی": "ي",
    "ۀ": "ه",
}

# تحويل الأرقام الهندية إلى عربية غربية
EASTERN_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
DIGIT_TRANSLATION = str.maketrans(EASTERN_DIGITS, WESTERN_DIGITS)
NORMALIZATION_TRANSLATION = str.maketrans(ARABIC_NORMALIZATION_MAP)


# ----------------------------------------------------------------------------- #
#                                  الإعدادات                                     #
# ----------------------------------------------------------------------------- #

@dataclass
class PreprocessConfig:
    """معاملات التحكم في خط أنابيب التنظيف."""

    min_answer_chars: int = 40         # أقل طول مقبول لنص الجواب
    max_answer_chars: int = 60_000     # قصّ النصوص المفرطة الطول
    remove_urls: bool = True
    remove_emails: bool = True
    drop_duplicates: bool = True
    strip_opening_formulas_in_search: bool = True
    keep_diacritics_in_display: bool = True


# ----------------------------------------------------------------------------- #
#                            دوال التنظيف الأساسية                                #
# ----------------------------------------------------------------------------- #

def strip_html(text: str) -> str:
    """إزالة وسوم HTML والكيانات (&nbsp; ...)."""
    if not text:
        return ""
    text = RE_SCRIPT_STYLE.sub(" ", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    return text


def clean_display_text(text: str, config: Optional[PreprocessConfig] = None) -> str:
    """
    تنظيف محافظ للنص المعروض: يحافظ على التشكيل والهمزات وعلامات الاقتباس القرآنية.
    """
    config = config or PreprocessConfig()
    if not isinstance(text, str) or not text:
        return ""

    text = strip_html(text)
    text = RE_CONTROL.sub(" ", text)

    if config.remove_urls:
        text = RE_URL.sub(" ", text)
    if config.remove_emails:
        text = RE_EMAIL.sub(" ", text)

    text = RE_TATWEEL.sub("", text)
    text = text.translate(DIGIT_TRANSLATION)
    text = RE_REPEATED_PUNCT.sub(r"\1", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)

    # تصحيح المسافات حول علامات الترقيم العربية
    text = re.sub(r"\s+([\.\،\؛\؟\!\:])", r"\1", text)
    text = re.sub(r"([\.\،\؛\؟\!\:])(?=[^\s\d])", r"\1 ", text)

    text = text.strip()
    if len(text) > (config.max_answer_chars or 60_000):
        text = text[: config.max_answer_chars].rsplit(" ", 1)[0] + " ..."
    return text


def normalize_arabic(text: str) -> str:
    """
    تطبيع قوي للنص العربي (لأغراض البحث والمقارنة فقط):
    إزالة التشكيل، توحيد الألف والياء والتاء المربوطة، إزالة الضوضاء.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = RE_DIACRITICS.sub("", text)
    text = text.translate(NORMALIZATION_TRANSLATION)
    text = text.translate(DIGIT_TRANSLATION)
    text = RE_NON_ARABIC_NOISE.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    return text.strip().lower()


def remove_opening_formulas(text: str) -> str:
    """إزالة العبارات الافتتاحية المكرّرة (من نص البحث فقط) لتقليل الضوضاء في التضمين."""
    if not text:
        return ""
    normalized_formulas = [normalize_arabic(f) for f in OPENING_FORMULAS]
    for formula in normalized_formulas:
        if formula and text.startswith(formula):
            text = text[len(formula):].strip(" ،.:")
    return text.strip()


def content_fingerprint(text: str) -> str:
    """بصمة MD5 للنص المطبّع — تُستخدم لكشف التكرار الحرفي وشبه الحرفي."""
    return hashlib.md5(normalize_arabic(text).encode("utf-8")).hexdigest()


def build_composite_text(row: pd.Series) -> str:
    """
    بناء النص المُركّب الذي سيُقطَّع لاحقاً.
    نضمّ السؤال والجواب لأن السؤال يحمل السياق الاستفهامي الذي يحسّن الاسترجاع.
    """
    parts: List[str] = []
    if row.get("title"):
        parts.append(f"العنوان: {row['title']}")
    if row.get("question"):
        parts.append(f"السؤال: {row['question']}")
    if row.get("answer"):
        parts.append(f"الجواب: {row['answer']}")
    return "\n\n".join(parts).strip()


# ----------------------------------------------------------------------------- #
#                              خط الأنابيب الرئيسي                                #
# ----------------------------------------------------------------------------- #

def preprocess_documents(
    documents: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
) -> pd.DataFrame:
    """
    تطبيق خط أنابيب التنظيف الكامل على DataFrame الوثائق.

    Returns:
        DataFrame منظّف يحتوي أعمدة إضافية:
            title_clean, question_clean, answer_clean, composite_text,
            search_text, fingerprint, char_count, word_count
    """
    config = config or PreprocessConfig()
    if documents.empty:
        LOGGER.warning("DataFrame المدخل فارغ.")
        return documents

    df = documents.copy()
    initial_count = len(df)
    LOGGER.info("بدء التنظيف على %d فتوى...", initial_count)

    # 1) التنظيف المحافظ للحقول النصية
    for src, dst in [
        ("title", "title_clean"),
        ("question", "question_clean"),
        ("answer", "answer_clean"),
    ]:
        source_series = df[src] if src in df.columns else pd.Series([""] * len(df))
        df[dst] = source_series.fillna("").astype(str).apply(
            lambda t: clean_display_text(t, config)
        )

    # 2) تصفية الفتاوى القصيرة/التالفة
    before = len(df)
    df = df[df["answer_clean"].str.len() >= config.min_answer_chars].reset_index(drop=True)
    LOGGER.info("حذف %d فتوى قصيرة (< %d محرف).", before - len(df), config.min_answer_chars)

    if df.empty:
        return df

    # 3) بناء النص المُركّب
    df["composite_text"] = df.apply(build_composite_text, axis=1)

    # 4) بناء نص البحث المطبّع
    df["search_text"] = df["composite_text"].apply(normalize_arabic)
    if config.strip_opening_formulas_in_search:
        df["search_text"] = df["search_text"].apply(remove_opening_formulas)

    # 5) كشف التكرار
    df["fingerprint"] = df["answer_clean"].apply(content_fingerprint)
    if config.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["fingerprint"], keep="first").reset_index(drop=True)
        LOGGER.info("حذف %d فتوى مكرّرة.", before - len(df))

    # 6) إحصاءات نصية للتوثيق الأكاديمي
    df["char_count"] = df["composite_text"].str.len()
    df["word_count"] = df["composite_text"].str.split().str.len().fillna(0).astype(int)

    LOGGER.info(
        "اكتمل التنظيف: %d → %d فتوى (نسبة الاحتفاظ %.1f%%).",
        initial_count, len(df), 100.0 * len(df) / max(initial_count, 1),
    )
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """تقرير جودة موجز يُستخدم في فصل المنهجية بالبحث الأكاديمي."""
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "المؤشر": [
            "عدد الفتاوى", "متوسط الكلمات", "الوسيط", "أقصر فتوى (كلمة)",
            "أطول فتوى (كلمة)", "عدد المصادر", "عدد التصنيفات",
        ],
        "القيمة": [
            len(df),
            round(float(df["word_count"].mean()), 1),
            int(df["word_count"].median()),
            int(df["word_count"].min()),
            int(df["word_count"].max()),
            int(df["source"].nunique()) if "source" in df.columns else 0,
            int(df["category"].nunique()) if "category" in df.columns else 0,
        ],
    })


def load_input(path: str) -> pd.DataFrame:
    """قراءة مخرجات المرحلة الأولى (Parquet أو CSV)."""
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".csv"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"لم يُعثر على ملف الإدخال: {path}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_output(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001
        path = os.path.splitext(path)[0] + ".csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
    LOGGER.info("تم حفظ %d فتوى منظّفة في: %s", len(df), path)
    return path


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="تنظيف وتجهيز نصوص الفتاوى.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--keep-duplicates", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = PreprocessConfig(
        min_answer_chars=args.min_chars,
        drop_duplicates=not args.keep_duplicates,
    )
    documents = load_input(args.input)
    cleaned = preprocess_documents(documents, config)
    if cleaned.empty:
        LOGGER.error("لم تنجُ أي فتوى من التنظيف — راجع معايير التصفية.")
        return 1

    save_output(cleaned, args.output)
    print("\n=== تقرير جودة البيانات ===")
    print(quality_report(cleaned).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
