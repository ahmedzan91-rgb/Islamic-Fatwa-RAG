# -*- coding: utf-8 -*-
"""
01_documents.py
===============
المرحلة الأولى من نظام الـ RAG الإسلامي: قراءة وتحميل ملفات الفتاوى (CSV).

المسؤوليات:
    1. اكتشاف كل ملفات CSV داخل مجلد البيانات (data/) بشكل تلقائي.
    2. قراءتها على دفعات (chunked reading) لدعم أكثر من 139 ألف فتوى دون استهلاك الذاكرة.
    3. الكشف التلقائي عن ترميز الملف (utf-8 / utf-8-sig / cp1256) وهو أمر شائع في البيانات العربية.
    4. توحيد أسماء الأعمدة المختلفة إلى مخطط (Schema) موحّد.
    5. حفظ النتيجة في ملف وسيط (Parquet / CSV) لتستهلكه المرحلة الثانية.

التشغيل من الطرفية:
    python 01_documents.py --input data --output artifacts/01_documents.parquet

الاستدعاء البرمجي:
    docs = load_documents("data")
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional

import pandas as pd

# ----------------------------------------------------------------------------- #
#                                  الإعدادات                                     #
# ----------------------------------------------------------------------------- #

LOGGER = logging.getLogger("islamic_rag.documents")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "artifacts", "01_documents.parquet"
)

# الترميزات المحتملة لملفات الفتاوى العربية، مرتّبة حسب الأولوية
CANDIDATE_ENCODINGS: List[str] = ["utf-8-sig", "utf-8", "cp1256", "windows-1256", "latin-1"]

# قاموس مطابقة أسماء الأعمدة: المفتاح = الاسم الموحّد، القيمة = الأسماء المحتملة في ملفات CSV
COLUMN_ALIASES: Dict[str, List[str]] = {
    "fatwa_id": [
        "fatwa_id", "id", "رقم الفتوى", "رقم_الفتوى", "fatwa_number", "no", "number",
        "الرقم", "معرف", "رقم",
    ],
    "title": [
        "title", "عنوان", "العنوان", "عنوان الفتوى", "عنوان_الفتوى", "subject", "الموضوع",
    ],
    "question": [
        "question", "سؤال", "السؤال", "نص السؤال", "نص_السؤال", "q", "query", "الاستفتاء",
        # مختصرات شائعة في مجموعات البيانات العربية المنشورة
        "ques", "quest", "questions", "question_text", "q_text", "qtext",
        "سؤال_الفتوى", "نص الاستفتاء", "السؤال الأصلي", "المسألة",
    ],
    "answer": [
        "answer", "جواب", "الجواب", "الإجابة", "الاجابة", "نص الجواب", "نص_الجواب",
        "a", "response", "الفتوى", "نص الفتوى", "content", "text", "النص",
        # مختصرات شائعة
        "ans", "answers", "answer_text", "a_text", "atext", "reply", "body",
        "الرد", "نص الإجابة", "الحكم", "فتوى", "المحتوى",
    ],
    "category": [
        "category", "قسم", "القسم", "التصنيف", "تصنيف", "section", "topic", "الموضوع الرئيسي",
        "باب", "الباب",
    ],
    "source": [
        "source", "مصدر", "المصدر", "الموقع", "site", "website", "المفتي", "mufti", "scholar",
        "الجهة",
    ],
    "date": [
        "date", "تاريخ", "التاريخ", "تاريخ الفتوى", "تاريخ_الفتوى", "published", "publish_date",
    ],
    "url": [
        "url", "link", "الرابط", "رابط", "المصدر الإلكتروني", "permalink",
    ],
}

# الأعمدة النهائية في المخطط الموحّد
UNIFIED_SCHEMA: List[str] = [
    "doc_id", "fatwa_id", "title", "question", "answer",
    "category", "source", "date", "url", "source_file",
]


# ----------------------------------------------------------------------------- #
#                                نموذج البيانات                                  #
# ----------------------------------------------------------------------------- #

@dataclass
class FatwaDocument:
    """تمثيل موحّد لوثيقة فتوى واحدة."""

    doc_id: str
    fatwa_id: str = ""
    title: str = ""
    question: str = ""
    answer: str = ""
    category: str = ""
    source: str = ""
    date: str = ""
    url: str = ""
    source_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadReport:
    """تقرير إحصائي عن عملية التحميل، يُستخدم للتوثيق الأكاديمي."""

    files_found: int = 0
    files_loaded: int = 0
    files_failed: List[str] = field(default_factory=list)
    rows_raw: int = 0
    rows_kept: int = 0
    encodings_used: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"الملفات المكتشفة: {self.files_found} | الملفات المحمّلة: {self.files_loaded} | "
            f"الصفوف الخام: {self.rows_raw:,} | الصفوف المقبولة: {self.rows_kept:,} | "
            f"الملفات الفاشلة: {len(self.files_failed)}"
        )


# ----------------------------------------------------------------------------- #
#                                دوال مساعدة                                     #
# ----------------------------------------------------------------------------- #

def _normalize_header(name: str) -> str:
    """تطبيع اسم العمود لمقارنته مع القاموس (إزالة المسافات والرموز والحالة)."""
    if name is None:
        return ""
    return (
        str(name)
        .replace("\ufeff", "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def build_column_mapping(columns: Iterable[str]) -> Dict[str, str]:
    """
    بناء خريطة {اسم العمود الأصلي -> الاسم الموحّد} اعتماداً على COLUMN_ALIASES.

    تُرجع خريطة جزئية؛ الأعمدة غير المعروفة تُتجاهل في المخطط النهائي.
    """
    normalized_aliases: Dict[str, str] = {}
    for unified, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_aliases[_normalize_header(alias)] = unified

    mapping: Dict[str, str] = {}
    taken: set = set()
    for col in columns:
        key = _normalize_header(col)
        unified = normalized_aliases.get(key)
        if unified and unified not in taken:
            mapping[col] = unified
            taken.add(unified)
    return mapping


def _open_maybe_gzip(path: str, encoding: str):
    """فتح الملف مع دعم شفّاف لـ gzip حسب الامتداد."""
    if path.lower().endswith(".gz"):
        import gzip  # noqa: PLC0415

        return gzip.open(path, "rt", encoding=encoding, errors="strict")
    return open(path, "r", encoding=encoding)


def _arabic_ratio(text: str) -> float:
    """نسبة المحارف العربية إلى مجموع الحروف — مؤشر صحة فكّ الترميز."""
    if not text:
        return 0.0
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    letters = sum(1 for ch in text if ch.isalpha())
    return arabic / max(letters, 1)


def detect_encoding(path: str) -> str:
    """
    الكشف عن ترميز الملف (مع دعم .gz).

    ملاحظة مهمة: لا يكفي تجريب الترميزات بالتتابع، لأن cp1256 و latin-1
    يفكّان **أي** تسلسل بايتات دون رمي استثناء، فيُختاران خطأً لملف UTF-8 سليم.
    لذلك نحكم على الجودة بنسبة المحارف العربية الناتجة، مع تفضيل UTF-8
    عند التقارب لأنه الأشيع في البيانات الحديثة.
    """
    # قراءة عيّنة بايتات (مع فكّ gzip إن لزم)
    try:
        if path.lower().endswith(".gz"):
            import gzip  # noqa: PLC0415

            with gzip.open(path, "rb") as fh:
                sample = fh.read(200_000)
        else:
            with open(path, "rb") as fh:
                sample = fh.read(200_000)
    except Exception:  # noqa: BLE001
        return "utf-8"

    if not sample:
        return "utf-8"

    # علامة BOM حاسمة
    if sample[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"

    # قصّ عند آخر سطر كامل لتفادي بتر محرف متعدد البايتات
    cut = sample.rfind(b"\n")
    if cut > 0:
        sample = sample[:cut]

    # ── الاختبار الحاسم: هل النصّ UTF-8 عربي في جوهره؟ ──
    # نفكّ بـ errors="replace" فلا يُسقطنا بايتٌ تالف واحد إلى ترميز خاطئ.
    # ملف مكشوط من الوِب قد يحوي بايتات فاسدة قليلة، وهذا لا يعني أنه cp1256.
    try:
        utf8_lenient = sample.decode("utf-8", errors="replace")
        bad_ratio = utf8_lenient.count("\ufffd") / max(len(utf8_lenient), 1)
        arabic = _arabic_ratio(utf8_lenient)
        # إن كان عربياً بوضوح والتلف طفيف (أقل من 2%) فهو UTF-8 قطعاً
        if arabic > 0.30 and bad_ratio < 0.02:
            return "utf-8"
    except Exception:  # noqa: BLE001
        pass

    best_encoding, best_score = "utf-8", -1.0
    for encoding in ("utf-8", "cp1256", "windows-1256", "latin-1"):
        try:
            decoded = sample.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # نعيد المحاولة بتسامح: البايتات التالفة القليلة لا تُبطل الترميز
            try:
                decoded = sample.decode(encoding, errors="replace")
            except Exception:  # noqa: BLE001
                continue

        score = _arabic_ratio(decoded)
        if encoding == "utf-8":
            score += 0.15  # مكافأة: UTF-8 هو الأشيع في البيانات الحديثة

        # عقوبة مضاعفة على محارف الاستبدال (دليل ترميز خاطئ)
        score -= 3.0 * decoded.count("\ufffd") / max(len(decoded), 1)

        # عقوبة على "الموجابيك": نمط Ø/Ù/Ã الناتج عن قراءة UTF-8 كـ cp1256
        mojibake = sum(decoded.count(ch) for ch in "ØÙÃÂðŸ™")
        score -= 2.0 * mojibake / max(len(decoded), 1)

        if score > best_score:
            best_encoding, best_score = encoding, score

    return best_encoding


def make_doc_id(source_file: str, fatwa_id: str, row_index: int, text: str) -> str:
    """
    توليد معرّف فريد ومستقر (deterministic) لكل فتوى.
    يعتمد على اسم الملف + رقم الفتوى + بصمة النص، لضمان عدم التصادم.
    """
    base = f"{os.path.basename(source_file)}::{fatwa_id}::{row_index}::{text[:160]}"
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:16]
    return f"fatwa_{digest}"


def discover_csv_files(input_path: str) -> List[str]:
    """اكتشاف ملفات CSV سواء كان المسار مجلداً أو ملفاً مفرداً."""
    if os.path.isfile(input_path):
        return [input_path]
    patterns = ["*.csv", "*.CSV", "*.tsv", "*.TSV", "*.csv.gz", "*.tsv.gz", "*.txt"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_path, "**", pattern), recursive=True))
    return sorted(set(files))


# ----------------------------------------------------------------------------- #
#                              المُحمِّل الرئيسي                                  #
# ----------------------------------------------------------------------------- #

def load_csv_file(
    path: str,
    chunksize: int = 20_000,
    max_rows: Optional[int] = None,
    report: Optional[LoadReport] = None,
) -> pd.DataFrame:
    """
    قراءة ملف CSV واحد وتحويله إلى المخطط الموحّد.

    Args:
        path: مسار ملف CSV.
        chunksize: حجم الدفعة عند القراءة (للملفات الضخمة).
        max_rows: حدّ أقصى للصفوف (مفيد أثناء التطوير/الاختبار).
        report: كائن التقرير لتحديث الإحصاءات.

    Returns:
        DataFrame بالمخطط الموحّد UNIFIED_SCHEMA.
    """
    encoding = detect_encoding(path)
    lowered = path.lower()
    base = lowered[:-3] if lowered.endswith(".gz") else lowered
    separator = "\t" if base.endswith(".tsv") else ","
    if report is not None:
        report.encodings_used[os.path.basename(path)] = encoding

    LOGGER.info("قراءة الملف: %s (الترميز: %s)", os.path.basename(path), encoding)

    collected: List[pd.DataFrame] = []
    total_rows = 0

    reader = pd.read_csv(
        path,
        encoding=encoding,
        sep=separator,
        compression="gzip" if lowered.endswith(".gz") else "infer",
        chunksize=chunksize,
        dtype=str,               # نقرأ كل شيء كنص لتفادي تحويل أرقام الفتاوى إلى float
        on_bad_lines="skip",     # تجاهل الأسطر التالفة بدل إيقاف العملية
        engine="python",
        keep_default_na=False,   # نمنع تحويل النص الفارغ إلى NaN
    )

    for chunk in reader:
        mapping = build_column_mapping(chunk.columns)
        if not mapping:
            LOGGER.warning("لا توجد أعمدة معروفة في %s — سيتم تجاهل الملف.", path)
            return pd.DataFrame(columns=UNIFIED_SCHEMA)

        chunk = chunk.rename(columns=mapping)

        # إضافة الأعمدة الناقصة كقيم فارغة
        for col in UNIFIED_SCHEMA:
            if col not in chunk.columns:
                chunk[col] = ""

        chunk["source_file"] = os.path.basename(path)
        chunk = chunk[UNIFIED_SCHEMA]
        collected.append(chunk)

        total_rows += len(chunk)
        if max_rows is not None and total_rows >= max_rows:
            break

    if not collected:
        return pd.DataFrame(columns=UNIFIED_SCHEMA)

    df = pd.concat(collected, ignore_index=True)
    if max_rows is not None:
        df = df.head(max_rows)

    if report is not None:
        report.rows_raw += len(df)

    # توليد المعرّفات الفريدة
    df["doc_id"] = [
        make_doc_id(path, str(fid), idx, str(ans))
        for idx, (fid, ans) in enumerate(zip(df["fatwa_id"], df["answer"]))
    ]
    return df


def load_documents(
    input_path: str = DEFAULT_DATA_DIR,
    max_rows_per_file: Optional[int] = None,
    drop_empty_answers: bool = True,
) -> pd.DataFrame:
    """
    تحميل كل ملفات الفتاوى ودمجها في DataFrame واحد بالمخطط الموحّد.

    Args:
        input_path: مجلد البيانات أو مسار ملف CSV.
        max_rows_per_file: حدّ أقصى للصفوف لكل ملف (None = بلا حدّ).
        drop_empty_answers: حذف الفتاوى التي لا تحتوي على نص إجابة.

    Returns:
        DataFrame موحّد يحتوي كل الفتاوى.
    """
    report = LoadReport()
    files = discover_csv_files(input_path)
    report.files_found = len(files)

    if not files:
        LOGGER.error("لم يتم العثور على أي ملف CSV في: %s", input_path)
        return pd.DataFrame(columns=UNIFIED_SCHEMA)

    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = load_csv_file(path, max_rows=max_rows_per_file, report=report)
            if not df.empty:
                frames.append(df)
                report.files_loaded += 1
        except Exception as exc:  # noqa: BLE001 — نريد الاستمرار رغم فشل ملف واحد
            LOGGER.exception("فشل تحميل الملف %s: %s", path, exc)
            report.files_failed.append(path)

    if not frames:
        return pd.DataFrame(columns=UNIFIED_SCHEMA)

    documents = pd.concat(frames, ignore_index=True)

    # تنظيف أولي خفيف: إزالة المسافات الزائدة من الحقول النصية
    for col in ["fatwa_id", "title", "question", "answer", "category", "source", "date", "url"]:
        documents[col] = documents[col].fillna("").astype(str).str.strip()

    if drop_empty_answers:
        before = len(documents)
        documents = documents[documents["answer"].str.len() > 0].reset_index(drop=True)
        LOGGER.info("حذف %d فتوى بدون نص إجابة.", before - len(documents))

    # إذا كان رقم الفتوى مفقوداً نستخدم معرّف الوثيقة كبديل
    documents.loc[documents["fatwa_id"] == "", "fatwa_id"] = documents["doc_id"]

    report.rows_kept = len(documents)
    LOGGER.info("تقرير التحميل: %s", report.summary())
    return documents


def save_documents(df: pd.DataFrame, output_path: str) -> str:
    """حفظ الوثائق في Parquet (أو CSV احتياطياً إذا لم تتوفر مكتبة pyarrow)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
    except Exception:  # noqa: BLE001
        output_path = os.path.splitext(output_path)[0] + ".csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        LOGGER.warning("تعذّر حفظ Parquet — تم الحفظ بصيغة CSV: %s", output_path)
    LOGGER.info("تم حفظ %d وثيقة في: %s", len(df), output_path)
    return output_path


# ----------------------------------------------------------------------------- #
#                                   CLI                                          #
# ----------------------------------------------------------------------------- #

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="تحميل ملفات الفتاوى (CSV) إلى مخطط موحّد.")
    parser.add_argument("--input", default=DEFAULT_DATA_DIR, help="مجلد البيانات أو ملف CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="مسار ملف الإخراج.")
    parser.add_argument("--max-rows", type=int, default=None, help="حدّ أقصى للصفوف لكل ملف.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    documents = load_documents(args.input, max_rows_per_file=args.max_rows)
    if documents.empty:
        LOGGER.error("لا توجد وثائق للحفظ. تأكد من وجود ملفات CSV في مجلد البيانات.")
        return 1

    save_documents(documents, args.output)

    print("\n=== عيّنة من الوثائق المحمّلة ===")
    with pd.option_context("display.max_colwidth", 60):
        print(documents[["fatwa_id", "title", "category", "source"]].head(5).to_string())
    print(f"\nإجمالي الفتاوى: {len(documents):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
