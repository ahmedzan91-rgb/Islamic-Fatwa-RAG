# -*- coding: utf-8 -*-
"""
08_data_manager.py
==================
وحدة إدارة رفع ملفات الفتاوى من الواجهة وحفظها بشكل دائم.

المسؤوليات:
    1. استقبال الملفات المرفوعة من Streamlit (CSV / TSV / XLSX / ZIP / GZ).
    2. التحقق من صحتها ومعاينة أعمدتها قبل الاعتماد.
    3. حفظها في مجلد data/ بأسماء آمنة (sanitized) دون الكتابة فوق ملف قائم.
    4. إدارة سجلّ الملفات (metadata) لعرضه في الواجهة.
    5. حذف الملفات وتنظيف المساحة.

╔══════════════════════════════════════════════════════════════════════════╗
║  ⚠️ تنبيه مهم على ديمومة الملفات (File Persistence)                        ║
╚══════════════════════════════════════════════════════════════════════════╝
    قرص Streamlit Community Cloud **مؤقّت (ephemeral)**:
    الملفات المرفوعة تبقى ما دامت الحاوية تعمل، لكنها تُمحى عند:
        • إعادة تشغيل التطبيق (reboot)
        • دفع تحديث جديد إلى المستودع
        • دخول التطبيق في سبات لعدم الاستخدام

    لذلك توفّر هذه الوحدة `PersistenceMode` لإعلام المستخدم بوضوح،
    وتقترح رفع البيانات إلى المستودع مباشرة للاستخدام الدائم.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger("islamic_rag.data_manager")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
REGISTRY_FILE = os.path.join(DEFAULT_DATA_DIR, "_uploads_registry.json")

ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".zip", ".gz"}
MAX_FILE_SIZE_MB = 500

RE_UNSAFE_CHARS = re.compile(r"[^\w\u0600-\u06FF.\-]+")


# ----------------------------------------------------------------------------- #
#                                نماذج البيانات                                  #
# ----------------------------------------------------------------------------- #

@dataclass
class UploadedFileInfo:
    """بيانات ملف مرفوع."""

    filename: str
    path: str
    size_bytes: int
    uploaded_at: str
    rows: int = 0
    columns: List[str] = field(default_factory=list)
    detected_encoding: str = ""
    checksum: str = ""
    status: str = "ok"
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def size_human(self) -> str:
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


@dataclass
class ValidationResult:
    """نتيجة فحص ملف قبل اعتماده."""

    is_valid: bool
    rows: int = 0
    columns: List[str] = field(default_factory=list)
    mapped_columns: Dict[str, str] = field(default_factory=dict)
    missing_critical: List[str] = field(default_factory=list)
    encoding: str = "utf-8"
    preview: Optional[pd.DataFrame] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- #
#                                دوال مساعدة                                     #
# ----------------------------------------------------------------------------- #

def sanitize_filename(name: str) -> str:
    """تنظيف اسم الملف من المحارف الخطرة مع الإبقاء على العربية."""
    name = os.path.basename(str(name)).strip()
    name = name.replace(" ", "_")
    name = RE_UNSAFE_CHARS.sub("_", name)
    name = re.sub(r"_{2,}", "_", name).strip("._")
    return name or f"upload_{int(time.time())}.csv"


def unique_path(directory: str, filename: str) -> str:
    """
    توليد مسار فريد لتفادي الكتابة فوق ملف موجود.
    مثال: fatwas.csv → fatwas_1.csv → fatwas_2.csv
    """
    os.makedirs(directory, exist_ok=True)
    candidate = os.path.join(directory, filename)
    if not os.path.exists(candidate):
        return candidate

    stem, ext = os.path.splitext(filename)
    for i in range(1, 1000):
        candidate = os.path.join(directory, f"{stem}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
    return os.path.join(directory, f"{stem}_{int(time.time())}{ext}")


def compute_checksum(data: bytes) -> str:
    """بصمة MD5 لكشف رفع الملف نفسه مرتين."""
    return hashlib.md5(data).hexdigest()[:16]


def _arabic_ratio(text: str) -> float:
    """نسبة المحارف العربية في النص — مؤشر على صحة فكّ الترميز."""
    if not text:
        return 0.0
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    letters = sum(1 for ch in text if ch.isalpha())
    return arabic / max(letters, 1)


def detect_encoding_from_bytes(data: bytes) -> str:
    """
    كشف ترميز البيانات العربية.

    ملاحظة مهمة: لا يكفي تجريب الترميزات بالتتابع، لأن cp1256 و latin-1
    يفكّان **أي** تسلسل بايتات دون خطأ، فيُختاران خطأً لملف UTF-8 سليم.
    لذلك نحكم على الجودة بنسبة المحارف العربية الناتجة، ونفضّل UTF-8
    عند التساوي لأنه الأشيع في البيانات الحديثة.
    """
    sample = data[:200_000]
    # قصّ عند آخر سطر كامل لتفادي بتر محرف متعدد البايتات
    cut = sample.rfind(b"\n")
    if cut > 0:
        sample = sample[:cut]

    # علامة BOM حاسمة
    if data[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"

    best_encoding, best_score = "utf-8", -1.0
    for encoding in ("utf-8", "cp1256", "windows-1256", "latin-1"):
        try:
            decoded = sample.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

        score = _arabic_ratio(decoded)
        # مكافأة UTF-8: إن فُكّ بلا خطأ فهو غالباً الصحيح
        if encoding == "utf-8":
            score += 0.15
        # عقوبة على محارف الاستبدال ورموز التحكم
        score -= decoded.count("\ufffd") / max(len(decoded), 1)

        if score > best_score:
            best_encoding, best_score = encoding, score

    return best_encoding


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ----------------------------------------------------------------------------- #
#                                فحص الملفات                                     #
# ----------------------------------------------------------------------------- #

def validate_tabular_bytes(
    data: bytes,
    filename: str,
    column_aliases: Optional[Dict[str, List[str]]] = None,
    preview_rows: int = 5,
) -> ValidationResult:
    """
    فحص ملف جدولي قبل اعتماده: قراءة عيّنة، كشف الأعمدة، مطابقتها بالمخطط الموحّد.

    Args:
        data: محتوى الملف بالبايتات.
        filename: اسم الملف (لتحديد الصيغة).
        column_aliases: قاموس مطابقة الأعمدة من الوحدة 01.
        preview_rows: عدد صفوف المعاينة.
    """
    result = ValidationResult(is_valid=False)
    extension = os.path.splitext(filename)[1].lower()

    payload = data
    if extension == ".gz":
        try:
            import gzip
            payload = gzip.decompress(data)
            filename = filename[:-3]
            extension = os.path.splitext(filename)[1].lower()
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"تعذّر فكّ ضغط gzip: {exc}")
            return result

    try:
        if extension in (".xlsx", ".xls"):
            frame = pd.read_excel(io.BytesIO(payload), dtype=str, nrows=2000)
            result.encoding = "binary/excel"
        else:
            encoding = detect_encoding_from_bytes(payload)
            result.encoding = encoding
            separator = "\t" if extension == ".tsv" else ","
            frame = pd.read_csv(
                io.BytesIO(payload),
                encoding=encoding,
                sep=separator,
                dtype=str,
                nrows=2000,
                on_bad_lines="skip",
                engine="python",
                keep_default_na=False,
            )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"تعذّرت قراءة الملف: {exc}")
        return result

    if frame.empty:
        result.errors.append("الملف فارغ أو لا يحتوي صفوفاً صالحة.")
        return result

    result.columns = [str(c) for c in frame.columns]
    result.preview = frame.head(preview_rows)

    # تقدير عدد الصفوف الكلي (قرأنا عيّنة فقط)
    if extension not in (".xlsx", ".xls"):
        try:
            result.rows = max(0, payload.count(b"\n") - 1)
        except Exception:  # noqa: BLE001
            result.rows = len(frame)
    else:
        result.rows = len(frame)

    # مطابقة الأعمدة بالمخطط الموحّد
    if column_aliases:
        normalized: Dict[str, str] = {}
        for unified, aliases in column_aliases.items():
            for alias in aliases:
                key = str(alias).strip().lower().replace(" ", "_").replace("-", "_")
                normalized[key] = unified

        for column in result.columns:
            key = (
                str(column).replace("\ufeff", "").strip().lower()
                .replace(" ", "_").replace("-", "_")
            )
            unified = normalized.get(key)
            if unified and unified not in result.mapped_columns.values():
                result.mapped_columns[column] = unified

        # العمود الحرج الوحيد هو نص الجواب
        if "answer" not in result.mapped_columns.values():
            result.missing_critical.append("answer (الجواب / نص الفتوى)")
        if "question" not in result.mapped_columns.values():
            result.warnings.append(
                "لم يُعثر على عمود السؤال — سيعتمد الاسترجاع على نص الجواب وحده."
            )

    if result.missing_critical:
        result.errors.append(
            "لم يُعثر على عمود نص الفتوى/الجواب. "
            "أعِد تسمية العمود أو أضِفه إلى COLUMN_ALIASES في 01_documents.py."
        )
        return result

    result.is_valid = True
    return result


# ----------------------------------------------------------------------------- #
#                                 حفظ الملفات                                    #
# ----------------------------------------------------------------------------- #

def extract_zip_bytes(data: bytes, target_dir: str) -> List[str]:
    """فكّ ضغط ZIP واستخراج الملفات الجدولية فقط."""
    saved: List[str] = []
    os.makedirs(target_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                extension = os.path.splitext(member)[1].lower()
                if extension not in (".csv", ".tsv", ".txt", ".xlsx", ".xls"):
                    continue
                # حماية من Zip Slip
                safe_name = sanitize_filename(os.path.basename(member))
                if not safe_name:
                    continue
                destination = unique_path(target_dir, safe_name)
                with archive.open(member) as source, open(destination, "wb") as out:
                    shutil.copyfileobj(source, out)
                saved.append(destination)
    except zipfile.BadZipFile:
        LOGGER.error("ملف ZIP تالف.")
    return saved


def save_uploaded_bytes(
    data: bytes,
    filename: str,
    data_dir: str = DEFAULT_DATA_DIR,
    validate: bool = True,
    column_aliases: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, str, Optional[UploadedFileInfo]]:
    """
    حفظ ملف مرفوع في مجلد البيانات.

    Returns:
        (نجح؟, رسالة, بيانات الملف)
    """
    os.makedirs(data_dir, exist_ok=True)

    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"صيغة غير مدعومة: {extension}", None

    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"الملف كبير جداً ({size_mb:.0f} MB > {MAX_FILE_SIZE_MB} MB)", None

    # معالجة ZIP بشكل خاص
    if extension == ".zip":
        extracted = extract_zip_bytes(data, data_dir)
        if not extracted:
            return False, "لم يُعثر على ملفات جدولية داخل الأرشيف.", None
        register_upload_batch(extracted, data_dir)
        return True, f"استُخرج {len(extracted)} ملف من الأرشيف.", None

    # الفحص قبل الحفظ
    validation: Optional[ValidationResult] = None
    if validate and extension not in (".gz",):
        validation = validate_tabular_bytes(data, filename, column_aliases)
        if not validation.is_valid:
            return False, " | ".join(validation.errors), None

    safe_name = sanitize_filename(filename)
    destination = unique_path(data_dir, safe_name)

    try:
        with open(destination, "wb") as fh:
            fh.write(data)
    except Exception as exc:  # noqa: BLE001
        return False, f"فشل الحفظ: {exc}", None

    info = UploadedFileInfo(
        filename=os.path.basename(destination),
        path=destination,
        size_bytes=len(data),
        uploaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rows=validation.rows if validation else 0,
        columns=validation.columns if validation else [],
        detected_encoding=validation.encoding if validation else "",
        checksum=compute_checksum(data),
    )
    append_to_registry(info, data_dir)
    LOGGER.info("حُفظ الملف: %s (%s)", info.filename, info.size_human)
    return True, f"تم حفظ «{info.filename}» بنجاح.", info


# ----------------------------------------------------------------------------- #
#                                 سجلّ الملفات                                    #
# ----------------------------------------------------------------------------- #

def _registry_path(data_dir: str = DEFAULT_DATA_DIR) -> str:
    return os.path.join(data_dir, "_uploads_registry.json")


def load_registry(data_dir: str = DEFAULT_DATA_DIR) -> List[dict]:
    """قراءة سجلّ الملفات المرفوعة."""
    path = _registry_path(data_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return []


def save_registry(records: List[dict], data_dir: str = DEFAULT_DATA_DIR) -> None:
    os.makedirs(data_dir, exist_ok=True)
    try:
        with open(_registry_path(data_dir), "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("تعذّر حفظ السجلّ: %s", exc)


def append_to_registry(info: UploadedFileInfo, data_dir: str = DEFAULT_DATA_DIR) -> None:
    records = load_registry(data_dir)
    records = [r for r in records if r.get("filename") != info.filename]
    records.append(info.to_dict())
    save_registry(records, data_dir)


def register_upload_batch(paths: List[str], data_dir: str = DEFAULT_DATA_DIR) -> None:
    """تسجيل مجموعة ملفات (بعد فكّ أرشيف مثلاً)."""
    for path in paths:
        try:
            info = UploadedFileInfo(
                filename=os.path.basename(path),
                path=path,
                size_bytes=os.path.getsize(path),
                uploaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            append_to_registry(info, data_dir)
        except Exception:  # noqa: BLE001
            continue


# ----------------------------------------------------------------------------- #
#                              إدارة مجلد البيانات                                #
# ----------------------------------------------------------------------------- #

def list_data_files(data_dir: str = DEFAULT_DATA_DIR) -> List[Dict[str, Any]]:
    """سرد كل ملفات البيانات الموجودة فعلياً على القرص."""
    if not os.path.isdir(data_dir):
        return []

    registry = {r.get("filename"): r for r in load_registry(data_dir)}
    files: List[Dict[str, Any]] = []

    for name in sorted(os.listdir(data_dir)):
        if name.startswith(("_", ".")):
            continue
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        extension = os.path.splitext(name)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            continue

        record = registry.get(name, {})
        try:
            size = os.path.getsize(path)
            modified = datetime.fromtimestamp(os.path.getmtime(path))
        except OSError:
            continue

        files.append({
            "الملف": name,
            "الحجم": human_size(size),
            "الصفوف (تقديري)": record.get("rows", "—") or "—",
            "الترميز": record.get("detected_encoding", "—") or "—",
            "تاريخ الإضافة": record.get("uploaded_at") or modified.strftime("%Y-%m-%d %H:%M"),
            "_path": path,
            "_size_bytes": size,
        })
    return files


def delete_data_file(path: str, data_dir: str = DEFAULT_DATA_DIR) -> Tuple[bool, str]:
    """حذف ملف بيانات مع تحديث السجلّ (مع حماية من الخروج عن المجلد)."""
    try:
        real_path = os.path.realpath(path)
        real_dir = os.path.realpath(data_dir)
        if not real_path.startswith(real_dir + os.sep):
            return False, "مسار غير مسموح."
        if not os.path.isfile(real_path):
            return False, "الملف غير موجود."

        name = os.path.basename(real_path)
        os.remove(real_path)
        save_registry(
            [r for r in load_registry(data_dir) if r.get("filename") != name], data_dir
        )
        return True, f"حُذف «{name}»."
    except Exception as exc:  # noqa: BLE001
        return False, f"فشل الحذف: {exc}"


def clear_data_directory(data_dir: str = DEFAULT_DATA_DIR) -> Tuple[int, int]:
    """حذف كل ملفات البيانات. يُرجع (عدد المحذوف, البايتات المحرَّرة)."""
    count, freed = 0, 0
    for entry in list_data_files(data_dir):
        ok, _ = delete_data_file(entry["_path"], data_dir)
        if ok:
            count += 1
            freed += entry["_size_bytes"]
    return count, freed


def data_directory_stats(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """إحصاءات موجزة عن مجلد البيانات."""
    files = list_data_files(data_dir)
    total_bytes = sum(f["_size_bytes"] for f in files)
    rows = [f["الصفوف (تقديري)"] for f in files if isinstance(f["الصفوف (تقديري)"], int)]
    return {
        "files": len(files),
        "total_size": human_size(total_bytes),
        "total_bytes": total_bytes,
        "estimated_rows": sum(rows) if rows else 0,
    }


# ----------------------------------------------------------------------------- #
#                            كشف بيئة التشغيل                                     #
# ----------------------------------------------------------------------------- #

def is_ephemeral_environment() -> bool:
    """
    كشف ما إذا كنا على بيئة سحابية بقرص مؤقّت (Streamlit Cloud).
    يُستخدم لعرض التحذير المناسب للمستخدم.
    """
    indicators = [
        os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud",
        os.path.exists("/mount/src"),           # مسار Streamlit Cloud المميّز
        "STREAMLIT_SHARING_MODE" in os.environ,
        os.environ.get("HOSTNAME", "").startswith("streamlit"),
    ]
    return any(indicators)


def persistence_note() -> Tuple[str, str]:
    """
    إرجاع (المستوى, الرسالة) لوصف ديمومة الملفات في البيئة الحالية.
    المستوى: "warning" أو "success".
    """
    if is_ephemeral_environment():
        return (
            "warning",
            "⚠️ **تنبيه على الديمومة:** أنت تعمل على Streamlit Cloud حيث القرص مؤقّت. "
            "الملفات المرفوعة تبقى ما دام التطبيق يعمل، لكنها **تُمحى** عند إعادة التشغيل "
            "أو دفع تحديث للمستودع. للاستخدام الدائم ارفع ملفات CSV إلى مجلد `data/` "
            "في مستودع GitHub مباشرة، أو ابنِ الفهرس محلياً وارفع مجلد `chroma_db/`.",
        )
    return (
        "success",
        "✅ **تشغيل محلي:** الملفات المرفوعة تُحفظ بشكل دائم في مجلد `data/`.",
    )


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def main() -> int:
    """عرض حالة مجلد البيانات."""
    print("=== حالة مجلد البيانات ===")
    stats = data_directory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("\n=== الملفات ===")
    files = list_data_files()
    if not files:
        print("  (لا توجد ملفات)")
    for entry in files:
        print(f"  • {entry['الملف']:40} {entry['الحجم']:>10}  {entry['تاريخ الإضافة']}")
    level, note = persistence_note()
    print(f"\n[{level}] {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
