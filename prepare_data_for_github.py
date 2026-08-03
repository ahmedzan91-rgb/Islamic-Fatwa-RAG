#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data_for_github.py
==========================
أداة تجهيز بيانات الفتاوى لرفعها إلى GitHub.

╔══════════════════════════════════════════════════════════════════════════╗
║  لماذا تحتاج هذه الأداة؟                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
    GitHub يرفض أي ملف يتجاوز 100 ميجابايت، ويحذّر عند تجاوز 50 ميجابايت.
    ملف 139 ألف فتوى بصيغة CSV الخام ≈ 300 ميجابايت → مرفوض قطعاً.
    لكن بضغط gzip ينكمش إلى ~50 ميجابايت → يمرّ بنجاح.

    هذه الأداة:
      1. تفحص أحجام ملفاتك وتصنّفها (آمن / تحذير / مرفوض).
      2. تضغطها بـ gzip (النظام يقرأ .csv.gz مباشرة بلا فكّ).
      3. تقسّم الملفات الضخمة جداً إلى أجزاء تحت الحدّ.
      4. تحدّث .gitignore ليسمح بالبيانات المضغوطة.
      5. تعرض تقريراً بما سيُرفع وحجمه الكلي.

الاستخدام:
    python prepare_data_for_github.py                    # فحص فقط (بلا تعديل)
    python prepare_data_for_github.py --compress         # ضغط الملفات
    python prepare_data_for_github.py --compress --split # ضغط + تقسيم الضخم
    python prepare_data_for_github.py --sample 20000     # إنشاء عيّنة خفيفة
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from typing import List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MB = 1024 * 1024
GITHUB_HARD_LIMIT = 100 * MB      # رفض قاطع
GITHUB_WARN_LIMIT = 50 * MB       # تحذير فقط
SAFE_TARGET = 45 * MB             # هدفنا عند التقسيم

TABULAR_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")


# ----------------------------------------------------------------------------- #
#                                دوال مساعدة                                     #
# ----------------------------------------------------------------------------- #

def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def verdict(size: int) -> Tuple[str, str]:
    """تصنيف الملف حسب حدود GitHub. يُرجع (رمز, وصف)."""
    if size >= GITHUB_HARD_LIMIT:
        return "❌", "مرفوض — يتجاوز 100 MB"
    if size >= GITHUB_WARN_LIMIT:
        return "⚠️ ", "تحذير — يمرّ لكن Git سينبّه"
    return "✅", "آمن"


def list_data_files(data_dir: str) -> List[str]:
    """سرد ملفات البيانات (غير المضغوطة) في المجلد."""
    if not os.path.isdir(data_dir):
        return []
    found = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path) or name.startswith((".", "_")):
            continue
        if name.lower().endswith(TABULAR_EXTENSIONS):
            found.append(path)
    return found


def list_compressed_files(data_dir: str) -> List[str]:
    """سرد الملفات المضغوطة الجاهزة للرفع."""
    if not os.path.isdir(data_dir):
        return []
    return [
        os.path.join(data_dir, n)
        for n in sorted(os.listdir(data_dir))
        if n.lower().endswith(".gz") and os.path.isfile(os.path.join(data_dir, n))
    ]


# ----------------------------------------------------------------------------- #
#                                  الفحص                                         #
# ----------------------------------------------------------------------------- #

def scan(data_dir: str) -> Tuple[List[str], List[str], int]:
    """
    فحص مجلد البيانات وطباعة تقرير.

    Returns:
        (الملفات الخام, الملفات المضغوطة, الحجم الكلي)
    """
    raw = list_data_files(data_dir)
    compressed = list_compressed_files(data_dir)

    print("=" * 74)
    print("  فحص مجلد البيانات:", data_dir)
    print("=" * 74)

    if not raw and not compressed:
        print("\n  ⚠️  لا توجد ملفات بيانات.")
        print(f"     ضع ملفات CSV في: {data_dir}\n")
        return [], [], 0

    total = 0

    if raw:
        print("\n  📄 ملفات خام (غير مضغوطة):\n")
        print(f"     {'الملف':<38}{'الحجم':>12}   الحالة")
        print("     " + "─" * 66)
        for path in raw:
            size = os.path.getsize(path)
            total += size
            icon, note = verdict(size)
            name = os.path.basename(path)
            display = name if len(name) <= 36 else name[:33] + "..."
            print(f"     {display:<38}{human(size):>12}   {icon} {note}")

    if compressed:
        print("\n  📦 ملفات مضغوطة (جاهزة للرفع):\n")
        print(f"     {'الملف':<38}{'الحجم':>12}   الحالة")
        print("     " + "─" * 66)
        for path in compressed:
            size = os.path.getsize(path)
            total += size
            icon, note = verdict(size)
            name = os.path.basename(path)
            display = name if len(name) <= 36 else name[:33] + "..."
            print(f"     {display:<38}{human(size):>12}   {icon} {note}")

    print("\n  " + "─" * 66)
    print(f"     الحجم الكلي: {human(total)}")

    blocked = [p for p in raw + compressed if os.path.getsize(p) >= GITHUB_HARD_LIMIT]
    if blocked:
        print(f"\n  ❌ {len(blocked)} ملف سيرفضه GitHub. شغّل: --compress (و --split عند اللزوم)")
    elif raw:
        print("\n  💡 ملفاتك الخام مستثناة في .gitignore. شغّل --compress لتجهيزها للرفع.")
    print()
    return raw, compressed, total


# ----------------------------------------------------------------------------- #
#                                  الضغط                                         #
# ----------------------------------------------------------------------------- #

def compress_file(path: str, level: int = 9, keep_original: bool = True) -> Optional[str]:
    """ضغط ملف بـ gzip. يُرجع مسار الملف المضغوط."""
    destination = path + ".gz"
    if os.path.exists(destination):
        print(f"     ⏭️  موجود مسبقاً: {os.path.basename(destination)}")
        return destination

    original_size = os.path.getsize(path)
    print(f"     🗜️  ضغط {os.path.basename(path)} ({human(original_size)})...", end="", flush=True)

    try:
        with open(path, "rb") as source, gzip.open(destination, "wb", compresslevel=level) as target:
            shutil.copyfileobj(source, target, length=4 * MB)
    except Exception as exc:  # noqa: BLE001
        print(f" فشل: {exc}")
        if os.path.exists(destination):
            os.remove(destination)
        return None

    new_size = os.path.getsize(destination)
    ratio = original_size / max(new_size, 1)
    icon, _ = verdict(new_size)
    print(f" → {human(new_size)} (ضغط {ratio:.1f}×) {icon}")

    if not keep_original:
        os.remove(path)
    return destination


def split_large_file(path: str, target_size: int = SAFE_TARGET) -> List[str]:
    """
    تقسيم ملف CSV كبير إلى أجزاء مع تكرار سطر الترويسة في كل جزء.
    يعمل بالتدفّق (streaming) فلا يستهلك ذاكرة كبيرة.
    """
    size = os.path.getsize(path)
    if size <= target_size:
        return [path]

    stem, extension = os.path.splitext(path)
    parts: List[str] = []

    print(f"     ✂️  تقسيم {os.path.basename(path)} ({human(size)})...")

    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as source:
            header = source.readline()
            index, written, handle = 1, 0, None

            for line in source:
                if handle is None:
                    part_path = f"{stem}_part{index:02d}{extension}"
                    handle = open(part_path, "w", encoding="utf-8-sig")
                    handle.write(header)
                    written = len(header.encode("utf-8"))
                    parts.append(part_path)

                handle.write(line)
                written += len(line.encode("utf-8"))

                if written >= target_size:
                    handle.close()
                    print(f"        ✓ {os.path.basename(parts[-1])} ({human(written)})")
                    handle, index = None, index + 1

            if handle is not None:
                handle.close()
                print(f"        ✓ {os.path.basename(parts[-1])} "
                      f"({human(os.path.getsize(parts[-1]))})")
    except Exception as exc:  # noqa: BLE001
        print(f"        فشل التقسيم: {exc}")
        return [path]

    return parts


def create_sample(source_path: str, rows: int, data_dir: str) -> Optional[str]:
    """إنشاء عيّنة خفيفة من ملف كبير (مفيدة للعرض والاختبار السريع)."""
    stem = os.path.splitext(os.path.basename(source_path))[0]
    destination = os.path.join(data_dir, f"{stem}_sample_{rows}.csv")

    print(f"     🎯 إنشاء عيّنة {rows:,} صف من {os.path.basename(source_path)}...")
    try:
        with open(source_path, "r", encoding="utf-8-sig", errors="replace") as source, \
             open(destination, "w", encoding="utf-8-sig") as target:
            target.write(source.readline())          # الترويسة
            for i, line in enumerate(source):
                if i >= rows:
                    break
                target.write(line)
    except Exception as exc:  # noqa: BLE001
        print(f"        فشل: {exc}")
        return None

    size = os.path.getsize(destination)
    icon, _ = verdict(size)
    print(f"        ✓ {os.path.basename(destination)} ({human(size)}) {icon}")
    return destination


# ----------------------------------------------------------------------------- #
#                              تحديث .gitignore                                  #
# ----------------------------------------------------------------------------- #

GITIGNORE_MARKER = "# ── بيانات الفتاوى المضغوطة (مسموح برفعها) ──"

GITIGNORE_RULES = f"""
{GITIGNORE_MARKER}
# الملفات الخام تبقى مستثناة (كبيرة جداً على GitHub)،
# لكن النسخ المضغوطة .gz مسموح برفعها — والنظام يقرؤها مباشرة.
!data/*.csv.gz
!data/*.tsv.gz
"""


def update_gitignore(base_dir: str) -> bool:
    """إضافة استثناء يسمح برفع الملفات المضغوطة."""
    path = os.path.join(base_dir, ".gitignore")
    content = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()

    if GITIGNORE_MARKER in content:
        print("     ⏭️  .gitignore محدَّث مسبقاً.")
        return False

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(GITIGNORE_RULES)
    print("     ✅ حُدّث .gitignore للسماح برفع ملفات .gz")
    return True


# ----------------------------------------------------------------------------- #
#                                  التعليمات                                     #
# ----------------------------------------------------------------------------- #

def print_push_instructions(files: List[str]) -> None:
    """طباعة أوامر الرفع الجاهزة."""
    if not files:
        return

    total = sum(os.path.getsize(f) for f in files)

    print("\n" + "=" * 74)
    print("  الخطوة التالية: الرفع إلى GitHub")
    print("=" * 74)
    print(f"\n  سيُرفع {len(files)} ملف بحجم كلي {human(total)}\n")
    print("  ┌" + "─" * 70 + "┐")
    print("  │ git add data/*.gz .gitignore" + " " * 41 + "│")
    print('  │ git commit -m "data: إضافة بيانات الفتاوى المضغوطة"' + " " * 18 + "│")
    print("  │ git push" + " " * 61 + "│")
    print("  └" + "─" * 70 + "┘")

    if total > 300 * MB:
        print("\n  ⚠️  الحجم الكلي كبير. GitHub يحدّ كل دفعة push بـ 2 GB.")
        print("     ارفع على دفعات: git add data/ملف_واحد.gz && git commit && git push")

    print("\n  💡 بعد الرفع، النظام يقرأ ملفات .gz مباشرة بلا فكّ ضغط:")
    print("     python run_pipeline.py --limit 2000\n")


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="تجهيز بيانات الفتاوى لرفعها إلى GitHub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="مجلد البيانات.")
    parser.add_argument("--compress", action="store_true", help="ضغط الملفات بـ gzip.")
    parser.add_argument("--split", action="store_true",
                        help="تقسيم الملفات التي تبقى فوق الحدّ بعد الضغط.")
    parser.add_argument("--delete-original", action="store_true",
                        help="حذف الملف الخام بعد الضغط (وفّر مساحة).")
    parser.add_argument("--sample", type=int, default=0,
                        help="إنشاء عيّنة بعدد الصفوف المحدّد بدل رفع الكل.")
    parser.add_argument("--level", type=int, default=9, choices=range(1, 10),
                        help="مستوى الضغط (1=سريع، 9=أقصى).")
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    raw, compressed, _ = scan(data_dir)

    # ─────────────── وضع العيّنة ─────────────── #
    if args.sample:
        if not raw:
            print("  ⚠️  لا توجد ملفات خام لأخذ عيّنة منها.\n")
            return 1
        print("=" * 74)
        print(f"  إنشاء عيّنات ({args.sample:,} صف)")
        print("=" * 74 + "\n")
        samples = []
        for path in raw:
            if path.lower().endswith((".xlsx", ".xls")):
                print(f"     ⏭️  تخطّي {os.path.basename(path)} (Excel غير مدعوم للعيّنة)")
                continue
            result = create_sample(path, args.sample, data_dir)
            if result:
                samples.append(result)
        print()
        update_gitignore(BASE_DIR)
        # العيّنات صغيرة عادةً — نرفعها كما هي
        if samples:
            path = os.path.join(BASE_DIR, ".gitignore")
            with open(path, "a", encoding="utf-8") as fh:
                for sample in samples:
                    fh.write(f"!data/{os.path.basename(sample)}\n")
            print(f"     ✅ أُضيفت {len(samples)} عيّنة إلى .gitignore")
        print_push_instructions(samples)
        return 0

    # ─────────────── وضع الفحص فقط ─────────────── #
    if not args.compress:
        print("  ℹ️  وضع الفحص فقط — لم يُعدَّل شيء.")
        print("     للضغط والتجهيز شغّل: python prepare_data_for_github.py --compress\n")
        return 0

    if not raw:
        print("  ⚠️  لا توجد ملفات خام للضغط.\n")
        if compressed:
            update_gitignore(BASE_DIR)
            print_push_instructions(compressed)
        return 0

    # ─────────────── الضغط ─────────────── #
    print("=" * 74)
    print("  ضغط الملفات")
    print("=" * 74 + "\n")

    ready: List[str] = list(compressed)
    for path in raw:
        if path.lower().endswith((".xlsx", ".xls")):
            print(f"     ⏭️  تخطّي {os.path.basename(path)} — حوّله إلى CSV أولاً.")
            continue

        # نُبقي الأصل مؤقتاً حتى نتأكد أننا لن نحتاجه للتقسيم،
        # ثم نحذفه في النهاية إن طُلب ذلك.
        result = compress_file(path, args.level, keep_original=True)
        if not result:
            continue

        compressed_size = os.path.getsize(result)
        needs_split = compressed_size >= GITHUB_HARD_LIMIT

        if needs_split and args.split:
            print(f"     ⚠️  {os.path.basename(result)} ما زال فوق الحدّ — التقسيم...")

            # حساب هدف تقسيم تكيّفي: الحدّ يُطبَّق على الحجم *المضغوط*،
            # لذا نضرب الهدف بنسبة الضغط الفعلية المقيسة لهذا الملف تحديداً،
            # فنحصل على أقل عدد ممكن من الأجزاء بدل تفتيتها بلا داعٍ.
            ratio = os.path.getsize(path) / max(compressed_size, 1)
            adaptive_target = int(SAFE_TARGET * ratio)
            print(f"        نسبة الضغط المقيسة: {ratio:.1f}× → "
                  f"حجم الجزء الخام: {human(adaptive_target)}")

            os.remove(result)
            for part in split_large_file(path, adaptive_target):
                if os.path.abspath(part) == os.path.abspath(path):
                    continue  # لم يحدث تقسيم فعلي
                part_gz = compress_file(part, args.level, keep_original=False)
                if part_gz:
                    ready.append(part_gz)
        elif needs_split:
            print(f"     ❌ {os.path.basename(result)} فوق الحدّ. أضِف --split")
            ready.append(result)
        else:
            ready.append(result)

        # حذف الأصل بعد انتهاء كل العمليات التي قد تحتاجه
        if args.delete_original and os.path.exists(path):
            os.remove(path)
            print(f"     🗑️  حُذف الأصل: {os.path.basename(path)}")

    print()
    update_gitignore(BASE_DIR)

    # ─────────────── التقرير النهائي ─────────────── #
    print("\n" + "=" * 74)
    print("  النتيجة")
    print("=" * 74 + "\n")

    uploadable = [f for f in ready if os.path.getsize(f) < GITHUB_HARD_LIMIT]
    blocked = [f for f in ready if os.path.getsize(f) >= GITHUB_HARD_LIMIT]

    for path in uploadable:
        icon, _ = verdict(os.path.getsize(path))
        print(f"     {icon} {os.path.basename(path):<44}{human(os.path.getsize(path)):>12}")
    for path in blocked:
        print(f"     ❌ {os.path.basename(path):<44}{human(os.path.getsize(path)):>12}  (أضِف --split)")

    print_push_instructions(uploadable)
    return 0 if not blocked else 1


if __name__ == "__main__":
    sys.exit(main())
