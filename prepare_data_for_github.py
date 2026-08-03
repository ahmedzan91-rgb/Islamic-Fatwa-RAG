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
      1. تبحث عن ملفاتك تلقائياً (حتى لو لم تكن في مجلد data/).
      2. تفحص أحجامها وتصنّفها (آمن / تحذير / مرفوض).
      3. تضغطها بـ gzip — والنظام يقرأ .csv.gz مباشرة بلا فكّ.
      4. تقسّم الضخم منها بحجم تكيّفي تحت الحدّ.
      5. تحدّث .gitignore وتطبع أوامر الرفع الجاهزة.

الاستخدام (ويندوز / لينكس / ماك):
    python prepare_data_for_github.py                     فحص فقط
    python prepare_data_for_github.py --find              البحث عن ملفات CSV
    python prepare_data_for_github.py --compress          ضغط
    python prepare_data_for_github.py --compress --split  ضغط وتقسيم
    python prepare_data_for_github.py --data-dir "C:\\path\\to\\csv"

⚠️ ملاحظة لمستخدمي ويندوز (CMD):
    لا تنسخ التعليقات العربية بعد الأمر — CMD لا يفهم "#" كتعليق
    وسيعتبرها وسيطاً غير معروف. انسخ الأمر وحده.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from typing import List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MB = 1024 * 1024
GITHUB_HARD_LIMIT = 100 * MB      # رفض قاطع
GITHUB_WARN_LIMIT = 50 * MB       # تحذير فقط
SAFE_TARGET = 45 * MB             # هدف الحجم المضغوط عند التقسيم

TABULAR_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
# ملفات نصية شائعة ليست بيانات — نستبعدها من البحث
SKIP_FILENAMES = {
    "requirements.txt", "requirements-full.txt", "packages.txt",
    "readme.txt", "license.txt", "changelog.txt", "notes.txt",
    "runtime.txt", "constraints.txt", "todo.txt",
}

SKIP_DIRECTORIES = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "chroma_db", "artifacts", ".idea", ".vscode", "env",
}


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
    """سرد ملفات البيانات الخام (غير المضغوطة) في المجلد."""
    if not os.path.isdir(data_dir):
        return []
    found = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path) or name.startswith((".", "_")):
            continue
        if name.lower() in SKIP_FILENAMES:
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
#                       اكتشاف مجلد البيانات ومستودع Git                          #
# ----------------------------------------------------------------------------- #

def find_repo_root(start: str) -> Optional[str]:
    """الصعود في الشجرة بحثاً عن جذر مستودع Git (مجلد .git)."""
    current = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def resolve_data_dir(explicit: Optional[str]) -> Tuple[str, str]:
    """
    تحديد مجلد البيانات تلقائياً.

    ترتيب البحث:
        1. المسار الصريح عبر --data-dir
        2. <مجلد السكربت>/data
        3. <مجلد السكربت> نفسه
        4. <مجلد العمل الحالي>/data
        5. <مجلد العمل الحالي> نفسه

    Returns:
        (المسار, سبب الاختيار)
    """
    if explicit:
        return os.path.abspath(explicit), "محدّد عبر --data-dir"

    cwd = os.getcwd()
    candidates = [
        (os.path.join(BASE_DIR, "data"), "مجلد data بجانب السكربت"),
        (BASE_DIR, "مجلد السكربت نفسه"),
        (os.path.join(cwd, "data"), "مجلد data في مسار التشغيل"),
        (cwd, "مسار التشغيل الحالي"),
    ]

    # نفضّل أول مجلد يحوي ملفات فعلية
    for path, reason in candidates:
        if list_data_files(path) or list_compressed_files(path):
            return os.path.abspath(path), reason

    return os.path.join(BASE_DIR, "data"), "الافتراضي (فارغ)"


def find_csv_files_recursive(root: str, max_depth: int = 3, limit: int = 300) -> List[str]:
    """بحث تكراري عن ملفات جدولية لمساعدة المستخدم على تحديد مكان بياناته."""
    results: List[str] = []
    root = os.path.abspath(root)
    root_depth = root.rstrip(os.sep).count(os.sep)

    for current, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d not in SKIP_DIRECTORIES]
        if current.rstrip(os.sep).count(os.sep) - root_depth >= max_depth:
            directories[:] = []
        for name in files:
            lowered = name.lower()
            if lowered in SKIP_FILENAMES:
                continue
            if lowered.endswith(TABULAR_EXTENSIONS) or lowered.endswith(".gz"):
                results.append(os.path.join(current, name))
                if len(results) >= limit:
                    return results
    return results


def run_find_mode(start: str) -> int:
    """وضع --find: البحث عن ملفات البيانات وعرض مواقعها."""
    print("=" * 74)
    print("  البحث عن ملفات البيانات")
    print("=" * 74)

    search_roots = []
    for candidate in (os.getcwd(), BASE_DIR, os.path.expanduser("~/Desktop"),
                      os.path.expanduser("~")):
        candidate = os.path.abspath(candidate)
        if candidate not in search_roots and os.path.isdir(candidate):
            search_roots.append(candidate)

    seen = set()
    found_any = False

    for root in search_roots[:3]:
        print(f"\n  🔎 البحث في: {root}")
        results = find_csv_files_recursive(root, max_depth=3)
        results = [p for p in results if p not in seen]
        if not results:
            print("     (لا شيء)")
            continue

        found_any = True
        by_directory: dict = {}
        for path in results:
            seen.add(path)
            by_directory.setdefault(os.path.dirname(path), []).append(path)

        for directory, paths in sorted(by_directory.items())[:10]:
            total = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
            print(f"\n     📁 {directory}")
            print(f"        {len(paths)} ملف — {human(total)}")
            for path in paths[:5]:
                size = os.path.getsize(path) if os.path.exists(path) else 0
                icon, _ = verdict(size)
                print(f"        {icon} {os.path.basename(path):<40}{human(size):>12}")
            if len(paths) > 5:
                print(f"        ... و{len(paths) - 5} ملف آخر")

    if not found_any:
        print("\n  ⚠️  لم يُعثر على أي ملف بيانات.")
        return 1

    print("\n" + "─" * 74)
    print("  💡 شغّل الأداة على المجلد الصحيح:")
    print('     python prepare_data_for_github.py --data-dir "المسار" --compress')
    print()
    return 0


# ----------------------------------------------------------------------------- #
#                                  الفحص                                         #
# ----------------------------------------------------------------------------- #

def scan(data_dir: str, reason: str = "") -> Tuple[List[str], List[str], int]:
    """فحص مجلد البيانات وطباعة تقرير."""
    raw = list_data_files(data_dir)
    compressed = list_compressed_files(data_dir)

    print("=" * 74)
    print("  فحص مجلد البيانات")
    print("=" * 74)
    print(f"\n  📂 {data_dir}")
    if reason:
        print(f"     ({reason})")

    if not raw and not compressed:
        print("\n  ⚠️  لا توجد ملفات بيانات في هذا المجلد.\n")
        print("  جرّب أحد الحلول:")
        print("     1. ابحث عن ملفاتك:")
        print("        python prepare_data_for_github.py --find")
        print("     2. حدّد المجلد يدوياً:")
        print('        python prepare_data_for_github.py --data-dir "C:\\مسار\\ملفاتي"')
        print("     3. انقل ملفات CSV إلى المجلد أعلاه.\n")
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
        print(f"\n  ❌ {len(blocked)} ملف سيرفضه GitHub.")
        print("     شغّل: python prepare_data_for_github.py --compress --split")
    elif raw:
        print("\n  💡 لتجهيز الملفات للرفع:")
        print("     python prepare_data_for_github.py --compress")
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
    print(f"     🗜️  ضغط {os.path.basename(path)} ({human(original_size)})...",
          end="", flush=True)

    try:
        with open(path, "rb") as source, \
             gzip.open(destination, "wb", compresslevel=level) as target:
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
            target.write(source.readline())
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


def update_gitignore(repo_root: Optional[str], data_dir: str) -> bool:
    """إضافة استثناء يسمح برفع الملفات المضغوطة (بمسار نسبي صحيح)."""
    if not repo_root:
        return False

    relative = os.path.relpath(data_dir, repo_root).replace(os.sep, "/")
    prefix = "" if relative == "." else f"{relative}/"

    path = os.path.join(repo_root, ".gitignore")
    content = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()

    rules = (
        f"\n{GITIGNORE_MARKER}\n"
        "# الخام كبير جداً على GitHub، لكن المضغوط يمرّ — والنظام يقرأ .gz مباشرة.\n"
        f"!{prefix}*.csv.gz\n"
        f"!{prefix}*.tsv.gz\n"
    )

    if f"!{prefix}*.csv.gz" in content:
        print("     ⏭️  .gitignore محدَّث مسبقاً.")
        return False

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(rules)
    print(f"     ✅ حُدّث .gitignore للسماح برفع: {prefix}*.csv.gz")
    return True


# ----------------------------------------------------------------------------- #
#                                  التعليمات                                     #
# ----------------------------------------------------------------------------- #

def print_push_instructions(files: List[str], repo_root: Optional[str],
                            data_dir: str) -> None:
    """طباعة أوامر الرفع الجاهزة، أو إرشاد المستخدم إن لم يكن داخل مستودع."""
    if not files:
        return

    total = sum(os.path.getsize(f) for f in files if os.path.exists(f))

    print("\n" + "=" * 74)
    print("  الخطوة التالية")
    print("=" * 74)
    print(f"\n  الملفات الجاهزة: {len(files)} — الحجم الكلي: {human(total)}")

    if not repo_root:
        print("\n  ⚠️  هذا المجلد ليس داخل مستودع Git — لذلك لا يمكن الرفع من هنا.\n")
        print("  ✅ لكن ملفاتك المضغوطة جاهزة في:")
        print(f"     {data_dir}\n")
        print("  " + "─" * 70)
        print("  الخطوات التالية (انسخ كل سطر وحده):")
        print("  " + "─" * 70 + "\n")
        print("  1) استنسخ المستودع على سطح المكتب:\n")
        print("     cd %USERPROFILE%\\Desktop")
        print("     git clone https://github.com/ahmedzan91-rgb/Islamic-Fatwa-RAG.git\n")
        print("  2) انسخ الملفات المضغوطة إلى مجلد data:\n")
        print(f'     copy "{data_dir}\\*.gz" "%USERPROFILE%\\Desktop\\Islamic-Fatwa-RAG\\data\\"\n')
        print("  3) ادخل المستودع وارفع:\n")
        print("     cd %USERPROFILE%\\Desktop\\Islamic-Fatwa-RAG")
        print("     python prepare_data_for_github.py")
        print("     git add data/*.gz .gitignore")
        print('     git commit -m "data: add fatwa dataset"')
        print("     git push\n")
        return

    relative = os.path.relpath(data_dir, repo_root).replace(os.sep, "/")
    prefix = "" if relative == "." else f"{relative}/"

    print("\n  نفّذ هذه الأوامر (من مجلد المستودع):\n")
    print("  ┌" + "─" * 70 + "┐")
    line1 = f"git add {prefix}*.gz .gitignore"
    line2 = 'git commit -m "data: add compressed fatwa dataset"'
    print(f"  │ {line1:<69}│")
    print(f"  │ {line2:<69}│")
    print(f"  │ {'git push':<69}│")
    print("  └" + "─" * 70 + "┘")

    if total > 300 * MB:
        print("\n  ⚠️  الحجم كبير. GitHub يحدّ كل دفعة push بـ 2 GB.")
        print("     ارفع على دفعات: أضِف ملفاً واحداً في كل commit.")

    print("\n  💡 بعد الرفع، النظام يقرأ ملفات .gz مباشرة بلا فكّ ضغط:")
    print("     python run_pipeline.py --limit 2000\n")


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="تجهيز بيانات الفتاوى لرفعها إلى GitHub.",
        epilog="ملاحظة لويندوز: لا تنسخ التعليقات بعد الأمر — CMD لا يفهم '#'.",
    )
    parser.add_argument("--data-dir", default=None,
                        help="مجلد البيانات (يُكتشف تلقائياً إن لم يُحدَّد).")
    parser.add_argument("--find", action="store_true",
                        help="البحث عن ملفات CSV في جهازك وعرض مواقعها.")
    parser.add_argument("--compress", action="store_true", help="ضغط الملفات بـ gzip.")
    parser.add_argument("--split", action="store_true",
                        help="تقسيم الملفات التي تبقى فوق الحدّ بعد الضغط.")
    parser.add_argument("--delete-original", action="store_true",
                        help="حذف الملف الخام بعد الضغط.")
    parser.add_argument("--sample", type=int, default=0,
                        help="إنشاء عيّنة بعدد الصفوف المحدّد.")
    parser.add_argument("--level", type=int, default=9, choices=range(1, 10),
                        help="مستوى الضغط (1=سريع، 9=أقصى).")
    args = parser.parse_args(argv)

    # ─────────────── وضع البحث ─────────────── #
    if args.find:
        return run_find_mode(os.getcwd())

    data_dir, reason = resolve_data_dir(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    repo_root = find_repo_root(data_dir)

    raw, compressed, _ = scan(data_dir, reason)

    if repo_root:
        print(f"  🐙 مستودع Git: {repo_root}\n")
    else:
        print("  ⚠️  لست داخل مستودع Git — سأجهّز الملفات فقط.\n")

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
                print(f"     ⏭️  تخطّي {os.path.basename(path)} (Excel — حوّله إلى CSV)")
                continue
            result = create_sample(path, args.sample, data_dir)
            if result:
                samples.append(result)
        print()
        if samples and repo_root:
            relative = os.path.relpath(data_dir, repo_root).replace(os.sep, "/")
            prefix = "" if relative == "." else f"{relative}/"
            with open(os.path.join(repo_root, ".gitignore"), "a", encoding="utf-8") as fh:
                fh.write(f"\n# عيّنات خفيفة للعرض\n")
                for sample in samples:
                    fh.write(f"!{prefix}{os.path.basename(sample)}\n")
            print(f"     ✅ أُضيفت {len(samples)} عيّنة إلى .gitignore")
        print_push_instructions(samples, repo_root, data_dir)
        return 0

    # ─────────────── وضع الفحص فقط ─────────────── #
    if not args.compress:
        if raw or compressed:
            print("  ℹ️  وضع الفحص فقط — لم يُعدَّل شيء.")
            print("     للضغط: python prepare_data_for_github.py --compress\n")
        return 0

    if not raw:
        print("  ⚠️  لا توجد ملفات خام للضغط.\n")
        if compressed:
            update_gitignore(repo_root, data_dir)
            print_push_instructions(compressed, repo_root, data_dir)
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

        # نُبقي الأصل حتى نتأكد أننا لن نحتاجه للتقسيم
        result = compress_file(path, args.level, keep_original=True)
        if not result:
            continue

        compressed_size = os.path.getsize(result)
        needs_split = compressed_size >= GITHUB_HARD_LIMIT

        if needs_split and args.split:
            print(f"     ⚠️  {os.path.basename(result)} ما زال فوق الحدّ — التقسيم...")

            # هدف تكيّفي: الحدّ يُطبَّق على الحجم *المضغوط*، فنضرب الهدف
            # بنسبة الضغط المقيسة لهذا الملف → أقل عدد أجزاء ممكن.
            ratio = os.path.getsize(path) / max(compressed_size, 1)
            adaptive_target = int(SAFE_TARGET * ratio)
            print(f"        نسبة الضغط المقيسة: {ratio:.1f}× → "
                  f"حجم الجزء الخام: {human(adaptive_target)}")

            os.remove(result)
            for part in split_large_file(path, adaptive_target):
                if os.path.abspath(part) == os.path.abspath(path):
                    continue
                part_gz = compress_file(part, args.level, keep_original=False)
                if part_gz:
                    ready.append(part_gz)
        elif needs_split:
            print(f"     ❌ {os.path.basename(result)} فوق الحدّ. أضِف --split")
            ready.append(result)
        else:
            ready.append(result)

        if args.delete_original and os.path.exists(path):
            os.remove(path)
            print(f"     🗑️  حُذف الأصل: {os.path.basename(path)}")

    print()
    update_gitignore(repo_root, data_dir)

    # ─────────────── التقرير النهائي ─────────────── #
    print("\n" + "=" * 74)
    print("  النتيجة")
    print("=" * 74 + "\n")

    # إزالة المكرّرات مع الحفاظ على الترتيب (قد يظهر الملف مرتين إن كان
    # مضغوطاً مسبقاً ثم أُعيد اكتشافه بعد ضغط أصله)
    unique_ready: List[str] = []
    seen_paths = set()
    for candidate in ready:
        key = os.path.abspath(candidate)
        if key not in seen_paths and os.path.exists(candidate):
            seen_paths.add(key)
            unique_ready.append(candidate)

    uploadable = [f for f in unique_ready if os.path.getsize(f) < GITHUB_HARD_LIMIT]
    blocked = [f for f in unique_ready if os.path.getsize(f) >= GITHUB_HARD_LIMIT]

    for path in uploadable:
        icon, _ = verdict(os.path.getsize(path))
        print(f"     {icon} {os.path.basename(path):<44}{human(os.path.getsize(path)):>12}")
    for path in blocked:
        print(f"     ❌ {os.path.basename(path):<44}"
              f"{human(os.path.getsize(path)):>12}  (أضِف --split)")

    print_push_instructions(uploadable, repo_root, data_dir)
    return 0 if not blocked else 1


if __name__ == "__main__":
    sys.exit(main())
