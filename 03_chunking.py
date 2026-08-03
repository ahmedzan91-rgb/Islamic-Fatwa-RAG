# -*- coding: utf-8 -*-
"""
03_chunking.py
==============
المرحلة الثالثة: تقسيم النصوص الشرعية إلى أجزاء (Chunks) مع الحفاظ الكامل على الـ metadata.

المنهجية المعتمدة (Sentence-Aware Recursive Chunking):
    التقطيع الأعمى بعدد المحارف يقطع الأحكام الشرعية في منتصفها ويفسد المعنى.
    لذلك نعتمد تقطيعاً هرمياً يحترم بنية النص العربي:
        الفقرة (\n\n)  →  الجملة (. ؟ ! ؛ :)  →  الفاصلة (،)  →  الكلمة
    مع تراكب (overlap) بين الأجزاء لضمان عدم ضياع السياق على الحدود.

سياسة الـ metadata:
    كل chunk يرث: رقم الفتوى، العنوان، السؤال (مقتطع)، المصدر، التصنيف، الرابط،
    بالإضافة إلى: chunk_index، chunk_total، is_question_chunk.
    السؤال يُضاف كترويسة سياقية (contextual header) لكل chunk — تقنية "Contextual
    Retrieval" التي ترفع دقة الاسترجاع لأن الأجزاء المتأخرة من الجواب تفقد مرجعها.

التشغيل:
    python 03_chunking.py --input artifacts/02_clean.parquet \
                          --output artifacts/03_chunks.parquet \
                          --chunk-size 900 --overlap 150
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterator, List, Optional

import pandas as pd

LOGGER = logging.getLogger("islamic_rag.chunking")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "artifacts", "02_clean.parquet")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "artifacts", "03_chunks.parquet")

# فواصل التقطيع الهرمي، من الأقوى (بنيوي) إلى الأضعف (لفظي)
HIERARCHICAL_SEPARATORS: List[str] = ["\n\n", "\n", "۔ ", ". ", "؟ ", "! ", "؛ ", ": ", "، ", " "]

RE_SENTENCE_SPLIT = re.compile(r"(?<=[\.\؟\!\؛])\s+")


# ----------------------------------------------------------------------------- #
#                                  الإعدادات                                     #
# ----------------------------------------------------------------------------- #

@dataclass
class ChunkConfig:
    """
    معاملات التقطيع.

    chunk_size:  الحجم المستهدف بالمحارف. 900 محرف ≈ 130-160 كلمة عربية،
                 وهو نطاق مناسب لنماذج التضمين متعددة اللغات (حدّ 512 توكن).
    chunk_overlap: التراكب بين الأجزاء المتتالية للحفاظ على السياق.
    min_chunk_size: أصغر جزء مقبول؛ الأجزاء الأصغر تُدمج مع سابقها.
    """

    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_size: int = 120
    add_question_header: bool = True
    question_header_max_chars: int = 220
    include_metadata_prefix: bool = True


@dataclass
class Chunk:
    """تمثيل جزء نصي واحد مع الـ metadata الكاملة."""

    chunk_id: str
    doc_id: str
    fatwa_id: str
    text: str                # النص النهائي الذي سيُضمَّن (يشمل الترويسة السياقية)
    raw_text: str            # النص الأصلي دون ترويسة (للعرض)
    chunk_index: int
    chunk_total: int
    title: str = ""
    question: str = ""
    category: str = ""
    source: str = ""
    url: str = ""
    date: str = ""
    char_count: int = 0
    word_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_chroma_metadata(self) -> Dict[str, str]:
        """
        تحويل إلى metadata متوافقة مع ChromaDB.
        ChromaDB تقبل فقط: str, int, float, bool — لذا نحوّل كل شيء إلى أنواع بسيطة.
        """
        return {
            "doc_id": str(self.doc_id),
            "fatwa_id": str(self.fatwa_id),
            "title": str(self.title)[:500],
            "question": str(self.question)[:1000],
            "category": str(self.category)[:200],
            "source": str(self.source)[:200],
            "url": str(self.url)[:500],
            "date": str(self.date)[:100],
            "chunk_index": int(self.chunk_index),
            "chunk_total": int(self.chunk_total),
            "char_count": int(self.char_count),
        }


# ----------------------------------------------------------------------------- #
#                              خوارزمية التقطيع                                   #
# ----------------------------------------------------------------------------- #

def _split_by_separator(text: str, separator: str) -> List[str]:
    """تقسيم مع الحفاظ على الفاصل ملحقاً بنهاية كل قطعة."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    result = [p + separator for p in parts[:-1]]
    if parts[-1]:
        result.append(parts[-1])
    return [p for p in result if p]


def recursive_split(text: str, chunk_size: int, separators: Optional[List[str]] = None) -> List[str]:
    """
    تقسيم هرمي تكراري: نجرّب الفاصل الأقوى أولاً، وإذا بقيت قطعة كبيرة
    ننزل إلى الفاصل التالي، حتى نصل إلى القطع بالمحارف كحل أخير.
    """
    separators = separators if separators is not None else HIERARCHICAL_SEPARATORS
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # حل أخير: قصّ صارم بالمحارف
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *remaining = separators
    pieces = _split_by_separator(text, separator)

    chunks: List[str] = []
    buffer = ""
    for piece in pieces:
        if len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            if len(piece) > chunk_size:
                chunks.extend(recursive_split(piece, chunk_size, remaining))
                buffer = ""
            else:
                buffer = piece
    if buffer.strip():
        chunks.append(buffer.strip())
    return [c for c in chunks if c.strip()]


def apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """
    إضافة تراكب بين الأجزاء: يُلحق بذيل الجزء السابق مقدارُ overlap محرف
    في بداية الجزء التالي، مع القطع عند حدود الكلمات لا وسطها.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        previous = chunks[i - 1]
        tail = previous[-overlap:]
        # نبدأ التراكب من أول مسافة لتفادي قطع الكلمة
        space_pos = tail.find(" ")
        if space_pos != -1:
            tail = tail[space_pos + 1:]
        overlapped.append((tail + " " + chunks[i]).strip())
    return overlapped


def merge_small_chunks(chunks: List[str], min_size: int) -> List[str]:
    """دمج الأجزاء الصغيرة جداً مع ما قبلها لتفادي أجزاء بلا معنى مستقل."""
    if not chunks:
        return []
    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_size:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged


def build_context_header(row: pd.Series, config: ChunkConfig) -> str:
    """
    بناء الترويسة السياقية التي تُسبق كل جزء.
    الهدف: أن يحمل كل chunk هويّته الشرعية حتى لو استُرجع منفرداً.
    """
    if not config.include_metadata_prefix:
        return ""

    lines: List[str] = []
    title = str(row.get("title_clean") or row.get("title") or "").strip()
    category = str(row.get("category") or "").strip()
    question = str(row.get("question_clean") or row.get("question") or "").strip()

    if title:
        lines.append(f"[الموضوع: {title}]")
    if category:
        lines.append(f"[التصنيف: {category}]")
    if config.add_question_header and question:
        snippet = question[: config.question_header_max_chars]
        if len(question) > config.question_header_max_chars:
            snippet = snippet.rsplit(" ", 1)[0] + "..."
        lines.append(f"[السؤال: {snippet}]")
    return "\n".join(lines)


def chunk_single_fatwa(row: pd.Series, config: ChunkConfig) -> List[Chunk]:
    """تقطيع فتوى واحدة إلى قائمة كائنات Chunk مع الـ metadata."""
    body = str(row.get("answer_clean") or row.get("answer") or "").strip()
    if not body:
        return []

    raw_pieces = recursive_split(body, config.chunk_size)
    raw_pieces = merge_small_chunks(raw_pieces, config.min_chunk_size)
    raw_pieces = apply_overlap(raw_pieces, config.chunk_overlap)

    if not raw_pieces:
        return []

    header = build_context_header(row, config)
    doc_id = str(row.get("doc_id", ""))
    fatwa_id = str(row.get("fatwa_id", doc_id))
    total = len(raw_pieces)

    chunks: List[Chunk] = []
    for idx, piece in enumerate(raw_pieces):
        embedded_text = f"{header}\n\n{piece}".strip() if header else piece
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}__c{idx:04d}",
                doc_id=doc_id,
                fatwa_id=fatwa_id,
                text=embedded_text,
                raw_text=piece,
                chunk_index=idx,
                chunk_total=total,
                title=str(row.get("title_clean") or row.get("title") or ""),
                question=str(row.get("question_clean") or row.get("question") or ""),
                category=str(row.get("category") or ""),
                source=str(row.get("source") or ""),
                url=str(row.get("url") or ""),
                date=str(row.get("date") or ""),
                char_count=len(piece),
                word_count=len(piece.split()),
            )
        )
    return chunks


# ----------------------------------------------------------------------------- #
#                              الواجهة العامة                                     #
# ----------------------------------------------------------------------------- #

def chunk_documents(
    documents: pd.DataFrame,
    config: Optional[ChunkConfig] = None,
    progress_every: int = 10_000,
) -> pd.DataFrame:
    """
    تقطيع كل الفتاوى وإرجاع DataFrame من الأجزاء.

    Returns:
        DataFrame بأعمدة Chunk كاملة، جاهز لمرحلة التضمين.
    """
    config = config or ChunkConfig()
    if documents.empty:
        LOGGER.warning("لا توجد وثائق للتقطيع.")
        return pd.DataFrame()

    LOGGER.info(
        "بدء التقطيع: %d فتوى | حجم الجزء=%d | التراكب=%d",
        len(documents), config.chunk_size, config.chunk_overlap,
    )

    records: List[dict] = []
    for i, (_, row) in enumerate(documents.iterrows(), start=1):
        for chunk in chunk_single_fatwa(row, config):
            records.append(chunk.to_dict())
        if progress_every and i % progress_every == 0:
            LOGGER.info("تمت معالجة %d/%d فتوى — الأجزاء حتى الآن: %d",
                        i, len(documents), len(records))

    chunks_df = pd.DataFrame(records)
    if chunks_df.empty:
        LOGGER.error("لم يُنتج التقطيع أي أجزاء.")
        return chunks_df

    LOGGER.info(
        "اكتمل التقطيع: %d فتوى → %d جزء (متوسط %.2f جزء/فتوى، متوسط %.0f محرف/جزء).",
        len(documents), len(chunks_df),
        len(chunks_df) / max(len(documents), 1),
        chunks_df["char_count"].mean(),
    )
    return chunks_df


def iter_chunk_batches(chunks_df: pd.DataFrame, batch_size: int = 1000) -> Iterator[pd.DataFrame]:
    """مولّد دفعات — تستخدمه مرحلة بناء قاعدة المتجهات لتفادي استهلاك الذاكرة."""
    for start in range(0, len(chunks_df), batch_size):
        yield chunks_df.iloc[start:start + batch_size]


def chunking_statistics(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """إحصاءات التقطيع للتوثيق الأكاديمي."""
    if chunks_df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "المؤشر": [
            "إجمالي الأجزاء", "عدد الفتاوى الأصلية", "متوسط الأجزاء لكل فتوى",
            "متوسط طول الجزء (محرف)", "أقصر جزء", "أطول جزء", "متوسط الكلمات لكل جزء",
        ],
        "القيمة": [
            len(chunks_df),
            int(chunks_df["doc_id"].nunique()),
            round(len(chunks_df) / max(chunks_df["doc_id"].nunique(), 1), 2),
            round(float(chunks_df["char_count"].mean()), 1),
            int(chunks_df["char_count"].min()),
            int(chunks_df["char_count"].max()),
            round(float(chunks_df["word_count"].mean()), 1),
        ],
    })


def load_input(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".csv"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"لم يُعثر على ملف الإدخال: {path}")
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(
        path, dtype=str, keep_default_na=False
    )


def save_output(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001
        path = os.path.splitext(path)[0] + ".csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
    LOGGER.info("تم حفظ %d جزء في: %s", len(df), path)
    return path


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="تقطيع نصوص الفتاوى مع حفظ الـ metadata.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--min-chunk", type=int, default=120)
    parser.add_argument("--no-question-header", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = ChunkConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_chunk_size=args.min_chunk,
        add_question_header=not args.no_question_header,
    )
    documents = load_input(args.input)
    chunks = chunk_documents(documents, config)
    if chunks.empty:
        return 1

    save_output(chunks, args.output)
    print("\n=== إحصاءات التقطيع ===")
    print(chunking_statistics(chunks).to_string(index=False))
    print("\n=== عيّنة من جزء واحد ===")
    print(chunks.iloc[0]["text"][:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
