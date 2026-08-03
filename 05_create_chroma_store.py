# -*- coding: utf-8 -*-
"""
05_create_chroma_store.py
=========================
المرحلة الخامسة: إنشاء قاعدة البيانات المتجهة ChromaDB وتخزين التضمينات فيها.

المسؤوليات:
    1. إنشاء/فتح عميل Chroma دائم (PersistentClient) على القرص.
    2. إنشاء المجموعة (Collection) بمسافة الجيب التمامي (cosine).
    3. الإدخال على دفعات (batched upsert) لدعم مئات الآلاف من الأجزاء.
    4. استئناف الفهرسة بعد الانقطاع (skip للمعرّفات الموجودة مسبقاً).
    5. التحقق من سلامة الفهرس وإصدار تقرير.

ملاحظة أداء:
    لفهرسة ~139 ألف فتوى (≈ 300-400 ألف جزء) على CPU قد تستغرق العملية ساعات.
    يُنصح بتشغيل هذا الملف مرة واحدة محلياً (أو على GPU) ثم رفع مجلد chroma_db
    مع المشروع، أو استخدام --limit أثناء التطوير.

التشغيل:
    python 05_create_chroma_store.py --input artifacts/03_chunks.parquet \
                                     --persist chroma_db \
                                     --collection islamic_fatwas \
                                     --batch-size 512
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger("islamic_rag.chroma")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "artifacts", "03_chunks.parquet")
DEFAULT_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_COLLECTION = "islamic_fatwas"


# ----------------------------------------------------------------------------- #
#              استيراد الوحدات المرقّمة (أسماؤها تبدأ بأرقام)                       #
# ----------------------------------------------------------------------------- #

def load_numbered_module(filename: str, alias: str):
    """
    استيراد ملف يبدأ اسمه برقم (لا تدعمه تعليمة import العادية).

    مهم: نتحقق من sys.modules أولاً. بدون هذا الفحص تُعاد تهيئة الوحدة
    مع كل استيراد فتنشأ نسخ متعددة لها متغيّرات عامة منفصلة — وهو ما كان
    يسبّب وجود أكثر من singleton لنموذج التضمين، فيُدرَّب أحدها ويُستخدم آخر.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"تعذّر تحميل الوحدة: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_vectors = load_numbered_module("04_vector_representation.py", "vector_representation")
EmbeddingConfig = _vectors.EmbeddingConfig
get_embedding_model = _vectors.get_embedding_model


# ----------------------------------------------------------------------------- #
#                                  الإعدادات                                     #
# ----------------------------------------------------------------------------- #

@dataclass
class ChromaConfig:
    """معاملات بناء قاعدة المتجهات."""

    persist_directory: str = DEFAULT_PERSIST_DIR
    collection_name: str = DEFAULT_COLLECTION
    batch_size: int = 512
    distance_metric: str = "cosine"      # cosine | l2 | ip
    reset_collection: bool = False       # حذف المجموعة قبل البناء
    skip_existing: bool = True           # تخطّي الأجزاء المفهرسة مسبقاً (للاستئناف)
    hnsw_construction_ef: int = 200
    hnsw_M: int = 32


# ----------------------------------------------------------------------------- #
#                              إدارة عميل Chroma                                  #
# ----------------------------------------------------------------------------- #

def get_chroma_client(persist_directory: str):
    """إنشاء عميل Chroma دائم على القرص."""
    import chromadb  # noqa: PLC0415
    from chromadb.config import Settings  # noqa: PLC0415

    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    LOGGER.info("عميل ChromaDB جاهز على المسار: %s", persist_directory)
    return client


def get_or_create_collection(client, config: ChromaConfig, embedding_function=None):
    """
    إنشاء أو فتح المجموعة.

    ملاحظة تصميمية: نمرّر التضمينات جاهزة (precomputed) عند الإضافة بدل الاعتماد
    على embedding_function داخل Chroma، لأن ذلك يمنحنا تحكماً في الدفعات وبادئات E5.
    دالة التضمين تُمرَّر فقط لتوثيق المجموعة ولتسهيل الاستعلام النصي المباشر.
    """
    metadata = {
        "hnsw:space": config.distance_metric,
        "hnsw:construction_ef": config.hnsw_construction_ef,
        "hnsw:M": config.hnsw_M,
        "description": "قاعدة متجهات الفتاوى والمسائل الشرعية",
    }
    if config.reset_collection:
        try:
            client.delete_collection(config.collection_name)
            LOGGER.warning("تم حذف المجموعة السابقة: %s", config.collection_name)
        except Exception:  # noqa: BLE001
            pass

    try:
        collection = client.get_or_create_collection(
            name=config.collection_name,
            metadata=metadata,
            embedding_function=embedding_function,
        )
    except Exception:  # noqa: BLE001 — توافق مع إصدارات Chroma الأقدم
        collection = client.get_or_create_collection(
            name=config.collection_name, metadata=metadata
        )
    LOGGER.info("المجموعة '%s' جاهزة — العناصر الحالية: %d",
                config.collection_name, collection.count())
    return collection


def get_existing_ids(collection, candidate_ids: Sequence[str], probe_batch: int = 5000) -> set:
    """جلب المعرّفات الموجودة مسبقاً في المجموعة (لدعم الاستئناف)."""
    existing: set = set()
    if collection.count() == 0:
        return existing
    for start in range(0, len(candidate_ids), probe_batch):
        batch = list(candidate_ids[start:start + probe_batch])
        try:
            result = collection.get(ids=batch, include=[])
            existing.update(result.get("ids", []))
        except Exception:  # noqa: BLE001
            continue
    return existing


# ----------------------------------------------------------------------------- #
#                                بناء الفهرس                                     #
# ----------------------------------------------------------------------------- #

def _row_to_metadata(row: pd.Series) -> dict:
    """تحويل صف الجزء إلى metadata متوافقة مع Chroma (أنواع بسيطة فقط)."""
    def s(key: str, limit: int = 500) -> str:
        value = row.get(key, "")
        return str(value)[:limit] if value is not None else ""

    def i(key: str) -> int:
        try:
            return int(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "doc_id": s("doc_id", 100),
        "fatwa_id": s("fatwa_id", 100),
        "title": s("title", 500),
        "question": s("question", 1000),
        "category": s("category", 200),
        "source": s("source", 200),
        "url": s("url", 500),
        "date": s("date", 100),
        "chunk_index": i("chunk_index"),
        "chunk_total": i("chunk_total"),
        "char_count": i("char_count"),
        "raw_text": s("raw_text", 4000),   # نحفظ النص الأصلي للعرض دون الترويسة
    }


def build_vector_store(
    chunks_df: pd.DataFrame,
    config: Optional[ChromaConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    limit: Optional[int] = None,
):
    """
    بناء قاعدة المتجهات وتخزين الأجزاء فيها.

    Args:
        chunks_df: مخرجات المرحلة الثالثة.
        config: إعدادات Chroma.
        embedding_config: إعدادات نموذج التضمين.
        limit: حدّ أقصى لعدد الأجزاء (للتطوير/الاختبار).

    Returns:
        كائن المجموعة (collection).
    """
    config = config or ChromaConfig()
    if chunks_df.empty:
        raise ValueError("لا توجد أجزاء للفهرسة.")

    if limit:
        chunks_df = chunks_df.head(limit).copy()

    model = get_embedding_model(embedding_config)
    LOGGER.info("نموذج التضمين: %s", model.info())

    # واجهة TF-IDF تتطلب تدريباً على المتن قبل الفهرسة، وإلا فالمتجهات بلا معنى.
    if model.needs_fitting():
        LOGGER.info("تدريب مُضمِّن TF-IDF على متن الفتاوى قبل الفهرسة...")
        model.fit_if_needed(chunks_df["text"].astype(str).tolist())

    client = get_chroma_client(config.persist_directory)
    collection = get_or_create_collection(client, config)

    all_ids = chunks_df["chunk_id"].astype(str).tolist()
    skip_ids: set = set()
    if config.skip_existing and not config.reset_collection:
        skip_ids = get_existing_ids(collection, all_ids)
        if skip_ids:
            LOGGER.info("تخطّي %d جزء مفهرس مسبقاً (وضع الاستئناف).", len(skip_ids))

    pending = chunks_df[~chunks_df["chunk_id"].astype(str).isin(skip_ids)].reset_index(drop=True)
    if pending.empty:
        LOGGER.info("كل الأجزاء مفهرسة بالفعل — لا حاجة لإعادة البناء.")
        return collection

    total = len(pending)
    LOGGER.info("بدء فهرسة %d جزء بحجم دفعة %d...", total, config.batch_size)
    started = time.time()
    indexed = 0

    for start in range(0, total, config.batch_size):
        batch = pending.iloc[start:start + config.batch_size]

        ids = batch["chunk_id"].astype(str).tolist()
        documents = batch["text"].astype(str).tolist()
        metadatas = [_row_to_metadata(row) for _, row in batch.iterrows()]

        try:
            embeddings = model.encode(documents, kind="passage").tolist()
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            indexed += len(ids)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("فشلت الدفعة %d-%d: %s", start, start + len(batch), exc)
            continue

        elapsed = time.time() - started
        rate = indexed / max(elapsed, 1e-6)
        remaining = (total - indexed) / max(rate, 1e-6)
        LOGGER.info(
            "التقدّم: %d/%d (%.1f%%) | %.1f جزء/ث | المتبقي ≈ %.1f دقيقة",
            indexed, total, 100.0 * indexed / total, rate, remaining / 60.0,
        )

    LOGGER.info(
        "اكتملت الفهرسة: %d جزء في %.1f دقيقة. إجمالي عناصر المجموعة: %d",
        indexed, (time.time() - started) / 60.0, collection.count(),
    )
    return collection


# ----------------------------------------------------------------------------- #
#                              التحقق والتشخيص                                   #
# ----------------------------------------------------------------------------- #

def verify_store(config: Optional[ChromaConfig] = None, sample_query: str = "ما حكم صيام يوم عرفة؟") -> dict:
    """التحقق من سلامة قاعدة المتجهات عبر استعلام تجريبي."""
    config = config or ChromaConfig()
    client = get_chroma_client(config.persist_directory)
    collection = get_or_create_collection(client, config)
    model = get_embedding_model()

    count = collection.count()
    report = {"collection": config.collection_name, "count": count, "status": "فارغة" if count == 0 else "جاهزة"}
    if count == 0:
        return report

    results = collection.query(
        query_embeddings=[model.embed_query(sample_query)],
        n_results=min(3, count),
        include=["documents", "metadatas", "distances"],
    )
    report["sample_query"] = sample_query
    report["top_hits"] = [
        {
            "fatwa_id": meta.get("fatwa_id", ""),
            "title": meta.get("title", "")[:80],
            "distance": round(float(dist), 4),
        }
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]
    return report


def load_input(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".csv"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"لم يُعثر على ملف الأجزاء: {path}")
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(
        path, dtype=str, keep_default_na=False
    )


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="بناء قاعدة المتجهات ChromaDB للفتاوى.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--persist", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="حدّ أقصى للأجزاء (تطوير).")
    parser.add_argument("--reset", action="store_true", help="حذف المجموعة قبل البناء.")
    parser.add_argument("--verify-only", action="store_true", help="التحقق فقط دون بناء.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = ChromaConfig(
        persist_directory=args.persist,
        collection_name=args.collection,
        batch_size=args.batch_size,
        reset_collection=args.reset,
    )

    if args.verify_only:
        print(verify_store(config))
        return 0

    chunks = load_input(args.input)
    build_vector_store(chunks, config, limit=args.limit)

    print("\n=== تقرير التحقق ===")
    report = verify_store(config)
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
