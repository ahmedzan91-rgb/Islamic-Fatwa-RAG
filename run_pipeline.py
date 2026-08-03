# -*- coding: utf-8 -*-
"""
run_pipeline.py
===============
مُشغّل خط الأنابيب الكامل: ينفّذ المراحل 01 → 05 بأمر واحد.
(ملف مساعد اختياري، لا يُغني عن الملفات المستقلة المطلوبة في المشروع.)

التشغيل:
    python run_pipeline.py --data-dir data --chunk-size 900 --overlap 150
    python run_pipeline.py --limit 5000        # وضع التطوير السريع
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from typing import Optional

LOGGER = logging.getLogger("islamic_rag.pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_numbered_module(filename: str, alias: str):
    """استيراد ملف يبدأ اسمه برقم."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, os.path.join(BASE_DIR, filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"تعذّر تحميل: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="تشغيل خط أنابيب الـ RAG الإسلامي كاملاً.")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--artifacts", default=os.path.join(BASE_DIR, "artifacts"))
    parser.add_argument("--persist", default=os.path.join(BASE_DIR, "chroma_db"))
    parser.add_argument("--collection", default="islamic_fatwas")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="حدّ الفتاوى لكل ملف (تطوير).")
    parser.add_argument("--reset", action="store_true", help="حذف المجموعة قبل البناء.")
    args = parser.parse_args(argv)

    os.makedirs(args.artifacts, exist_ok=True)
    started = time.time()

    # ------------------------- المرحلة 1: التحميل ------------------------- #
    LOGGER.info("═══ المرحلة 1/5: قراءة ملفات الفتاوى ═══")
    documents_module = load_numbered_module("01_documents.py", "documents")
    documents = documents_module.load_documents(args.data_dir, max_rows_per_file=args.limit)
    if documents.empty:
        LOGGER.error("لا توجد بيانات في %s — ضع ملفات CSV هناك أولاً.", args.data_dir)
        return 1
    documents_module.save_documents(documents, os.path.join(args.artifacts, "01_documents.parquet"))

    # ------------------------- المرحلة 2: التنظيف ------------------------- #
    LOGGER.info("═══ المرحلة 2/5: تنظيف النصوص الشرعية ═══")
    preprocessing_module = load_numbered_module("02_preprocessing.py", "preprocessing")
    cleaned = preprocessing_module.preprocess_documents(documents)
    if cleaned.empty:
        LOGGER.error("لم تنجُ أي فتوى من التنظيف.")
        return 1
    preprocessing_module.save_output(cleaned, os.path.join(args.artifacts, "02_clean.parquet"))

    # ------------------------- المرحلة 3: التقطيع ------------------------- #
    LOGGER.info("═══ المرحلة 3/5: تقطيع النصوص ═══")
    chunking_module = load_numbered_module("03_chunking.py", "chunking")
    chunk_config = chunking_module.ChunkConfig(
        chunk_size=args.chunk_size, chunk_overlap=args.overlap
    )
    chunks = chunking_module.chunk_documents(cleaned, chunk_config)
    if chunks.empty:
        LOGGER.error("لم يُنتج التقطيع أي أجزاء.")
        return 1
    chunking_module.save_output(chunks, os.path.join(args.artifacts, "03_chunks.parquet"))

    # --------------------- المرحلة 4: نموذج التضمين ----------------------- #
    LOGGER.info("═══ المرحلة 4/5: تحميل نموذج التضمين ═══")
    vectors_module = load_numbered_module("04_vector_representation.py", "vector_representation")
    model = vectors_module.get_embedding_model()
    LOGGER.info("النموذج: %s", model.info())

    # -------------------- المرحلة 5: قاعدة المتجهات ----------------------- #
    LOGGER.info("═══ المرحلة 5/5: بناء قاعدة المتجهات ChromaDB ═══")
    store_module = load_numbered_module("05_create_chroma_store.py", "create_chroma_store")
    store_config = store_module.ChromaConfig(
        persist_directory=args.persist,
        collection_name=args.collection,
        batch_size=args.batch_size,
        reset_collection=args.reset,
    )
    collection = store_module.build_vector_store(chunks, store_config)

    elapsed = (time.time() - started) / 60.0
    LOGGER.info("═══ اكتمل خط الأنابيب في %.1f دقيقة ═══", elapsed)
    print("\n" + "=" * 60)
    print(f"  الفتاوى المعالَجة : {len(cleaned):,}")
    print(f"  الأجزاء المفهرسة : {collection.count():,}")
    print(f"  قاعدة المتجهات   : {args.persist}")
    print(f"  الزمن الكلي      : {elapsed:.1f} دقيقة")
    print("=" * 60)
    print("\n▶ الخطوة التالية:  streamlit run streamlit_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
