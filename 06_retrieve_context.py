# -*- coding: utf-8 -*-
"""
06_retrieve_context.py
======================
المرحلة السادسة: استرجاع النصوص الشرعية الأكثر صلة بالسؤال (Context Retrieval).

المعمارية المطبَّقة (Multi-Stage Retrieval):
    1. البحث الدلالي الكثيف (Dense Search) عبر ChromaDB — يلتقط المعنى لا اللفظ.
    2. توسيع الاستعلام (Query Expansion) بالمرادفات الفقهية — "حكم/فتوى/يجوز".
    3. إعادة الترتيب (Re-ranking) بمزيج من:
         - درجة التشابه الدلالي
         - تداخل الكلمات المفتاحية (lexical overlap) لتفادي الانحراف الدلالي
         - تنويع المصادر (MMR-lite) لتفادي هيمنة فتوى واحدة على السياق
    4. تجميع الأجزاء المتجاورة من الفتوى نفسها (Parent Document Merging).
    5. حدّ أدنى للثقة (relevance threshold) — إن لم يتجاوزه أي نتيجة نُبلّغ بعدم وجود سند،
       وهو ضروري شرعاً لمنع النموذج من التأليف.

التشغيل:
    python 06_retrieve_context.py --query "ما حكم صيام يوم عرفة لغير الحاج؟" --top-k 5
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("islamic_rag.retrieval")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_numbered_module(filename: str, alias: str):
    """استيراد ملف يبدأ اسمه برقم."""
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"تعذّر تحميل الوحدة: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_preprocessing = load_numbered_module("02_preprocessing.py", "preprocessing")
_vectors = load_numbered_module("04_vector_representation.py", "vector_representation")
_store = load_numbered_module("05_create_chroma_store.py", "create_chroma_store")

normalize_arabic = _preprocessing.normalize_arabic
get_embedding_model = _vectors.get_embedding_model
ChromaConfig = _store.ChromaConfig
get_chroma_client = _store.get_chroma_client
get_or_create_collection = _store.get_or_create_collection


# ----------------------------------------------------------------------------- #
#                                  الإعدادات                                     #
# ----------------------------------------------------------------------------- #

# كلمات وقف عربية شائعة تُستبعد من حساب التداخل اللفظي
ARABIC_STOPWORDS = {
    "في", "من", "على", "الى", "الي", "عن", "مع", "هذا", "هذه", "ذلك", "التي", "الذي",
    "ما", "لا", "ان", "او", "ثم", "قد", "كان", "يكون", "هل", "هو", "هي", "به", "له",
    "كل", "بعد", "قبل", "عند", "حتي", "حتى", "اذا", "لكن", "بين", "غير", "سوي", "كما",
    "و", "ف", "ب", "ل", "ك",
}

# مرادفات فقهية لتوسيع الاستعلام
FIQH_SYNONYMS: Dict[str, List[str]] = {
    "حكم": ["فتوى", "يجوز", "حلال", "حرام", "مشروعية"],
    "صلاة": ["الصلوات", "فريضة", "نافلة", "ركعة"],
    "صيام": ["صوم", "افطار", "رمضان"],
    "زكاة": ["نصاب", "صدقة", "مال"],
    "طلاق": ["فراق", "عدة", "خلع"],
    "زواج": ["نكاح", "عقد", "مهر", "خطبة"],
    "بيع": ["شراء", "تجارة", "معاملة", "عقد"],
    "ربا": ["فائدة", "قرض", "بنك"],
    "حج": ["عمرة", "احرام", "طواف", "منسك"],
    "طهارة": ["وضوء", "غسل", "نجاسة", "تيمم"],
}


@dataclass
class RetrievalConfig:
    """معاملات الاسترجاع."""

    persist_directory: str = os.path.join(BASE_DIR, "chroma_db")
    collection_name: str = "islamic_fatwas"
    top_k: int = 5                       # عدد النتائج النهائية
    fetch_k: int = 25                    # عدد المرشحين قبل إعادة الترتيب
    relevance_threshold: float = 0.28    # أدنى درجة تشابه مقبولة (0-1)
    semantic_weight: float = 0.75        # وزن التشابه الدلالي في الدرجة المركّبة
    lexical_weight: float = 0.25         # وزن التداخل اللفظي
    max_per_fatwa: int = 2               # أقصى عدد أجزاء من الفتوى نفسها (تنويع)
    enable_query_expansion: bool = True
    merge_adjacent_chunks: bool = True
    max_context_chars: int = 9000        # سقف حجم السياق المرسل للنموذج


@dataclass
class RetrievedChunk:
    """نتيجة استرجاع واحدة مع درجاتها وبياناتها الوصفية."""

    chunk_id: str
    text: str
    fatwa_id: str = ""
    doc_id: str = ""
    title: str = ""
    question: str = ""
    category: str = ""
    source: str = ""
    url: str = ""
    date: str = ""
    chunk_index: int = 0
    distance: float = 1.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> str:
        """صياغة مرجع الفتوى للاقتباس في الإجابة."""
        parts = []
        if self.fatwa_id:
            parts.append(f"فتوى رقم {self.fatwa_id}")
        if self.source:
            parts.append(f"المصدر: {self.source}")
        if self.category:
            parts.append(f"التصنيف: {self.category}")
        return " — ".join(parts) if parts else "مصدر غير محدّد"


@dataclass
class RetrievalResult:
    """حزمة نتيجة الاسترجاع الكاملة التي تُسلَّم لمرحلة البرومبت."""

    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    context: str = ""
    has_sufficient_context: bool = False
    max_score: float = 0.0
    total_candidates: int = 0

    def sources_table(self) -> List[Dict[str, Any]]:
        """جدول المصادر لعرضه في واجهة Streamlit."""
        return [
            {
                "رقم الفتوى": c.fatwa_id,
                "العنوان": c.title[:70],
                "التصنيف": c.category,
                "المصدر": c.source,
                "درجة التطابق": round(c.final_score, 3),
                "الرابط": c.url,
            }
            for c in self.chunks
        ]


# ----------------------------------------------------------------------------- #
#                              دوال مساعدة                                       #
# ----------------------------------------------------------------------------- #

def tokenize_arabic(text: str) -> List[str]:
    """تفكيك النص العربي إلى كلمات دلالية بعد التطبيع وحذف كلمات الوقف."""
    normalized = normalize_arabic(text)
    tokens = re.findall(r"[\u0600-\u06FF0-9a-zA-Z]+", normalized)
    return [t for t in tokens if len(t) > 1 and t not in ARABIC_STOPWORDS]


def expand_query(query: str) -> str:
    """توسيع الاستعلام بمرادفات فقهية لرفع معدل الاسترجاع (recall)."""
    tokens = set(tokenize_arabic(query))
    expansions: List[str] = []
    for key, synonyms in FIQH_SYNONYMS.items():
        if normalize_arabic(key) in tokens:
            expansions.extend(synonyms)
    if not expansions:
        return query
    return f"{query} {' '.join(dict.fromkeys(expansions))}"


def lexical_overlap(query_tokens: Sequence[str], text: str) -> float:
    """معامل جاكارد المرجّح بين كلمات الاستعلام وكلمات النص المسترجع."""
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize_arabic(text))
    if not text_tokens:
        return 0.0
    matched = sum(1 for t in set(query_tokens) if t in text_tokens)
    return matched / len(set(query_tokens))


def distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """تحويل مسافة Chroma إلى درجة تشابه في المجال [0,1]."""
    if metric == "cosine":
        return max(0.0, min(1.0, 1.0 - float(distance)))
    return 1.0 / (1.0 + float(distance))


# ----------------------------------------------------------------------------- #
#                              المسترجِع الرئيسي                                  #
# ----------------------------------------------------------------------------- #

class FatwaRetriever:
    """
    مسترجِع الفتاوى: يغلّف ChromaDB ونموذج التضمين ومنطق إعادة الترتيب.

    الاستخدام:
        retriever = FatwaRetriever()
        result = retriever.retrieve("ما حكم صيام يوم عرفة؟")
        print(result.context)
    """

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.config = config or RetrievalConfig()
        self.model = get_embedding_model()
        chroma_config = ChromaConfig(
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name,
        )
        self.client = get_chroma_client(chroma_config.persist_directory)
        self.collection = get_or_create_collection(self.client, chroma_config)
        LOGGER.info("المسترجِع جاهز — عناصر المجموعة: %d", self.count())

    # ------------------------------ الأساسيات ----------------------------- #

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:  # noqa: BLE001
            return 0

    def _query_chroma(self, query_text: str, n_results: int, where: Optional[dict] = None) -> dict:
        """تنفيذ الاستعلام المتجهي على ChromaDB."""
        embedding = self.model.embed_query(query_text)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": max(1, min(n_results, max(self.count(), 1))),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    # ------------------------------ الاسترجاع ------------------------------ #

    def _to_chunks(self, raw: dict, query_tokens: Sequence[str]) -> List[RetrievedChunk]:
        """تحويل مخرجات Chroma الخام إلى كائنات RetrievedChunk مع حساب الدرجات."""
        chunks: List[RetrievedChunk] = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return chunks

        ids = raw["ids"][0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            display_text = metadata.get("raw_text") or document or ""
            semantic = distance_to_similarity(distance)
            lexical = lexical_overlap(query_tokens, display_text)
            final = (
                self.config.semantic_weight * semantic
                + self.config.lexical_weight * lexical
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=display_text,
                    fatwa_id=str(metadata.get("fatwa_id", "")),
                    doc_id=str(metadata.get("doc_id", "")),
                    title=str(metadata.get("title", "")),
                    question=str(metadata.get("question", "")),
                    category=str(metadata.get("category", "")),
                    source=str(metadata.get("source", "")),
                    url=str(metadata.get("url", "")),
                    date=str(metadata.get("date", "")),
                    chunk_index=int(metadata.get("chunk_index", 0) or 0),
                    distance=float(distance),
                    semantic_score=round(semantic, 4),
                    lexical_score=round(lexical, 4),
                    final_score=round(final, 4),
                )
            )
        return chunks

    def _diversify(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """تنويع النتائج: تحديد عدد الأجزاء المأخوذة من الفتوى الواحدة."""
        seen: Dict[str, int] = {}
        selected: List[RetrievedChunk] = []
        for chunk in chunks:
            key = chunk.doc_id or chunk.fatwa_id or chunk.chunk_id
            count = seen.get(key, 0)
            if count >= self.config.max_per_fatwa:
                continue
            seen[key] = count + 1
            selected.append(chunk)
            if len(selected) >= self.config.top_k:
                break
        return selected

    def _merge_adjacent(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """دمج الأجزاء المتجاورة من الفتوى نفسها لاستعادة تسلسل الحكم الشرعي."""
        if not self.config.merge_adjacent_chunks:
            return chunks

        by_doc: Dict[str, List[RetrievedChunk]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk.doc_id or chunk.fatwa_id, []).append(chunk)

        merged: List[RetrievedChunk] = []
        for group in by_doc.values():
            group.sort(key=lambda c: c.chunk_index)
            current = group[0]
            for nxt in group[1:]:
                if nxt.chunk_index == current.chunk_index + 1:
                    current.text = f"{current.text}\n{nxt.text}"
                    current.final_score = max(current.final_score, nxt.final_score)
                else:
                    merged.append(current)
                    current = nxt
            merged.append(current)

        merged.sort(key=lambda c: c.final_score, reverse=True)
        return merged

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """
        استرجاع السياق الشرعي الأنسب للسؤال.

        Args:
            query: سؤال المستخدم.
            top_k: عدد النتائج المطلوبة (يتجاوز الإعداد الافتراضي).
            category_filter: تقييد البحث بتصنيف معيّن.
            source_filter: تقييد البحث بمصدر معيّن.
        """
        query = (query or "").strip()
        if not query:
            return RetrievalResult(query=query, has_sufficient_context=False)

        if top_k:
            self.config.top_k = top_k

        if self.count() == 0:
            LOGGER.warning("قاعدة المتجهات فارغة — شغّل 05_create_chroma_store.py أولاً.")
            return RetrievalResult(query=query, has_sufficient_context=False)

        search_query = expand_query(query) if self.config.enable_query_expansion else query
        query_tokens = tokenize_arabic(query)

        where: Dict[str, Any] = {}
        if category_filter:
            where["category"] = category_filter
        if source_filter:
            where["source"] = source_filter

        raw = self._query_chroma(search_query, self.config.fetch_k, where or None)
        candidates = self._to_chunks(raw, query_tokens)
        total_candidates = len(candidates)

        # إعادة الترتيب بالدرجة المركّبة
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        selected = self._diversify(candidates)
        selected = self._merge_adjacent(selected)[: self.config.top_k]

        max_score = max((c.final_score for c in selected), default=0.0)
        sufficient = bool(selected) and max_score >= self.config.relevance_threshold

        result = RetrievalResult(
            query=query,
            chunks=selected,
            context=self.format_context(selected),
            has_sufficient_context=sufficient,
            max_score=round(max_score, 4),
            total_candidates=total_candidates,
        )
        LOGGER.info(
            "الاسترجاع: %d مرشح → %d نتيجة | أعلى درجة=%.3f | سياق كافٍ=%s",
            total_candidates, len(selected), max_score, sufficient,
        )
        return result

    # ------------------------------ التنسيق -------------------------------- #

    def format_context(self, chunks: Sequence[RetrievedChunk]) -> str:
        """
        تنسيق السياق المسترجع بصيغة مرقّمة قابلة للاقتباس،
        بحيث يستطيع النموذج الإشارة إلى [مصدر ١] و[مصدر ٢] في إجابته.
        """
        if not chunks:
            return ""

        blocks: List[str] = []
        used_chars = 0
        for i, chunk in enumerate(chunks, start=1):
            header = f"### [مصدر {i}] {chunk.citation()}"
            if chunk.title:
                header += f"\nالعنوان: {chunk.title}"
            if chunk.question:
                snippet = chunk.question[:300]
                header += f"\nنص السؤال الأصلي: {snippet}"
            block = f"{header}\nنص الفتوى:\n{chunk.text}\n"

            if used_chars + len(block) > self.config.max_context_chars:
                remaining = self.config.max_context_chars - used_chars
                if remaining > 400:
                    blocks.append(block[:remaining] + "\n[...]")
                break
            blocks.append(block)
            used_chars += len(block)

        return "\n---\n".join(blocks).strip()

    def list_categories(self, limit: int = 5000) -> List[str]:
        """جلب قائمة التصنيفات المتاحة (لفلاتر واجهة Streamlit)."""
        try:
            data = self.collection.get(limit=limit, include=["metadatas"])
            values = {
                str(m.get("category", "")).strip()
                for m in data.get("metadatas", []) if m and m.get("category")
            }
            return sorted(v for v in values if v)
        except Exception:  # noqa: BLE001
            return []

    def list_sources(self, limit: int = 5000) -> List[str]:
        """جلب قائمة المصادر المتاحة."""
        try:
            data = self.collection.get(limit=limit, include=["metadatas"])
            values = {
                str(m.get("source", "")).strip()
                for m in data.get("metadatas", []) if m and m.get("source")
            }
            return sorted(v for v in values if v)
        except Exception:  # noqa: BLE001
            return []


# ----------------------------------------------------------------------------- #
#                              واجهة مبسّطة                                       #
# ----------------------------------------------------------------------------- #

_RETRIEVER_SINGLETON: Optional[FatwaRetriever] = None


def get_retriever(config: Optional[RetrievalConfig] = None) -> FatwaRetriever:
    """إرجاع نسخة مشتركة من المسترجِع (Singleton) لتسريع Streamlit."""
    global _RETRIEVER_SINGLETON  # noqa: PLW0603
    if _RETRIEVER_SINGLETON is None:
        _RETRIEVER_SINGLETON = FatwaRetriever(config)
    return _RETRIEVER_SINGLETON


def retrieve_context(query: str, top_k: int = 5) -> RetrievalResult:
    """دالة اختصار للاستدعاء السريع من الوحدات الأخرى."""
    return get_retriever().retrieve(query, top_k=top_k)


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="استرجاع السياق الشرعي من قاعدة المتجهات.")
    parser.add_argument("--query", required=True, help="السؤال الشرعي.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--persist", default=os.path.join(BASE_DIR, "chroma_db"))
    parser.add_argument("--collection", default="islamic_fatwas")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    retriever = FatwaRetriever(
        RetrievalConfig(
            persist_directory=args.persist,
            collection_name=args.collection,
            top_k=args.top_k,
        )
    )
    result = retriever.retrieve(args.query, category_filter=args.category, source_filter=args.source)

    print(f"\nالسؤال: {result.query}")
    print(f"سياق كافٍ: {result.has_sufficient_context} | أعلى درجة: {result.max_score}")
    print("\n=== المصادر ===")
    for row in result.sources_table():
        print(row)
    print("\n=== السياق المُنسَّق ===")
    print(result.context[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
