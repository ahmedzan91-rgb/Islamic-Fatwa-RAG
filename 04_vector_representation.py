# -*- coding: utf-8 -*-
"""
04_vector_representation.py
===========================
المرحلة الرابعة: إعداد نموذج التضمين المتجهي (Embedding Model) الداعم للغة العربية.

النماذج المدعومة (مرتّبة حسب التوصية للنصوص الشرعية العربية):
    1. intfloat/multilingual-e5-base     — توازن ممتاز بين الجودة والحجم (768 بُعد).
       يتطلب بادئات "query: " و "passage: " وهي مطبَّقة تلقائياً هنا.
    2. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 — خفيف وسريع (384 بُعد)
       مناسب للنشر على Streamlit Cloud بموارد محدودة.
    3. UBC-NLP/ARBERTv2 / CAMeL-Lab      — نماذج عربية أصيلة (تتطلب mean pooling).

التصميم:
    - واجهة موحّدة `EmbeddingModel` تُخفي اختلافات النماذج.
    - Singleton + تخزين مؤقت (cache) لتفادي إعادة تحميل النموذج في Streamlit.
    - دعم التطبيع L2 لتمكين استخدام مسافة الجيب التمامي (cosine) بكفاءة.
    - وضع احتياطي (fallback) قائم على TF-IDF إذا تعذّر تحميل sentence-transformers،
      حتى لا يتعطّل المشروع في بيئات بلا إنترنت.

التشغيل (اختبار سريع):
    python 04_vector_representation.py --test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger("islamic_rag.embeddings")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ----------------------------------------------------------------------------- #
#                                  الإعدادات                                     #
# ----------------------------------------------------------------------------- #

DEFAULT_MODEL_NAME = os.environ.get(
    "ISLAMIC_RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
)

LIGHTWEIGHT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# النماذج التي تتطلب بادئات E5
E5_FAMILY_PREFIXES = ("intfloat/e5", "intfloat/multilingual-e5")


@dataclass
class EmbeddingConfig:
    """معاملات نموذج التضمين."""

    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 64
    normalize: bool = True          # تطبيع L2 → cosine == dot product
    device: Optional[str] = None    # None = كشف تلقائي (cuda إن وُجدت)
    max_seq_length: int = 512
    show_progress: bool = True
    use_e5_prefixes: bool = True


# ----------------------------------------------------------------------------- #
#                             نموذج التضمين الرئيسي                               #
# ----------------------------------------------------------------------------- #

class EmbeddingModel:
    """
    غلاف موحّد فوق SentenceTransformer مع دعم بادئات E5 والتطبيع.

    الاستخدام:
        model = EmbeddingModel()
        passage_vectors = model.embed_documents(["نص الفتوى ..."])
        query_vector    = model.embed_query("ما حكم كذا؟")
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension: Optional[int] = None
        self._backend: str = "sentence-transformers"
        self._load()

    # ------------------------------ التحميل ------------------------------- #

    def _resolve_device(self) -> str:
        if self.config.device:
            return self.config.device
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    @staticmethod
    def _read_dimension(model) -> int:
        """قراءة أبعاد المتجه مع دعم اختلاف أسماء الدوال بين إصدارات المكتبة."""
        for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            getter = getattr(model, attr, None)
            if callable(getter):
                try:
                    value = getter()
                    if value:
                        return int(value)
                except Exception:  # noqa: BLE001
                    continue
        return 768

    def _load(self) -> None:
        """تحميل النموذج مع سلسلة احتياطية عند الفشل."""
        device = self._resolve_device()
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            LOGGER.info("تحميل نموذج التضمين: %s على %s", self.config.model_name, device)
            self._model = SentenceTransformer(self.config.model_name, device=device)
            try:
                self._model.max_seq_length = self.config.max_seq_length
            except Exception:  # noqa: BLE001
                pass
            self._dimension = self._read_dimension(self._model)
            LOGGER.info("تم التحميل بنجاح — أبعاد المتجه: %d", self._dimension)
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("تعذّر تحميل %s (%s). المحاولة بالنموذج الخفيف...",
                           self.config.model_name, exc)

        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(LIGHTWEIGHT_MODEL_NAME, device=device)
            self.config.model_name = LIGHTWEIGHT_MODEL_NAME
            self._dimension = self._read_dimension(self._model)
            LOGGER.info("تم تحميل النموذج الخفيف — أبعاد المتجه: %d", self._dimension)
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("تعذّر تحميل أي نموذج عصبي (%s). التحوّل إلى الوضع الاحتياطي TF-IDF.", exc)

        self._backend = "tfidf"
        self._model = _TfidfFallback()
        self._dimension = self._model.dimension

    # ------------------------------ التضمين -------------------------------- #

    def _is_e5(self) -> bool:
        return self.config.use_e5_prefixes and self.config.model_name.lower().startswith(
            E5_FAMILY_PREFIXES
        )

    def _prefix(self, texts: Sequence[str], kind: str) -> List[str]:
        """إضافة بادئات E5 المطلوبة ('query: ' / 'passage: ')."""
        if not self._is_e5():
            return list(texts)
        tag = "query: " if kind == "query" else "passage: "
        return [f"{tag}{t}" for t in texts]

    def encode(self, texts: Sequence[str], kind: str = "passage") -> np.ndarray:
        """تضمين قائمة نصوص وإرجاع مصفوفة numpy بالشكل (n, dim)."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        prepared = self._prefix(texts, kind)

        if self._backend == "tfidf":
            return self._model.encode(prepared)

        vectors = self._model.encode(
            prepared,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=self.config.show_progress and len(prepared) > 256,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """تضمين وثائق/أجزاء (passages) — الواجهة المتوافقة مع LangChain و Chroma."""
        return self.encode(texts, kind="passage").tolist()

    def embed_query(self, text: str) -> List[float]:
        """تضمين استعلام واحد."""
        return self.encode([text], kind="query")[0].tolist()

    # ------------------------------ خصائص --------------------------------- #

    @property
    def dimension(self) -> int:
        return int(self._dimension or 384)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "backend": self._backend,
            "dimension": self.dimension,
            "normalize": self.config.normalize,
            "batch_size": self.config.batch_size,
            "uses_e5_prefixes": self._is_e5(),
        }


# ----------------------------------------------------------------------------- #
#                        الوضع الاحتياطي (TF-IDF)                                 #
# ----------------------------------------------------------------------------- #

class _TfidfFallback:
    """
    بديل خفيف قائم على TF-IDF + SVD (LSA) عند تعذّر تحميل النماذج العصبية.
    يضمن استمرار عمل النظام في البيئات المعزولة عن الإنترنت.
    """

    def __init__(self, dimension: int = 384) -> None:
        from sklearn.decomposition import TruncatedSVD  # noqa: PLC0415
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        self.dimension = dimension
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=60_000
        )
        self._svd = TruncatedSVD(n_components=dimension, random_state=42)
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> None:
        matrix = self._vectorizer.fit_transform(texts)
        n_components = min(self.dimension, max(2, min(matrix.shape) - 1))
        self._svd.n_components = n_components
        self._svd.fit(matrix)
        self.dimension = n_components
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(list(texts) if len(texts) > 1 else list(texts) * 2)
        matrix = self._vectorizer.transform(texts)
        vectors = self._svd.transform(matrix).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


# ----------------------------------------------------------------------------- #
#                       غلاف دالة التضمين الخاص بـ ChromaDB                        #
# ----------------------------------------------------------------------------- #

class ChromaEmbeddingFunction:
    """
    محوّل يجعل EmbeddingModel متوافقاً مع بروتوكول EmbeddingFunction في ChromaDB.
    ChromaDB تستدعي: embedding_function(input=[...]) وتتوقع List[List[float]].
    """

    def __init__(self, model: Optional[EmbeddingModel] = None) -> None:
        self.model = model or get_embedding_model()

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
        return self.model.embed_documents(list(input))

    def name(self) -> str:
        """مطلوب في إصدارات ChromaDB الحديثة لتوثيق دالة التضمين."""
        return f"islamic_rag::{self.model.model_name}"


# ----------------------------------------------------------------------------- #
#                              Singleton / Cache                                 #
# ----------------------------------------------------------------------------- #

_MODEL_SINGLETON: Optional[EmbeddingModel] = None


def get_embedding_model(config: Optional[EmbeddingConfig] = None, force_reload: bool = False) -> EmbeddingModel:
    """
    إرجاع نسخة واحدة مشتركة من النموذج (Singleton).
    ضروري في Streamlit لتفادي إعادة تحميل النموذج مع كل إعادة تشغيل للسكربت.
    """
    global _MODEL_SINGLETON  # noqa: PLW0603
    if _MODEL_SINGLETON is None or force_reload:
        _MODEL_SINGLETON = EmbeddingModel(config)
    return _MODEL_SINGLETON


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """حساب تشابه الجيب التمامي بين متجهين."""
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denominator = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denominator)


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def _run_self_test(model: EmbeddingModel) -> None:
    """اختبار ذاتي: يتحقق أن التشابه الدلالي بين نصين شرعيين مترابطين أعلى من غير المترابطين."""
    samples = [
        "ما حكم صيام يوم عرفة لغير الحاج؟",
        "صيام يوم عرفة سنة مؤكدة لغير الحاج ويكفّر سنتين.",
        "ما هي شروط صحة عقد البيع في الفقه الإسلامي؟",
    ]
    vectors = model.encode(samples, kind="passage")
    print("\n=== معلومات النموذج ===")
    for key, value in model.info().items():
        print(f"  {key}: {value}")
    print(f"\nشكل المصفوفة: {vectors.shape}")
    print(f"تشابه (سؤال عرفة ↔ جواب عرفة):  {cosine_similarity(vectors[0], vectors[1]):.4f}")
    print(f"تشابه (سؤال عرفة ↔ عقد البيع): {cosine_similarity(vectors[0], vectors[2]):.4f}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="إعداد واختبار نموذج التضمين العربي.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test", action="store_true", help="تشغيل الاختبار الذاتي.")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    model = get_embedding_model(EmbeddingConfig(model_name=args.model, batch_size=args.batch_size))
    if args.test:
        _run_self_test(model)
    else:
        print(model.info())
    return 0


if __name__ == "__main__":
    sys.exit(main())
