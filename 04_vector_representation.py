# -*- coding: utf-8 -*-
"""
04_vector_representation.py
===========================
المرحلة الرابعة: إعداد نموذج التضمين المتجهي (Embedding Model) الداعم للغة العربية.

╔══════════════════════════════════════════════════════════════════════════╗
║  معمارية الواجهات الخلفية المتدرّجة (Tiered Backend Architecture)          ║
╚══════════════════════════════════════════════════════════════════════════╝

يكتشف النظام البيئة تلقائياً ويختار أفضل واجهة خلفية متاحة:

  المستوى 1 │ neural  │ sentence-transformers + torch
            │         │ النموذج: intfloat/multilingual-e5-base (768 بُعد)
            │         │ الجودة: ممتازة | الذاكرة: ~1.2 GB | يتطلب: torch
            │
  المستوى 2 │ neural  │ النموذج الخفيف MiniLM متعدد اللغات (384 بُعد)
            │  -lite  │ الجودة: جيدة | الذاكرة: ~450 MB | يتطلب: torch
            │
  المستوى 3 │ tfidf   │ TF-IDF على n-grams حرفية + SVD (LSA)
            │         │ الجودة: مقبولة للعربية | الذاكرة: ~80 MB | بلا torch

لماذا هذا التدرّج؟
    Streamlit Community Cloud يفرض حدّ 1 GB للذاكرة، ويستخدم حالياً Python 3.14
    التي لا تتوفر لها wheels لـ torch — ما يجعل تثبيته يفشل أو يستنزف البناء.
    المستوى 3 يضمن أن التطبيق **يعمل دائماً** حتى بلا مكتبات عصبية.

ملاحظة على جودة المستوى 3 للعربية:
    استخدام n-grams حرفية (2-4) مع `analyzer="char_wb"` مناسب للعربية تحديداً،
    لأنه يلتقط الجذور والسوابق واللواحق (ال، ون، ات...) دون تجذيع صريح،
    ويقاوم اختلاف التشكيل والإملاء. مع إعادة الترتيب اللفظية في المرحلة 06
    تصبح النتائج مقبولة عملياً وإن كانت دون المستوى العصبي دلالياً.

التشغيل:
    python 04_vector_representation.py --test
    python 04_vector_representation.py --backend tfidf --test
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- #
#                                  الإعدادات                                     #
# ----------------------------------------------------------------------------- #

DEFAULT_MODEL_NAME = os.environ.get(
    "ISLAMIC_RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
)
LIGHTWEIGHT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# فرض واجهة خلفية معيّنة عبر متغير البيئة: neural | tfidf | auto
FORCED_BACKEND = os.environ.get("ISLAMIC_RAG_BACKEND", "auto").strip().lower()

E5_FAMILY_PREFIXES = ("intfloat/e5", "intfloat/multilingual-e5")

DEFAULT_TFIDF_STATE = os.path.join(BASE_DIR, "artifacts", "tfidf_embedder.pkl")


@dataclass
class EmbeddingConfig:
    """معاملات نموذج التضمين."""

    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 64
    normalize: bool = True
    device: Optional[str] = None
    max_seq_length: int = 512
    show_progress: bool = True
    use_e5_prefixes: bool = True
    backend: str = FORCED_BACKEND          # auto | neural | tfidf
    tfidf_dimension: int = 384
    tfidf_state_path: str = DEFAULT_TFIDF_STATE


# ----------------------------------------------------------------------------- #
#                          كشف توفّر المكتبات العصبية                             #
# ----------------------------------------------------------------------------- #

def neural_stack_available() -> bool:
    """
    التحقق من توفّر torch + sentence-transformers فعلياً (لا مجرد وجود الاسم).
    نستخدم find_spec لتفادي تكلفة الاستيراد الكامل عند عدم الحاجة.
    """
    try:
        import importlib.util  # noqa: PLC0415

        for package in ("torch", "sentence_transformers"):
            if importlib.util.find_spec(package) is None:
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- #
#                     المستوى 3: مُضمِّن TF-IDF قابل للحفظ                        #
# ----------------------------------------------------------------------------- #

class TfidfEmbedder:
    """
    مُضمِّن قائم على TF-IDF (n-grams حرفية) + SVD، بلا أي اعتماد على torch.

    مصمَّم خصيصاً للعربية:
      - `char_wb` مع n-grams (2,4) يلتقط الأنماط الصرفية العربية.
      - يقاوم اختلاف التشكيل والإملاء دون تجذيع صريح.
      - قابل للحفظ والتحميل، فلا يُعاد تدريبه مع كل إقلاع للتطبيق.

    ملاحظة مهمة: يجب استدعاء fit() على المتن قبل الفهرسة،
    وإلا فلن تكون المتجهات ذات معنى.
    """

    def __init__(self, dimension: int = 384, state_path: Optional[str] = None) -> None:
        self.dimension = dimension
        self.state_path = state_path or DEFAULT_TFIDF_STATE
        self._vectorizer = None
        self._svd = None
        self._fitted = False

    # ------------------------------ التدريب ------------------------------- #

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        """تدريب المُضمِّن على متن النصوص."""
        from sklearn.decomposition import TruncatedSVD  # noqa: PLC0415
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        corpus = [t for t in texts if t and t.strip()]
        if len(corpus) < 2:
            corpus = (corpus or ["نص افتراضي"]) * 2

        LOGGER.info("تدريب مُضمِّن TF-IDF على %d نص...", len(corpus))
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=60_000,
            sublinear_tf=True,
            min_df=1,
        )
        matrix = self._vectorizer.fit_transform(corpus)

        n_components = int(min(self.dimension, max(2, min(matrix.shape) - 1)))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(matrix)

        self.dimension = n_components
        self._fitted = True
        explained = float(self._svd.explained_variance_ratio_.sum())
        LOGGER.info(
            "اكتمل التدريب — الأبعاد: %d | التباين المُفسَّر: %.1f%%",
            n_components, 100.0 * explained,
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------ التضمين -------------------------------- #

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """تحويل النصوص إلى متجهات مطبّعة (L2)."""
        if not self._fitted:
            raise RuntimeError(
                "مُضمِّن TF-IDF غير مُدرَّب.\n"
                "السبب المرجّح: بُني الفهرس على جهاز آخر ولم تُرفع حالة المُضمِّن معه.\n"
                "الحل: أعِد بناء الفهرس من تبويب «📤 البيانات والفهرسة»، "
                "أو ارفع ملف artifacts/tfidf_embedder.pkl مع مجلد chroma_db."
            )
        matrix = self._vectorizer.transform(list(texts))
        vectors = self._svd.transform(matrix).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    # --------------------------- الحفظ والتحميل ---------------------------- #

    def save(self, path: Optional[str] = None) -> str:
        """حفظ حالة المُضمِّن على القرص."""
        path = path or self.state_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "svd": self._svd,
                    "dimension": self.dimension,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        LOGGER.info("حُفظت حالة مُضمِّن TF-IDF في: %s", path)
        return path

    def load(self, path: Optional[str] = None) -> bool:
        """تحميل حالة محفوظة. يُرجع True عند النجاح."""
        path = path or self.state_path
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                state = pickle.load(fh)
            self._vectorizer = state["vectorizer"]
            self._svd = state["svd"]
            self.dimension = int(state["dimension"])
            self._fitted = True
            LOGGER.info("حُمّلت حالة مُضمِّن TF-IDF (%d بُعد) من: %s", self.dimension, path)
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("تعذّر تحميل حالة TF-IDF: %s", exc)
            return False


# ----------------------------------------------------------------------------- #
#                             نموذج التضمين الرئيسي                               #
# ----------------------------------------------------------------------------- #

class EmbeddingModel:
    """
    واجهة موحّدة فوق كل الواجهات الخلفية.

    الاستخدام:
        model = EmbeddingModel()
        model.fit_if_needed(corpus_texts)          # ضروري لواجهة tfidf فقط
        passages = model.embed_documents([...])
        query    = model.embed_query("ما حكم ...؟")
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension: Optional[int] = None
        self._backend: str = "unknown"
        self._load()

    # ------------------------------ التحميل ------------------------------- #

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

    def _try_load_neural(self, model_name: str) -> bool:
        """محاولة تحميل نموذج عصبي. يُرجع True عند النجاح."""
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            device = self._resolve_device()
            LOGGER.info("تحميل نموذج التضمين العصبي: %s على %s", model_name, device)
            self._model = SentenceTransformer(model_name, device=device)
            try:
                self._model.max_seq_length = self.config.max_seq_length
            except Exception:  # noqa: BLE001
                pass
            self._dimension = self._read_dimension(self._model)
            self.config.model_name = model_name
            self._backend = "sentence-transformers"
            LOGGER.info("✅ تم التحميل — أبعاد المتجه: %d", self._dimension)
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("تعذّر تحميل %s: %s", model_name, exc)
            return False

    def _load_tfidf(self) -> None:
        """
        تفعيل واجهة TF-IDF مع محاولة استعادة حالة محفوظة.

        نبحث في عدة مواضع لأن الفهرس قد يُبنى على جهاز ويُستخدم على آخر:
          1. المسار المُعدّ (artifacts/)
          2. داخل مجلد chroma_db (يُرفع عادةً مع الفهرس)
          3. جذر المشروع
        """
        self._model = TfidfEmbedder(
            dimension=self.config.tfidf_dimension,
            state_path=self.config.tfidf_state_path,
        )

        for candidate in self._state_search_paths():
            if self._model.load(candidate):
                break

        self._dimension = self._model.dimension
        self._backend = "tfidf"
        LOGGER.info(
            "تم تفعيل واجهة TF-IDF (بلا torch) — مُدرَّب: %s | الأبعاد: %d",
            self._model.is_fitted, self._dimension,
        )

    def _state_search_paths(self) -> List[str]:
        """المواضع المحتملة لحالة مُضمِّن TF-IDF، مرتّبة حسب الأولوية."""
        name = "tfidf_embedder.pkl"
        return [
            self.config.tfidf_state_path,
            os.path.join(BASE_DIR, "chroma_db", name),
            os.path.join(BASE_DIR, "artifacts", name),
            os.path.join(BASE_DIR, name),
        ]

    def _load(self) -> None:
        """اختيار وتحميل الواجهة الخلفية وفق الإعداد والبيئة."""
        backend = self.config.backend

        if backend == "tfidf":
            self._load_tfidf()
            return

        if backend in ("auto", "neural"):
            if not neural_stack_available():
                if backend == "neural":
                    LOGGER.error(
                        "طُلبت الواجهة العصبية لكن torch/sentence-transformers غير مثبّتة. "
                        "ثبّت requirements-full.txt أو استخدم backend=tfidf."
                    )
                else:
                    LOGGER.info(
                        "المكتبات العصبية غير متوفّرة (torch/sentence-transformers) — "
                        "التحوّل إلى واجهة TF-IDF الخفيفة."
                    )
                self._load_tfidf()
                return

            if self._try_load_neural(self.config.model_name):
                return
            if self._try_load_neural(LIGHTWEIGHT_MODEL_NAME):
                return

            LOGGER.warning("فشل تحميل كل النماذج العصبية — التحوّل إلى TF-IDF.")
            self._load_tfidf()
            return

        LOGGER.warning("واجهة غير معروفة '%s' — استخدام auto.", backend)
        self.config.backend = "auto"
        self._load()

    # --------------------------- تدريب TF-IDF ------------------------------ #

    def needs_fitting(self) -> bool:
        """هل تحتاج الواجهة الحالية إلى تدريب قبل الاستخدام؟"""
        return self._backend == "tfidf" and not self._model.is_fitted

    def _save_state_everywhere(self) -> None:
        """
        حفظ حالة المُضمِّن في artifacts/ و chroma_db/ معاً.

        السبب: مجلد chroma_db هو ما يُرفع عادةً مع المشروع، فحفظ الحالة بداخله
        يضمن أن الفهرس ومُضمِّنه يسافران معاً. بدون ذلك يصبح الفهرس عديم القيمة
        على جهاز آخر، لأن متجهات TF-IDF لا معنى لها دون المفردات التي دُرِّبت عليها.
        """
        for path in {
            self.config.tfidf_state_path,
            os.path.join(BASE_DIR, "chroma_db", "tfidf_embedder.pkl"),
        }:
            try:
                self._model.save(path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("تعذّر حفظ حالة المُضمِّن في %s: %s", path, exc)

    def fit_if_needed(self, corpus: Sequence[str], save: bool = True) -> None:
        """
        تدريب المُضمِّن على المتن إن كانت الواجهة تتطلب ذلك.
        تُستدعى من مرحلة بناء الفهرس (05) قبل الفهرسة.
        """
        if self._backend != "tfidf":
            return
        if self._model.is_fitted:
            return
        self._model.fit(corpus)
        self._dimension = self._model.dimension
        if save:
            self._save_state_everywhere()

    def refit(self, corpus: Sequence[str], save: bool = True) -> None:
        """إعادة تدريب إجبارية (عند تغيّر المتن جذرياً)."""
        if self._backend != "tfidf":
            return
        self._model.fit(corpus)
        self._dimension = self._model.dimension
        if save:
            self._save_state_everywhere()

    # ------------------------------ التضمين -------------------------------- #

    def _is_e5(self) -> bool:
        return (
            self._backend == "sentence-transformers"
            and self.config.use_e5_prefixes
            and self.config.model_name.lower().startswith(E5_FAMILY_PREFIXES)
        )

    def _prefix(self, texts: Sequence[str], kind: str) -> List[str]:
        """إضافة بادئات E5 المطلوبة ('query: ' / 'passage: ')."""
        if not self._is_e5():
            return list(texts)
        tag = "query: " if kind == "query" else "passage: "
        return [f"{tag}{t}" for t in texts]

    def encode(self, texts: Sequence[str], kind: str = "passage") -> np.ndarray:
        """تضمين قائمة نصوص وإرجاع مصفوفة (n, dim)."""
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
        """تضمين وثائق/أجزاء — الواجهة المتوافقة مع LangChain و Chroma."""
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
        if self._backend == "tfidf":
            return "tfidf-char-ngram-svd"
        return self.config.model_name

    @property
    def is_neural(self) -> bool:
        return self._backend == "sentence-transformers"

    def info(self) -> dict:
        return {
            "backend": self._backend,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "is_neural": self.is_neural,
            "needs_fitting": self.needs_fitting(),
            "normalize": self.config.normalize,
            "uses_e5_prefixes": self._is_e5(),
        }

    def quality_note(self) -> str:
        """وصف موجز لجودة الواجهة الحالية — يُعرض في الواجهة الرسومية."""
        if self._backend == "sentence-transformers":
            if "e5-base" in self.config.model_name:
                return "جودة ممتازة — تضمين عصبي دلالي (768 بُعد)"
            return "جودة جيدة — تضمين عصبي خفيف"
        return (
            "جودة مقبولة — تضمين لفظي (TF-IDF) بلا مكتبات عصبية. "
            "للحصول على فهم دلالي أعمق ثبّت requirements-full.txt محلياً."
        )


# ----------------------------------------------------------------------------- #
#                       غلاف دالة التضمين الخاص بـ ChromaDB                        #
# ----------------------------------------------------------------------------- #

class ChromaEmbeddingFunction:
    """محوّل يجعل EmbeddingModel متوافقاً مع بروتوكول EmbeddingFunction في ChromaDB."""

    def __init__(self, model: Optional[EmbeddingModel] = None) -> None:
        self.model = model or get_embedding_model()

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002
        return self.model.embed_documents(list(input))

    def name(self) -> str:
        return f"islamic_rag::{self.model.model_name}"


# ----------------------------------------------------------------------------- #
#                              Singleton / Cache                                 #
# ----------------------------------------------------------------------------- #

_MODEL_SINGLETON: Optional[EmbeddingModel] = None


def get_embedding_model(
    config: Optional[EmbeddingConfig] = None, force_reload: bool = False
) -> EmbeddingModel:
    """إرجاع نسخة واحدة مشتركة من النموذج (Singleton)."""
    global _MODEL_SINGLETON  # noqa: PLW0603
    if _MODEL_SINGLETON is None or force_reload:
        _MODEL_SINGLETON = EmbeddingModel(config)
    return _MODEL_SINGLETON


def reset_embedding_model() -> None:
    """إعادة تعيين النسخة المشتركة (بعد إعادة بناء الفهرس مثلاً)."""
    global _MODEL_SINGLETON  # noqa: PLW0603
    _MODEL_SINGLETON = None


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """حساب تشابه الجيب التمامي بين متجهين."""
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denominator = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denominator)


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def _run_self_test(model: EmbeddingModel) -> None:
    """اختبار ذاتي: التشابه بين نصين شرعيين مترابطين يجب أن يفوق غير المترابطين."""
    samples = [
        "ما حكم صيام يوم عرفة لغير الحاج؟",
        "صيام يوم عرفة سنة مؤكدة لغير الحاج ويكفّر سنتين.",
        "ما هي شروط صحة عقد البيع في الفقه الإسلامي؟",
        "يشترط لصحة البيع التراضي وأن يكون المبيع معلوماً مقدوراً على تسليمه.",
    ]

    if model.needs_fitting():
        LOGGER.info("الواجهة تتطلب تدريباً — تدريب على عيّنة الاختبار.")
        model.fit_if_needed(samples, save=False)

    vectors = model.encode(samples, kind="passage")

    print("\n=== معلومات النموذج ===")
    for key, value in model.info().items():
        print(f"  {key:18}: {value}")
    print(f"\n  ملاحظة الجودة  : {model.quality_note()}")
    print(f"\nشكل المصفوفة: {vectors.shape}")

    related_1 = cosine_similarity(vectors[0], vectors[1])
    related_2 = cosine_similarity(vectors[2], vectors[3])
    unrelated = cosine_similarity(vectors[0], vectors[2])

    print(f"\n  تشابه مترابط  (عرفة ↔ جواب عرفة) : {related_1:+.4f}")
    print(f"  تشابه مترابط  (بيع  ↔ جواب بيع)  : {related_2:+.4f}")
    print(f"  تشابه غير مترابط (عرفة ↔ بيع)     : {unrelated:+.4f}")

    passed = related_1 > unrelated and related_2 > unrelated
    print(f"\n  نتيجة الاختبار: {'✅ ناجح' if passed else '⚠️ ضعيف التمييز'}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="إعداد واختبار نموذج التضمين العربي.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--backend", default=FORCED_BACKEND,
                        choices=["auto", "neural", "tfidf"])
    parser.add_argument("--test", action="store_true", help="تشغيل الاختبار الذاتي.")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    model = get_embedding_model(
        EmbeddingConfig(
            model_name=args.model, batch_size=args.batch_size, backend=args.backend
        )
    )
    if args.test:
        _run_self_test(model)
    else:
        print(model.info())
        print(model.quality_note())
    return 0


if __name__ == "__main__":
    sys.exit(main())
