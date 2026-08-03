# -*- coding: utf-8 -*-
"""
07_prompting.py
===============
المرحلة السابعة: بناء البرومبت المحكم وإرسال الاستعلام للنموذج عبر OpenRouter API.

مبادئ هندسة البرومبت في السياق الشرعي (Guardrailed Prompting):
    1. التقيّد المطلق بالسياق (Groundedness): يُمنع النموذج من الإفتاء من معرفته الداخلية.
    2. الإحالة الإلزامية (Mandatory Citation): كل حكم يجب أن يُنسب إلى [مصدر ن].
    3. الامتناع عند الجهل (Abstention): إن لم يكفِ السياق يصرّح بذلك ويحيل إلى أهل العلم.
    4. إبراز الخلاف الفقهي (Ikhtilaf Awareness): عرض أقوال أهل العلم عند تعدّدها.
    5. إخلاء المسؤولية (Disclaimer): التنبيه أن الإجابة نقل آلي لا فتوى شخصية.

الأمان:
    لا يوجد أي مفتاح API في هذا الملف. المفتاح يُقرأ حصراً من:
        1) st.secrets  (Streamlit TOML Secrets)   ← الأولوية على السحابة
        2) متغيرات البيئة (Environment Variables) ← للتشغيل المحلي

التشغيل:
    python 07_prompting.py --query "ما حكم صيام يوم عرفة لغير الحاج؟"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import requests

LOGGER = logging.getLogger("islamic_rag.prompting")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"


# ----------------------------------------------------------------------------- #
#           قراءة المفتاح والنموذج بأمان (Environment ثم Streamlit Secrets)        #
# ----------------------------------------------------------------------------- #

# 1) المحاولة الأولى: متغيرات البيئة (التشغيل المحلي)
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "")

# 2) المحاولة الثانية: أسرار Streamlit (النشر على Streamlit Cloud)
#    الآلية المطلوبة في مواصفات المشروع — محاطة بـ try/except لأن st.secrets
#    ترمي استثناءً عند غياب ملف secrets.toml أو عند التشغيل خارج Streamlit.
try:
    import streamlit as st  # noqa: PLC0415

    if not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
except Exception:
    pass

# 3) النموذج الافتراضي إن لم يُحدَّد في أي مصدر
if not OPENROUTER_MODEL:
    OPENROUTER_MODEL = "openai/gpt-4o-mini"


def get_api_credentials() -> Dict[str, str]:
    """
    إرجاع بيانات الاعتماد الحالية (مع إعادة المحاولة من st.secrets عند الحاجة).
    تُستدعى وقت التنفيذ لا وقت الاستيراد، لضمان التقاط الأسرار بعد إقلاع Streamlit.
    """
    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    model = OPENROUTER_MODEL or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    try:
        import streamlit as st  # noqa: PLC0415

        if not api_key:
            api_key = st.secrets.get("OPENROUTER_API_KEY", "")
        secret_model = st.secrets.get("OPENROUTER_MODEL", "")
        if secret_model:
            model = secret_model
    except Exception:
        pass

    return {"api_key": api_key or "", "model": model or "openai/gpt-4o-mini"}


def mask_key(key: str) -> str:
    """إخفاء المفتاح عند عرضه في السجلات أو الواجهة."""
    if not key:
        return "غير مُعرَّف"
    return f"{key[:6]}{'*' * 10}{key[-4:]}" if len(key) > 12 else "*" * len(key)


# ----------------------------------------------------------------------------- #
#                          استيراد وحدة الاسترجاع                                 #
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


# ----------------------------------------------------------------------------- #
#                                 قوالب البرومبت                                  #
# ----------------------------------------------------------------------------- #

SYSTEM_PROMPT = """أنت "المُعين الشرعي"، مساعد بحثي متخصص في عرض الفتاوى والمسائل الشرعية الإسلامية، مبني على قاعدة بيانات موثّقة من الفتاوى.

# دورك الدقيق
أنت لست مفتياً، بل ناقل أمين ومُلخِّص دقيق لما ورد في الفتاوى المرفقة في "السياق". مهمتك عرض ما قاله أهل العلم كما ورد، لا إنشاء حكم جديد.

# القواعد الملزِمة (لا يجوز مخالفتها بأي حال)
1. **التقيّد بالسياق**: أجب حصراً من نصوص الفتاوى الواردة في قسم "السياق المسترجع". يُمنع منعاً باتاً استخدام معلوماتك الداخلية لإصدار حكم شرعي غير موجود في السياق.
2. **الإحالة الإلزامية**: اذكر بعد كل حكم أو معلومة رقم مصدرها هكذا: [مصدر 1]. ولا تذكر مصدراً لم يرد في السياق.
3. **الامتناع عند عدم الكفاية**: إذا كان السياق لا يجيب على السؤال أو كان بعيداً عنه، فقل صراحة: "لم أجد في قاعدة الفتاوى المتاحة ما يجيب على هذا السؤال بشكل مباشر"، ثم انصح بمراجعة دار إفتاء معتبرة. ولا تحاول التخمين إطلاقاً.
4. **إبراز الخلاف**: إذا تعدّدت أقوال أهل العلم في السياق، اعرضها جميعاً منسوبةً لمصادرها دون ترجيح شخصي منك.
5. **الأمانة في النقل**: لا تُحرّف نص الفتوى، ولا تُعمّم حكماً خاصاً بحالة معيّنة على كل الحالات، ونبّه على القيود والشروط التي ذكرتها الفتوى.
6. **حدود الاختصاص**: إذا كان السؤال خارج المجال الشرعي (طبي، قانوني وضعي، تقني...) فاعتذر بلطف ووضّح تخصصك.
7. **الأدب في الخطاب**: التزم لغة عربية فصيحة، هادئة، محترمة، خالية من التشدّد أو التهوين، وابدأ بالتحية المناسبة عند الاقتضاء.

# بنية الإجابة المطلوبة
اكتب إجابتك بصيغة Markdown وفق الترتيب التالي:

**الخلاصة:** (جملة أو جملتان تلخّصان الحكم مباشرة)

**التفصيل:**
(شرح مرتّب بالنقاط، مع الإحالة [مصدر ن] بعد كل نقطة، وذكر الأدلة والشروط كما وردت)

**الخلاف الفقهي:** (يُذكر فقط إن وُجد في السياق)

**المصادر المعتمدة:**
(قائمة بأرقام الفتاوى المستشهد بها)

> **تنبيه:** هذه الإجابة نقل آلي من قاعدة فتاوى، وليست فتوى شخصية. للمسائل الخاصة والنوازل يُرجع إلى أهل العلم المختصين ودور الإفتاء المعتبرة."""


USER_PROMPT_TEMPLATE = """# السياق المسترجع من قاعدة الفتاوى

{context}

# ------------------------------------------------------------

# سؤال المستخدم
{question}

# التعليمات النهائية
أجب على السؤال أعلاه معتمداً **حصراً** على السياق المسترجع، ملتزماً ببنية الإجابة والقواعد المذكورة في تعليمات النظام. تذكّر: الإحالة إلى [مصدر ن] إلزامية بعد كل حكم، والامتناع واجب إن لم يكفِ السياق."""


NO_CONTEXT_PROMPT_TEMPLATE = """لم يُعثر في قاعدة الفتاوى على نصوص ذات صلة كافية بسؤال المستخدم التالي:

"{question}"

مهمتك الآن محدودة جداً: اكتب رداً عربياً مهذّباً ومختصراً (٣-٥ أسطر) يتضمّن:
1. الإفصاح بوضوح أنه لم يُعثر على فتوى مطابقة في قاعدة البيانات المتاحة.
2. اقتراح إعادة صياغة السؤال بمصطلحات فقهية أدق، أو تحديد المذهب/الحالة.
3. النصح بمراجعة دار إفتاء معتبرة.

**ممنوع منعاً باتاً** إصدار أي حكم شرعي أو ذكر أي معلومة فقهية من معرفتك الداخلية."""


# ----------------------------------------------------------------------------- #
#                                  الإعدادات                                     #
# ----------------------------------------------------------------------------- #

@dataclass
class GenerationConfig:
    """معاملات توليد الإجابة."""

    model: str = ""                 # فارغ = يُقرأ من الأسرار/البيئة
    temperature: float = 0.2        # منخفضة لضمان الالتزام بالنص الشرعي
    max_tokens: int = 1400
    top_p: float = 0.9
    timeout: int = 90
    max_retries: int = 3
    retry_backoff: float = 2.0
    stream: bool = False
    site_url: str = "https://islamic-rag.streamlit.app"
    site_name: str = "Islamic Fatwa RAG"


@dataclass
class GenerationResult:
    """نتيجة التوليد مع البيانات التشخيصية."""

    answer: str = ""
    model_used: str = ""
    success: bool = False
    error: str = ""
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    grounded: bool = False


# ----------------------------------------------------------------------------- #
#                                بناء البرومبت                                    #
# ----------------------------------------------------------------------------- #

def build_messages(
    question: str,
    context: str,
    has_context: bool = True,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    بناء قائمة الرسائل المرسلة للنموذج.

    Args:
        question: سؤال المستخدم.
        context: السياق المسترجع من المرحلة السادسة.
        has_context: هل السياق كافٍ؟ (يحدّد أي قالب يُستخدم)
        history: محادثة سابقة اختيارية [{"role": "...", "content": "..."}]
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for turn in history[-6:]:  # آخر ٣ تبادلات فقط لضبط حجم البرومبت
            if turn.get("role") in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})

    if has_context and context.strip():
        user_content = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    else:
        user_content = NO_CONTEXT_PROMPT_TEMPLATE.format(question=question)

    messages.append({"role": "user", "content": user_content})
    return messages


def estimate_tokens(text: str) -> int:
    """تقدير تقريبي لعدد التوكنات في النص العربي (≈ 2.2 محرف/توكن)."""
    return int(len(text) / 2.2)


# ----------------------------------------------------------------------------- #
#                              عميل OpenRouter                                    #
# ----------------------------------------------------------------------------- #

class OpenRouterClient:
    """
    عميل خفيف لواجهة OpenRouter المتوافقة مع OpenAI.
    يدعم إعادة المحاولة الأسّية ومعالجة أخطاء الحصص (429) والانقطاع.
    """

    def __init__(self, config: Optional[GenerationConfig] = None) -> None:
        self.config = config or GenerationConfig()
        credentials = get_api_credentials()
        self.api_key = credentials["api_key"]
        self.model = self.config.model or credentials["model"]

    # ------------------------------ الرؤوس -------------------------------- #

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # الرؤوس التالية اختيارية لكنها موصى بها من OpenRouter للتعريف بالتطبيق
            "HTTP-Referer": self.config.site_url,
            "X-Title": self.config.site_name,
        }

    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------ التوليد ------------------------------- #

    def chat(self, messages: List[Dict[str, str]]) -> GenerationResult:
        """إرسال الرسائل إلى OpenRouter وإرجاع نتيجة التوليد."""
        if not self.is_configured():
            return GenerationResult(
                success=False,
                error=(
                    "مفتاح OPENROUTER_API_KEY غير مُعرَّف. "
                    "أضِفه في أسرار Streamlit (Secrets) أو في متغيرات البيئة."
                ),
                model_used=self.model,
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": False,
        }

        started = time.time()
        last_error = ""

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(
                    OPENROUTER_ENDPOINT,
                    headers=self._headers(),
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.config.timeout,
                )

                # ضمان فكّ UTF-8: بعض الاستجابات تأتي بلا charset فيفترض
                # requests ترميز latin-1 ويشوّه النص العربي.
                response.encoding = "utf-8"

                if response.status_code == 200:
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    answer = (choice.get("message") or {}).get("content", "").strip()
                    usage = data.get("usage") or {}
                    return GenerationResult(
                        answer=answer,
                        model_used=data.get("model", self.model),
                        success=bool(answer),
                        error="" if answer else "استجابة فارغة من النموذج.",
                        latency_seconds=round(time.time() - started, 2),
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                        total_tokens=int(usage.get("total_tokens", 0) or 0),
                    )

                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"خطأ مؤقت {response.status_code}: {response.text[:200]}"  # response.encoding مضبوط أعلاه
                    wait = self.config.retry_backoff ** attempt
                    LOGGER.warning("%s — إعادة المحاولة بعد %.1f ثانية.", last_error, wait)
                    time.sleep(wait)
                    continue

                if response.status_code == 401:
                    return GenerationResult(
                        success=False, model_used=self.model,
                        error="مفتاح API غير صالح (401). تحقّق من OPENROUTER_API_KEY.",
                    )
                if response.status_code == 402:
                    return GenerationResult(
                        success=False, model_used=self.model,
                        error="الرصيد غير كافٍ (402). جرّب نموذجاً مجانياً مثل :free.",
                    )

                return GenerationResult(
                    success=False, model_used=self.model,
                    error=f"خطأ {response.status_code}: {response.text[:300]}",
                    latency_seconds=round(time.time() - started, 2),
                )

            except requests.exceptions.Timeout:
                last_error = f"انتهت المهلة بعد {self.config.timeout} ثانية."
                time.sleep(self.config.retry_backoff ** attempt)
            except requests.exceptions.RequestException as exc:
                last_error = f"خطأ في الاتصال: {exc}"
                time.sleep(self.config.retry_backoff ** attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = f"خطأ غير متوقع: {exc}"
                break

        return GenerationResult(
            success=False, model_used=self.model,
            error=last_error or "فشل الاتصال بعد كل المحاولات.",
            latency_seconds=round(time.time() - started, 2),
        )

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """بثّ الإجابة تدريجياً (SSE) — يُستخدم في واجهة Streamlit لتحسين التجربة."""
        if not self.is_configured():
            yield "⚠️ مفتاح OPENROUTER_API_KEY غير مُعرَّف."
            return

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": True,
        }

        try:
            with requests.post(
                OPENROUTER_ENDPOINT,
                headers=self._headers(),
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=self.config.timeout,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    response.encoding = "utf-8"
                    yield f"⚠️ خطأ {response.status_code}: {response.text[:200]}"
                    return

                # ── حاسم للعربية ──
                # لا نستخدم decode_unicode=True لأن requests يفترض latin-1
                # عند غياب charset من ترويسة SSE (وفق RFC 2616)، فيشوّه
                # كل حرف عربي: "الخلاصة" تصبح "Ø§ÙØ®ÙØ§ØµØ©".
                # نقرأ البايتات الخام ونفكّها بـ UTF-8 صراحةً، مع مخزن مؤقت
                # يحمي المحارف متعددة البايتات من البتر على حدود الدفعات.
                buffer = b""
                for raw_chunk in response.iter_content(chunk_size=None):
                    if not raw_chunk:
                        continue
                    buffer += raw_chunk

                    while b"\n" in buffer:
                        raw_line, buffer = buffer.split(b"\n", 1)
                        line = raw_line.decode("utf-8", errors="replace").strip()

                        if not line or not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(chunk)
                            delta = (data.get("choices") or [{}])[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:  # noqa: BLE001
            yield f"\n\n⚠️ انقطع البثّ: {exc}"

    def list_free_models(self) -> List[str]:
        """جلب النماذج المجانية المتاحة على OpenRouter (تنتهي بـ :free)."""
        try:
            response = requests.get(OPENROUTER_MODELS_ENDPOINT, timeout=20)
            if response.status_code != 200:
                return []
            response.encoding = "utf-8"
            models = response.json().get("data", [])
            return sorted(m["id"] for m in models if str(m.get("id", "")).endswith(":free"))
        except Exception:  # noqa: BLE001
            return []


# ----------------------------------------------------------------------------- #
#                            التحقق من التأصيل                                    #
# ----------------------------------------------------------------------------- #

def looks_corrupted(text: str) -> bool:
    """
    فحص أخير قبل عرض الإجابة: هل هي مشوّهة (موجابيك)؟

    خطّ دفاع أخير. حتى بعد ضبط الترميز، قد يعيد مزوّد أو وسيط نصاً تالفاً.
    عرض نص غير مقروء أسوأ من رسالة خطأ صريحة، خاصة في سياق شرعي
    قد يُساء فيه فهم كلمة مشوّهة.
    """
    if not text or len(text) < 30:
        return False
    sample = text[:1500]
    latin_ext = sum(1 for ch in sample if ch in "ØÙÚÛÃÂÐÑðŸ™Œ")
    arabic = sum(1 for ch in sample if "\u0600" <= ch <= "\u06FF")
    # نص فيه كثافة عالية من اللاتينية الممدودة ولا عربية تُذكر = مشوّه
    return latin_ext > 15 and arabic < len(sample) * 0.10


def verify_groundedness(answer: str, num_sources: int) -> bool:
    """
    تحقّق آلي بسيط: هل استشهدت الإجابة فعلاً بالمصادر؟
    يُستخدم كمؤشر جودة يُعرض في الواجهة (Guardrail ما بعد التوليد).
    """
    if not answer or num_sources == 0:
        return False
    import re as _re
    citations = _re.findall(r"\[\s*مصدر\s*(\d+)\s*\]", answer)
    if not citations:
        return False
    return any(1 <= int(c) <= num_sources for c in citations)


# ----------------------------------------------------------------------------- #
#                          خط الأنابيب الكامل (RAG)                               #
# ----------------------------------------------------------------------------- #

def answer_question(
    question: str,
    top_k: int = 5,
    generation_config: Optional[GenerationConfig] = None,
    retrieval_config: Optional[Any] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> GenerationResult:
    """
    خط الأنابيب الكامل: استرجاع → بناء برومبت → توليد → تحقّق.

    Returns:
        GenerationResult يحوي الإجابة والمصادر والمؤشرات التشخيصية.
    """
    retrieval_module = load_numbered_module("06_retrieve_context.py", "retrieve_context")
    retriever = retrieval_module.get_retriever(retrieval_config)
    retrieval = retriever.retrieve(question, top_k=top_k)

    messages = build_messages(
        question=question,
        context=retrieval.context,
        has_context=retrieval.has_sufficient_context,
        history=history,
    )

    client = OpenRouterClient(generation_config)
    result = client.chat(messages)
    result.sources = retrieval.sources_table()
    result.grounded = verify_groundedness(result.answer, len(retrieval.chunks))
    return result


# ----------------------------------------------------------------------------- #
#                                    CLI                                         #
# ----------------------------------------------------------------------------- #

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="توليد إجابة شرعية مؤصَّلة عبر OpenRouter.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--show-prompt", action="store_true", help="عرض البرومبت دون إرسال.")
    parser.add_argument("--list-free-models", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    credentials = get_api_credentials()
    print(f"النموذج: {credentials['model']} | المفتاح: {mask_key(credentials['api_key'])}")

    if args.list_free_models:
        for model_id in OpenRouterClient().list_free_models():
            print(model_id)
        return 0

    if args.show_prompt:
        retrieval_module = load_numbered_module("06_retrieve_context.py", "retrieve_context")
        retrieval = retrieval_module.get_retriever().retrieve(args.query, top_k=args.top_k)
        for message in build_messages(args.query, retrieval.context, retrieval.has_sufficient_context):
            print(f"\n===== [{message['role']}] =====\n{message['content'][:3000]}")
        return 0

    config = GenerationConfig(model=args.model, temperature=args.temperature)
    result = answer_question(args.query, top_k=args.top_k, generation_config=config)

    print("\n=== الإجابة ===")
    print(result.answer if result.success else f"فشل: {result.error}")
    print("\n=== المؤشرات ===")
    print(f"النموذج: {result.model_used} | الزمن: {result.latency_seconds}s | "
          f"التوكنات: {result.total_tokens} | مؤصَّلة: {result.grounded}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
