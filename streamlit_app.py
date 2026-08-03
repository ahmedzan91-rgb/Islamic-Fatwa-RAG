# -*- coding: utf-8 -*-
"""
streamlit_app.py
================
واجهة المستخدم التفاعلية لنظام الـ RAG الإسلامي للإجابة على الفتاوى والمسائل الشرعية.

تربط هذه الواجهة كل المراحل السابقة (01 → 07) في تطبيق واحد:
    - لوحة جانبية لضبط معاملات الاسترجاع والتوليد.
    - محادثة تفاعلية مع بثّ الإجابة تدريجياً.
    - عرض المصادر الشرعية المستشهد بها مع درجات التطابق.
    - أدوات بناء الفهرس مباشرة من الواجهة (للتشغيل المحلي).
    - مؤشر التأصيل (Groundedness) لضمان الشفافية الأكاديمية.

التشغيل:
    streamlit run streamlit_app.py

الأمان:
    لا يحتوي هذا الملف على أي مفتاح API. المفاتيح تُقرأ من st.secrets أو متغيرات البيئة.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from typing import Any, Dict, List, Optional

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- #
#                            إعداد الصفحة (يجب أن يسبق أي أمر st)                 #
# ----------------------------------------------------------------------------- #

st.set_page_config(
    page_title="المُعين الشرعي | نظام RAG للفتاوى",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------- #
#                     قراءة الأسرار (Streamlit TOML Secrets)                      #
# ----------------------------------------------------------------------------- #

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "")

# الآلية المطلوبة في مواصفات المشروع لقراءة المفتاح بأمان على Streamlit Cloud
try:
    if not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
except Exception:
    pass

if not OPENROUTER_MODEL:
    OPENROUTER_MODEL = "openai/gpt-4o-mini"

# نمرّر القيم إلى البيئة كي تلتقطها الوحدة 07 عند استيرادها
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_MODEL"] = OPENROUTER_MODEL


# ----------------------------------------------------------------------------- #
#                     تحميل الوحدات المرقّمة (01 → 07)                            #
# ----------------------------------------------------------------------------- #

def load_numbered_module(filename: str, alias: str):
    """
    استيراد ملف بايثون يبدأ اسمه برقم (مثل 06_retrieve_context.py).
    تعليمة import العادية لا تدعم الأسماء التي تبدأ بأرقام، فنستخدم importlib.
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


@st.cache_resource(show_spinner="⏳ جارٍ تحميل وحدات النظام...")
def load_pipeline_modules() -> Dict[str, Any]:
    """تحميل كل وحدات خط الأنابيب مرة واحدة وتخزينها في الذاكرة المؤقتة."""
    return {
        "documents": load_numbered_module("01_documents.py", "documents"),
        "preprocessing": load_numbered_module("02_preprocessing.py", "preprocessing"),
        "chunking": load_numbered_module("03_chunking.py", "chunking"),
        "vectors": load_numbered_module("04_vector_representation.py", "vector_representation"),
        "store": load_numbered_module("05_create_chroma_store.py", "create_chroma_store"),
        "retrieval": load_numbered_module("06_retrieve_context.py", "retrieve_context"),
        "prompting": load_numbered_module("07_prompting.py", "prompting"),
    }


@st.cache_resource(show_spinner="🧠 جارٍ تحميل نموذج التضمين العربي...")
def get_cached_retriever(persist_dir: str, collection: str, top_k: int, threshold: float):
    """إنشاء المسترجِع مرة واحدة (مكلف: يحمّل النموذج + يفتح قاعدة المتجهات)."""
    modules = load_pipeline_modules()
    retrieval_module = modules["retrieval"]
    config = retrieval_module.RetrievalConfig(
        persist_directory=persist_dir,
        collection_name=collection,
        top_k=top_k,
        relevance_threshold=threshold,
    )
    return retrieval_module.FatwaRetriever(config)


# ----------------------------------------------------------------------------- #
#                                  التنسيق (CSS)                                  #
# ----------------------------------------------------------------------------- #

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Cairo', 'Segoe UI', sans-serif; }
    .main .block-container { direction: rtl; text-align: right; padding-top: 2rem; max-width: 1200px; }
    section[data-testid="stSidebar"] { direction: rtl; text-align: right; }
    .stChatMessage { direction: rtl; text-align: right; }

    .app-header {
        background: linear-gradient(135deg, #0f5132 0%, #1a7a4c 50%, #14532d 100%);
        padding: 2rem 1.5rem; border-radius: 16px; color: #fff;
        text-align: center; margin-bottom: 1.5rem;
        box-shadow: 0 6px 24px rgba(15,81,50,0.25);
    }
    .app-header h1 { font-family: 'Amiri', serif; font-size: 2.4rem; margin: 0 0 .4rem 0; }
    .app-header p  { margin: 0; opacity: .92; font-size: 1rem; }

    .source-card {
        background: #f8faf9; border-right: 5px solid #1a7a4c; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: .85rem; direction: rtl; text-align: right;
        box-shadow: 0 2px 8px rgba(0,0,0,.05);
    }
    .source-card .meta { font-size: .82rem; color: #4b5563; margin-bottom: .5rem; }
    .source-card .body { font-size: .92rem; line-height: 1.9; color: #1f2937; }

    .badge {
        display: inline-block; padding: .18rem .65rem; border-radius: 999px;
        font-size: .74rem; font-weight: 600; margin-left: .35rem;
    }
    .badge-green { background: #d1fae5; color: #065f46; }
    .badge-amber { background: #fef3c7; color: #92400e; }
    .badge-red   { background: #fee2e2; color: #991b1b; }
    .badge-blue  { background: #dbeafe; color: #1e40af; }

    .disclaimer {
        background: #fffbeb; border: 1px solid #fcd34d; border-radius: 10px;
        padding: .9rem 1.1rem; font-size: .88rem; color: #78350f;
        direction: rtl; text-align: right; margin-top: 1rem;
    }
    .answer-box {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 1.4rem 1.6rem; line-height: 2.05; font-size: 1.02rem;
        direction: rtl; text-align: right;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- #
#                              حالة الجلسة                                        #
# ----------------------------------------------------------------------------- #

def init_session_state() -> None:
    defaults = {
        "messages": [],
        "last_sources": [],
        "last_metrics": {},
        "index_built": False,
        "query_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ----------------------------------------------------------------------------- #
#                              الترويسة                                           #
# ----------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="app-header">
        <h1>🕌 المُعين الشرعي</h1>
        <p>نظام استرجاع وتوليد معزّز (RAG) للإجابة على الفتاوى والمسائل الشرعية</p>
        <p style="font-size:.85rem; opacity:.8; margin-top:.5rem;">
            مبني على قاعدة بيانات تضم أكثر من ١٣٩ ألف فتوى — مشروع أكاديمي
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- #
#                             اللوحة الجانبية                                     #
# ----------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ الإعدادات")

    # --- حالة الاتصال ---
    st.subheader("🔐 حالة الاتصال")
    if OPENROUTER_API_KEY:
        masked = f"{OPENROUTER_API_KEY[:6]}{'*' * 8}{OPENROUTER_API_KEY[-4:]}" \
            if len(OPENROUTER_API_KEY) > 12 else "*" * len(OPENROUTER_API_KEY)
        st.success(f"✅ المفتاح مُهيَّأ\n\n`{masked}`")
    else:
        st.error("❌ لم يُعثر على `OPENROUTER_API_KEY`")
        with st.expander("كيف أضيف المفتاح؟"):
            st.markdown(
                "**على Streamlit Cloud:** من `Settings → Secrets` أضف:\n"
                "```toml\n"
                'OPENROUTER_API_KEY = "sk-or-v1-..."\n'
                'OPENROUTER_MODEL = "openai/gpt-4o-mini"\n'
                "```\n"
                "**محلياً:** أنشئ ملف `.streamlit/secrets.toml` بنفس المحتوى."
            )

    model_name = st.text_input(
        "🤖 نموذج التوليد (OpenRouter)",
        value=OPENROUTER_MODEL,
        help="مثال: openai/gpt-4o-mini أو meta-llama/llama-3.3-70b-instruct:free",
    )

    st.divider()

    # --- إعدادات الاسترجاع ---
    st.subheader("🔍 معاملات الاسترجاع")
    top_k = st.slider("عدد الفتاوى المسترجعة (Top-K)", 1, 15, 5,
                      help="كلما زاد العدد اتّسع السياق وزادت التكلفة.")
    threshold = st.slider("حدّ الثقة الأدنى", 0.0, 1.0, 0.28, 0.02,
                          help="إن لم تتجاوزه أي نتيجة، يمتنع النظام عن الإفتاء.")
    collection_name = st.text_input("اسم المجموعة", value="islamic_fatwas")
    persist_dir = st.text_input("مسار قاعدة المتجهات", value=os.path.join(BASE_DIR, "chroma_db"))

    st.divider()

    # --- إعدادات التوليد ---
    st.subheader("🎛️ معاملات التوليد")
    temperature = st.slider("درجة الحرارة (Temperature)", 0.0, 1.0, 0.2, 0.05,
                            help="القيم المنخفضة أدقّ والتزاماً بالنص الشرعي.")
    max_tokens = st.slider("أقصى طول للإجابة (توكن)", 300, 4000, 1400, 100)
    use_streaming = st.checkbox("بثّ الإجابة تدريجياً", value=True)
    show_context = st.checkbox("إظهار السياق الخام المُرسل للنموذج", value=False)

    st.divider()

    # --- حالة قاعدة المتجهات ---
    st.subheader("🗄️ قاعدة المتجهات")
    try:
        retriever = get_cached_retriever(persist_dir, collection_name, top_k, threshold)
        chunk_count = retriever.count()
        if chunk_count > 0:
            st.success(f"✅ جاهزة — {chunk_count:,} جزء مفهرس")
            st.session_state.index_built = True
        else:
            st.warning("⚠️ المجموعة فارغة — شغّل مراحل البناء أولاً.")
            st.session_state.index_built = False
    except Exception as exc:  # noqa: BLE001
        st.error(f"تعذّر فتح قاعدة المتجهات:\n\n`{exc}`")
        retriever = None
        st.session_state.index_built = False

    with st.expander("🏗️ بناء الفهرس (تشغيل محلي)"):
        st.caption("ينفّذ المراحل 01 → 05 على ملفات CSV في مجلد `data/`.")
        build_limit = st.number_input("حدّ الفتاوى (0 = الكل)", 0, 200_000, 2000, 500)
        chunk_size = st.number_input("حجم الجزء (محرف)", 300, 2000, 900, 50)
        chunk_overlap = st.number_input("التراكب (محرف)", 0, 500, 150, 25)
        if st.button("🚀 ابدأ بناء الفهرس"):
            modules = load_pipeline_modules()
            progress = st.progress(0.0, text="المرحلة 1/4: قراءة ملفات CSV...")
            try:
                docs_module = modules["documents"]
                data_dir = os.path.join(BASE_DIR, "data")
                documents = docs_module.load_documents(
                    data_dir,
                    max_rows_per_file=int(build_limit) or None,
                )
                if documents.empty:
                    st.error("لم يُعثر على بيانات في مجلد `data/`.")
                else:
                    progress.progress(0.25, text="المرحلة 2/4: تنظيف النصوص...")
                    cleaned = modules["preprocessing"].preprocess_documents(documents)

                    progress.progress(0.5, text="المرحلة 3/4: تقطيع النصوص...")
                    chunk_config = modules["chunking"].ChunkConfig(
                        chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
                    )
                    chunks = modules["chunking"].chunk_documents(cleaned, chunk_config)

                    progress.progress(0.7, text="المرحلة 4/4: بناء قاعدة المتجهات...")
                    store_config = modules["store"].ChromaConfig(
                        persist_directory=persist_dir, collection_name=collection_name
                    )
                    modules["store"].build_vector_store(chunks, store_config)

                    progress.progress(1.0, text="اكتمل البناء ✅")
                    st.success(f"تمت فهرسة {len(chunks):,} جزء من {len(cleaned):,} فتوى.")
                    st.cache_resource.clear()
            except Exception as exc:  # noqa: BLE001
                st.error(f"فشل البناء: {exc}")

    st.divider()
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_metrics = {}
        st.rerun()

    st.caption(f"عدد الاستعلامات في هذه الجلسة: {st.session_state.query_count}")


# ----------------------------------------------------------------------------- #
#                              التبويبات الرئيسية                                 #
# ----------------------------------------------------------------------------- #

tab_chat, tab_sources, tab_about = st.tabs(["💬 المحادثة", "📚 المصادر", "ℹ️ عن النظام"])


# ------------------------------ تبويب المحادثة ------------------------------- #

with tab_chat:
    # أمثلة جاهزة
    st.markdown("**أسئلة مقترحة للتجربة:**")
    example_cols = st.columns(3)
    examples = [
        "ما حكم صيام يوم عرفة لغير الحاج؟",
        "ما هي شروط صحة عقد الزواج؟",
        "ما حكم التعامل مع البنوك الربوية؟",
    ]
    clicked_example: Optional[str] = None
    for col, example in zip(example_cols, examples):
        with col:
            if st.button(example, key=f"ex_{example[:15]}"):
                clicked_example = example

    st.divider()

    # عرض تاريخ المحادثة
    for message in st.session_state.messages:
        avatar = "🕌" if message["role"] == "assistant" else "🧑"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=False)

    user_question = st.chat_input("اكتب سؤالك الشرعي هنا...") or clicked_example

    if user_question:
        st.session_state.query_count += 1
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar="🕌"):
            if not st.session_state.index_built or retriever is None:
                st.error(
                    "⚠️ قاعدة المتجهات غير جاهزة. الرجاء بناء الفهرس أولاً "
                    "من اللوحة الجانبية أو بتشغيل الملفات `01` إلى `05`."
                )
            elif not OPENROUTER_API_KEY:
                st.error("⚠️ مفتاح `OPENROUTER_API_KEY` غير مُهيَّأ — راجع اللوحة الجانبية.")
            else:
                modules = load_pipeline_modules()
                prompting = modules["prompting"]

                # 1) الاسترجاع
                with st.spinner("🔍 جارٍ البحث في قاعدة الفتاوى..."):
                    started = time.time()
                    retriever.config.top_k = top_k
                    retriever.config.relevance_threshold = threshold
                    retrieval = retriever.retrieve(user_question, top_k=top_k)
                    retrieval_time = time.time() - started

                st.session_state.last_sources = retrieval.chunks

                if not retrieval.has_sufficient_context:
                    st.warning(
                        f"⚠️ لم يُعثر على فتاوى ذات صلة كافية "
                        f"(أعلى درجة تطابق: {retrieval.max_score:.3f} < {threshold:.2f}). "
                        "سيتم إبلاغك بذلك دون إصدار حكم."
                    )

                if show_context and retrieval.context:
                    with st.expander("🔎 السياق الخام المُرسل للنموذج"):
                        st.text(retrieval.context[:6000])

                # 2) بناء البرومبت
                history = [
                    m for m in st.session_state.messages[:-1]
                    if m["role"] in {"user", "assistant"}
                ]
                messages = prompting.build_messages(
                    question=user_question,
                    context=retrieval.context,
                    has_context=retrieval.has_sufficient_context,
                    history=history,
                )

                # 3) التوليد
                generation_config = prompting.GenerationConfig(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=int(max_tokens),
                )
                client = prompting.OpenRouterClient(generation_config)

                generation_started = time.time()
                if use_streaming:
                    placeholder = st.empty()
                    answer_parts: List[str] = []
                    for token in client.chat_stream(messages):
                        answer_parts.append(token)
                        placeholder.markdown(
                            f'<div class="answer-box">{"".join(answer_parts)}▌</div>',
                            unsafe_allow_html=True,
                        )
                    answer = "".join(answer_parts).strip()
                    placeholder.markdown(
                        f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True
                    )
                    total_tokens = 0
                else:
                    with st.spinner("✍️ جارٍ صياغة الإجابة..."):
                        result = client.chat(messages)
                    answer = result.answer if result.success else f"⚠️ {result.error}"
                    total_tokens = result.total_tokens
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                generation_time = time.time() - generation_started

                # 4) مؤشرات الجودة
                grounded = prompting.verify_groundedness(answer, len(retrieval.chunks))
                st.session_state.last_metrics = {
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "chunks": len(retrieval.chunks),
                    "max_score": retrieval.max_score,
                    "grounded": grounded,
                    "tokens": total_tokens,
                }

                metric_cols = st.columns(4)
                metric_cols[0].metric("⏱️ الاسترجاع", f"{retrieval_time:.2f}s")
                metric_cols[1].metric("✍️ التوليد", f"{generation_time:.2f}s")
                metric_cols[2].metric("📚 المصادر", len(retrieval.chunks))
                metric_cols[3].metric("🎯 أعلى تطابق", f"{retrieval.max_score:.3f}")

                if grounded:
                    st.markdown(
                        '<span class="badge badge-green">✓ إجابة مؤصَّلة بالمصادر</span>',
                        unsafe_allow_html=True,
                    )
                elif retrieval.has_sufficient_context:
                    st.markdown(
                        '<span class="badge badge-amber">⚠ لم تُرصد إحالات صريحة للمصادر</span>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="disclaimer">📌 <b>تنبيه شرعي:</b> هذه الإجابة نقل آلي من '
                    'قاعدة فتاوى ولا تُعدّ فتوى شخصية. النوازل والمسائل الخاصة يُرجع فيها '
                    'إلى أهل العلم المختصين ودور الإفتاء المعتبرة.</div>',
                    unsafe_allow_html=True,
                )

                st.session_state.messages.append({"role": "assistant", "content": answer})


# ------------------------------ تبويب المصادر -------------------------------- #

with tab_sources:
    st.subheader("📚 المصادر الشرعية المسترجعة للسؤال الأخير")
    sources = st.session_state.get("last_sources", [])
    if not sources:
        st.info("لم يُطرح سؤال بعد. اطرح سؤالاً في تبويب المحادثة لعرض مصادره هنا.")
    else:
        for i, chunk in enumerate(sources, start=1):
            score_class = (
                "badge-green" if chunk.final_score >= 0.6
                else "badge-amber" if chunk.final_score >= 0.35
                else "badge-red"
            )
            with st.expander(
                f"[مصدر {i}] {chunk.title[:80] or 'فتوى رقم ' + str(chunk.fatwa_id)} "
                f"— تطابق {chunk.final_score:.3f}",
                expanded=(i == 1),
            ):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="meta">
                            <span class="badge badge-blue">رقم الفتوى: {chunk.fatwa_id}</span>
                            <span class="badge {score_class}">التطابق: {chunk.final_score:.3f}</span>
                            <span class="badge badge-blue">دلالي: {chunk.semantic_score:.3f}</span>
                            <span class="badge badge-blue">لفظي: {chunk.lexical_score:.3f}</span>
                            <br/><br/>
                            <b>المصدر:</b> {chunk.source or '—'} &nbsp;|&nbsp;
                            <b>التصنيف:</b> {chunk.category or '—'} &nbsp;|&nbsp;
                            <b>التاريخ:</b> {chunk.date or '—'}
                        </div>
                        <div class="body">{chunk.text[:2500]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if chunk.question:
                    st.caption(f"**نص السؤال الأصلي:** {chunk.question[:400]}")
                if chunk.url:
                    st.markdown(f"🔗 [الرابط الأصلي للفتوى]({chunk.url})")

        # تنزيل المصادر
        import json as _json
        st.download_button(
            "⬇️ تنزيل المصادر (JSON)",
            data=_json.dumps([c.to_dict() for c in sources], ensure_ascii=False, indent=2),
            file_name="fatwa_sources.json",
            mime="application/json",
        )


# ------------------------------ تبويب عن النظام ------------------------------- #

with tab_about:
    st.subheader("ℹ️ عن النظام والمعمارية")

    st.markdown(
        """
        ### 🏗️ معمارية خط الأنابيب (RAG Pipeline)

        | المرحلة | الملف | الوظيفة |
        |---|---|---|
        | 1 | `01_documents.py` | قراءة ملفات الفتاوى (CSV) وتوحيد المخطط |
        | 2 | `02_preprocessing.py` | تنظيف النصوص وتطبيع العربية وحذف التكرار |
        | 3 | `03_chunking.py` | تقطيع هرمي يحترم الجُمل مع حفظ الـ metadata |
        | 4 | `04_vector_representation.py` | نموذج التضمين العربي (multilingual-e5) |
        | 5 | `05_create_chroma_store.py` | بناء قاعدة المتجهات ChromaDB |
        | 6 | `06_retrieve_context.py` | استرجاع متعدد المراحل مع إعادة ترتيب |
        | 7 | `07_prompting.py` | البرومبت المحكم + OpenRouter API |
        | 8 | `streamlit_app.py` | واجهة المستخدم التفاعلية |

        ### 🛡️ ضوابط السلامة الشرعية
        - **التقيّد بالسياق:** يُمنع النموذج من الإفتاء من معرفته الداخلية.
        - **الإحالة الإلزامية:** كل حكم منسوب إلى `[مصدر ن]` مع رقم الفتوى.
        - **الامتناع عند الجهل:** حدّ ثقة أدنى يمنع التأليف عند غياب السند.
        - **التحقق البعدي:** مؤشر التأصيل يفحص وجود الإحالات في الإجابة.

        ### 🔐 إدارة المفاتيح
        لا يوجد أي مفتاح مكتوب في الشيفرة. تُقرأ القيم من `st.secrets` أو متغيرات البيئة:
        ```toml
        # .streamlit/secrets.toml
        OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxx"
        OPENROUTER_MODEL   = "openai/gpt-4o-mini"
        ```
        """
    )

    st.divider()
    modules_loaded = "pipeline" in str(st.session_state)
    info_cols = st.columns(3)
    info_cols[0].metric("📄 المراحل", "7 + واجهة")
    info_cols[1].metric("🗄️ قاعدة المتجهات", "ChromaDB")
    info_cols[2].metric("🤖 مزوّد النموذج", "OpenRouter")

    with st.expander("🧪 تشخيص البيئة"):
        try:
            modules = load_pipeline_modules()
            model = modules["vectors"].get_embedding_model()
            st.json(model.info())
        except Exception as exc:  # noqa: BLE001
            st.error(f"تعذّر تحميل نموذج التضمين: {exc}")
        st.write("**مسار قاعدة المتجهات:**", persist_dir)
        st.write("**المجموعة:**", collection_name)
        st.write("**النموذج التوليدي:**", model_name)


# ----------------------------------------------------------------------------- #
#                                  التذييل                                        #
# ----------------------------------------------------------------------------- #

st.markdown(
    """
    <hr style="margin-top:2.5rem; opacity:.25;"/>
    <div style="text-align:center; color:#6b7280; font-size:.85rem; padding-bottom:1.5rem;">
        🕌 المُعين الشرعي — نظام RAG للفتاوى الإسلامية | مشروع أكاديمي<br/>
        <span style="font-size:.78rem;">
            وَقُل رَّبِّ زِدْنِي عِلْمًا — والله تعالى أعلم
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
