# -*- coding: utf-8 -*-
"""
streamlit_app.py
================
واجهة المستخدم التفاعلية لنظام الـ RAG الإسلامي للإجابة على الفتاوى والمسائل الشرعية.

تربط هذه الواجهة كل المراحل (01 → 08) في تطبيق واحد:
    💬 المحادثة   — سؤال وجواب مؤصَّل بالمصادر مع بثّ تدريجي
    📤 البيانات   — رفع ملفات الفتاوى وحفظها وبناء الفهرس
    📚 المصادر    — عرض الفتاوى المستشهد بها ودرجات التطابق
    ℹ️ عن النظام  — المعمارية والتشخيص

التشغيل:
    streamlit run streamlit_app.py

الأمان:
    لا يحتوي هذا الملف على أي مفتاح API. المفاتيح تُقرأ من st.secrets أو متغيرات البيئة.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ----------------------------------------------------------------------------- #
#                    إعداد الصفحة (يجب أن يسبق أي أمر st آخر)                     #
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

# الآلية المعتمدة لقراءة المفتاح بأمان على Streamlit Cloud
try:
    if not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
except Exception:
    pass

if not OPENROUTER_MODEL:
    OPENROUTER_MODEL = "openai/gpt-4o-mini"

# تمرير القيم إلى البيئة كي تلتقطها الوحدة 07
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_MODEL"] = OPENROUTER_MODEL


# ----------------------------------------------------------------------------- #
#                     تحميل الوحدات المرقّمة (01 → 08)                            #
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
    """تحميل كل وحدات خط الأنابيب مرة واحدة."""
    return {
        "documents": load_numbered_module("01_documents.py", "documents"),
        "preprocessing": load_numbered_module("02_preprocessing.py", "preprocessing"),
        "chunking": load_numbered_module("03_chunking.py", "chunking"),
        "vectors": load_numbered_module("04_vector_representation.py", "vector_representation"),
        "store": load_numbered_module("05_create_chroma_store.py", "create_chroma_store"),
        "retrieval": load_numbered_module("06_retrieve_context.py", "retrieve_context"),
        "prompting": load_numbered_module("07_prompting.py", "prompting"),
        "data_manager": load_numbered_module("08_data_manager.py", "data_manager"),
    }


@st.cache_resource(show_spinner="🧠 جارٍ تهيئة محرّك البحث الدلالي...")
def get_cached_retriever(persist_dir: str, collection: str, cache_key: int = 0):
    """إنشاء المسترجِع مرة واحدة. cache_key يسمح بإبطال المخبأ بعد إعادة البناء."""
    modules = load_pipeline_modules()
    retrieval_module = modules["retrieval"]
    config = retrieval_module.RetrievalConfig(
        persist_directory=persist_dir, collection_name=collection
    )
    return retrieval_module.FatwaRetriever(config)


# ----------------------------------------------------------------------------- #
#                                  التنسيق (CSS)                                  #
# ----------------------------------------------------------------------------- #

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Cairo', 'Segoe UI', sans-serif; }
    .main .block-container { direction: rtl; text-align: right; padding-top: 2rem; max-width: 1250px; }
    section[data-testid="stSidebar"] { direction: rtl; text-align: right; }
    .stChatMessage { direction: rtl; text-align: right; }
    div[data-testid="stFileUploader"] { direction: rtl; text-align: right; }

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
    .badge-gray  { background: #f3f4f6; color: #374151; }

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
    .step-card {
        background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: .8rem;
    }
    .step-card h4 { margin: 0 0 .5rem 0; color: #0f5132; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
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
        "index_built": False,
        "query_count": 0,
        "retriever_cache_key": 0,
        "upload_feedback": [],
        "build_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()
MODULES = load_pipeline_modules()
DATA_MANAGER = MODULES["data_manager"]


# ----------------------------------------------------------------------------- #
#                              الترويسة                                           #
# ----------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="app-header">
        <h1>🕌 المُعين الشرعي</h1>
        <p>نظام استرجاع وتوليد معزّز (RAG) للإجابة على الفتاوى والمسائل الشرعية</p>
        <p style="font-size:.85rem; opacity:.8; margin-top:.5rem;">
            مبني على قاعدة فتاوى موثّقة — مشروع أكاديمي
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

    # ------------------------- حالة الاتصال ------------------------- #
    st.subheader("🔐 حالة الاتصال")
    if OPENROUTER_API_KEY:
        masked = (
            f"{OPENROUTER_API_KEY[:6]}{'*' * 8}{OPENROUTER_API_KEY[-4:]}"
            if len(OPENROUTER_API_KEY) > 12 else "*" * len(OPENROUTER_API_KEY)
        )
        st.success(f"✅ المفتاح مُهيَّأ\n\n`{masked}`")
    else:
        st.error("❌ لم يُعثر على `OPENROUTER_API_KEY`")
        with st.expander("كيف أضيف المفتاح؟"):
            st.markdown(
                "**على Streamlit Cloud:** `Settings → Secrets`\n"
                "```toml\n"
                'OPENROUTER_API_KEY = "sk-or-v1-..."\n'
                'OPENROUTER_MODEL = "openai/gpt-4o-mini"\n'
                "```\n"
                "**محلياً:** أنشئ `.streamlit/secrets.toml` بنفس المحتوى."
            )

    model_name = st.text_input(
        "🤖 نموذج التوليد",
        value=OPENROUTER_MODEL,
        help="مثال: openai/gpt-4o-mini أو meta-llama/llama-3.3-70b-instruct:free",
    )

    st.divider()

    # ------------------------- محرّك التضمين ------------------------- #
    st.subheader("🧠 محرّك التضمين")
    try:
        embedding_model = MODULES["vectors"].get_embedding_model()
        info = embedding_model.info()
        if info["is_neural"]:
            st.success(f"عصبي · {info['dimension']} بُعد")
            st.caption(f"`{info['model_name']}`")
        else:
            st.warning(f"لفظي (TF-IDF) · {info['dimension']} بُعد")
            st.caption(
                "المكتبات العصبية غير مثبّتة في هذه البيئة. "
                "النظام يعمل بجودة مقبولة. للجودة الكاملة: "
                "`pip install -r requirements-full.txt` محلياً."
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"تعذّر تهيئة محرّك التضمين: {exc}")
        embedding_model = None

    st.divider()

    # ------------------------- معاملات الاسترجاع ------------------------- #
    st.subheader("🔍 معاملات الاسترجاع")
    top_k = st.slider("عدد الفتاوى المسترجعة", 1, 15, 5)
    threshold = st.slider("حدّ الثقة الأدنى", 0.0, 1.0, 0.28, 0.02,
                          help="إن لم تتجاوزه أي نتيجة، يمتنع النظام عن الإفتاء.")
    collection_name = st.text_input("اسم المجموعة", value="islamic_fatwas")
    persist_dir = st.text_input("مسار قاعدة المتجهات", value=os.path.join(BASE_DIR, "chroma_db"))

    st.divider()

    # ------------------------- معاملات التوليد ------------------------- #
    st.subheader("🎛️ معاملات التوليد")
    temperature = st.slider("درجة الحرارة", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("أقصى طول للإجابة", 300, 4000, 1400, 100)
    use_streaming = st.checkbox("بثّ الإجابة تدريجياً", value=True)
    show_context = st.checkbox("إظهار السياق الخام", value=False)

    st.divider()

    # ------------------------- حالة الفهرس ------------------------- #
    st.subheader("🗄️ قاعدة المتجهات")
    retriever = None
    try:
        retriever = get_cached_retriever(
            persist_dir, collection_name, st.session_state.retriever_cache_key
        )
        chunk_count = retriever.count()
        ready, ready_reason = retriever.is_ready()
        if ready:
            st.success(f"✅ جاهزة — {chunk_count:,} جزء")
            st.session_state.index_built = True
        elif chunk_count > 0:
            # الفهرس موجود لكن المُضمِّن غير مُدرَّب
            st.error(f"⚠️ {chunk_count:,} جزء مفهرس — لكن غير قابل للاستعلام")
            st.caption(ready_reason)
            st.session_state.index_built = False
        else:
            st.warning("⚠️ فارغة — ارفع بيانات وابنِ الفهرس من تبويب «البيانات».")
            st.session_state.index_built = False
    except Exception as exc:  # noqa: BLE001
        st.error(f"تعذّر فتح القاعدة:\n\n`{exc}`")
        st.session_state.index_built = False

    st.divider()
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.caption(f"الاستعلامات في هذه الجلسة: {st.session_state.query_count}")


# ----------------------------------------------------------------------------- #
#                              التبويبات الرئيسية                                 #
# ----------------------------------------------------------------------------- #

tab_chat, tab_data, tab_sources, tab_about = st.tabs(
    ["💬 المحادثة", "📤 البيانات والفهرسة", "📚 المصادر", "ℹ️ عن النظام"]
)


# ═══════════════════════════ تبويب البيانات ═══════════════════════════ #

with tab_data:
    st.subheader("📤 رفع ملفات الفتاوى وبناء الفهرس")

    # تنبيه الديمومة حسب البيئة
    level, note = DATA_MANAGER.persistence_note()
    (st.warning if level == "warning" else st.success)(note)

    st.markdown("---")

    # ─────────────────── الخطوة 1: الرفع ─────────────────── #
    st.markdown("#### 1️⃣ رفع الملفات")

    uploaded_files = st.file_uploader(
        "اختر ملفات الفتاوى",
        type=["csv", "tsv", "txt", "xlsx", "xls", "zip"],
        accept_multiple_files=True,
        help=(
            "الصيغ المدعومة: CSV · TSV · Excel · ZIP (يُفكّ تلقائياً). "
            "الحد الأقصى 500 ميجابايت للملف. "
            "الترميز العربي (utf-8 / windows-1256) يُكتشف تلقائياً."
        ),
    )

    col_upload, col_validate = st.columns([1, 1])
    with col_upload:
        do_validate = st.checkbox("فحص الأعمدة قبل الحفظ", value=True)
    with col_validate:
        save_clicked = st.button("💾 حفظ الملفات المرفوعة", type="primary")

    if save_clicked:
        if not uploaded_files:
            st.warning("لم تختر أي ملف بعد.")
        else:
            documents_module = MODULES["documents"]
            aliases = getattr(documents_module, "COLUMN_ALIASES", None)
            feedback: List[tuple] = []

            progress = st.progress(0.0, text="جارٍ الحفظ...")
            for i, uploaded in enumerate(uploaded_files, start=1):
                progress.progress(
                    i / len(uploaded_files), text=f"معالجة: {uploaded.name}"
                )
                try:
                    payload = uploaded.getvalue()
                    ok, message, info = DATA_MANAGER.save_uploaded_bytes(
                        data=payload,
                        filename=uploaded.name,
                        data_dir=DATA_DIR,
                        validate=do_validate,
                        column_aliases=aliases,
                    )
                    feedback.append((ok, uploaded.name, message, info))
                except Exception as exc:  # noqa: BLE001
                    feedback.append((False, uploaded.name, f"خطأ: {exc}", None))

            progress.empty()
            st.session_state.upload_feedback = [
                (ok, name, msg) for ok, name, msg, _ in feedback
            ]

            succeeded = sum(1 for ok, *_ in feedback if ok)
            if succeeded:
                st.success(f"✅ حُفظ {succeeded} من {len(feedback)} ملف.")
            for ok, name, message, info in feedback:
                if ok:
                    detail = ""
                    if info and info.rows:
                        detail = f" — ~{info.rows:,} صف · ترميز {info.detected_encoding}"
                    st.success(f"**{name}**: {message}{detail}")
                else:
                    st.error(f"**{name}**: {message}")

    # معاينة الملفات قبل الحفظ
    if uploaded_files and not save_clicked:
        with st.expander(f"🔎 معاينة الملفات المختارة ({len(uploaded_files)})"):
            documents_module = MODULES["documents"]
            aliases = getattr(documents_module, "COLUMN_ALIASES", None)
            for uploaded in uploaded_files[:5]:
                st.markdown(f"**📄 {uploaded.name}** — {DATA_MANAGER.human_size(uploaded.size)}")
                if uploaded.name.lower().endswith(".zip"):
                    st.caption("أرشيف مضغوط — سيُفكّ عند الحفظ.")
                    continue
                try:
                    result = DATA_MANAGER.validate_tabular_bytes(
                        uploaded.getvalue(), uploaded.name, aliases
                    )
                    if result.is_valid:
                        st.caption(
                            f"✅ صالح · ~{result.rows:,} صف · ترميز `{result.encoding}` · "
                            f"{len(result.columns)} عمود"
                        )
                        if result.mapped_columns:
                            mapped = " · ".join(
                                f"`{k}` → **{v}**" for k, v in result.mapped_columns.items()
                            )
                            st.caption(f"الأعمدة المتعرّف عليها: {mapped}")
                        for warning in result.warnings:
                            st.caption(f"⚠️ {warning}")
                        if result.preview is not None:
                            st.dataframe(result.preview, use_container_width=True, height=180)
                    else:
                        st.error(" | ".join(result.errors))
                        st.caption(f"الأعمدة الموجودة: {', '.join(result.columns[:15])}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"تعذّرت المعاينة: {exc}")
                st.markdown("---")

    st.markdown("---")

    # ─────────────────── الخطوة 2: الملفات المحفوظة ─────────────────── #
    st.markdown("#### 2️⃣ الملفات المحفوظة")

    data_files = DATA_MANAGER.list_data_files(DATA_DIR)
    stats = DATA_MANAGER.data_directory_stats(DATA_DIR)

    stat_cols = st.columns(3)
    stat_cols[0].metric("📁 عدد الملفات", stats["files"])
    stat_cols[1].metric("💾 الحجم الكلي", stats["total_size"])
    stat_cols[2].metric(
        "📊 الصفوف التقديرية",
        f"{stats['estimated_rows']:,}" if stats["estimated_rows"] else "—",
    )

    if not data_files:
        st.info("لا توجد ملفات بعد. ارفع ملفات CSV من الأعلى، أو ضعها في مجلد `data/` بالمستودع.")
    else:
        display_rows = [
            {k: v for k, v in row.items() if not k.startswith("_")} for row in data_files
        ]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

        with st.expander("🗑️ حذف ملف"):
            names = [row["الملف"] for row in data_files]
            to_delete = st.selectbox("اختر الملف", names, key="delete_select")
            if st.button("تأكيد الحذف", key="delete_btn"):
                target = next(r for r in data_files if r["الملف"] == to_delete)
                ok, message = DATA_MANAGER.delete_data_file(target["_path"], DATA_DIR)
                (st.success if ok else st.error)(message)
                st.rerun()

    st.markdown("---")

    # ─────────────────── الخطوة 3: بناء الفهرس ─────────────────── #
    st.markdown("#### 3️⃣ بناء الفهرس")
    st.caption("ينفّذ المراحل 01 → 05: القراءة ← التنظيف ← التقطيع ← التضمين ← الفهرسة.")

    build_cols = st.columns(3)
    with build_cols[0]:
        build_limit = st.number_input(
            "حدّ الفتاوى لكل ملف (0 = الكل)", 0, 500_000, 2000, 500,
            help="ابدأ برقم صغير للتأكد من تطابق الأعمدة قبل البناء الكامل.",
        )
    with build_cols[1]:
        chunk_size = st.number_input("حجم الجزء (محرف)", 300, 2000, 900, 50)
    with build_cols[2]:
        chunk_overlap = st.number_input("التراكب (محرف)", 0, 500, 150, 25)

    reset_index = st.checkbox(
        "حذف الفهرس الحالي قبل البناء", value=False,
        help="فعّلها عند تغيير البيانات جذرياً. بدونها تُضاف الفتاوى الجديدة للفهرس الموجود.",
    )

    if st.button("🚀 ابدأ بناء الفهرس", type="primary", disabled=not data_files):
        status = st.status("جارٍ بناء الفهرس...", expanded=True)
        try:
            # المرحلة 1
            status.write("**1/5** — قراءة ملفات CSV...")
            documents = MODULES["documents"].load_documents(
                DATA_DIR, max_rows_per_file=int(build_limit) or None
            )
            if documents.empty:
                status.update(label="فشل البناء", state="error")
                st.error(
                    "لم يُعثر على بيانات صالحة. تحقّق من أن ملفاتك تحتوي عمود "
                    "الجواب/نص الفتوى، أو أضِف أسماء أعمدتك إلى `COLUMN_ALIASES`."
                )
            else:
                status.write(f"✅ حُمّلت {len(documents):,} فتوى")

                # المرحلة 2
                status.write("**2/5** — تنظيف النصوص الشرعية...")
                cleaned = MODULES["preprocessing"].preprocess_documents(documents)
                status.write(f"✅ بقيت {len(cleaned):,} فتوى بعد التنظيف")

                # المرحلة 3
                status.write("**3/5** — تقطيع النصوص...")
                chunk_config = MODULES["chunking"].ChunkConfig(
                    chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
                )
                chunks = MODULES["chunking"].chunk_documents(cleaned, chunk_config)
                status.write(f"✅ أُنتج {len(chunks):,} جزء")

                # المرحلة 4
                status.write("**4/5** — تهيئة محرّك التضمين...")
                model = MODULES["vectors"].get_embedding_model()
                if model.needs_fitting():
                    status.write("↳ تدريب مُضمِّن TF-IDF على المتن...")
                status.write(f"✅ {model.quality_note()}")

                # المرحلة 5
                status.write("**5/5** — بناء قاعدة المتجهات (قد يستغرق وقتاً)...")
                store_config = MODULES["store"].ChromaConfig(
                    persist_directory=persist_dir,
                    collection_name=collection_name,
                    reset_collection=reset_index,
                )
                collection = MODULES["store"].build_vector_store(chunks, store_config)
                indexed = collection.count()

                status.update(label=f"✅ اكتمل البناء — {indexed:,} جزء مفهرس", state="complete")
                st.success(
                    f"تمت فهرسة **{indexed:,}** جزء من **{len(cleaned):,}** فتوى. "
                    "انتقل إلى تبويب المحادثة لطرح أسئلتك."
                )
                st.balloons()

                # إبطال المخبأ ليلتقط الفهرس الجديد
                st.session_state.retriever_cache_key += 1
                get_cached_retriever.clear()

        except Exception as exc:  # noqa: BLE001
            status.update(label="فشل البناء", state="error")
            st.error(f"حدث خطأ أثناء البناء: {exc}")
            with st.expander("تفاصيل تقنية"):
                import traceback
                st.code(traceback.format_exc())

    if not data_files:
        st.caption("⬆️ ارفع ملفات أولاً لتفعيل زر البناء.")


# ═══════════════════════════ تبويب المحادثة ═══════════════════════════ #

with tab_chat:
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

    for message in st.session_state.messages:
        avatar = "🕌" if message["role"] == "assistant" else "🧑"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    user_question = st.chat_input("اكتب سؤالك الشرعي هنا...") or clicked_example

    if user_question:
        st.session_state.query_count += 1
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar="🕌"):
            if retriever is None:
                st.error("⚠️ تعذّر فتح قاعدة المتجهات — راجع اللوحة الجانبية.")
            elif not st.session_state.index_built:
                _, reason = retriever.is_ready()
                st.error(f"⚠️ {reason}")
                if retriever.count() > 0:
                    st.info(
                        "💡 **لماذا حدث هذا؟** متجهات TF-IDF تعتمد على المفردات التي "
                        "دُرِّب عليها المُضمِّن. عند بناء الفهرس على جهاز ونقله إلى آخر "
                        "دون ملف الحالة، تصبح المتجهات بلا مرجع. إعادة البناء تحلّ المشكلة "
                        "نهائياً وتحفظ الحالة داخل `chroma_db/` تلقائياً."
                    )
            elif not OPENROUTER_API_KEY:
                st.error("⚠️ مفتاح `OPENROUTER_API_KEY` غير مُهيَّأ — راجع اللوحة الجانبية.")
            else:
                prompting = MODULES["prompting"]

                # 1) الاسترجاع
                with st.spinner("🔍 جارٍ البحث في قاعدة الفتاوى..."):
                    started = time.time()
                    retriever.config.top_k = top_k
                    retriever.config.relevance_threshold = threshold
                    retrieval = retriever.retrieve(user_question, top_k=top_k)
                    retrieval_time = time.time() - started

                st.session_state.last_sources = retrieval.chunks

                if getattr(retrieval, "error", ""):
                    st.error(f"⚠️ {retrieval.error}")
                elif not retrieval.has_sufficient_context:
                    st.warning(
                        f"⚠️ لم يُعثر على فتاوى ذات صلة كافية "
                        f"(أعلى تطابق {retrieval.max_score:.3f} < {threshold:.2f}). "
                        "سيُبلَّغ بذلك دون إصدار حكم."
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
                    model=model_name, temperature=temperature, max_tokens=int(max_tokens)
                )
                client = prompting.OpenRouterClient(generation_config)

                generation_started = time.time()
                total_tokens = 0
                if use_streaming:
                    placeholder = st.empty()
                    parts: List[str] = []
                    for token in client.chat_stream(messages):
                        parts.append(token)
                        placeholder.markdown(
                            f'<div class="answer-box">{"".join(parts)}▌</div>',
                            unsafe_allow_html=True,
                        )
                    answer = "".join(parts).strip()
                    placeholder.markdown(
                        f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True
                    )
                else:
                    with st.spinner("✍️ جارٍ صياغة الإجابة..."):
                        result = client.chat(messages)
                    answer = result.answer if result.success else f"⚠️ {result.error}"
                    total_tokens = result.total_tokens
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                generation_time = time.time() - generation_started

                # 4) مؤشرات الجودة
                grounded = prompting.verify_groundedness(answer, len(retrieval.chunks))
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
                        '<span class="badge badge-amber">⚠ لم تُرصد إحالات صريحة</span>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="disclaimer">📌 <b>تنبيه شرعي:</b> هذه الإجابة نقل آلي من '
                    'قاعدة فتاوى ولا تُعدّ فتوى شخصية. النوازل والمسائل الخاصة يُرجع فيها '
                    'إلى أهل العلم المختصين ودور الإفتاء المعتبرة.</div>',
                    unsafe_allow_html=True,
                )

                st.session_state.messages.append({"role": "assistant", "content": answer})


# ═══════════════════════════ تبويب المصادر ═══════════════════════════ #

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
            title = chunk.title[:80] or f"فتوى رقم {chunk.fatwa_id}"
            with st.expander(
                f"[مصدر {i}] {title} — تطابق {chunk.final_score:.3f}", expanded=(i == 1)
            ):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="meta">
                            <span class="badge badge-blue">رقم الفتوى: {chunk.fatwa_id}</span>
                            <span class="badge {score_class}">التطابق: {chunk.final_score:.3f}</span>
                            <span class="badge badge-gray">دلالي: {chunk.semantic_score:.3f}</span>
                            <span class="badge badge-gray">لفظي: {chunk.lexical_score:.3f}</span>
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

        st.download_button(
            "⬇️ تنزيل المصادر (JSON)",
            data=json.dumps([c.to_dict() for c in sources], ensure_ascii=False, indent=2),
            file_name="fatwa_sources.json",
            mime="application/json",
        )


# ═══════════════════════════ تبويب عن النظام ═══════════════════════════ #

with tab_about:
    st.subheader("ℹ️ عن النظام والمعمارية")

    st.markdown(
        """
        ### 🏗️ خط الأنابيب (RAG Pipeline)

        | المرحلة | الملف | الوظيفة |
        |---|---|---|
        | 1 | `01_documents.py` | قراءة ملفات الفتاوى (CSV) وتوحيد المخطط |
        | 2 | `02_preprocessing.py` | تنظيف النصوص وتطبيع العربية وحذف التكرار |
        | 3 | `03_chunking.py` | تقطيع هرمي يحترم الجُمل مع حفظ الـ metadata |
        | 4 | `04_vector_representation.py` | التضمين المتكيّف (عصبي / TF-IDF) |
        | 5 | `05_create_chroma_store.py` | بناء قاعدة المتجهات ChromaDB |
        | 6 | `06_retrieve_context.py` | استرجاع متعدد المراحل مع إعادة ترتيب |
        | 7 | `07_prompting.py` | البرومبت المحكم + OpenRouter API |
        | 8 | `08_data_manager.py` | إدارة رفع الملفات وحفظها |

        ### 🛡️ ضوابط السلامة الشرعية
        - **التقيّد بالسياق:** يُمنع النموذج من الإفتاء من معرفته الداخلية.
        - **الإحالة الإلزامية:** كل حكم منسوب إلى `[مصدر ن]` مع رقم الفتوى.
        - **الامتناع عند الجهل:** حدّ ثقة أدنى يمنع التأليف عند غياب السند.
        - **التحقق البعدي:** مؤشر التأصيل يفحص وجود الإحالات في الإجابة.
        """
    )

    st.divider()
    st.markdown("### 🧪 تشخيص البيئة")

    diag_cols = st.columns(2)
    with diag_cols[0]:
        st.markdown("**محرّك التضمين**")
        try:
            st.json(MODULES["vectors"].get_embedding_model().info())
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    with diag_cols[1]:
        st.markdown("**البيئة**")
        st.json({
            "python": sys.version.split()[0],
            "بيئة مؤقتة": DATA_MANAGER.is_ephemeral_environment(),
            "مسار البيانات": DATA_DIR,
            "مسار الفهرس": persist_dir,
            "المجموعة": collection_name,
            "النموذج التوليدي": model_name,
        })

    with st.expander("📦 المكتبات المثبّتة"):
        for package in ["streamlit", "pandas", "numpy", "chromadb",
                        "sklearn", "torch", "sentence_transformers"]:
            spec = importlib.util.find_spec(package)
            if spec:
                try:
                    module = importlib.import_module(package)
                    version = getattr(module, "__version__", "—")
                    st.markdown(f"✅ `{package}` — {version}")
                except Exception:  # noqa: BLE001
                    st.markdown(f"✅ `{package}`")
            else:
                st.markdown(f"➖ `{package}` — غير مثبّتة")


# ----------------------------------------------------------------------------- #
#                                  التذييل                                        #
# ----------------------------------------------------------------------------- #

st.markdown(
    """
    <hr style="margin-top:2.5rem; opacity:.25;"/>
    <div style="text-align:center; color:#6b7280; font-size:.85rem; padding-bottom:1.5rem;">
        🕌 المُعين الشرعي — نظام RAG للفتاوى الإسلامية | مشروع أكاديمي<br/>
        <span style="font-size:.78rem;">وَقُل رَّبِّ زِدْنِي عِلْمًا — والله تعالى أعلم</span>
    </div>
    """,
    unsafe_allow_html=True,
)
