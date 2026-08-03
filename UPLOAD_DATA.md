<div align="center">

# 📤 دليل رفع بيانات الفتاوى إلى GitHub

**من جهازك المحلي إلى المستودع — خطوة بخطوة**

</div>

---

## 🪟 لمستخدمي ويندوز — اقرأ هذا أولاً

### ⚠️ لا تنسخ التعليقات مع الأمر

في CMD الويندوز، الرمز `#` **ليس** تعليقاً كما في لينكس:

```cmd
REM ❌ خطأ — سيعطي: unrecognized arguments
python prepare_data_for_github.py --compress --split  # ضغط وتقسيم

REM ✅ صحيح — انسخ الأمر وحده
python prepare_data_for_github.py --compress --split
```

### ⚠️ `fatal: not a git repository`

هذا يعني أن المجلد الذي تقف فيه **ليس نسخة من المستودع**. أمر `git pull` يعمل
فقط داخل مجلد مستنسخ. استنسخ المستودع أولاً:

```cmd
cd %USERPROFILE%\Desktop
git clone https://github.com/ahmedzan91-rgb/Islamic-Fatwa-RAG.git
cd Islamic-Fatwa-RAG
```

### 📁 أين أضع ملفات CSV؟

الأداة تبحث تلقائياً في: مجلد `data/` بجانب السكربت ← مجلد السكربت نفسه ←
مجلد التشغيل الحالي. فلا يهم أين وضعتها بالضبط.

**إن لم تجدها، ابحث عنها:**

```cmd
python prepare_data_for_github.py --find
```

**أو حدّد المجلد صراحةً (لاحظ علامتَي التنصيص للمسارات ذات المسافات):**

```cmd
python prepare_data_for_github.py --data-dir "C:\Users\Admin\Desktop\مشروع الفتوي AI\Data" --compress
```

---

## ⚡ الطريقة السريعة (٣ أوامر)

```bash
# 1) ضع ملفات CSV في مجلد data/ ثم افحصها
python prepare_data_for_github.py

# 2) اضغطها وجهّزها للرفع
python prepare_data_for_github.py --compress --split

# 3) ارفعها
git add data/*.gz .gitignore
git commit -m "data: إضافة بيانات الفتاوى"
git push
```

انتهى. النظام يقرأ ملفات `.gz` مباشرة بلا فكّ ضغط.

---

## 🚧 لماذا لا أرفع CSV مباشرة؟

GitHub يفرض حدوداً صارمة على أحجام الملفات [(المصدر)](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github):

| الحجم | ما يحدث |
|---|---|
| أقل من 50 MB | ✅ يُرفع بلا مشاكل |
| 50 – 100 MB | ⚠️ يُرفع مع تحذير من Git |
| **أكثر من 100 MB** | ❌ **يرفض GitHub الدفع تماماً** |
| رفع من المتصفح | 25 MB كحدّ أقصى |

**والمشكلة:** ملف 139 ألف فتوى بصيغة CSV الخام ≈ **300 ميجابايت** → مرفوض قطعاً.

**والحل:** الضغط بـ gzip. النص العربي ينضغط بنسبة **5–9×**:

| الصيغة | الحجم المتوقع (139 ألف فتوى) | الحكم |
|---|:-:|:-:|
| CSV خام | ~300 MB | ❌ مرفوض |
| **CSV.GZ** | **~50 MB** | ✅ **يمرّ** |

> 📊 هذه أرقام مقيسة فعلياً على نص عربي متنوّع، وليست تقديرات.

---

## 🛠️ الأداة: `prepare_data_for_github.py`

### الفحص أولاً (لا تعديل)

```bash
python prepare_data_for_github.py
```

```
==========================================================================
  فحص مجلد البيانات: /path/to/data
==========================================================================

  📄 ملفات خام (غير مضغوطة):

     الملف                                        الحجم   الحالة
     ──────────────────────────────────────────────────────────
     fatwas_139k.csv                            298.4 MB   ❌ مرفوض — يتجاوز 100 MB

  ❌ 1 ملف سيرفضه GitHub. شغّل: --compress (و --split عند اللزوم)
```

### الضغط

```bash
python prepare_data_for_github.py --compress
```

يُنشئ `fatwas_139k.csv.gz` بجانب الأصل، ويحدّث `.gitignore` تلقائياً للسماح برفعه.

### الضغط + التقسيم

إذا بقي الملف فوق 100 MB حتى بعد الضغط:

```bash
python prepare_data_for_github.py --compress --split
```

يقسّم الملف إلى أجزاء (`_part01`, `_part02`...) **مع تكرار سطر الترويسة في كل جزء**،
ثم يضغطها. حجم الجزء **تكيّفي**: يقيس نسبة الضغط الفعلية لملفك ويحسب الهدف
على أساسها، فيُنتج أقل عدد ممكن من الأجزاء.

> ✅ **مُختبَر:** 40,000 صف → 9 أجزاء → إعادة قراءتها أعطت 40,000 صف بالضبط، بلا فقد.

### خيارات إضافية

| الخيار | الوظيفة |
|---|---|
| `--delete-original` | حذف CSV الخام بعد الضغط (توفير مساحة) |
| `--sample 20000` | إنشاء عيّنة خفيفة بدل رفع كل البيانات |
| `--level 1..9` | مستوى الضغط (1 سريع · 9 أقصى · الافتراضي 9) |
| `--data-dir PATH` | مجلد بيانات مخصّص |

---

## 🎯 السيناريوهات الشائعة

<details open>
<summary><b>السيناريو 1: بياناتي أقل من 50 MB</b></summary>

الأبسط — اضغطها ثم ارفع:

```bash
python prepare_data_for_github.py --compress
git add data/*.gz .gitignore
git commit -m "data: إضافة بيانات الفتاوى"
git push
```

</details>

<details>
<summary><b>السيناريو 2: بياناتي 139 ألف فتوى (~300 MB)</b></summary>

```bash
python prepare_data_for_github.py --compress --split
git add data/*.gz .gitignore
git commit -m "data: إضافة 139 ألف فتوى مضغوطة"
git push
```

الحجم المتوقع بعد الضغط: **~50 MB** — يمرّ في دفعة واحدة.

</details>

<details>
<summary><b>السيناريو 3: أريد عيّنة خفيفة فقط للعرض</b></summary>

مفيد إن كان المستودع للعرض الأكاديمي ولا تريد إثقاله:

```bash
python prepare_data_for_github.py --sample 20000
git add data/*_sample_*.csv .gitignore
git commit -m "data: عيّنة 20 ألف فتوى للعرض"
git push
```

ثم احتفظ بالبيانات الكاملة محلياً لبناء الفهرس.

</details>

<details>
<summary><b>السيناريو 4: أريد رفع الفهرس الجاهز بدل البيانات (موصى به)</b></summary>

**هذا أفضل خيار للنشر على Streamlit Cloud** — تحصل على جودة عصبية دون تثبيت `torch` هناك:

```bash
# 1) ابنِ الفهرس محلياً بالجودة العصبية الكاملة
pip install -r requirements-full.txt
python run_pipeline.py

# 2) اسمح برفع الفهرس
echo '!chroma_db/**' >> .gitignore

# 3) تحقّق من الحجم أولاً — الفهرس قد يكون كبيراً
du -sh chroma_db/

# 4) ارفع
git add chroma_db/ .gitignore
git commit -m "index: إضافة قاعدة المتجهات الجاهزة"
git push
```

> ⚠️ فهرس 139 ألف فتوى قد يتجاوز 1 GB. إن تجاوز الحدّ، ارفع البيانات المضغوطة
> بدلاً منه ودع التطبيق يبني الفهرس بواجهة TF-IDF الخفيفة.

</details>

---

## 🐙 إعداد Git لأول مرة

إن لم تكن ربطت مستودعك بعد:

```bash
cd مسار/المشروع

git init
git branch -M main
git remote add origin https://github.com/ahmedzan91-rgb/Islamic-Fatwa-RAG.git

# اسحب الشيفرة الموجودة أولاً لتفادي التعارض
git pull origin main --allow-unrelated-histories
```

### المصادقة

GitHub ألغى المصادقة بكلمة المرور. استخدم **Personal Access Token**:

1. `GitHub → Settings → Developer settings → Personal access tokens`
2. اختر **Fine-grained token** (أكثر أماناً من Classic)
3. حدّده بمستودع واحد + صلاحية `Contents: Read and write` فقط
4. عند طلب كلمة المرور في `git push`، الصق التوكن

**لتفادي إدخاله كل مرة:**

```bash
# لينكس / ماك
git config --global credential.helper store    # يحفظه على القرص
git config --global credential.helper cache    # أو مؤقتاً في الذاكرة

# ويندوز
git config --global credential.helper manager
```

> 🔒 **لا تكتب التوكن داخل أي ملف في المستودع** ولا في رابط الـ remote.
> إن سُرّب، ألغِه فوراً من نفس الصفحة.

---

## ✅ التحقق بعد الرفع

```bash
# ما الذي رُفع فعلاً؟
git ls-files data/

# حجم المستودع
git count-objects -vH | grep size-pack

# تأكّد أن الخام مستثنى والمضغوط مسموح
git check-ignore -v data/fatwas.csv        # يجب أن يظهر مستثنى
git check-ignore -v data/fatwas.csv.gz     # يجب ألا يظهر شيء
```

**اختبر أن النظام يقرأ البيانات المرفوعة:**

```bash
python 01_documents.py --input data
python run_pipeline.py --limit 2000
```

---

## 🚨 حلّ المشاكل

<details>
<summary><b>❌ <code>File exceeds GitHub's file size limit of 100.00 MB</code></b></summary>

الملف كبير. اضغطه وقسّمه:

```bash
python prepare_data_for_github.py --compress --split
```

**إن كان الملف قد دخل تاريخ Git بالفعل**، لا يكفي حذفه — يجب تنظيف التاريخ:

```bash
# إن كان في آخر كوميت لم يُدفع بعد
git rm --cached data/big.csv
git commit --amend -C HEAD

# إن كان أعمق في التاريخ (الأداة الموصى بها من GitHub)
pip install git-filter-repo
git filter-repo --path data/big.csv --invert-paths --force
```

</details>

<details>
<summary><b>⚠️ ملفات <code>.gz</code> لا تُرفع رغم الضغط</b></summary>

`.gitignore` يستثنيها. الأداة تصلح هذا تلقائياً، وللتأكد يدوياً:

```bash
grep -n "csv.gz" .gitignore     # يجب أن ترى: !data/*.csv.gz
git check-ignore -v data/x.csv.gz
git add -f data/*.gz            # حلّ إسعافي
```

</details>

<details>
<summary><b>🐌 الدفع بطيء جداً أو ينقطع</b></summary>

```bash
# ارفع ملفاً واحداً في كل كوميت
git add data/part01.csv.gz && git commit -m "data: جزء 1" && git push

# ارفع حجم المخزن المؤقت لـ HTTP
git config --global http.postBuffer 524288000
```

> GitHub يحدّ كل عملية `push` بـ **2 GB**.

</details>

<details>
<summary><b>🔤 النصوص العربية تظهر مشوّهة بعد الرفع</b></summary>

مشكلة ترميز في الملف الأصلي. النظام يكتشف `utf-8` و `windows-1256` تلقائياً،
لكن للتحويل اليدوي:

```bash
iconv -f WINDOWS-1256 -t UTF-8 old.csv > new.csv
```

```python
import pandas as pd
pd.read_csv("old.csv", encoding="cp1256") \
  .to_csv("new.csv", index=False, encoding="utf-8-sig")
```

</details>

<details>
<summary><b>🤔 هل أرفع البيانات أصلاً؟</b></summary>

**لا ترفعها إن:**
- كانت مملوكة لجهة أخرى ولا تملك حقّ إعادة نشرها ⚠️
- تتجاوز 500 MB مضغوطة (استخدم عيّنة + رابط للمصدر)

**ارفعها إن:**
- أردت أن يعمل التطبيق على Streamlit Cloud مباشرة
- أردت إعادة إنتاج النتائج أكاديمياً (reproducibility)

> ⚖️ **تنبيه حقوقي:** بيانات الفتاوى ملك لناشريها الأصليين. تأكّد من شروط
> الاستخدام قبل إعادة النشر، وأضِف نسبةً واضحة للمصدر في المستودع.

</details>

---

<div align="center">
<sub>🕌 المُعين الشرعي — نظام RAG للفتاوى الإسلامية</sub>
</div>
