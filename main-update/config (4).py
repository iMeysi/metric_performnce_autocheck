# وضعیت اعتبارسنجی

## جمع‌بندی

**۲۹ آزمون واحد/یکپارچه با موفقیت اجرا شده‌اند، اما مدل واقعی TensorFlow اجرا نشده است.** این بسته نمونهٔ مهندسی‌شدهٔ قابل بررسی و ادامهٔ اعتبارسنجی است؛ موفقیت تست‌های adapter آزمایشی به معنی آماده‌بودن تولیدی یا اثبات صحت مدل نیست.

## آنچه اجرا شد

| بررسی | نتیجه | دامنه و محدودیت |
|---|---|---|
| `python -m unittest discover -s tests -v` | ۲۹ از ۲۹ موفق | هسته، HTTP و چرخهٔ artifact با backend آزمایشی؛ SPOT واقعی upstream |
| `python -m compileall -q pad_omni tests scripts app.py` | موفق | کامپایل نحوی در Python 3.11، نه اجرای TensorFlow |
| بررسی AST با grammar نسخهٔ 3.6 | موفق | فقط syntax فایل‌های خود پروژه؛ نه سازگاری تمام APIها یا dependencyها |
| parse فایل‌های YAML | موفق | YAML معتبر؛ نه اعتبارسنجی schema Kubernetes یا پذیرش توسط OpenShift |
| بررسی لینک‌های محلی مستندات پروژه | موفق | موجودبودن فایل‌های مقصد |
| `python scripts/benchmark_data.py` | موفق | دادهٔ مصنوعی، پیش‌پردازش و SPOT واقعی؛ بدون مدل |

خروجی‌های خام:

- [unit-http-tests.txt](../reports/unit-http-tests.txt)
- [benchmark-data.json](../reports/benchmark-data.json)
- [static-checks.json](../reports/static-checks.json)

### دامنهٔ تست‌ها

- override متغیرهای PAD، مدت‌ها، سقف تاریخچه و حساسیت signature به ترتیب queryها.
- قطعه‌بندی query_range، هم‌ترازی timestamp، schema ثابت و کنترل ابعاد.
- رد timestamp نامعتبر، series تکراری، افزایش cardinality و پاسخ بیش از بودجه.
- رد warning/partial/error و پاسخ غیر-matrix از API Prometheus.
- forward-fill محدود، missing budget و ممنوعیت صادرکردن استنتاج با آخرین نقطهٔ ناقص.
- رد دادهٔ نامتناهی و سرریز تبدیل به float32.
- ذخیره/بازیابی scaler، ستون ثابت و هم‌ترازی پنجره‌ها.
- اجرای SPOT واقعی vendored و رد دم ناکافی یا منحط.
- قرارداد train → persist → restore → infer با backend آزمایشی.
- نوشتن اتمیک JSON، بررسی کاندید قبل از انتشار، حذف کاندید ناموفق و حفظ نسخهٔ قبلی.
- بررسی انتقال timeout به فراخوانی subprocess؛ این تست جای آزمون timeout/OOM واقعی TF را نمی‌گیرد.
- fail-closed بودن متریک‌ها، TTL و نتیجهٔ ناقص؛ خطای بازآموزی نتیجهٔ معتبر قبلی را حذف نمی‌کند.
- سرور HTTP مصنوعی Prometheus، آموزش/بازیابی آزمایشی و exporter واقعی HTTP؛ سپس قطع/خرابی داده و تغییر readiness.

در آزمون‌های چرخهٔ مدل، backend آزمایشی جای TF را می‌گیرد. بنابراین حتی تستی با نام `candidate_restore_verified_before_promotion` صحت سازوکار فراخوانی و حفظ artifact را می‌سنجد، نه توانایی restore کردن checkpoint واقعی TensorFlow.

### بنچمارک ثبت‌شدهٔ مسیر داده

آخرین اجرای ثبت‌شده:

| مقدار | نتیجه |
|---|---:|
| Python | 3.11.16 |
| NumPy | 1.26.4 |
| ردیف‌ها | ۴۳۲۱ |
| ویژگی‌ها | ۳ |
| پنجره‌ها | ۴۲۲۲ |
| زمان پیش‌پردازش | حدود ۰٫۰۲۰۵ ثانیه |
| زمان POT | حدود ۲٫۴۸۵ ثانیه |
| peak RSS پردازش بنچمارک | ۲۴۹٬۷۱۶٬۷۳۶ بایت، حدود ۲۳۸ MiB |

این اعداد فقط یک اجرا در محیط اشتراکی Linux/x86_64 هستند؛ با بار ماشین تغییر می‌کنند. RSS شامل interpreter و کتابخانه‌های بارگذاری‌شده است، **نه RSS مدل، workerهای هم‌زمان یا کل pod**. این اندازه‌گیری برای انتخاب limits تولیدی کافی نیست و latency استنتاج یا throughput مدل را نشان نمی‌دهد.

در خروجی تست، warning مربوط به ترکیب dependencyهای `requests` در محیط میزبان و deprecationهای matplotlib/pyparsing وجود دارد. آزمون‌ها با وجود آن‌ها موفق‌اند، ولی محیط میزبان همان محیط قفل‌شدهٔ مرجع Python 3.6 نیست. فایل لاگ warningها را برای شفافیت حفظ کرده است.

## آنچه هنوز اجرا نشده است

- نصب و resolve کامل `requirements.txt` در Python 3.6 و اجرای `pip check`.
- ساخت Dockerfile و اسکن امنیتی image نهایی.
- آموزش/کالیبراسیون/restore/استنتاج با **OmniAnomaly واقعی و TensorFlow 1.12**.
- اجرای `scripts/smoke_real.py` در محیط هدف.
- سنجش peak RAM و زمان مدل واقعی برای پروفایل ۳روزه یا ۱۵روزه.
- آزمون روی Prometheus واقعی با حجم/کیفیت دادهٔ تولیدی.
- استقرار OpenShift، دسترسی Thanos، RBAC، CA/token rotation، PVC/SCC و discovery واقعی ServiceMonitor.
- ارزیابی precision، recall، F1، تأخیر کشف یا نرخ هشدار کاذب روی دادهٔ برچسب‌خوردهٔ مستقل.

پس نباید از این گزارش نتیجه گرفت که مدل دقیق، runtime امن، build قابل تکرار یا deployment آمادهٔ تولید است. [راهنمای اصلی](README.fa.md) فرمان‌ها و چک‌لیست پذیرش باقی‌مانده را ارائه می‌کند.

## بازتولید گزارش‌ها

از ریشهٔ پروژه، در محیط مناسب مسیر داده:

```bash
python -m unittest discover -s tests -v > reports/unit-http-tests.txt 2>&1
python -m compileall -q pad_omni tests scripts app.py
python scripts/benchmark_data.py > reports/benchmark-data.json
```

برای smoke واقعی، پس از build موفق و در همان image:

```bash
docker run --rm pad-omnianomaly:0.1 python scripts/smoke_real.py --epochs 1
```

خروجی این فرمان را به‌عنوان گزارش جدید و جداگانه نگه دارید؛ گزارش بنچمارک داده را با نتیجهٔ واقعی مدل اشتباه نگیرید. موفقیت یک epoch روی دادهٔ مصنوعی نیز آزمون کیفیت تشخیص یا ظرفیت عملیاتی نیست.
