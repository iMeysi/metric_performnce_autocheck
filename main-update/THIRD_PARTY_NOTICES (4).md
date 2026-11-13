# PAD + OmniAnomaly — تشخیص ناهنجاری چندمتغیره

این پروژه یک **سرویس همراه/جایگزین exporter مدل PAD** است: متریک‌های Prometheus را در گروه‌های چندمتغیره جمع می‌کند، با OmniAnomaly مدل می‌سازد و امتیاز و پرچم ناهنجاری را دوباره به Prometheus می‌دهد.

> **وضعیت: نمونهٔ مهندسی‌شده، نه انتشار تأییدشده برای تولید.** آزمون‌های واحد و HTTP با backend آزمایشی اجرا شده‌اند؛ اجرای مدل واقعی TensorFlow 1.12، ساخت image و استقرار OpenShift هنوز تأیید نشده‌اند. محدودکردن پنجره و ابعاد داده، تضمین جلوگیری از OOM مدل نیست.

## راهنمای اصلی

**[راهنمای گام‌به‌گام فارسی: معماری، کد، تنظیمات، Docker و OpenShift](docs/README.fa.md)**

- [نتایج آزمون و محدودیت‌های اعتبارسنجی](docs/VALIDATION.fa.md)
- [منابع، نسخه‌ها و مجوزها](THIRD_PARTY_NOTICES.md)
- [پیکربندی نمونه](config.yaml)
- [متغیرهای محیطی نمونه](.env.example)
- [استقرار OpenShift](deploy/deployment.yaml)

## معماری در یک نگاه

```text
Prometheus / Thanos query_range
    → هم‌ترازی UTC و ماتریس ویژگی‌های ثابت
    → پاک‌سازی علّی و استانداردسازی براساس بخش آموزش
    → subprocess: OmniAnomaly / TF1
    → کالیبراسیون مستقل POT و ذخیرهٔ checkpoint + scaler + manifest
    → restore و inference در subprocess تازه
    → انتشار مدل و /metrics → Prometheus → Alertmanager
```

TensorFlow در پردازش HTTP وارد نمی‌شود؛ هر عملیات آموزش یا استنتاج در پردازش جدا اجرا می‌شود. این جداسازی در **همان محیط Python image** است، نه دو محیط dependency مستقل.

## شروع

۱. در `config.yaml` آدرس Prometheus و PromQLهای مربوط به یک موجودیت را تنظیم کنید.

۲. در محیط دارای Docker، image مرجع قدیمی را بسازید و پیش از استفاده، آزمون واقعی را اجرا کنید:

```bash
docker build -t pad-omnianomaly:0.1 .
docker run --rm pad-omnianomaly:0.1 python -m unittest discover -s tests -v
docker run --rm pad-omnianomaly:0.1 python scripts/smoke_real.py --epochs 1
```

این فرمان‌ها **دستورالعمل اجرای آتی** هستند، نه ادعای موفقیت build یا smoke. image پایه Python 3.6 و TensorFlow 1.12 قدیمی و خارج از پشتیبانی‌اند؛ استفادهٔ تولیدی نیازمند بررسی امنیت و تصمیم دربارهٔ مهاجرت است. وابستگی‌های GitHub هنگام build دریافت می‌شوند؛ ZIP یک بستهٔ نصب آفلاین نیست.

۳. سرویس را با ذخیره‌سازی پایدار اجرا کنید. پیش از اجرا، `YOUR_PROMETHEUS_HOST` را با آدرس واقعی قابل دسترس از داخل کانتینر جایگزین کنید و queryهای `config.yaml` را مطابق متریک‌های خودتان تغییر دهید:

```bash
docker volume create pad-omni-models
docker run -d --name pad-omni \
  -p 127.0.0.1:8080:8080 \
  --mount type=volume,src=pad-omni-models,dst=/data \
  --mount type=bind,src="$(pwd)/config.yaml",dst=/app/config.yaml,readonly \
  -e FLT_PROM_URL=http://YOUR_PROMETHEUS_HOST:9090 \
  pad-omnianomaly:0.1
```

`localhost` داخل کانتینر، میزبان Docker نیست. روی Linux و Docker جدید، برای Prometheus میزبان می‌توانید به فرمان بالا `--add-host=host.docker.internal:host-gateway` اضافه و آدرس را `http://host.docker.internal:9090` کنید؛ Prometheus باید روی interface قابل دسترس گوش کند و firewall اجازه دهد. برای TLS/token راهنمای کامل را ببینید. volume باید برای کاربر کانتینر قابل‌نوشتن باشد.

۴. لاگ و سلامت را بررسی کنید:

```bash
docker logs -f pad-omni
# فرمان‌های زیر را در ترمینال دیگری اجرا کنید:
curl -i http://localhost:8080/healthz
curl -i http://localhost:8080/readyz
curl -s http://localhost:8080/metrics
```

پاسخ 503 از `/readyz` پیش از پایان آموزش اولیه طبیعی است؛ اگر ادامه یافت لاگ‌ها را بررسی کنید. تنظیم پیش‌فرض نیازمند تاریخچهٔ حدود سه روز از متریک‌های انتخابی است. موفقیت `/healthz` به معنی آماده‌بودن مدل نیست.

۵. توقف و شروع مجدد:

```bash
docker stop pad-omni
docker start pad-omni
```

برای نصب مستقیم یا OpenShift، [راهنمای کامل](docs/README.fa.md) را دنبال کنید. اجرای دستورها از ریشهٔ پوشهٔ استخراج‌شدهٔ `pad-omnianomaly` است.

## ساختار

```text
app.py                       نقطهٔ ورود سرویس
pad_omni/
  config.py                  بارگذاری و محدودیت‌های تنظیمات
  data.py                    query_range، DataFrame و کنترل داده
  preprocess.py              استانداردسازی و پنجره‌بندی مرجع
  backend.py                 adapter مدل واقعی TensorFlow 1
  pot.py                     آستانهٔ ثابت با SPOT initialization
  worker.py                  یک عملیات train یا infer
  service.py                 زمان‌بندی، انتشار مدل، HTTP و متریک‌ها
vendor/OmniAnomaly/           کد مدل upstream و مجوز آن
vendor/SOURCES.json          نسخه‌های منابع بررسی‌شده
scripts/                     smoke واقعی و بنچمارک مسیر داده
tests/                       تست‌های واحد و یکپارچهٔ HTTP
deploy/                      manifestهای استقرار و پایش
docs/                        راهنمای فارسی و وضعیت اعتبارسنجی
reports/                     خروجی ثبت‌شدهٔ آزمون‌ها
```

## سازگاری با PAD

نام چند متغیر `FLT_*` حفظ شده است، ولی این بسته fork کامل PAD یا plugin سازگار با همهٔ APIها و dashboardهای آن نیست. مدل‌های Prophet/Fourier در این سرویس اجرا نمی‌شوند. برای پرهیز از تداخل وابستگی‌ها، از image قدیمی PAD به‌عنوان پایه استفاده نشده است. جزئیات تغییر پیش‌فرض‌ها، جداکنندهٔ فهرست queryها و نام متریک‌های جدید در راهنما آمده است.
