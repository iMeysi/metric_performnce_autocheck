# راهنمای فارسی ادغام PAD و OmniAnomaly

## ۱. دامنه و تصمیم‌های طراحی

این بسته منطق موردنیاز PAD برای جمع‌آوری، آموزش دوره‌ای و انتشار نتیجه را به‌صورت سرویس همراه پیاده می‌کند و از کد واقعی OmniAnomaly برای مدل چندمتغیره استفاده می‌کند. ادغام در سطح جریان داده و عملیات است؛ کد کامل PAD و مدل‌های Prophet/Fourier در بسته کپی نشده‌اند. برای مهاجرت، سرویس فعلی PAD را ابتدا حفظ کنید، این exporter را کنار آن راه بیندازید و پس از ارزیابی، scrape و هشدارهای مصرف‌کننده را تغییر دهید.

تفاوت‌های عمدی با پیشنهاد اولیه:

- **محیط مدل:** Python 3.6 و TensorFlow CPU 1.12؛ نصب روی Python جدید یا image فعلی PAD مفروض نیست. پردازش سرویس و workerها از یک interpreter استفاده می‌کنند، ولی TF فقط داخل worker بارگذاری می‌شود.
- **پیش‌پردازش:** استانداردسازی میانگین/انحراف معیار، نه MinMax؛ پارامترهای آن فقط از بخش آموزش یاد گرفته می‌شوند.
- **API واقعی:** `OmniAnomaly` به‌تنهایی متد `fit()` ندارد؛ آموزش با `Trainer.fit()` و امتیاز با `Predictor.get_score()` انجام می‌شود.
- **POT:** کالیبراسیون روی بخش زمانی جدا از دادهٔ آموزش و validation است. از برچسب واقعی، جست‌وجوی بهترین F1 و point adjustment در تصمیم آنلاین استفاده نمی‌شود.
- **پروفایل سبک‌تر:** پیش‌فرض بسته ۳ روز، بازآموزی ۱۲۰ دقیقه، `hidden=dense=128` است؛ پروفایل ۱۵ روز/۱۵ دقیقه/۵۰۰ در بخش تنظیمات توضیح داده شده است.
- **HTTP client:** به‌جای `prometheus-api-client`، از `requests` با کنترل حجم پاسخ و زمان استفاده شده است.

این تغییرها باید در ارزیابی سازگاری لحاظ شوند. مدل چندمتغیره یک ناهنجاری برای کل گروه می‌دهد، نه پیش‌بینی یا توضیح علت برای هر متریک.

## ۲. معماری سه‌لایه

### لایهٔ دریافت داده — `pad_omni/data.py`

۱. هر گروه شامل چند PromQL متعلق به یک موجودیت پایش‌شده است؛ برای مثال نرخ درخواست، متوسط latency و حافظهٔ یک سرویس.
۲. درخواست‌های `query_range` روی شبکهٔ زمانی مشترک با گام `step_seconds` و در قطعات `chunk_rows` انجام می‌شوند.
۳. هر ترکیب **query + label-set** یک ویژگی مستقل است. ترتیب ویژگی‌ها در manifest ذخیره می‌شود؛ تنها تعداد queryها تعیین‌کنندهٔ تعداد ویژگی‌ها نیست.
۴. خروجی `collect_frame()` یک `pandas.DataFrame` با index زمان UTC است. دادهٔ نامتناهی به missing تبدیل می‌شود.
۵. فقط forward-fill محدود مجاز است. backfill، استفاده از آینده، حذف ردیف و جایگزینی missing با صفر انجام نمی‌شود. در استنتاج، آخرین نقطهٔ ارزیابی باید برای همهٔ ویژگی‌ها موجود باشد.

نمونهٔ استفاده از API داخلی در محیط نصب‌شده:

```python
import time
from pad_omni.config import load
from pad_omni.data import Prometheus, collect_frame, clean

c = load('config.yaml')
g = c['groups'][0]
end = int((time.time() - c['lag_seconds']) // c['step_seconds']) * c['step_seconds']
start = end - (c['rolling_seconds'] // c['step_seconds']) * c['step_seconds']
client = Prometheus(c)
try:
    frame = collect_frame(client, g['queries'], start, end, c)
finally:
    client.close()
values = clean(frame.values, c)  # شکل: (T, D)، float32
```

اضافه‌شدن یک series هنگام استنتاج به ابعاد مدل اضافه نمی‌شود؛ schema قبلی حفظ می‌شود و ویژگی‌های جدید نادیده گرفته می‌شوند، مگر آنکه تعداد seriesهای کشف‌شده از بودجه عبور کند. حذف ویژگی قدیمی باعث missing و شکست اعتبارسنجی می‌شود. برای تغییر عمدی schema باید مدل جدید آموزش داده شود.

### لایهٔ آموزش — `worker.py`، `preprocess.py` و `backend.py`

داده به‌ترتیب زمان به ۷۰٪ اولیه و ۳۰٪ انتهایی تقسیم می‌شود. Trainer داخل ۷۰٪ اولیه، ۲۰٪ انتهایی را برای validation کنار می‌گذارد؛ در نتیجه تقریباً:

| بخش | سهم کل تاریخچه | کاربرد |
|---|---:|---|
| آموزش | ۵۶٪ | یادگیری scaler و وزن‌های مدل |
| validation | ۱۴٪ | اعتبارسنجی داخلی Trainer |
| calibration | ۳۰٪ | تعیین آستانه پس از آموزش |

گردکردن تعداد ردیف‌ها ممکن است نسبت دقیق را تغییر دهد. مقدارهای واقعی در manifest ثبت می‌شوند. برای هر بخش، کف تعداد پنجره‌ها کنترل می‌شود؛ calibration حداقل `min_calibration` امتیاز نیاز دارد.

استانداردسازی هر ستون:

```text
z = (x - mean_train) / std_train
```

برای ستون تقریباً ثابت، scale برابر ۱ قرار می‌گیرد. میانگین و scale در `scaler.npz` ذخیره می‌شوند. خروجی باید finite و قابل نمایش در float32 باشد.

**نکتهٔ شکل ورودی:** API عمومی `Trainer.fit()` ماتریس دوبعدی `(T, D)` می‌گیرد. کد upstream آن را به batchهای سه‌بعدی `(B, W, D)` تبدیل می‌کند. ساخت یک tensor بزرگ برای تمام پنجره‌ها لازم نیست. تابع مرجع زیر برای فهم و تست همین شکل است، نه آرگومان مستقیم Trainer:

```python
from pad_omni.preprocess import Standardizer, window_batches

# fit_values تنها بخش آموزش است؛ all_values می‌تواند validation/calibration هم داشته باشد.
scaler = Standardizer().fit(fit_values)
z = scaler.transform(all_values)
for batch in window_batches(z, window=100, batch_size=32):
    assert batch.shape[1:] == (100, z.shape[1])
```

مدل recurrent VAE با latent متصل و normalizing flow ساخته می‌شود. `window_length=100`، `z_dim=3`، `nf_layers=20` و `n_z=8` در پروفایل نمونه‌اند. learning rate برابر `0.001` و gradient clipping برابر `10` در adapter مشخص شده‌اند. هر بازآموزی از ابتداست؛ warm start یا fine-tune اجرا نمی‌شود.

### لایهٔ امتیاز، انتشار و سرویس — `pot.py` و `service.py`

`Predictor` برای آخرین نقطهٔ هر پنجره log reconstruction probability مشترک را می‌دهد. علامت آن معکوس می‌شود:

```text
score = - joint reconstruction log probability
anomaly = int(score > threshold)
```

پس امتیاز بزرگ‌تر نامعمول‌تر است؛ مقدار مطلق بین مدل‌ها یا گروه‌ها الزاماً قابل مقایسه نیست. این امتیاز احتمال وقوع حادثه یا درصد اطمینان نیست.

آستانه با `SPOT.initialize(min_extrema=False)` روی دم بالایی امتیازهای calibration ساخته می‌شود. آستانه **ثابت تا بازآموزی بعدی** است؛ الگوریتم online/dynamic SPOT اجرا نمی‌شود. `pot_q` احتمال خطر اسمی مدل دم است و تضمین نرخ خطای واقعی نیست. دم ناکافی یا منحط موجب رد مدل جدید می‌شود؛ fallback خاموش به quantile ساده وجود ندارد.

چرخهٔ انتشار برای هر گروه:

```text
جمع‌آوری تاریخچه → scaler → Trainer → calibration/POT
→ ذخیره در v-... → بررسی manifest
→ restore و استنتاج با دادهٔ جاری در subprocess تازه
→ جایگزینی اتمیک فایل CURRENT → پاک‌کردن نسخهٔ قدیمی
```

اگر آموزش، POT یا بررسی کاندید شکست بخورد، `CURRENT` تغییر نمی‌کند. مدل قبلی فقط تا وقتی معتبر و جوان‌تر از `max_model_age` باشد خدمت می‌دهد. فایل JSON با فایل موقت، fsync فایل و `os.replace` منتشر می‌شود؛ این سازوکار ادعای تراکنش توزیع‌شده یا تضمین دوام در هر نوع خرابی storage نیست.

## ۳. تنظیم PromQL و منابع

نمونهٔ `config.yaml` برای یک گروه:

```yaml
prometheus_url: http://prometheus:9090
model_dir: /data/models
step_seconds: 60
poll_seconds: 60
lag_seconds: 120
rolling_seconds: 259200
retrain_seconds: 7200
parallelism: 1
window_length: 100
hidden: 128
dense: 128
z_dim: 3
max_rows: 10000
max_features: 32
groups:
  - name: api
    queries:
      - 'sum(rate(http_requests_total{job="api"}[5m]))'
      - 'sum(rate(http_request_duration_seconds_sum{job="api"}[5m])) / clamp_min(sum(rate(http_request_duration_seconds_count{job="api"}[5m])), 0.001)'
      - 'sum(process_resident_memory_bytes{job="api"})'
```

این نام متریک‌ها مثال‌اند؛ ابتدا هر query را در Prometheus خودتان اجرا کنید. در صورت استفاده از counter، معمولاً rate لازم است. aggregation باید معنای عملیاتی درست داشته باشد؛ حذف labelها بدون توجه به معنای سرویس می‌تواند چند موجودیت نامربوط را مخلوط کند. labelهای ناپایدار مثل pod UID باعث ناپایداری schema می‌شوند. برای چند سرویس، چند گروه مستقل با نام یکتا بسازید؛ از ساخت ویژگی به‌ازای هر pod در گروه بزرگ پرهیز کنید.

### متغیرهای سازگار با PAD

متغیر محیطی تنظیم YAML را override می‌کند. فایل `.env.example` صرفاً نمونه است و برنامه آن را خودکار نمی‌خواند.

| متغیر | رفتار در این بسته |
|---|---|
| `FLT_PROM_URL` | آدرس پایهٔ API Prometheus/Thanos |
| `FLT_METRICS_LIST` | queryهای جداشده با **سمی‌کالن `;`**؛ تمام گروه‌های YAML را با گروه `default` جایگزین می‌کند |
| `FLT_ROLLING_TRAINING_WINDOW_SIZE` | مدت با `s/m/h/d`، مثلاً `3d` یا `15d` |
| `FLT_RETRAINING_INTERVAL_MINUTES` | فاصلهٔ بازآموزی به دقیقه؛ پس از پایان تلاش قبلی برنامه‌ریزی می‌شود |
| `FLT_PARALLELISM` | سقف هم‌زمانی در pool آموزش و، جداگانه، در pool استنتاج |
| `FLT_PROM_ACCESS_TOKEN` | Bearer token مستقیم؛ برای استفادهٔ عملیاتی فایل token ترجیح دارد |
| `PROM_TOKEN_FILE` | فایل token که در هر درخواست دوباره خوانده می‌شود؛ اولویت با این فایل است |
| `PROM_CA_FILE` | مسیر CA برای TLS؛ اعتبارسنجی TLS خاموش نمی‌شود |

سمی‌کالن انتخاب شده چون کاما بخشی از PromQL است؛ این رفتار drop-in برای همهٔ قالب‌های قدیمی PAD نیست. مقدارهای دیگر `FLT_*` الزاماً پشتیبانی نمی‌شوند. متغیرها باید به subprocessها نیز برسند؛ سرویس به‌صورت عادی محیط خود را به آن‌ها منتقل می‌کند.

### پروفایل نزدیک‌تر به درخواست اولیه

برای تاریخچهٔ ۱۵ روز، بازآموزی ۱۵ دقیقه و ابعاد مخفی upstream، ابتدا در YAML تغییر دهید:

```yaml
rolling_seconds: 1296000
retrain_seconds: 900
max_rows: 21601
hidden: 500
dense: 500
```

اگر override محیطی می‌گذارید، مقدارهای `15d` و `15` را برای دو متغیر متناظر تنظیم کنید. افزایش `max_rows` ضروری است؛ تغییر فقط `FLT_ROLLING_TRAINING_WINDOW_SIZE` با سقف فعلی عمداً رد می‌شود. ۲۱۶۰۱ ردیف حاصل هر دقیقه در ۱۵ روز با دو endpoint است. این پروفایل **از نظر RAM/CPU یا زمان آموزش تأیید نشده** و پیشنهاد ظرفیت تولیدی نیست.

### بودجه‌ها و محدودیت‌ها

| گزینه | پیش‌فرض | توضیح |
|---|---:|---|
| `max_groups` | 8 | تعداد گروه‌ها |
| `max_features` | 32 | تعداد series/ویژگی‌های کشف‌شده در هر گروه |
| `max_rows` | 10000 | تعداد ردیف‌های تاریخچه |
| `chunk_rows` | 1000 | اندازهٔ هر قطعهٔ query_range |
| `max_response_bytes` | 16777216 | سقف بدنهٔ یک پاسخ، نه کل RAM |
| `fill_limit` | 2 | تعداد forward-fill متوالی مجاز |
| `max_missing_fraction` | 0.02 | نسبت missing در کل ماتریس، پیش از fill |
| `request_timeout` | 30 | timeout درخواست؛ retryهای محدود نیز وجود دارند |
| `train_timeout` | 3600 | سقف زمانی subprocess آموزش |
| `inference_timeout` | 120 | سقف زمانی subprocess استنتاج/بررسی کاندید |
| `metric_ttl` | 240 | مهلت اعتبار آخرین نتیجه |
| `max_model_age` | 86400 | بیشینهٔ عمر مدل، ثانیه |
| `tf_threads` | 1 | threadهای درون/بین عملیات TF |
| `pot_level` / `pot_q` | 0.95 / 0.001 | شروع دم و خطر اسمی POT |
| `min_peaks` / `min_calibration` | 20 / 500 | کف دادهٔ کالیبراسیون |

`parallelism=1` لزوماً یعنی فقط یک worker نیست: یک آموزش یا بررسی کاندید می‌تواند هم‌زمان با یک استنتاج جاری اجرا شود. در حالت کلی تا حدود `2 × parallelism` worker مدل ممکن است فعال باشند. محدودیت threadهای TF و BLAS جدا از این عدد است. RSS گزارش‌شده برای یک worker است، نه مجموع pod.

پنجرهٔ غلتان و batchها مصرف داده را محدود می‌کنند، اما تخصیص حافظهٔ TF، optimizer، flowها و پردازش JSON همچنان هزینه دارند. سقف سخت RAM در Deployment اعمال می‌شود و می‌تواند OOMKill ایجاد کند. پیش از تولید، peak مجموع pod را در بدترین هم‌زمانی اندازه بگیرید. کوچک‌کردن batch، hidden/dense، تعداد ویژگی‌ها یا تاریخچه را پس از ارزیابی کیفیت انجام دهید.

## ۴. ساخت و اجرای محلی

### گزینهٔ توصیه‌شده برای آزمون: image جداگانهٔ legacy

Dockerfile یک **مرجع build هنوز آزموده‌نشده** بر پایهٔ `python:3.6.15-buster` است؛ CPU و معماری رایج x86_64 را هدف می‌گیرد. سازگاری ARM، GPU یا سخت‌افزارهای فاقد قابلیت‌های لازم wheel قدیمی تضمین نشده است.

```bash
docker build -t pad-omnianomaly:0.1 .
docker run --rm pad-omnianomaly:0.1 python -m unittest discover -s tests -v
docker run --rm pad-omnianomaly:0.1 python scripts/smoke_real.py --epochs 1
```

آخرین فرمان مدل واقعی را روی Prometheus HTTP مصنوعی آموزش می‌دهد و در پردازش تازه restore می‌کند؛ دادهٔ تولیدی را فراخوانی نمی‌کند و tokenهای موروثی را حذف می‌کند. شکست آن را نادیده نگیرید. موفقیت آن نیز صحت تشخیص یا ظرفیت در محیط واقعی را ثابت نمی‌کند.

برای سرویس محلی با volume پایدار:

```bash
docker volume create pad-omni-models
docker run --rm --name pad-omni -p 127.0.0.1:8080:8080 \
  --mount type=volume,src=pad-omni-models,dst=/data \
  --mount type=bind,src="$(pwd)/config.yaml",dst=/app/config.yaml,readonly \
  -e FLT_PROM_URL=http://YOUR_PROMETHEUS_HOST:9090 \
  pad-omnianomaly:0.1
```

آدرس Prometheus باید از داخل container قابل دسترسی باشد؛ `localhost` داخل container به میزبان اشاره نمی‌کند. برای TLS/token، فایل‌ها را read-only mount و `PROM_CA_FILE`/`PROM_TOKEN_FILE` را تنظیم کنید. writable بودن `/data` برای UID اجرایی را بررسی کنید؛ در volume موجود ممکن است آماده‌سازی مجوزها لازم شود.

### نصب مستقیم، فقط در محیط ایزولهٔ Python 3.6

```bash
python3.6 -m venv .venv
. .venv/bin/activate
python -m pip install pip==21.3.1 setuptools==59.6.0 wheel==0.37.1
python -m pip install -r requirements.txt
python -m pip check
# model_dir را در config.yaml به یک مسیر قابل‌نوشتن تغییر دهید.
python app.py --config config.yaml
```

`requirements.txt` قفل کامل همهٔ وابستگی‌های گذرا با hash نیست. دو dependency از commit ثابت GitHub دریافت می‌شوند. build به دسترسی شبکه و موجودبودن artifactهای قدیمی وابسته است؛ برای بازتولید قابل‌اتکا، پس از build موفق، dependency lock/hash و digest image را ثبت و artifactها را در registry داخلی نگه دارید. محیط قدیمی خارج از پشتیبانی امنیتی است؛ آن را با مجوزهای محدود و شبکهٔ کنترل‌شده اجرا کنید. نوسازی dependencyها باید همراه تست API و ارزیابی مجدد مدل باشد، نه فقط تغییر شمارهٔ نسخه‌ها.

### بررسی HTTP

```bash
curl -i http://localhost:8080/healthz
curl -i http://localhost:8080/readyz
curl -s http://localhost:8080/metrics
```

`/healthz` فقط زنده‌بودن HTTP را نشان می‌دهد. `/readyz` تا زمانی که **همهٔ گروه‌ها** نتیجهٔ تازه و مدل معتبر نداشته باشند 503 می‌دهد؛ زمان آموزش اولیه بخشی از این حالت است. ناهنجاری‌بودن داده به‌خودی‌خود readiness را false نمی‌کند.

## ۵. استقرار OpenShift

manifestها قالب‌اند و در cluster واقعی آزموده نشده‌اند. پیش‌نیازها: image ساخته و اسکن‌شده، دسترسی registry، StorageClass مناسب، endpoint مجاز Prometheus و در صورت نیاز User Workload Monitoring و CRDهای Prometheus Operator.

### ۵.۱ namespace، image و ConfigMap

```bash
oc new-project pad-omni
# نام registry/سازمان را جایگزین کنید.
docker tag pad-omnianomaly:0.1 quay.io/YOUR_ORG/pad-omnianomaly:0.1
docker push quay.io/YOUR_ORG/pad-omnianomaly:0.1

oc -n pad-omni create configmap pad-omni-config \
  --from-file=config.yaml=./config.yaml --dry-run=client -o yaml \
  | oc -n pad-omni apply -f -
```

در `deploy/deployment.yaml` آدرس image را ترجیحاً با digest تغییر دهید. مقدار `FLT_PROM_URL` داخل Deployment بر YAML اولویت دارد؛ endpoint فعلی قالب، Thanos داخلی است و باید با cluster تطبیق داده شود. requests/limits فعلی تنها نقطهٔ شروع تخمینی‌اند، نه نتیجهٔ بنچمارک TF.

### ۵.۲ دسترسی به داده و TLS

`deploy/rbac-monitoring.yaml` اختیاری و نیازمند تأیید مدیر است. نقش `cluster-monitoring-view` دید cluster-wide به داده‌های پایش می‌دهد. اگر سیاست شما محدود به namespace است، از endpoint و Role/RoleBinding محدودتر مورد تأیید مدیر استفاده کنید؛ این فایل را بدون بررسی اعمال نکنید. namespace داخل subject باید با namespace استقرار برابر باشد.

```bash
# فقط پس از بررسی و تأیید مدیر:
oc apply -f deploy/rbac-monitoring.yaml
oc -n pad-omni apply -f deploy/deployment.yaml
```

ConfigMap مربوط به service CA با annotation تزریق OpenShift ساخته شده است. سرویس از token چرخشی ServiceAccount و CA mountشده استفاده می‌کند. آماده‌شدن CA و سازگاری CA با endpoint مقصد را کنترل کنید. برای Prometheus خارجی، CA و روش احراز هویت مناسب همان مقصد را جایگزین کنید؛ خطای certificate را با خاموش‌کردن verify حل نکنید.

### ۵.۳ بررسی pod و storage

```bash
oc -n pad-omni get pods,pvc,svc
oc -n pad-omni logs deployment/pad-omni --tail=100
oc -n pad-omni rollout status deployment/pad-omni --timeout=20m
oc -n pad-omni port-forward service/pad-omni 8080:8080
```

timeout دستور rollout فقط زمان انتظار CLI است؛ اگر آموزش اولیه طولانی‌تر باشد، لازم است افزایش یابد. writable بودن PVC برای UID تصادفی OpenShift را بررسی کنید. manifest UID ثابت یا SCC دارای امتیاز بالا درخواست نمی‌کند؛ رفتار fsGroup/مجوز storage به SCC و provisioner cluster وابسته است. image برای گروه ۰ آماده شده، اما این به‌تنهایی مجوز PVC موجود را تضمین نمی‌کند.

Deployment دارای یک replica و `Recreate` است؛ قفل `flock` روی model directory مانع writer هم‌زمان می‌شود. برای scale-out، گروه‌ها را میان Deploymentهای مستقل و volumeهای مستقل تقسیم کنید؛ افزایش replicas روی PVC فعلی راهکار HA نیست. مدل‌ها و JSONهای ذخیره‌شده trusted artifact هستند؛ نوشتن روی PVC را محدود کنید. امضای تنظیمات در manifest جای امضای رمزنگاری‌شدهٔ artifact را نمی‌گیرد.

برای اعمال تغییر تنظیمات، ConfigMap را دوباره بسازید/اعمال و سپس restart کنید؛ hot reload پشتیبانی نمی‌شود:

```bash
oc -n pad-omni create configmap pad-omni-config \
  --from-file=config.yaml=./config.yaml --dry-run=client -o yaml \
  | oc -n pad-omni apply -f -
oc -n pad-omni rollout restart deployment/pad-omni
```

### ۵.۴ scrape و هشدار

در صورت فعال‌بودن Operator و User Workload Monitoring:

```bash
oc -n pad-omni apply -f deploy/monitoring.yaml
```

selectorها، labelهای لازم برای discovery و سیاست namespace را در cluster خود بررسی کنید. Service دارای `publishNotReadyAddresses: true` است تا scrape خطایابی در دورهٔ ناآماده‌بودن مدل ممکن باشد؛ کشف و scrape targetهای not-ready را در Prometheus واقعی تأیید کنید.

سرویس HTTP احراز هویت یا TLS داخلی ندارد؛ Route عمومی برای آن ایجاد نشده است. با NetworkPolicy و سیاست‌های سازمانی دسترسی را محدود کنید. خروجی مدل و نام گروه‌ها می‌توانند اطلاعات عملیاتی حساس باشند.

## ۶. متریک‌ها و معناشناسی fail-closed

تمام متریک‌های زیر label به نام `group` دارند:

| نام | معنی |
|---|---|
| `pad_omni_ready` | وجود نتیجهٔ تازه و معتبر، ۰ یا ۱ |
| `pad_omni_training` | آموزش در صف یا در حال اجرا |
| `pad_omni_anomaly_score` | منفی log reconstruction probability مشترک |
| `pad_omni_anomaly_threshold` | آستانهٔ ثابت POT مدل جاری |
| `pad_omni_anomaly` | ۱ فقط برای `score > threshold` |
| `pad_omni_last_evaluation_timestamp_seconds` | زمان نقطهٔ ارزیابی Prometheus |
| `pad_omni_model_trained_timestamp_seconds` | زمان پایان ساخت مدل |
| `pad_omni_training_seconds` | زمان ساخت آخرین مدل منتشرشده، شامل دریافت و POT؛ بدون بررسی تازهٔ کاندید |
| `pad_omni_training_peak_rss_bytes` | peak RSS worker آموزش آخرین مدل منتشرشده |
| `pad_omni_inference_seconds` | زمان استنتاج موفق، شامل دریافت داده و ساخت graph/restore |
| `pad_omni_inference_peak_rss_bytes` | peak RSS worker استنتاج موفق |
| `pad_omni_training_errors_total` | شمارندهٔ تلاش‌های ناموفق آموزش، از شروع همین سرویس |
| `pad_omni_inference_errors_total` | شمارندهٔ استنتاج‌های ناموفق، از شروع همین سرویس |

وقتی نتیجه نامعتبر، منقضی یا غیرقابل‌دریافت شود، متریک‌های امتیاز، آستانه و پرچم منتشر نمی‌شوند و `ready=0` می‌شود. **نبود پرچم به معنی سالم‌بودن سیستم نیست.** readiness، خطا و سلامت exporter باید جداگانه هشدار داشته باشند. خطای بازآموزی به‌تنهایی استنتاج معتبر مدل قبلی را حذف نمی‌کند.

نمونهٔ عبارت هشدار:

```promql
pad_omni_anomaly == 1
pad_omni_ready == 0
up{service="pad-omni"} == 0
```

در قالب، ماندگاری ناهنجاری ۵ دقیقه و نبود مدل ۱۵ دقیقه است. این‌ها سیاست نمونه‌اند. `up == 0` targetی را که به‌کلی از discovery حذف شده کشف نمی‌کند؛ برای نبود target موردانتظار، alert مناسب `absent()` یا پایش inventory را نیز مطابق labelهای واقعی سازمان اضافه کنید. خروجی فعلی با نام متریک‌های PAD قدیمی یکسان نیست؛ dashboard و recording ruleهای آن باید مهاجرت کنند.

## ۷. عیب‌یابی

| پیام/علامت | اقدام پیشنهادی |
|---|---|
| `prometheus_url / FLT_PROM_URL required` | URL معتبر با http/https و اولویت env را بررسی کنید |
| `rolling history exceeds max_rows` | مدت/گام را کاهش دهید یا آگاهانه سقف ردیف را افزایش دهید |
| `feature cardinality budget exceeded` | PromQL را aggregate کنید و labelهای ناپایدار را حذف/محدود کنید |
| `Prometheus response exceeds byte budget` | query را محدود یا chunk را کوچک‌تر کنید |
| `Prometheus error or partial/warning response` | warning و partial result سمت Prometheus/Thanos را برطرف کنید |
| `query_range must return a matrix` | endpoint و نوع پاسخ API را بررسی کنید |
| `latest evaluation point missing` | scrape، lag، پنجرهٔ rate و موجودبودن آخرین داده را بررسی کنید |
| `unresolved gaps` / `missing-data budget exceeded` | ابتدا کیفیت داده را درست کنید؛ missing را کورکورانه صفر نکنید |
| `history too short` | تاریخچهٔ کامل کافی برای train/validation/calibration تهیه کنید |
| `insufficient or degenerate POT tail` | تنوع calibration و کف نمونه‌ها را بررسی کنید؛ برچسب آنلاین اضافه نکنید |
| `artifact/config signature mismatch` | تنظیمات مؤثر مدل تغییر کرده؛ مدل جدید نیاز است |
| 401/403 | token و مجوز endpoint را بررسی کنید |
| خطای TLS | CA و نام میزبان مقصد را تطبیق دهید |
| `Permission denied` روی `/data` | UID، SCC، گروه volume و مجوز PVC را بررسی کنید |
| `OOMKilled` یا timeout | peak کل pod، هم‌زمانی، مدل و بودجهٔ زمان را اندازه بگیرید |

در شکست آموزش اولیه، تا تلاش بعدی مدل ندارید؛ فاصلهٔ retry برابر `retrain_seconds` است، نه retry فوری. پس از رفع مشکل می‌توان با restart تلاش اولیه را دوباره آغاز کرد. لاگ worker در stderr باقی می‌ماند؛ stdout پرحجم مدل در اجرای سرویس حذف می‌شود.

## ۸. آزمون و پذیرش پیش از تولید

```bash
python -m unittest discover -s tests -v
python -m compileall -q pad_omni tests scripts app.py
python scripts/benchmark_data.py
# فقط در runtime قدیمی سازگار با TensorFlow 1.12:
python scripts/smoke_real.py --epochs 1
```

سه فرمان اول در محیط موجود اجرا شده‌اند؛ فرمان آخر هنوز اجرا/تأیید نشده است. [گزارش اعتبارسنجی](VALIDATION.fa.md) دامنهٔ دقیق آزمون‌ها را تفکیک می‌کند.

چک‌لیست پذیرش:

- [ ] build و `pip check` روی معماری هدف؛ ثبت digest، dependencyها و اسکن امنیتی.
- [ ] smoke واقعی، اجرای همهٔ تست‌ها در Python 3.6 و بررسی restore پس از restart.
- [ ] مقایسهٔ آنلاین روی تاریخچهٔ نمایندهٔ محیط با baseline؛ ارزیابی false positive/false negative روی دادهٔ برچسب‌خوردهٔ جداگانه.
- [ ] سنجش RAM کل pod و زمان آموزش/استنتاج با بیشترین ابعاد و هم‌زمانی موردنیاز.
- [ ] آزمون قطع Prometheus، TLS/token نامعتبر، دادهٔ ناقص، drift schema، timeout و OOMKill.
- [ ] بررسی اینکه در شکست کاندید `CURRENT` قبلی حفظ شود و دادهٔ منقضی هرگز به‌عنوان نتیجهٔ تازه صادر نشود.
- [ ] بررسی PVC/SCC، scrape هنگام not-ready و هشدارهای نبود target و نبود نتیجه.
- [ ] تعریف سیاست بازآموزی و جلوگیری از آلوده‌شدن تاریخچهٔ یادگیری با رخدادهای غیرعادی طولانی.

seed NumPy و TF ثابت است، ولی تضمین bitwise reproducibility بین سخت‌افزارها، کتابخانه‌ها یا زمان‌های جمع‌آوری متفاوت وجود ندارد. برای آزمایش قابل بازتولید، snapshot داده و queryها، نسخهٔ تنظیمات و تصویر runtime را نیز نگه دارید.
