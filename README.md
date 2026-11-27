# PAD Modern — نسخه آزمایشی Python 3.15 / Windows

**این شاخه بازپیاده‌سازی آزمایشی مدل است، نه ارتقای تأییدشده یا نسخه جدید رسمی OmniAnomaly.** backend قدیمی TF1 با مدل PyTorch جایگزین شده است. رفتار و دقت معادل upstream ادعا نمی‌شود. نسخه قبلی جداگانه حفظ شده است.

هدف نصب: **Python 3.15.2، Windows x64 استاندارد دارای GIL، بدون Docker و WSL**. نصب واقعی روی Python 3.15/Windows و اجرای مدل در محیط ساخت بسته تأیید نشده است. وجود کد و فایل requirements اثبات سازگاری نیست.

## نصب

Python 3.15.2 x64 را از python.org نصب کنید. ZIP را در پوشه تازه استخراج کنید؛ از venv و مدل‌های نسخه قبلی استفاده نکنید. PowerShell را در ریشه پروژه باز کنید:

```powershell
.\scripts\windows\setup.ps1 -Python 'C:\Path\To\Python315\python.exe'
notepad config.windows.yaml
.\scripts\windows\start.ps1
```

مسیر Python نمونه است. `setup.ps1` نسخه/معماری را کنترل می‌کند، محیط مجازی می‌سازد، تمام wheelها را با `--only-binary=:all:` دریافت می‌کند، نصب و `pip check` و تست‌ها و smoke واقعی مدل را اجرا می‌کند. در هر خطا متوقف می‌شود. شبکه و فضای کافی برای wheelهای PyTorch لازم است. اگر wheel سازگار برای یکی از وابستگی‌ها موجود نباشد، نصب متوقف خواهد شد؛ نسخه Python را خودسرانه عوض نکنید و خطا را ثبت کنید.

بعد از موفقیت کامل، فهرست نسخه‌های نصب‌شده در `requirements.resolved.windows.txt` ثبت می‌شود. این فایل lock دارای hash نیست؛ برای بازتولید، wheelhouse و hash فایل‌ها را نیز نگه دارید. setup سیاست اجرای PowerShell را تغییر نمی‌دهد و نیاز به Administrator ندارد. اگر اجرای اسکریپت طبق سیاست سازمان مسدود است، با مدیر هماهنگ کنید.

در `config.windows.yaml` آدرس Prometheus و PromQLها را تنظیم کنید. `localhost:9090` یعنی Prometheus روی همین Windows. متغیر محیطی `FLT_PROM_URL` در صورت وجود بر YAML اولویت دارد. مدل جدید در `models-modern` ذخیره می‌شود. تاریخچه پیش‌فرض سه روز است. برای توقف Ctrl+C بزنید؛ عملیات درحال اجرا ممکن است تا timeout زمان ببرد. نصب Windows Service انجام نمی‌شود.

```powershell
curl.exe -i http://localhost:8080/healthz
curl.exe -i http://localhost:8080/readyz
curl.exe -s http://localhost:8080/metrics
```

تا آموزش اولیه، readiness می‌تواند 503 باشد. سرویس HTTP روی همه interfaceها گوش می‌کند و احراز هویت داخلی ندارد؛ Windows Firewall را به مبدأهای مجاز محدود کنید.

## مدل جدید دقیقاً چیست؟

- encoder از نوع GRU؛ posterior بازگشتی وابسته به latent قبلی؛ planar normalizing flow با قید وارون‌پذیری.
- prior اولیه استاندارد Gaussian و transition prior آموختنی بر اساس latent قبلی.
- decoder از نوع GRU با خروجی Gaussian قطری، آموزش با منفی ELBO و gradient clipping.
- جداسازی زمانی train/validation و انتخاب وزن با بهترین validation؛ calibration بیرون داده fit باقی می‌ماند.
- امتیاز: منفی میانگین Monte Carlo از log-probability بازسازی مشترک ویژگی‌های آخر پنجره. این log-probability حاشیه‌ای دقیق نیست. drawها برای هر پنجره ثابت‌اند تا وابسته به ترکیب batch نباشند.
- آستانه POT، مسیر Prometheus، بررسی restore در subprocess جدید و انتشار کاندید همچنان حفظ شده‌اند.

این طراحی inspired-by است، نه ترجمه خط‌به‌خط توزیع‌ها، loss و آموزش upstream. هیچ checkpoint قدیمی قابل استفاده نیست؛ signature شامل شناسه backend و seed شده است تا artifact قدیمی رد شود. scaler، checkpoint و POT را با هم دوباره بسازید. `torch_threads` جای `tf_threads` را گرفته است.

TensorFlow، TensorFlow Probability، zhusuan و tfsnippet در نصب جدید استفاده نمی‌شوند. کد vendored قدیمی فقط برای provenance و الگوریتم SPOT باقی است؛ مدل قدیمی اجرا نمی‌شود. مسیر CPU استفاده می‌شود؛ GPU در این نسخه پیاده نشده است.

## وضعیت اعتبارسنجی

- ۲۹ تست هسته/HTTP با backend آزمایشی پس از تغییرات روی Python 3.11/Linux اجرا شدند؛ نتیجه در `reports/modern-unit-tests.txt` است.
- دو تست مدل واقعی برای fit/save/restore، سازگاری batch و ورودی نامعتبر اضافه شده‌اند؛ در نبود PyTorch **skip** می‌شوند، نه اینکه صحت مدل اثبات شده باشد.
- نصب PyTorch در محیط ساخت بسته کامل نشد: endpoint CPU خطای proxy/timeout داد؛ نصب جایگزین نیز به timeout خورد. بنابراین smoke واقعی و تست‌های مدل اجرا نشده‌اند.
- Windows، Python 3.15، resolver وابستگی‌های جدید، PowerShell و کیفیت تشخیص هنوز تأیید نشده‌اند. حافظه گزارش‌شده worker، محدودیت سخت منابع یا حافظه کل سرویس نیست.
- گزارش‌های `reports/legacy` و دیگر مستندات قدیمی مربوط به TF1 هستند و دستور نصب این شاخه نیستند. Dockerfile و manifestهای deployment قدیمی از این شاخه حذف شده‌اند.

پیش از استفاده عملی باید روی Windows هدف نصب، smoke واقعی، timeout، رقابت قفل دو سرویس، restart، و دقت روی داده برچسب‌خورده بررسی شود. model code بدون اجرای واقعی «آماده تولید» محسوب نمی‌شود.

## منابع نسخه هدف

طبق بررسی وب هنگام ساخت بسته:
- Python پایدار 3.15.2: https://www.python.org/downloads/release/python-3152/
- اعلام پشتیبانی Python 3.15 در PyTorch 2.15: https://pytorch.org/blog/pytorch-2-15-release-blog/

اعلام پشتیبانی عمومی جای بررسی wheelهای Windows همه dependencyها را نمی‌گیرد؛ setup عمداً این موضوع را در محیط مقصد کنترل می‌کند.
