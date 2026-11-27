# نصب بومی Windows — بدون Docker و WSL

وضعیت: پورت آزمایشی. اجرای Windows، نصب وابستگی‌ها و مدل واقعی در Windows هنوز تأیید نشده‌اند. ۲۹ تست قبلی پس از تغییرات روی Linux/Python 3.11 موفق شدند؛ این نتیجه اثبات سازگاری runtime ویندوز نیست.

## پیش‌نیازها

- Windows x64؛ هدف اولیه Windows 10/11 روی پردازنده x86-64 است، نه ARM.
- Python **3.6 x64** با قابلیت venv و pip. Python جدیدتر را جایگزین نکنید. مسیر واقعی python.exe را به setup بدهید.
- دسترسی اینترنت به PyPI و GitHub. بسته آفلاین نیست.
- محیط محدود و غیرتولیدی: Python 3.6 و TensorFlow 1.12 خارج از پشتیبانی امنیتی هستند.

## نصب

ZIP را استخراج کنید و PowerShell را در پوشه پروژه باز کنید:

```powershell
.\scripts\windows\setup.ps1 -Python 'C:\Python36\python.exe'
```

مسیر بالا نمونه است. اگر سیاست سازمان اجرای اسکریپت را منع می‌کند، با مدیر هماهنگ کنید؛ نیاز به اجرای Administrator یا غیرفعال‌کردن سیاست سراسری نیست. اسکریپت محیط `.venv` می‌سازد، وابستگی‌ها را نصب می‌کند و سپس تست‌ها و smoke مدل واقعی را اجرا می‌کند. با هر خطا متوقف می‌شود. فعال‌کردن venv لازم نیست.

نصب wheelهای قدیمی یا بارگذاری DLL ممکن است شکست بخورد. در خطای `No matching distribution` یا `DLL load failed`، متن کامل خطا، نسخه Python و معماری را ثبت کنید؛ نسخه TensorFlow را خودسرانه بالا نبرید. این بسته تضمین نمی‌کند همه artifactهای قدیمی امروز قابل دریافت باشند.

## تنظیم و اجرا

```powershell
notepad config.windows.yaml
```

`prometheus_url` را به آدرس واقعی Prometheus و queryها را به متریک‌های موجود تغییر دهید. آدرس پیش‌فرض `http://localhost:9090` برای Prometheus روی همین Windows است. اگر متغیر محیطی `FLT_PROM_URL` تعریف شده باشد، بر YAML اولویت دارد. `model_dir: ./models` مسیر محلی ذخیره مدل است؛ اسکریپت همواره از ریشه پروژه اجرا می‌شود. تاریخچه پیش‌فرض سه روز است.

```powershell
.\scripts\windows\start.ps1
```

سرویس در همین ترمینال اجرا می‌شود؛ برای توقف Ctrl+C بزنید. پایان عملیات در حال اجرا ممکن است تا timeout طول بکشد. این نسخه Windows Service یا راه‌اندازی خودکار پس از reboot نصب نمی‌کند.

در ترمینال دیگر:

```powershell
curl.exe -i http://localhost:8080/healthz
curl.exe -i http://localhost:8080/readyz
curl.exe -s http://localhost:8080/metrics
```

503 در readiness پیش از آموزش اولیه طبیعی است؛ healthz موفق به معنی سلامت مدل نیست. HTTP سرویس روی همه interfaceها گوش می‌کند؛ دسترسی پورت 8080 را با Windows Firewall فقط به مبدأهای موردنیاز محدود کنید. API احراز هویت داخلی ندارد.

## تغییرات پورت

- قفل انحصاری مدل: `msvcrt.locking` در Windows و `flock` در Unix؛ خطای قفل نادیده گرفته نمی‌شود.
- peak working set ویندوز از `psutil`؛ این مقدار حافظه worker است، نه مجموع حافظه سرویس و پردازش‌ها و نه محدودکننده سخت حافظه.
- TensorFlow فقط در subprocess با timeout باقی می‌ماند؛ مدل بازنویسی نشده است.
- config و اسکریپت‌های PowerShell مستقل از مسیرهای Linux اضافه شده‌اند.

پیش از استفاده واقعی باید نصب، smoke، رقابت دو سرویس بر سر یک model_dir، timeout، restart و آموزش/استنتاج با Prometheus واقعی روی Windows آزموده شوند. تست‌های adapter صحت مدل را تضمین نمی‌کنند. برای دیگر جزئیات به [راهنمای اصلی](README.fa.md) و [محدودیت‌های اعتبارسنجی](VALIDATION.fa.md) مراجعه کنید.
