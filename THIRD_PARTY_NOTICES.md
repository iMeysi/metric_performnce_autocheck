# منابع و اطلاعیهٔ کد ثالث

## OmniAnomaly

- منبع: https://github.com/NetManAIOps/OmniAnomaly
- commit بررسی‌شده: `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`
- کد مدل و SPOT مورد استفاده در `vendor/OmniAnomaly/` قرار دارد. این پوشه زیرمجموعهٔ لازم از repository است، نه نسخهٔ کامل داده‌ها و artifactهای آزمایش مقاله.
- مجوز upstream: MIT، با حق نشر `Copyright (c) 2021 NetManAIOps-OmniAnomaly`.
- متن کامل مجوز بدون حذف در [vendor/OmniAnomaly/LICENSE](vendor/OmniAnomaly/LICENSE) آمده است.
- `main.py` و README upstream برای مرجع همراه بسته‌اند؛ نقطهٔ ورود این سرویس `app.py` در ریشه است. این سرویس از مسیر label-aware ارزیابی upstream برای تصمیم آنلاین استفاده نمی‌کند.

## PAD

- منبع طراحی و بررسی سازگاری تنظیمات: https://github.com/AICoE/prometheus-anomaly-detector
- commit بررسی‌شده: `822abeccf33da2459e35cfb38de762ed3a8b6a56`
- این بسته fork کامل PAD نیست و image یا درخت کامل کد PAD را توزیع نمی‌کند. ثبت revision به معنی برابری کامل رفتار و API این سرویس با PAD نیست.

## وابستگی‌های مدل دریافت‌شده هنگام build

| پروژه | منبع | revision |
|---|---|---|
| zhusuan | https://github.com/thu-ml/zhusuan | `4386b2a12ae4f4ed8e694e504e51d7dcdfd6f22a` |
| tfsnippet | https://github.com/haowen-xu/tfsnippet | `7b43abdbdd29f1914dbc11b961b5d45b9de76653` |

این دو وابستگی در ZIP vendored نشده‌اند؛ URLهای commit ثابت در `requirements.txt` قرار دارند. سایر dependencyها و image پایه نیز هنگام build دریافت می‌شوند و شرایط مجوز و امنیتی خودشان را دارند. این فایل SBOM یا ممیزی جامع مجوز نیست؛ برای توزیع image تولیدی، مجوزهای تمام dependencyهای نصب‌شده و image پایه را بررسی و حفظ کنید.

فهرست ماشینی revisionهای بررسی‌شده در [vendor/SOURCES.json](vendor/SOURCES.json) است. نسخه‌های ثابت به‌تنهایی تضمین سازگاری، امنیت، یا قابلیت build امروز نیستند؛ وضعیت آزمون در [docs/VALIDATION.fa.md](docs/VALIDATION.fa.md) ثبت شده است.
