# النبض

النبض أداة محلية لتقدير استقبال المحتوى قبل نشره. النسخة الحالية تستخدم `tweet_eval/sentiment` من Hugging Face كبيانات اجتماعية كبيرة: الإيجابي يعد مؤشراً لاحتمال استقبال أفضل، والسلبي/المحايد يحتاج تحسيناً.

## أوامر

```powershell
python -m alnabd.cli download --limit 12000
python -m alnabd.cli train --input data\external\tweet_eval_sentiment_12000.jsonl
python -m alnabd.cli score --text "new easy tool to save time"
python -m alnabd.cli batch --input data\external\tweet_eval_sentiment_12000.jsonl
```

هذه نواة محلية وليست نموذج انتشار تجاري نهائي؛ المرحلة التالية تربطها ببيانات Reach Optimizer الفعلية للمنصات الخليجية.

## آخر نتائج

- الاختبارات الذاتية: 2/2 ناجحة.
- بيانات الإنترنت: Hugging Face `cardiffnlp/tweet_eval` config `sentiment` بعدد 12,000 منشور.
- التدريب: 12,000 سجل، 4,680 إيجابي، 7,320 ضعيف/محايد، 20,000 ميزة، threshold=0.6271.
- Benchmark أولي in-sample: Accuracy=90.48%، Precision=89.01%، Recall=86.22%، Specificity=93.20%، F1=87.59%.
- Stress: 36,000 تقييم، 0 أخطاء، p99=0.092ms، peak memory=1.20MB.

## تحسينات إنتاجية 2026-07-04

- `train()` يرفض الآن بيانات التدريب الفارغة أو ذات صنف واحد، ويرفض `max_features` أو `alpha` غير الصالحة.
- النموذج يضبط `threshold` تلقائياً لتعظيم F1 على بيانات التدريب بدلاً من الاعتماد على 0.5 ثابتة.
- اختيار العتبة يتم بتمرير مرتب O(n log n)، لذلك يبقى مناسباً لملفات الاختبار الكبيرة مثل 12,000 سجل.

## التشغيل المؤسسي (Enterprise) — v1.0.0

- **خدمة تقييم HTTP**: `python -m alnabd.cli serve` → `POST /api/score {"text": "..."}` يعيد score/decision/أسباب/rewrite.
- **النموذج يحمل مرة واحدة** عند الإقلاع (`ALNABD_MODEL`، افتراضي `models\alnabd_pulse_model.json`).
- **نقاط فحص**: `/api/health` (مفتوح) · `/api/version` · `/api/metrics`.
- **تهيئة عبر البيئة**: متغيرات `ALNABD_*` — انظر `docs/OPERATIONS.md`.
- **مصادقة**: `ALNABD_API_KEY` → ترويسة `X-API-Key`. **سجلات JSON**: `logs\alnabd.service.jsonl`.
