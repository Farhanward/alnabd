def markdown(summary: dict, title: str = "تقرير النبض") -> str:
    m = summary.get("metrics", {})
    l = summary.get("latency_ms", {})
    mem = summary.get("memory_mb", {})
    return "\n".join([
        f"# {title}", "",
        f"- السجلات: `{summary.get('records')}`",
        f"- المعالجة: `{summary.get('processed')}`",
        f"- الأخطاء: `{summary.get('errors')}`",
        f"- الانهيار: `{'PASS' if summary.get('collapse_check', {}).get('passed') else 'FAIL'}`",
        "",
        "## الدقة", "",
        f"- Accuracy: `{m.get('accuracy', 0):.4f}`",
        f"- Precision: `{m.get('precision', 0):.4f}`",
        f"- Recall: `{m.get('recall', 0):.4f}`",
        f"- Specificity: `{m.get('specificity', 0):.4f}`",
        f"- F1: `{m.get('f1', 0):.4f}`",
        f"- TP/TN/FP/FN: `{m.get('tp')}/{m.get('tn')}/{m.get('fp')}/{m.get('fn')}`",
        "",
        "## الأداء", "",
        f"- p99: `{l.get('p99', 0):.4f}ms`",
        f"- peak memory: `{mem.get('peak', 0):.4f}MB`",
        f"- elapsed: `{summary.get('elapsed_seconds', 0):.2f}s`",
        "",
    ])
