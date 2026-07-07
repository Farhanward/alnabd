from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import evaluate
from .core import rewrite_text, score_text
from .datasets import fetch_tweet_eval
from .model import PulseModel, train
from .reports import markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="alnabd", description="النبض: توقع استقبال المحتوى وإعادة صياغته.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--out", default="data/external/tweet_eval_sentiment_12000.jsonl")
    dl.add_argument("--limit", type=int, default=12000)
    tr = sub.add_parser("train")
    tr.add_argument("--input", required=True)
    tr.add_argument("--out", default="models/alnabd_pulse_model.json")
    score = sub.add_parser("score")
    score.add_argument("--model", default="models/alnabd_pulse_model.json")
    score.add_argument("--text", required=True)
    batch = sub.add_parser("batch")
    batch.add_argument("--input", required=True)
    batch.add_argument("--model", default="models/alnabd_pulse_model.json")
    batch.add_argument("--json-out", default="reports/alnabd_benchmark.json")
    batch.add_argument("--report", default="reports/alnabd_benchmark.md")
    stress = sub.add_parser("stress")
    stress.add_argument("--input", required=True)
    stress.add_argument("--model", default="models/alnabd_pulse_model.json")
    stress.add_argument("--repeat", type=int, default=3)
    stress.add_argument("--json-out", default="reports/alnabd_stress.json")
    stress.add_argument("--report", default="reports/alnabd_stress.md")
    serve = sub.add_parser("serve")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    sub.add_parser("version")
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        from .service import run_server

        run_server(host=args.host, port=args.port)
        return 0
    if args.cmd == "version":
        from .version import __version__

        print(json.dumps({"service": "alnabd", "version": __version__}, ensure_ascii=False))
        return 0
    if args.cmd == "download":
        print(json.dumps(fetch_tweet_eval(args.out, limit=args.limit), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "train":
        print(json.dumps(train(args.input, args.out), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "score":
        model = PulseModel.load(args.model)
        print(json.dumps(rewrite_text(args.text, model), ensure_ascii=False, indent=2))
        return 0
    if args.cmd in {"batch", "stress"}:
        summary = evaluate(args.input, args.model, repeat=getattr(args, "repeat", 1))
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(args.report).write_text(markdown(summary, "تقرير ضغط النبض" if args.cmd == "stress" else "تقرير النبض"), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["collapse_check"]["passed"] else 2
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
