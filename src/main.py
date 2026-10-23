"""Command-line entry points for the product-title classification project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.dataset import audit_records, load_records, normalize_title
from src.inference import benchmark_file, evaluate_model, predict_file, protect_output, write_json
from src.modeling import train_model


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="商品标题分类：字符 TF-IDF + LinearSVC")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="检查本地数据，不修改原始文件")
    audit.add_argument("--data-dir", type=Path, default=Path("data"))
    audit.add_argument("--output", type=Path, default=Path("artifacts/data_audit.json"))
    train = commands.add_parser("train", help="仅使用训练集选模并训练最终模型")
    train.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/tfidf"))
    train.add_argument("--validation-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="评估冻结模型，披露训练测试标题重叠")
    evaluate.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    evaluate.add_argument("--output-dir", type=Path)
    predict = commands.add_parser("predict", help="分批预测 JSONL 或 CSV")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark", help="测量吞吐和采样峰值 RSS")
    benchmark.add_argument("--input", type=Path, default=Path("data/train.jsonl"))
    benchmark.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    for command in (evaluate, predict, benchmark):
        command.add_argument("--model-dir", type=Path, default=Path("artifacts/tfidf"))
        command.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            protect_output(args.output, [args.data_dir / "train.jsonl", args.data_dir / "test.jsonl"])
            training = load_records(args.data_dir / "train.jsonl")
            testing = load_records(args.data_dir / "test.jsonl")
            training_titles = {normalize_title(row["product_name"]) for row in training}
            overlap_titles = {normalize_title(row["product_name"]) for row in testing} & training_titles
            report = {"train": audit_records(training), "test": audit_records(testing),
                      "overlap_unique_titles": len(overlap_titles),
                      "overlap_test_rows": sum(normalize_title(row["product_name"]) in training_titles for row in testing)}
            write_json(args.output, report)
        elif args.command == "train":
            report = train_model(args.data, args.output_dir, args.validation_size, args.seed)
        elif args.command == "evaluate":
            report = evaluate_model(args.model_dir, args.data, args.output_dir, args.batch_size)
        elif args.command == "predict":
            report = predict_file(args.model_dir, args.input, args.output, args.batch_size)
        else:
            protect_output(args.output, [args.input])
            report = benchmark_file(args.model_dir, args.input, args.batch_size)
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
