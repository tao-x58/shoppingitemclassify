"""Evaluation and bounded-memory prediction for the selected classifier."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

import psutil

from src.dataset import load_records, normalize_title
from src.modeling import classification_metrics, load_model


def protect_output(path: str | Path, inputs) -> None:
    path = Path(path)
    for source in map(Path, inputs):
        if path.resolve() == source.resolve() or (path.exists() and source.exists() and path.samefile(source)):
            raise ValueError(f"Output must not overwrite input data: {source}")


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_title_batches(path: str | Path, batch_size: int = 4096) -> Iterator[list[dict]]:
    """Read JSONL/CSV without accumulating the input file in memory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    path = Path(path)
    batch: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".jsonl":
            def json_rows():
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
                    yield line_no, row
            rows = json_rows()
        elif path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream, strict=True)
            try:
                fields = reader.fieldnames
            except csv.Error as exc:
                raise ValueError(f"{path}: invalid CSV header: {exc}") from exc
            if not fields or "product_name" not in fields or len(set(fields)) != len(fields):
                raise ValueError(f"{path}: CSV requires a unique product_name column")
            def csv_rows():
                try:
                    for row in reader:
                        yield reader.line_num, row
                except csv.Error as exc:
                    raise ValueError(f"{path}:{reader.line_num}: invalid CSV: {exc}") from exc
            rows = csv_rows()
        else:
            raise ValueError("Input format must be .jsonl or .csv")
        row_id = 0
        for line_no, row in rows:
            if not isinstance(row, dict) or None in row:
                raise ValueError(f"{path}:{line_no}: expected an object with valid columns")
            title = row.get("product_name")
            if not isinstance(title, str) or not normalize_title(title):
                raise ValueError(f"{path}:{line_no}: product_name must be a nonempty string")
            row_id += 1
            batch.append({"row_id": row_id, "product_name": title})
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def predict_file(model_dir: str | Path, input_path: str | Path,
                 output_path: str | Path, batch_size: int = 4096) -> dict:
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    if output_path.suffix.lower() not in {".jsonl", ".csv"}:
        raise ValueError("Output format must be .jsonl or .csv")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = load_model(model_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=output_path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = None
            if output_path.suffix.lower() == ".csv":
                writer = csv.DictWriter(stream, fieldnames=["row_id", "product_name", "predicted_category"])
                writer.writeheader()
            for batch in iter_title_batches(input_path, batch_size):
                predictions = model.predict([row["product_name"] for row in batch])
                for row, label in zip(batch, predictions, strict=True):
                    result = {**row, "predicted_category": str(label)}
                    if writer:
                        writer.writerow(result)
                    else:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += len(batch)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    seconds = time.perf_counter() - started
    return {"rows": count, "output": str(output_path), "batch_size": batch_size,
            "seconds_excluding_model_load": seconds,
            "rows_per_second": count / seconds if seconds else None}


def evaluate_model(model_dir: str | Path, test_path: str | Path,
                   output_dir: str | Path | None = None, batch_size: int = 4096) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model_dir, test_path = Path(model_dir), Path(test_path)
    output_dir = Path(output_dir) if output_dir else model_dir / "evaluation"
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    test_hash = _sha256(test_path)
    if metadata.get("data_sha256") == test_hash:
        raise ValueError("Evaluation file is identical to the training file")
    records = load_records(test_path)
    if not records:
        raise ValueError("Evaluation data is empty")
    model = load_model(model_dir)
    seen = set(json.loads((model_dir / "training_titles.json").read_text(encoding="utf-8")))
    predictions: list[str] = []
    for offset in range(0, len(records), batch_size):
        predictions.extend(str(label) for label in model.predict(
            [row["product_name"] for row in records[offset:offset + batch_size]]))
    truth = [row["category"] for row in records]
    overlap = [normalize_title(row["product_name"]) in seen for row in records]

    def subset_metrics(flag: bool):
        indices = [i for i, is_seen in enumerate(overlap) if is_seen == flag]
        return classification_metrics([truth[i] for i in indices], [predictions[i] for i in indices]) if indices else None

    report = {"schema_version": 1, "test_file": str(test_path), "test_sha256": test_hash,
              "rows": len(records), "overlap_rows": sum(overlap),
              "overlap_unique_titles": len({normalize_title(row["product_name"])
                                             for row, flag in zip(records, overlap) if flag}),
              "metrics": classification_metrics(truth, predictions),
              "unseen_title_metrics": subset_metrics(False), "seen_title_metrics": subset_metrics(True),
              "note": "Official test rows are preserved; overlap subsets are diagnostics, not model selection."}
    write_json(output_dir / "test_metrics.json", report)
    with (output_dir / "test_errors.jsonl").open("w", encoding="utf-8") as stream:
        for i, (row, label) in enumerate(zip(records, predictions, strict=True), 1):
            if row["category"] != label:
                stream.write(json.dumps({"row_id": i, **row, "predicted_category": label,
                                         "title_seen_in_training": overlap[i - 1]}, ensure_ascii=False) + "\n")
    return report


def benchmark_file(model_dir: str | Path, input_path: str | Path,
                   batch_size: int = 4096) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    started = time.perf_counter()
    model = load_model(model_dir)
    load_seconds = time.perf_counter() - started
    first = next(iter_title_batches(input_path, batch_size), None)
    if first is None:
        raise ValueError("Benchmark input is empty")
    model.predict([row["product_name"] for row in first])
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    count, characters = 0, 0
    started = time.perf_counter()
    try:
        for batch in iter_title_batches(input_path, batch_size):
            model.predict([row["product_name"] for row in batch])
            count += len(batch)
            characters += sum(len(row["product_name"]) for row in batch)
            peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        seconds = time.perf_counter() - started
        stop.set()
        sampler.join()
    rate = count / seconds
    return {"schema_version": 1, "rows": count, "batch_size": batch_size,
            "seconds": seconds, "rows_per_second": rate,
            "mean_title_characters": characters / count, "model_load_seconds": load_seconds,
            "sampled_peak_rss_mib": peak[0] / 1024**2, "baseline_rss_mib": baseline / 1024**2,
            "estimated_seconds_for_10_million": 10_000_000 / rate,
            "measured_10_million": count == 10_000_000,
            "measurement_scope": "Warm model; includes input parsing, feature extraction and prediction; excludes output serialization.",
            "estimate_limit": "Extrapolated from this input size and title distribution; not an actual ten-million-row run.",
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
                            "total_memory_gib": psutil.virtual_memory().total / 1024**3,
                            "process_threads": process.num_threads(),
                            "thread_environment": {key: os.environ.get(key) for key in
                                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}}
