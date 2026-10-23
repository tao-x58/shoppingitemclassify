import csv
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.dataset import normalize_title
from src.inference import benchmark_file, evaluate_model, iter_title_batches, predict_file


@pytest.fixture
def saved_model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                                         preprocessor=normalize_title), LinearSVC())
    model.fit(["草莓软糖", "橙子软糖", "薄荷牙膏", "儿童牙膏"], ["软糖", "软糖", "牙膏", "牙膏"])
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (model_dir / "training_titles.json").write_text(json.dumps(["草莓软糖", "薄荷牙膏"]), encoding="utf-8")
    return model_dir


def write_rows(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_prediction_order_and_labels_are_independent_of_batch_size(saved_model, tmp_path):
    names = ["薄荷牙膏", "草莓软糖", "儿童牙膏", "橙子软糖", "薄荷牙膏"]
    input_path = tmp_path / "products.jsonl"
    write_rows(input_path, [{"product_name": name} for name in names])
    outputs = []
    for size in (1, 3, 100):
        path = tmp_path / f"predictions-{size}.jsonl"
        summary = predict_file(saved_model, input_path, path, size)
        assert summary["rows"] == len(names)
        outputs.append([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])
    assert outputs[0] == outputs[1] == outputs[2]
    assert [row["row_id"] for row in outputs[0]] == list(range(1, 6))
    assert [row["product_name"] for row in outputs[0]] == names
    assert [row["predicted_category"] for row in outputs[0]] == ["牙膏", "软糖", "牙膏", "软糖", "牙膏"]


def test_csv_quotes_bom_and_csv_output(saved_model, tmp_path):
    input_path = tmp_path / "products.csv"
    with input_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["product_name"])
        writer.writeheader()
        writer.writerow({"product_name": '薄荷牙膏,"清新"'})
        writer.writerow({"product_name": "草莓软糖"})
    output_path = tmp_path / "predictions.csv"
    predict_file(saved_model, input_path, output_path, 1)
    with output_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["product_name"] == '薄荷牙膏,"清新"'
    assert [row["predicted_category"] for row in rows] == ["牙膏", "软糖"]


def test_invalid_late_input_does_not_replace_existing_output(saved_model, tmp_path):
    input_path = tmp_path / "bad.jsonl"
    write_rows(input_path, [{"product_name": "牙膏"}, {"product_name": "  "}])
    output_path = tmp_path / "existing.jsonl"
    output_path.write_text("existing output", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        predict_file(saved_model, input_path, output_path, 1)
    assert output_path.read_text(encoding="utf-8") == "existing output"
    assert not list(tmp_path.glob("*.tmp"))


def test_evaluation_discloses_seen_and_unseen_subsets(saved_model, tmp_path):
    test_path = tmp_path / "test.jsonl"
    write_rows(test_path, [{"product_name": "草莓软糖", "category": "软糖"},
                           {"product_name": "儿童牙膏", "category": "牙膏"}])
    report = evaluate_model(saved_model, test_path, batch_size=1)
    assert report["rows"] == 2
    assert report["overlap_rows"] == report["overlap_unique_titles"] == 1
    assert report["metrics"]["accuracy"] == 1
    assert report["unseen_title_metrics"]["per_class"]["牙膏"]["support"] == 1
    assert (saved_model / "evaluation" / "test_errors.jsonl").read_text(encoding="utf-8") == ""


def test_benchmark_measures_actual_rows_and_labels_estimate(saved_model, tmp_path):
    input_path = tmp_path / "input.jsonl"
    write_rows(input_path, [{"product_name": "牙膏"}, {"product_name": "软糖"}])
    report = benchmark_file(saved_model, input_path, 1)
    assert report["rows"] == 2
    assert report["rows_per_second"] > 0
    assert report["sampled_peak_rss_mib"] >= report["baseline_rss_mib"] > 0
    assert report["measured_10_million"] is False
    assert report["estimated_seconds_for_10_million"] == pytest.approx(10_000_000 / report["rows_per_second"])


def test_streaming_batches_and_input_protection(saved_model, tmp_path):
    input_path = tmp_path / "products.jsonl"
    write_rows(input_path, [{"product_name": "牙膏"}] * 7)
    assert [len(batch) for batch in iter_title_batches(input_path, 3)] == [3, 3, 1]
    with pytest.raises(ValueError, match="different files"):
        predict_file(saved_model, input_path, input_path)
    with pytest.raises(ValueError, match="positive"):
        list(iter_title_batches(input_path, 0))


def test_malformed_csv_does_not_silently_merge_records(saved_model, tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text('product_name\n"toothpaste\nsoft-candy\n', encoding="utf-8")
    target = tmp_path / "output.jsonl"
    target.write_text("previous result", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid CSV"):
        predict_file(saved_model, source, target, 1)
    assert target.read_text(encoding="utf-8") == "previous result"


@pytest.mark.parametrize("command", ["audit", "benchmark"])
def test_cli_rejects_raw_input_as_report_output(tmp_path, command):
    source = tmp_path / "train.jsonl"
    write_rows(source, [{"product_name": "薄荷牙膏", "category": "牙膏"}])
    original = source.read_bytes()
    arguments = ["--data-dir", str(tmp_path)] if command == "audit" else ["--input", str(source)]
    result = subprocess.run([sys.executable, "-m", "src.main", command, *arguments,
                             "--output", str(source)], capture_output=True)
    assert result.returncode == 2
    assert b"overwrite input data" in result.stderr
    assert source.read_bytes() == original
