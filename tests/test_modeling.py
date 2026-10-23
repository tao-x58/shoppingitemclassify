import hashlib
import json

import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.dataset import CATEGORIES, normalize_title
from src.modeling import classification_metrics, load_model, train_model


def test_train_quarantines_conflicts_and_persists_without_reading_test(tmp_path):
    records = [
        {"product_name": f"品牌{variant} {category} 商品{variant}号250克", "category": category}
        for category in CATEGORIES
        for variant in range(12)
    ]
    conflict_title = "冲突果汁1.5L"
    records.extend(
        {"product_name": conflict_title, "category": category}
        for category in CATEGORIES[:2]
    )
    train_path = tmp_path / "train.jsonl"
    source = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    train_path.write_text(source, encoding="utf-8")
    # Training must not inspect the official test file, including its labels.
    (tmp_path / "test.jsonl").write_text("invalid test data", encoding="utf-8")
    output_dir = tmp_path / "model"

    result = train_model(train_path, output_dir)
    json.dumps(result, allow_nan=False)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "audit_train.json").read_text(encoding="utf-8"))
    training_titles = json.loads(
        (output_dir / "training_titles.json").read_text(encoding="utf-8")
    )
    split = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    clean_records = records[:-2]
    train_titles = {
        normalize_title(clean_records[index]["product_name"])
        for index in split["train_indices"]
    }
    validation_titles = {
        normalize_title(clean_records[index]["product_name"])
        for index in split["validation_indices"]
    }
    assert train_path.read_text(encoding="utf-8") == source
    assert metadata["data_sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert metadata["schema_version"] == 1
    assert metadata["counts"]["excluded_rows"] == audit["excluded_rows"] == 2
    assert metadata["counts"]["final_refit_rows"] == 120
    assert normalize_title(conflict_title) not in training_titles
    assert set(training_titles) == train_titles | validation_titles
    assert not train_titles & validation_titles
    assert len(json.loads((output_dir / "experiments.json").read_text(encoding="utf-8"))) == 3
    assert len(result["validation_metrics"]["confusion_matrix"]) == len(CATEGORIES)
    model = load_model(output_dir)
    sample = [clean_records[index]["product_name"] for index in range(0, 120, 12)]
    predictions = model.predict(sample).tolist()
    assert predictions == list(CATEGORIES)
    assert load_model(output_dir).predict(sample).tolist() == predictions


def test_metrics_expose_missed_minority_despite_high_accuracy():
    majority, minority = CATEGORIES[:2]
    metrics = classification_metrics([majority] * 9 + [minority], [majority] * 10)
    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["per_class"][minority]["recall"] == 0
    assert metrics["per_class"][minority]["support"] == 1
    assert metrics["macro_f1"] < metrics["accuracy"]
    assert list(metrics["per_class"]) == list(CATEGORIES)
    assert metrics["confusion_matrix"][1][0] == 1
    assert sum(map(sum, metrics["confusion_matrix"])) == 10


def test_unconverged_training_fails_explicitly(tmp_path, monkeypatch):
    records = [
        {"product_name": f"{category}款式{index}商品", "category": category}
        for category in CATEGORIES
        for index in range(10)
    ]
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    def unconverged_fit(self, *args, **kwargs):
        import warnings

        warnings.warn("Synthetic nonconvergence", ConvergenceWarning)

    monkeypatch.setattr(Pipeline, "fit", unconverged_fit)
    with pytest.raises(RuntimeError, match="did not converge"):
        train_model(data_path, tmp_path / "model")
    assert not (tmp_path / "model" / "model.joblib").exists()
