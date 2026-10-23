"""Train and persist the character TF-IDF + LinearSVC classifier."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Sequence
import warnings

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.dataset import (
    CATEGORIES,
    load_records,
    normalize_title,
    quarantine_conflicts,
    split_records,
)


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Return metrics with all ten categories in a fixed, disclosed order."""
    y_true, y_pred = list(y_true), list(y_pred)
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Expected equally sized, nonempty true and predicted labels.")
    unknown = (set(y_true) | set(y_pred)) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")
    labels = list(CATEGORIES)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "labels": labels,
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1-score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _pipeline(config: dict, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    preprocessor=normalize_title,
                    sublinear_tf=True,
                    norm="l2",
                    max_features=100_000,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=config["C"],
                    class_weight=config["class_weight"],
                    random_state=seed,
                    max_iter=10_000,
                ),
            ),
        ]
    )


def _fit(model: Pipeline, titles: list[str], labels: list[str]) -> float:
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(titles, labels)
        except ConvergenceWarning as exc:
            raise RuntimeError(
                "LinearSVC did not converge within 10000 iterations. "
                "Training stopped; no unconverged candidate was selected."
            ) from exc
    return time.perf_counter() - started


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train_model(
    data_path: str | Path,
    output_dir: str | Path,
    validation_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Select on training-file validation data, then refit on all clean rows.

    Conflicting normalized titles are excluded, with their original row numbers
    retained in the audit. The official test file is never read here.
    """
    started = time.perf_counter()
    data_path, output_dir = Path(data_path), Path(output_dir)
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    records = load_records(data_path)
    clean_records, audit = quarantine_conflicts(records)
    train_indices, validation_indices = split_records(
        clean_records, validation_size=validation_size, random_state=seed
    )
    train_indices = [int(index) for index in train_indices]
    validation_indices = [int(index) for index in validation_indices]
    train_titles = [clean_records[index]["product_name"] for index in train_indices]
    train_labels = [clean_records[index]["category"] for index in train_indices]
    validation_titles = [
        clean_records[index]["product_name"] for index in validation_indices
    ]
    validation_labels = [
        clean_records[index]["category"] for index in validation_indices
    ]
    candidates = [
        {"C": 1.0, "class_weight": None},
        {"C": 3.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ]
    experiments = []
    best_config, best_metrics, best_predictions = None, None, None
    for config in candidates:
        model = _pipeline(config, seed)
        fit_seconds = _fit(model, train_titles, train_labels)
        predict_started = time.perf_counter()
        predictions = model.predict(validation_titles).tolist()
        predict_seconds = time.perf_counter() - predict_started
        metrics = classification_metrics(validation_labels, predictions)
        experiments.append(
            {
                "config": config,
                "status": "converged",
                "fit_seconds": fit_seconds,
                "validation_predict_seconds": predict_seconds,
                "metrics": metrics,
            }
        )
        # Candidate order is the deterministic tie breaker.
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_config, best_metrics, best_predictions = config, metrics, predictions

    final_model = _pipeline(best_config, seed)
    refit_seconds = _fit(
        final_model,
        [record["product_name"] for record in clean_records],
        [record["category"] for record in clean_records],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(final_model, model_path)
    counts = {
        "source_rows": len(records),
        "excluded_rows": len(records) - len(clean_records),
        "clean_rows": len(clean_records),
        "selection_train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "final_refit_rows": len(clean_records),
        "clean_class_counts": dict(
            Counter(record["category"] for record in clean_records)
        ),
    }
    metadata = {
        "schema_version": 1,
        "model_type": "character_tfidf_linear_svc",
        "source_file": str(data_path.resolve()),
        "data_sha256": data_hash,
        "seed": seed,
        "validation_size": validation_size,
        "selection_metric": "macro_f1",
        "tie_breaker": "first candidate in experiments.json",
        "labels": list(CATEGORIES),
        "best_config": best_config,
        "vectorizer_config": {
            "analyzer": "char",
            "ngram_range": [2, 4],
            "preprocessor": "src.dataset.normalize_title",
            "sublinear_tf": True,
            "norm": "l2",
            "max_features": 100_000,
            "dtype": "float32",
        },
        "classifier_config": {
            **best_config,
            "random_state": seed,
            "max_iter": 10_000,
        },
        "counts": counts,
        "validation_metrics": best_metrics,
        "final_model_training": "Refit on every clean training-file row after selection.",
        "final_refit_seconds": refit_seconds,
        "model_size_bytes": model_path.stat().st_size,
        "vocabulary_size": len(final_model.named_steps["tfidf"].vocabulary_),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "total_train_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "audit_train.json", audit)
    _write_json(output_dir / "experiments.json", experiments)
    _write_json(output_dir / "validation_metrics.json", best_metrics)
    _write_json(
        output_dir / "split_manifest.json",
        {
            "schema_version": 1,
            "index_basis": "Zero-based indices into clean rows after quarantine, in original order.",
            "purpose": "Model selection only; the saved model is refit on all clean rows.",
            "data_sha256": data_hash,
            "seed": seed,
            "validation_size": validation_size,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
        },
    )
    _write_json(
        output_dir / "training_titles.json",
        sorted({normalize_title(record["product_name"]) for record in clean_records}),
    )
    with (output_dir / "validation_errors.jsonl").open("w", encoding="utf-8") as stream:
        for index, truth, prediction in zip(
            validation_indices, validation_labels, best_predictions, strict=True
        ):
            if truth != prediction:
                stream.write(
                    json.dumps(
                        {
                            "clean_index": index,
                            "product_name": clean_records[index]["product_name"],
                            "category": truth,
                            "prediction": prediction,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    _write_json(output_dir / "metadata.json", metadata)
    return {
        "schema_version": 1,
        "model_dir": str(output_dir.resolve()),
        "best_config": best_config,
        "validation_metrics": best_metrics,
        "counts": counts,
        "model_size_bytes": metadata["model_size_bytes"],
        "total_train_seconds": metadata["total_train_seconds"],
    }


def load_model(model_dir: str | Path) -> Pipeline:
    """Load a locally produced model directory; joblib files must be trusted."""
    model = joblib.load(Path(model_dir) / "model.joblib")
    if not isinstance(model, Pipeline):
        raise ValueError("model.joblib does not contain an sklearn Pipeline.")
    return model
