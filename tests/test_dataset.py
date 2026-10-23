"""Offline dataset tests: every fixture is written under pytest's tmp_path."""

import copy
import json
from collections import Counter

import pytest

import src.dataset as dataset_module
from src.dataset import (
    CATEGORIES,
    ProductClassificationDataset,
    audit_records,
    load_records,
    normalize_title,
    quarantine_conflicts,
    split_records,
)


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_categories_follow_readme_order():
    assert CATEGORIES == (
        "即饮奶茶", "即饮茶", "硬糖", "牙膏", "非冷藏即饮果汁",
        "冷藏即饮果汁", "白酒", "方便面", "软糖", "口香糖",
    )


def test_normalize_title_preserves_punctuation():
    assert normalize_title(" ＡＢＣ　牙膏\t100ＭＬ\n（薄荷）! ") == "abc牙膏100ml(薄荷)!"


def test_load_records_handles_bom_blank_lines_and_preserves_values(tmp_path):
    path = tmp_path / "input.jsonl"
    record = {"product_name": " Ａ 牙膏 ", "category": "牙膏", "extra": 7}
    path.write_text(
        "\n" + json.dumps(record, ensure_ascii=False) + "\n \t\n",
        encoding="utf-8-sig",
    )
    original = path.read_bytes()
    assert load_records(path) == [record]
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("bad_line", "message"),
    [
        ("{broken", "invalid JSON"),
        ("[]", "JSON object"),
        ('{"category": "牙膏"}', "product_name"),
        ('{"product_name": 12, "category": "牙膏"}', "product_name"),
        ('{"product_name": "", "category": "牙膏"}', "product_name"),
        ('{"product_name": "　 ", "category": "牙膏"}', "product_name"),
        ('{"product_name": "牙膏", "category": null}', "category"),
        ('{"product_name": "牙膏", "category": " "}', "category"),
        ('{"product_name": "牙膏", "category": "其他"}', "unknown category"),
    ],
)
def test_load_records_rejects_invalid_rows_with_path_and_line(tmp_path, bad_line, message):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n\n" + bad_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message) as error:
        load_records(path)
    assert f"{path}:3:" in str(error.value)


def test_audit_and_quarantine_keep_consistent_duplicates_and_sources(tmp_path):
    records = [
        {"product_name": " Ａ果汁 ", "category": "冷藏即饮果汁"},
        {"product_name": "薄荷牙膏", "category": "牙膏"},
        {"product_name": "a 果汁", "category": "非冷藏即饮果汁"},
        {"product_name": "薄荷 牙膏", "category": "牙膏"},
    ]
    path = tmp_path / "conflicts.jsonl"
    write_jsonl(path, records)
    source_bytes = path.read_bytes()
    original_records = copy.deepcopy(records)

    clean, audit = quarantine_conflicts(records)

    assert clean == [records[1], records[3]]
    assert records == original_records
    assert path.read_bytes() == source_bytes
    assert audit["rows"] == 4
    assert audit["class_counts"]["牙膏"] == 2
    assert audit["class_counts"]["白酒"] == 0
    assert audit["duplicates"] == 2
    assert audit["lengths"]["max"] == max(len(row["product_name"]) for row in records)
    assert audit["conflicts"][0]["normalized_title"] == "a果汁"
    assert audit["conflicts"][0]["rows"] == [1, 3]
    assert set(audit["conflicts"][0]["categories"]) == {"冷藏即饮果汁", "非冷藏即饮果汁"}
    assert audit["excluded_rows"] == 2
    assert [record["row"] for record in audit["excluded_records"]] == [1, 3]
    assert all("normalized_title" in record for record in audit["excluded_records"])


def test_empty_audit_and_quarantine():
    clean, audit = quarantine_conflicts([])
    assert clean == []
    assert audit["rows"] == audit["duplicates"] == audit["excluded_rows"] == 0
    assert audit["conflicts"] == audit["excluded_records"] == []
    assert audit["lengths"] == {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}


def test_audit_length_statistics():
    records = [
        {"product_name": "a" * length, "category": "牙膏"}
        for length in [1, 2, 3, 4, 5]
    ]
    assert audit_records(records)["lengths"] == {
        "min": 1, "max": 5, "mean": 3, "median": 3, "p95": 4.8,
    }


def test_split_groups_normalized_titles_without_leakage():
    records = [
        {"product_name": f"{category}{index}Ａ", "category": category}
        for category in ("牙膏", "白酒")
        for index in range(10)
    ]
    records += [
        {"product_name": "牙膏 0a", "category": "牙膏"},
        {"product_name": "白酒 0a", "category": "白酒"},
    ]
    original = copy.deepcopy(records)
    train_indices, validation_indices = split_records(records)
    assert split_records(records) == (train_indices, validation_indices)
    assert records == original
    assert sorted(train_indices + validation_indices) == list(range(len(records)))
    assert set(train_indices).isdisjoint(validation_indices)
    train_titles = {normalize_title(records[index]["product_name"]) for index in train_indices}
    validation_titles = {
        normalize_title(records[index]["product_name"]) for index in validation_indices
    }
    assert train_titles.isdisjoint(validation_titles)
    group_labels = {
        normalize_title(record["product_name"]): record["category"] for record in records
    }
    assert Counter(group_labels[title] for title in validation_titles) == {"牙膏": 2, "白酒": 2}
    assert Counter(group_labels[title] for title in train_titles) == {"牙膏": 8, "白酒": 8}


def test_split_rejects_unquarantined_conflicts():
    records = [
        {"product_name": "果汁", "category": "冷藏即饮果汁"},
        {"product_name": "果 汁", "category": "非冷藏即饮果汁"},
    ]
    with pytest.raises(ValueError, match="quarantine conflicts"):
        split_records(records)


def test_split_reports_too_few_unique_titles():
    records = [
        {"product_name": "牙膏", "category": "牙膏"},
        {"product_name": "牙 膏", "category": "牙膏"},
        {"product_name": "白酒1", "category": "白酒"},
        {"product_name": "白酒2", "category": "白酒"},
    ]
    with pytest.raises(ValueError, match="at least 2 unique normalized titles.*牙膏"):
        split_records(records)


def test_split_reports_validation_fraction_constraints():
    records = [
        {"product_name": f"{category}{index}", "category": category}
        for category in ("牙膏", "白酒")
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="cannot create a stratified title-group split"):
        split_records(records, validation_size=0.2)


@pytest.mark.parametrize("validation_size", [0, 1, -0.1, 1.1, True])
def test_split_rejects_invalid_fraction(validation_size):
    with pytest.raises(ValueError, match="validation_size"):
        split_records([], validation_size=validation_size)


def test_split_rejects_empty_data():
    with pytest.raises(ValueError, match="empty dataset"):
        split_records([])


@pytest.mark.parametrize("split", ["train", "test"])
def test_dataset_local_interfaces(tmp_path, monkeypatch, split):
    records = [
        {"product_name": "薄荷牙膏", "category": "牙膏"},
        {"product_name": "茉莉花茶", "category": "即饮茶"},
    ]
    write_jsonl(tmp_path / f"{split}.jsonl", records)
    monkeypatch.setattr(dataset_module, "DATA_DIR", tmp_path)
    dataset = ProductClassificationDataset(split=split)
    assert dataset._dataset is None
    assert len(dataset) == 2
    assert list(dataset) == records
    assert dataset[0] == records[0]
    assert dataset.dataset == records
    assert dataset.dataframe().to_dict(orient="records") == records
    assert dataset.dataset is dataset._dataset


@pytest.mark.parametrize("split", [None, "", "validation", "TRAIN"])
def test_dataset_rejects_invalid_split(split):
    with pytest.raises(ValueError, match="split"):
        ProductClassificationDataset(split=split)
