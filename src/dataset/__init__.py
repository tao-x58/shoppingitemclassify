"""Local data loading, auditing, and leakage-aware data splitting."""

import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


CATEGORIES = (
    "即饮奶茶",
    "即饮茶",
    "硬糖",
    "牙膏",
    "非冷藏即饮果汁",
    "冷藏即饮果汁",
    "白酒",
    "方便面",
    "软糖",
    "口香糖",
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def normalize_title(title: str) -> str:
    """Normalize width and case, removing whitespace but retaining punctuation."""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return "".join(character for character in normalized if not character.isspace())


def load_records(path: str | Path) -> list[dict]:
    """Read validated JSONL without altering the source file or record contents."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{location}: each record must be a JSON object")
            title = record.get("product_name")
            if not isinstance(title, str) or not normalize_title(title):
                raise ValueError(f"{location}: product_name must be a non-empty string")
            category = record.get("category")
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"{location}: category must be a non-empty string")
            if category not in CATEGORIES:
                raise ValueError(f"{location}: unknown category {category!r}")
            records.append(record)
    return records


def _title_groups(records: list[dict]) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[normalize_title(record["product_name"])].append(index)
    return dict(groups)


def _percentile(values: list[int], fraction: float) -> float:
    """Calculate a percentile by linear interpolation between ordered values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def audit_records(records: list[dict]) -> dict:
    """Describe records; row numbers are 1-based positions in ``records``.

    Duplicate counts include every extra occurrence of a normalized title,
    including conflicting labels. Lengths count characters in original titles.
    """
    counts = Counter(record["category"] for record in records)
    lengths = [len(record["product_name"]) for record in records]
    groups = _title_groups(records)
    conflicts = []
    for normalized_title, indices in groups.items():
        categories = sorted({records[index]["category"] for index in indices})
        if len(categories) > 1:
            conflicts.append(
                {
                    "normalized_title": normalized_title,
                    "rows": [index + 1 for index in indices],
                    "categories": categories,
                    "records": [
                        {
                            "row": index + 1,
                            "product_name": records[index]["product_name"],
                            "category": records[index]["category"],
                        }
                        for index in indices
                    ],
                }
            )
    return {
        "rows": len(records),
        "class_counts": {category: counts[category] for category in CATEGORIES},
        "lengths": {
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "mean": mean(lengths) if lengths else 0.0,
            "median": median(lengths) if lengths else 0.0,
            "p95": _percentile(lengths, 0.95),
        },
        "duplicates": sum(len(indices) - 1 for indices in groups.values()),
        "conflicts": conflicts,
    }


def quarantine_conflicts(records: list[dict]) -> tuple[list[dict], dict]:
    """Exclude all rows of conflicting normalized titles, leaving sources intact."""
    audit = audit_records(records)
    excluded_indices = {
        row - 1 for conflict in audit["conflicts"] for row in conflict["rows"]
    }
    audit["excluded_rows"] = len(excluded_indices)
    audit["excluded_records"] = [
        {
            "row": index + 1,
            "product_name": record["product_name"],
            "category": record["category"],
            "normalized_title": normalize_title(record["product_name"]),
        }
        for index, record in enumerate(records)
        if index in excluded_indices
    ]
    clean_records = [
        record for index, record in enumerate(records) if index not in excluded_indices
    ]
    return clean_records, audit


def split_records(
    records: list[dict], validation_size: float = 0.2, random_state: int = 42
) -> tuple[list[int], list[int]]:
    """Stratify unique normalized titles, then map groups to source indices.

    The validation fraction applies to title groups, so the final fraction of
    rows can differ when groups have unequal sizes. Returned indices are sorted.
    """
    if isinstance(validation_size, bool) or not 0 < validation_size < 1:
        raise ValueError("validation_size must be a fraction strictly between 0 and 1")
    if not records:
        raise ValueError("cannot split an empty dataset")
    groups = _title_groups(records)
    titles = list(groups)
    labels = []
    for title, indices in groups.items():
        categories = {records[index]["category"] for index in indices}
        if len(categories) != 1:
            raise ValueError(
                f"conflicting labels for normalized title {title!r}; "
                "quarantine conflicts before splitting"
            )
        labels.append(next(iter(categories)))
    group_counts = Counter(labels)
    insufficient = {category: count for category, count in group_counts.items() if count < 2}
    if insufficient:
        raise ValueError(
            "each category needs at least 2 unique normalized titles for a stratified "
            f"split; insufficient category counts: {insufficient}"
        )

    from sklearn.model_selection import train_test_split

    try:
        train_titles, validation_titles = train_test_split(
            titles,
            test_size=validation_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "cannot create a stratified title-group split with "
            f"validation_size={validation_size} and category group counts "
            f"{dict(group_counts)}: {error}"
        ) from error
    train_indices = sorted(index for title in train_titles for index in groups[title])
    validation_indices = sorted(
        index for title in validation_titles for index in groups[title]
    )
    return train_indices, validation_indices


class ProductClassificationDataset:
    """Read local JSONL first, downloading through ModelScope only if absent."""

    def __init__(self, split="train"):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            local_path = DATA_DIR / f"{self.split}.jsonl"
            if local_path.is_file():
                self._dataset = load_records(local_path)
            else:
                from modelscope.msdatasets import MsDataset

                self._dataset = MsDataset.load(
                    "winwin_inc/product-classification-hiring-demo", split=self.split
                )
        return self._dataset

    def __iter__(self):
        return iter(self.dataset)

    def dataframe(self):
        if isinstance(self.dataset, list):
            import pandas as pd

            return pd.DataFrame.from_records(
                self.dataset, columns=["product_name", "category"]
            )
        return self.dataset.to_pandas()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
