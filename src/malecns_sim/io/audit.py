"""Audit the released annotation schema and categorical values without inferring roles."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_sim.cns.download import DEFAULT_DATA_DIR, FILES
from malecns_sim.cns.graph.store import write_json

CLASSIFICATION_FIELDS = (
    "status",
    "statusLabel",
    "superclass",
    "class",
    "subclass",
    "type",
    "supertype",
    "somaSide",
    "rootSide",
    "somaNeuromere",
    "entryNerve",
    "exitNerve",
    "receptorType",
    "mancType",
    "flywireType",
    "hemibrainType",
    "itoleeHl",
    "trumanHl",
    "serialMotif",
    "dimorphism",
    "fruDsx",
)


def _json_value(value):
    return None if pd.isna(value) else str(value)


def _distinct_count(series: pd.Series) -> int:
    if series.dtype != object:
        return int(series.nunique(dropna=False))
    hashable = series.map(
        lambda value: tuple(value.tolist()) if isinstance(value, np.ndarray) else value
    )
    return int(hashable.nunique(dropna=False))


def audit_annotations(path: str | Path) -> dict:
    frame = pd.read_feather(path)
    columns = [
        {
            "name": name,
            "dtype": str(frame[name].dtype),
            "nulls": int(frame[name].isna().sum()),
            "distinct_including_null": _distinct_count(frame[name]),
        }
        for name in frame.columns
    ]
    classifications = {}
    for field in CLASSIFICATION_FIELDS:
        if field not in frame:
            continue
        counts = frame[field].value_counts(dropna=False, sort=True)
        rows = [
            {"value": _json_value(value), "count": int(count)} for value, count in counts.items()
        ]
        rows.sort(key=lambda row: (-row["count"], row["value"] is not None, row["value"] or ""))
        classifications[field] = rows
    return {
        "dataset": "MaleCNS v1.0",
        "rows": len(frame),
        "available_annotation_columns": list(frame.columns),
        "column_schema": columns,
        "classification_fields": classifications,
        "interpretation": (
            "Values are reported exactly as released (JSON null represents missing data). "
            "This audit does not assign normalized roles or infer anatomy from label text."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("outputs/io-annotation-audit.json"))
    args = parser.parse_args()
    report = audit_annotations(args.data_dir / FILES["annotations"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print("Available annotation columns\n" + "-" * 32)
    print("\n".join(report["available_annotation_columns"]))
    print("\nClassification fields\n" + "-" * 32)
    for field, rows in report["classification_fields"].items():
        preview = ", ".join(f"{row['value']!r}: {row['count']}" for row in rows[:12])
        print(f"{field} ({len(rows)} distinct): {preview}")
    print(f"\nFull distinct-value audit: {args.output}")


if __name__ == "__main__":
    main()
