"""Audit annotation membership and policy differences before building a graph."""

import argparse
import json
from pathlib import Path

import pandas as pd

from malecns_sim.cns.download import DEFAULT_DATA_DIR, FILES
from malecns_sim.cns.graph.policy import exact_ids, load_policy, select_nodes

REFERENCE_URL = (
    "https://research.google/pubs/sexual-dimorphism-in-the-complete-connectome-"
    "of-the-drosophila-male-central-nervous-system/"
)
AUDIT_COLUMNS = [
    "status",
    "statusLabel",
    "superclass",
    "class",
    "subclass",
    "type",
    "somaSide",
    "rootSide",
]


def counts(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    # String labels + JSON null preserve the difference between missing and empty.
    values = frame[columns].astype("string")
    grouped = values.groupby(columns, dropna=False, observed=True, sort=True).size()
    rows = []
    for labels, count in grouped.items():
        labels = labels if isinstance(labels, tuple) else (labels,)
        rows.append(
            {
                **{c: None if pd.isna(v) else str(v) for c, v in zip(columns, labels)},
                "count": int(count),
            }
        )
    return rows


def audit_annotations(frame: pd.DataFrame, policies: list[dict] | None = None) -> dict:
    ids = exact_ids(frame["bodyId"].to_numpy())
    if len(set(ids)) != len(ids):
        raise ValueError("Annotation body IDs are not unique")
    policies = policies or [load_policy(name) for name in ("published", "traced", "all-annotated")]
    assigned = select_nodes(frame, {"column": "superclass", "assigned": True})
    traced = frame.status.eq("Traced").fillna(False).to_numpy(dtype=bool)
    masks = {p["id"]: select_nodes(frame, p["selection"]) for p in policies}
    columns = [c for c in AUDIT_COLUMNS if c in frame]
    return {
        "dataset": "MaleCNS v1.0",
        "reference_neurons": 166691,
        "reference_url": REFERENCE_URL,
        "total_annotations": len(frame),
        "traced_count": int(traced.sum()),
        "glia_count": int(frame.status.eq("Glia").sum()),
        "assigned_superclass_count": int(assigned.sum()),
        "traced_without_superclass": int((traced & ~assigned).sum()),
        "assigned_superclass_not_traced": int((assigned & ~traced).sum()),
        "columns": {c: counts(frame, [c]) for c in columns},
        "combinations": {
            "status_superclass": counts(frame, ["status", "superclass"]),
            "taxonomy_and_side": counts(frame, [c for c in columns if c != "statusLabel"]),
        },
        "side_fields": {
            "side": "Not supplied as a single released column; no inferred side is invented.",
            "available": [c for c in ("side", "somaSide", "rootSide") if c in frame],
        },
        "policies": {
            p["id"]: {
                "policy": p,
                "nodes": int(masks[p["id"]].sum()),
                "difference_from_reference": int(masks[p["id"]].sum()) - p["reference_neurons"],
                "membership_verified": p["reference_membership_verified"],
            }
            for p in policies
        },
        "policy_differences": {
            f"{left}_minus_{right}": int((masks[left] & ~masks[right]).sum())
            for left in masks
            for right in masks
            if left != right
        },
        "interpretation": (
            "Assigned neuronal superclass is a provisional membership proxy, not proof of the "
            "published set. Counts cannot establish body-ID membership equivalence. The release "
            "has uncertain classes and non-Traced neuronal annotations. Any difference from "
            "166691 remains unresolved until an authoritative versioned membership list or "
            "publication-specific selection is available; no nodes are removed to force a match."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("outputs/census.json"))
    args = parser.parse_args()
    report = audit_annotations(pd.read_feather(args.data_dir / FILES["annotations"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"columns", "combinations"}}, indent=2
        )
    )
    for name in ("status", "superclass"):
        print(f"{name} counts: {json.dumps(report['columns'][name])}")
    print(f"Full combinations and labels: {args.output}")


if __name__ == "__main__":
    main()
