"""Audit only released sensory annotations; do not infer receptor tuning."""

import argparse
from pathlib import Path

from malecns_sim.cns.graph.store import write_json
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.inspect import DEFAULT_CATALOGUE

FIELDS = ("class", "subclass", "type", "receptorType", "entryNerve", "side", "somaNeuromere")


def _coverage(frame, fields=FIELDS):
    return {
        field: {
            "present": field in frame,
            "nonempty": int(frame[field].notna().sum()) if field in frame else 0,
            "missing": int(frame[field].isna().sum()) if field in frame else len(frame),
            "values": (
                {
                    str(key): int(value)
                    for key, value in frame[field]
                    .fillna("<null>")
                    .value_counts()
                    .sort_index()
                    .items()
                }
                if field in frame
                else {}
            ),
        }
        for field in fields
    }


def audit_sensory(catalogue: IOCatalog) -> dict:
    sensory = catalogue.sensory().table.to_pandas()
    lf = sensory[(sensory["side"] == "L") & (sensory["entryNerve"] == "ProLN")]
    proprio = lf[lf["class"] == "mechanosensory_proprioceptive"]
    groups = {}
    for name, mask in {
        "chordotonal": proprio["subclass"] == "chordotonal organ",
        "hair_plate": proprio["subclass"] == "hair plate",
        "campaniform": proprio["subclass"] == "campaniform sensilla",
        "other_proprioceptive": ~proprio["subclass"].isin(
            ["chordotonal organ", "hair plate", "campaniform sensilla"]
        ),
        "unknown": lf["class"] == "unknown_sensory",
    }.items():
        selected = proprio[mask] if name != "unknown" else lf[mask]
        groups[name] = {
            "neurons": len(selected),
            "body_ids": [int(value) for value in selected.body_id],
            "type_counts": {
                str(k): int(v) for k, v in selected.type.fillna("<null>").value_counts().items()
            },
        }
    return {
        "schema": 1,
        "total_sensory_neurons": len(sensory),
        "lf_selector": {"side": "L", "entryNerve": "ProLN"},
        "lf_neurons": len(lf),
        "lf_proprioceptive_neurons": len(proprio),
        "coverage": _coverage(sensory),
        "lf_proprioceptive_coverage": _coverage(proprio),
        "lf_groups": groups,
        "interpretation": (
            "L + ProLN is an explicit released laterality/entry-nerve selection. "
            "The released table does not label these cells claw, hook or club; "
            "no feature tuning is inferred by this audit."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/bridge/sensory-catalogue-audit.json")
    )
    args = parser.parse_args()
    report = audit_sensory(IOCatalog.load(args.catalogue))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print("MaleCNS LF sensory catalogue\n" + "-" * 32)
    print(f"LF (L + ProLN): {report['lf_neurons']}")
    print(f"LF proprioceptive: {report['lf_proprioceptive_neurons']}")
    for name, group in report["lf_groups"].items():
        print(f"{name}: {group['neurons']}")
    print(f"Audit: {args.output}")


if __name__ == "__main__":
    main()
