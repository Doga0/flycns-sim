"""Query exact roles and released annotation values from the I/O catalogue."""

import argparse
import json
from pathlib import Path

from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.inspect import DEFAULT_CATALOGUE
from malecns_sim.io.roles import NeuronRole

DISPLAY_FIELDS = (
    "node_index",
    "body_id",
    "role",
    "superclass",
    "class",
    "subclass",
    "type",
    "side",
    "nerve",
    "body_region",
    "sensory_system",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--role", choices=[role.value for role in NeuronRole])
    parser.add_argument("--type")
    parser.add_argument("--side")
    parser.add_argument("--class", dest="cell_class")
    parser.add_argument("--subclass")
    parser.add_argument("--sensory-system")
    parser.add_argument("--nerve")
    parser.add_argument("--body-region")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    criteria = {
        key: value
        for key, value in {
            "role": args.role,
            "type": args.type,
            "side": args.side,
            "class": args.cell_class,
            "subclass": args.subclass,
            "sensory_system": args.sensory_system,
            "nerve": args.nerve,
            "body_region": args.body_region,
        }.items()
        if value is not None
    }
    try:
        result = IOCatalog.load(args.catalogue).filter(**criteria)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"I/O catalogue query failed: {error}") from None
    rows = result.table.select(DISPLAY_FIELDS).slice(0, args.limit).to_pylist()
    report = {"criteria": criteria, "matches": len(result), "shown": len(rows), "neurons": rows}
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
