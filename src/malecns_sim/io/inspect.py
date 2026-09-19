"""Print a compact summary of a built MaleCNS I/O catalogue."""

import argparse
import json
from pathlib import Path

from malecns_sim.io.store import load_catalogue_artifacts

DEFAULT_CATALOGUE = Path("data/processed/malecns-v1.0/published-v1/io-v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    args = parser.parse_args()
    try:
        manifest, _ = load_catalogue_artifacts(args.catalogue)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"I/O catalogue inspect failed: {error}") from None
    counts = manifest["role_counts"]
    print("MaleCNS I/O catalogue\n")
    print(f"Total graph neurons:       {counts['total_graph_neurons']}")
    for role in (
        "sensory",
        "motor",
        "descending",
        "ascending",
        "interneuron",
        "endocrine",
        "other",
    ):
        print(f"{role.capitalize():<27}{counts['roles'][role]}")
    print(f"Mapped role rules:         {counts['mapped_role_rules']}")
    print(f"Unknown role rules:        {counts['unknown_role_rules']}")
    print("\nSuperclass to role source counts:")
    print(json.dumps(counts["source_values"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
