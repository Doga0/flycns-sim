"""Print raw table counts, schemas, status labels and transmitter labels."""

import argparse
from pathlib import Path

from malecns_sim.cns.dataset import CNSDataset
from malecns_sim.cns.download import DEFAULT_DATA_DIR, FILES


def describe(dataset: CNSDataset) -> None:
    for name in FILES:
        table = getattr(dataset, name)
        print(f"\n{name}: {len(table):,} rows")
        print("Columns: " + ", ".join(f"{col} ({dtype})" for col, dtype in table.dtypes.items()))
        for column in table.columns:
            label = column.lower()
            if "status" in label or label.endswith("_nt") or label == "ground_truth":
                print(f"{column} values (including missing):")
                print(table[column].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    try:
        describe(CNSDataset.load(args.data_dir))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Dataset inspection failed: {error}\n")


if __name__ == "__main__":
    main()
