"""Read-only storage for the deterministic MaleCNS I/O catalogue artifact."""

import json
from pathlib import Path

import pyarrow.parquet as pq


def load_catalogue_artifacts(directory: str | Path):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("catalogue_schema") != 1:
        raise ValueError("Unsupported I/O catalogue schema")
    return manifest, pq.read_table(directory / "neurons.parquet")
