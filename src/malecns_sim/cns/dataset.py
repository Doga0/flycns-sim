"""Read the released tables unchanged. No joins, thresholds, or neuron selection."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from malecns_sim.cns.download import DEFAULT_DATA_DIR, FILES

REQUIRED_COLUMNS = {
    "annotations": {"bodyId", "status", "statusLabel"},
    "neurotransmitters": {"body", "predicted_nt", "celltype_predicted_nt", "consensus_nt"},
    "connections": {"body_pre", "body_post", "weight"},
}


@dataclass
class CNSDataset:
    annotations: pd.DataFrame
    neurotransmitters: pd.DataFrame
    connections: pd.DataFrame

    @classmethod
    def load(cls, directory: Path = DEFAULT_DATA_DIR) -> "CNSDataset":
        directory = Path(directory)
        missing = [filename for filename in FILES.values() if not (directory / filename).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing MaleCNS files in {directory}: {', '.join(missing)}. "
                "Run python -m malecns_sim.cns.download first."
            )
        tables = {}
        for name, filename in FILES.items():
            try:
                table = pd.read_feather(directory / filename)
            except Exception as error:
                raise ValueError(f"Cannot read Feather table {filename}: {error}") from error
            if table.empty or not table.columns.is_unique:
                raise ValueError(f"Empty table or duplicate columns: {filename}")
            absent = REQUIRED_COLUMNS[name] - set(table.columns)
            if absent:
                raise ValueError(f"Missing columns in {filename}: {', '.join(sorted(absent))}")
            tables[name] = table
        return cls(**tables)
