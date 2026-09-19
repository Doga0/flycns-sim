"""Fast exact-match queries over normalized roles and preserved MaleCNS metadata."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from malecns_sim.io.roles import NeuronRole
from malecns_sim.io.store import load_catalogue_artifacts


@dataclass(frozen=True)
class IOQueryResult:
    table: pa.Table

    def __len__(self) -> int:
        return self.table.num_rows

    @property
    def node_indices(self) -> np.ndarray:
        return self.table["node_index"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)

    @property
    def body_ids(self) -> np.ndarray:
        return self.table["body_id"].to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)

    def to_pylist(self) -> list[dict]:
        return self.table.to_pylist()


class IOCatalog:
    def __init__(self, directory: Path, manifest: dict, neurons: pa.Table):
        self.directory = directory
        self.manifest = manifest
        self.neurons = neurons

    @classmethod
    def load(cls, directory: str | Path) -> "IOCatalog":
        directory = Path(directory)
        manifest, neurons = load_catalogue_artifacts(directory)
        return cls(directory, manifest, neurons)

    def filter(self, **criteria) -> IOQueryResult:
        unknown = set(criteria) - set(self.neurons.column_names)
        if unknown:
            raise KeyError(f"Unknown catalogue fields: {sorted(unknown)}")
        mask = pa.array([True] * self.neurons.num_rows)
        for field, value in criteria.items():
            if isinstance(value, NeuronRole):
                value = value.value
            column = self.neurons[field]
            condition = pc.is_null(column) if value is None else pc.equal(column, value)
            mask = pc.and_(mask, condition)
        return IOQueryResult(self.neurons.filter(mask))

    def sensory(self, **criteria) -> IOQueryResult:
        return self.filter(role=NeuronRole.SENSORY, **criteria)

    def motor(self, **criteria) -> IOQueryResult:
        return self.filter(role=NeuronRole.MOTOR, **criteria)

    def descending(self, **criteria) -> IOQueryResult:
        return self.filter(role=NeuronRole.DESCENDING, **criteria)

    def ascending(self, **criteria) -> IOQueryResult:
        return self.filter(role=NeuronRole.ASCENDING, **criteria)

    def by_type(self, cell_type: str) -> IOQueryResult:
        return self.filter(type=cell_type)

    def by_side(self, side: str) -> IOQueryResult:
        return self.filter(side=side)
