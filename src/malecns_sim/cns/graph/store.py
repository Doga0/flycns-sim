"""Portable, read-only, memory-mapped COO/CSR storage and downstream queries."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ARRAY_DTYPES = {
    "body_ids": "<u8",
    "src": "<i4",
    "dst": "<i4",
    "contacts": "<u4",
    "csr_indptr": "<i8",
    "csr_indices": "<i4",
    "csr_contacts": "<u4",
}


def file_hashes(path: Path) -> dict:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return {"bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    # No platform newlines, paths, timestamps, or host identifiers in artifacts.
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(repr=False)
class CNSGraph:
    directory: Path
    manifest: dict
    nodes: pa.Table
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, directory: str | Path) -> "CNSGraph":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("graph_schema") != 1:
            raise ValueError("Unsupported graph schema")
        arrays = {
            name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ARRAY_DTYPES
        }
        return cls(directory, manifest, pq.read_table(directory / "nodes.parquet"), arrays)

    @property
    def body_ids(self) -> np.ndarray:
        return self.arrays["body_ids"]

    def node_index(self, body_id: int) -> int:
        if isinstance(body_id, (bool, np.bool_)) or not isinstance(body_id, (int, np.integer)):
            raise TypeError("body_id must be an integer")
        if body_id < 0 or body_id > np.iinfo(np.uint64).max:
            raise KeyError(body_id)
        index = int(np.searchsorted(self.body_ids, np.uint64(body_id)))
        if index == len(self.body_ids) or int(self.body_ids[index]) != body_id:
            raise KeyError(body_id)
        return index

    def outgoing(self, node_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return read-only views, valid while this graph remains open."""
        if isinstance(node_index, (bool, np.bool_)) or not isinstance(
            node_index, (int, np.integer)
        ):
            raise TypeError("node_index must be an integer")
        if not 0 <= node_index < len(self.body_ids):
            raise IndexError(node_index)
        start, end = self.arrays["csr_indptr"][node_index : node_index + 2]
        return self.arrays["csr_indices"][start:end], self.arrays["csr_contacts"][start:end]

    def close(self) -> None:
        # Release mapped files explicitly, particularly before Windows renames/removals.
        for array in self.arrays.values():
            if isinstance(array, np.memmap):
                array._mmap.close()
        self.arrays.clear()

    def node_metadata(self, node_index: int) -> dict:
        if not 0 <= node_index < len(self.body_ids):
            raise IndexError(node_index)

        return self.nodes.slice(node_index, 1).to_pylist()[0]

    @property
    def num_nodes(self) -> int:
        return int(self.manifest["nodes"])

    @property
    def num_edges(self) -> int:
        return int(self.manifest["edges"])

    @property
    def num_contacts(self) -> int:
        return int(self.manifest["contacts"])

    def __enter__(self) -> "CNSGraph":
        return self

    def __exit__(self, *args) -> None:
        self.close()
