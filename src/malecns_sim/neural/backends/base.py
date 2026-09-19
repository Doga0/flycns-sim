"""Backend-independent graph view and neural engine contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from malecns_sim.cns.graph.store import CNSGraph, file_hashes
from malecns_sim.neural.stimulus import Stimulus


@dataclass(repr=False)
class NeuralGraph:
    body_ids: np.ndarray
    node_indices: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    contacts: np.ndarray
    neurotransmitters: list
    provenance: dict

    @classmethod
    def from_graph(cls, graph: CNSGraph) -> "NeuralGraph":
        return cls(
            graph.body_ids.copy(),
            np.arange(len(graph.body_ids), dtype=np.int32),
            graph.arrays["src"],
            graph.arrays["dst"],
            graph.arrays["contacts"],
            graph.nodes["consensus_nt"].to_pylist(),
            {
                "graph_manifest_sha256": file_hashes(graph.directory / "manifest.json")["sha256"],
                "graph_manifest": graph.manifest,
                "selection": {"type": "full"},
            },
        )

    def validate(self) -> None:
        n = len(self.body_ids)
        if any(
            array.ndim != 1
            for array in (self.body_ids, self.node_indices, self.src, self.dst, self.contacts)
        ):
            raise ValueError("Graph arrays must be one-dimensional")
        if n == 0 or self.body_ids.dtype != np.uint64 or len(np.unique(self.body_ids)) != n:
            raise ValueError("Neural graph needs unique uint64 body IDs")
        if (
            len(self.node_indices) != n
            or self.node_indices.dtype.kind not in "iu"
            or len(np.unique(self.node_indices)) != n
            or np.any(self.node_indices < 0)
            or np.any(self.node_indices > 2**31 - 1)
        ):
            raise ValueError("Neural graph needs unique parent node indices")
        if len(self.neurotransmitters) != n or not (
            len(self.src) == len(self.dst) == len(self.contacts)
        ):
            raise ValueError("Graph array lengths disagree")
        for start in range(0, len(self.src), 1_000_000):
            src, dst, contacts = (
                x[start : start + 1_000_000] for x in (self.src, self.dst, self.contacts)
            )
            if (
                src.dtype.kind not in "iu"
                or dst.dtype.kind not in "iu"
                or np.any(src < 0)
                or np.any(dst < 0)
                or np.any(src >= n)
                or np.any(dst >= n)
            ):
                raise ValueError("Unknown neural graph endpoint")
            if (
                contacts.dtype.kind not in "iu"
                or np.any(contacts <= 0)
                or np.any(contacts > 2**32 - 1)
            ):
                raise ValueError("Contacts must be positive uint32-compatible counts")


class NeuralBackend(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> None: ...

    @abstractmethod
    def schedule(self, stimulus: Stimulus) -> None: ...

    @abstractmethod
    def stimulate(self, node_indices, rate_hz: float) -> None: ...

    @abstractmethod
    def silence(self, node_indices) -> None: ...

    @abstractmethod
    def run(self, duration_ms: float) -> None: ...

    @abstractmethod
    def get_spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return spike times in ms and local neural node indices."""
