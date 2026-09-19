import numpy as np
import pytest

from malecns_sim.neural.backends.base import NeuralGraph


@pytest.fixture
def graph_factory():
    def make(n=2, edges=((0, 1, 250),), nts=None):
        return NeuralGraph(
            np.arange(2**63, 2**63 + n, dtype=np.uint64),
            np.arange(n, dtype=np.int32) * 3 + 7,
            np.array([e[0] for e in edges], dtype=np.int32),
            np.array([e[1] for e in edges], dtype=np.int32),
            np.array([e[2] for e in edges], dtype=np.uint32),
            nts or ["acetylcholine"] * n,
            {"selection": {"type": "synthetic"}},
        )

    return make
