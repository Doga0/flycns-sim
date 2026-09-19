import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from malecns_sim.cns.graph.store import CNSGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.config import LIFConfig
from malecns_sim.neural.experiments.selection import choose_source, extract_subgraph
from malecns_sim.neural.stimulus import Stimulus


def test_induced_subgraph_cap_and_global_mapping(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"nodes": 5}), encoding="utf-8")
    arrays = {
        "body_ids": np.array([10, 20, 30, 40, 50], dtype=np.uint64),
        "csr_indptr": np.array([0, 0, 1, 3, 3, 4]),
        "csr_indices": np.array([0, 1, 4, 3], dtype=np.int32),
        "csr_contacts": np.array([5, 20, 10, 5], dtype=np.uint32),
    }
    stored = CNSGraph(
        tmp_path, {"nodes": 5}, pa.table({"consensus_nt": ["acetylcholine"] * 5}), arrays
    )
    small = extract_subgraph(stored, 2, max_nodes=4)
    small.validate()
    np.testing.assert_array_equal(small.node_indices, [0, 1, 2, 4])
    np.testing.assert_array_equal(small.body_ids, [10, 20, 30, 50])
    np.testing.assert_array_equal(small.src, [1, 2, 2])
    np.testing.assert_array_equal(small.dst, [0, 1, 3])
    np.testing.assert_array_equal(small.contacts, [5, 20, 10])
    assert small.provenance["selection"]["hop_audit"][1]["omitted_by_cap"] == 1
    second = extract_subgraph(stored, 2, max_nodes=4)
    assert small.provenance == second.provenance


@pytest.mark.graph
def test_real_malecns_subgraph_spikes_and_mapping():
    directory = Path("data/processed/malecns-v1.0/published-v1")
    if not (directory / "manifest.json").exists():
        pytest.skip("Build the published-v1 graph first")
    with CNSGraph.load(directory) as stored:
        source = choose_source(stored, LIFConfig())
        small = extract_subgraph(stored, source, max_nodes=500)
        np.testing.assert_array_equal(small.body_ids, stored.body_ids[small.node_indices])
        assert (
            small.neurotransmitters
            == stored.nodes.take(small.node_indices)["consensus_nt"].to_pylist()
        )
        backend = Brian2LIFBackend(small, seed=42)
        local = int(np.flatnonzero(small.node_indices == source)[0])
        backend.schedule(Stimulus((local,), stop_ms=100))
        backend.run(100)
        times, nodes = backend.get_spikes()
        assert len(times) > 0
        assert len(np.unique(nodes)) > 1
        assert np.isfinite(times).all()
