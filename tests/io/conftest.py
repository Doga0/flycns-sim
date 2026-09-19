import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from malecns_sim.cns.graph.store import ARRAY_DTYPES


@pytest.fixture
def graph_dir(tmp_path):
    directory = tmp_path / "published-v1"
    directory.mkdir()
    nodes = pa.table(
        {
            "node_index": pa.array(np.arange(7, dtype=np.int32)),
            "body_id": pa.array(np.arange(100, 107, dtype=np.uint64)),
            "superclass": [
                "vnc_sensory",
                "vnc_motor",
                "descending_neuron",
                "ascending_neuron",
                "cb_intrinsic",
                "cb_endocrine",
                "new_value",
            ],
            "class": ["mechanosensory_proprioceptive", None, None, None, None, None, None],
            "subclass": ["leg", "fl", "xn", "BA", None, None, None],
            "type": ["SNch01", "Ti flexor MN", "DNa01", "AN01", "IN01", None, None],
            "somaSide": [None, "L", "R", "L", "R", "M", None],
            "rootSide": ["R", None, None, None, None, None, None],
            "entryNerve": ["ProLN", None, None, None, None, None, None],
            "exitNerve": [None, "ProLN", None, None, None, None, None],
            "somaNeuromere": [None, "T1", "LB", "T1", "T2", "CG", None],
            "receptorType": [None] * 7,
            "mancType": [None] * 7,
            "flywireType": [None] * 7,
            "hemibrainType": [None] * 7,
            "consensus_nt": ["acetylcholine"] * 7,
        }
    )
    pq.write_table(nodes, directory / "nodes.parquet")
    manifest = {
        "graph_schema": 1,
        "nodes": 7,
        "edges": 0,
        "contacts": 0,
        "node_policy": {"id": "fixture"},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    sizes = {
        "body_ids": 7,
        "src": 0,
        "dst": 0,
        "contacts": 0,
        "csr_indptr": 8,
        "csr_indices": 0,
        "csr_contacts": 0,
    }
    for name, dtype in ARRAY_DTYPES.items():
        values = np.zeros(sizes[name], dtype=dtype)
        if name == "body_ids":
            values[:] = np.arange(100, 107, dtype=np.uint64)
        np.save(directory / f"{name}.npy", values, allow_pickle=False)
    return directory
