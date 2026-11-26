import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import yaml

from malecns_sim.cns.download import FILES
from malecns_sim.cns.graph.build import build_graph
from malecns_sim.cns.graph.policy import load_policy

# 0,1 sensory; 2 excitatory, 3 inhibitory, 4 unsigned interneurons; 5,6 motor; 7 isolated.
EDGES = [
    (0, 5, 3),
    (0, 2, 2),
    (2, 2, 1),
    (2, 5, 4),
    (2, 3, 6),
    (3, 5, 2),
    (1, 3, 5),
    (3, 6, 1),
    (1, 4, 7),
    (4, 6, 9),
]
SIGNS = np.array([1, 1, 1, -1, 0, 1, 1, 1], dtype=np.int8)

# Body IDs for the same topology, built through the real graph builder.
BODY_IDS = [101, 102, 201, 202, 203, 301, 302, 401]
NTS = [
    "acetylcholine",
    "acetylcholine",
    "acetylcholine",
    "gaba",
    None,
    "acetylcholine",
    "acetylcholine",
    "acetylcholine",
]


@pytest.fixture
def built_graph(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "bodyId": np.array(BODY_IDS, dtype=np.uint64),
            "status": ["Traced"] * 8,
            "superclass": ["vnc_sensory"] * 2
            + ["vnc_intrinsic"] * 3
            + ["vnc_motor"] * 2
            + ["cb_intrinsic"],
            "type": ["S1", "S2", "IN_a", "IN_b", "IN_c", "MN_a", "MN_b", "other"],
            "somaSide": ["L"] * 8,
        }
    ).to_feather(raw / FILES["annotations"])
    pd.DataFrame(
        {
            "body": np.array(BODY_IDS, dtype=np.uint64),
            "consensus_nt": NTS,
            "predicted_nt_confidence": [0.9] * 8,
        }
    ).to_feather(raw / FILES["neurotransmitters"])
    pre, post, weight = zip(*[(BODY_IDS[s], BODY_IDS[d], c) for s, d, c in EDGES])
    feather.write_feather(
        pa.table(
            {
                "body_pre": pa.array(pre, type=pa.uint64()),
                "body_post": pa.array(post, type=pa.uint64()),
                "weight": pa.array(weight, type=pa.int64()),
            }
        ),
        raw / FILES["connections"],
    )
    graph = tmp_path / "graph"
    build_graph(raw, graph, load_policy("published"))
    sensory = tmp_path / "sensory.yaml"
    sensory.write_text(
        yaml.safe_dump(
            {
                "mapping_version": "sensory-bridge-v1",
                "channels": [{"name": "probe", "malecns": {"body_ids": [101, 102]}}],
            }
        )
    )
    motor = tmp_path / "motor.yaml"
    motor.write_text(
        yaml.safe_dump(
            {
                "mapping_version": "motor-bridge-v1",
                "channels": [
                    {
                        "name": "joint",
                        "male_cns": {
                            "flexor": {"body_ids": [301]},
                            "extensor": {"body_ids": [302]},
                        },
                    }
                ],
            }
        )
    )
    return graph, sensory, motor
