import json
import math

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from malecns_sim.analysis.pathway_audit import run_audit
from malecns_sim.analysis.sensory_motor_paths import (
    SIGN_CLASSES,
    UNREACHED,
    Connectivity,
    direct_drive_estimate,
    enumerate_paths,
    hop_distances,
    psp_peak_per_mv,
    sign_class,
    walk_counts,
)
from malecns_sim.neural.config import LIFConfig

# Same topology as the `built_graph` fixture in conftest.py (keep both in sync).
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


@pytest.fixture
def conn():
    return Connectivity.from_edges(8, EDGES)


def brute_force_walks(max_hops, sources, targets):
    adjacency = {}
    for s, d, c in EDGES:
        adjacency.setdefault(s, []).append((d, c))
    walks = []
    for source in sources:
        frontier = [([source], [])]
        for _ in range(max_hops):
            frontier = [
                (nodes + [d], contacts + [c])
                for nodes, contacts in frontier
                for d, c in adjacency.get(nodes[-1], [])
            ]
            walks += [w for w in frontier if w[0][-1] in targets]
    return walks


def test_hop_distances_forward_and_reverse(conn):
    forward = hop_distances(conn, [0, 1], 3)
    assert forward.tolist() == [0, 0, 1, 1, 1, 1, 2, UNREACHED]
    backward = hop_distances(conn, [5, 6], 3, reverse=True)
    assert backward[[0, 1, 2, 3, 4, 5, 6, 7]].tolist() == [1, 2, 1, 1, 1, 0, 0, UNREACHED]
    assert hop_distances(conn, [0], 1)[6] == UNREACHED


def test_sign_class_uses_presynaptic_parity():
    assert sign_class([1, 1]) == "excitatory"
    assert sign_class([1, -1]) == "net_inhibitory"
    assert sign_class([1, -1, -1]) == "net_excitatory_via_inhibition"
    assert sign_class([1, -1, 0]) == "muted"


def test_walk_counts_match_brute_force_by_sign_class(conn):
    counts = walk_counts(conn, SIGNS, [0, 1], [5, 6], 4)
    walks = brute_force_walks(4, {0, 1}, {5, 6})
    expected = np.zeros_like(counts)
    for nodes, _ in walks:
        k = SIGN_CLASSES.index(sign_class(SIGNS[nodes[:-1]]))
        expected[len(nodes) - 2, k, [5, 6].index(nodes[-1])] += 1
    np.testing.assert_array_equal(counts, expected)
    # The direct synapse and the self-loop walk 0>2>2>5 are both counted.
    assert counts[0, 0].tolist() == [1, 0]
    assert counts[2, 0, 0] == 1


def test_enumeration_lists_exactly_the_counted_walks(conn):
    listed = list(enumerate_paths(conn, [0, 1], [5, 6], 3, limit=100))
    expected = brute_force_walks(3, {0, 1}, {5, 6})
    assert sorted(map(str, listed)) == sorted(str((n, c)) for n, c in expected)
    assert ([0, 2, 2, 5], [2, 1, 4]) in listed
    with pytest.raises(ValueError, match="More than 2 paths"):
        list(enumerate_paths(conn, [0, 1], [5, 6], 3, limit=2))


def test_invalid_populations_and_signs_are_rejected(conn):
    with pytest.raises(ValueError, match="unique"):
        hop_distances(conn, [0, 0], 2)
    with pytest.raises(ValueError, match="unknown node"):
        walk_counts(conn, SIGNS, [0], [99], 2)
    with pytest.raises(ValueError, match="signs"):
        walk_counts(conn, np.full(8, 2), [0], [5], 2)


def test_psp_peak_matches_numerical_integration():
    dt, tm, ts = 0.001, 20.0, 5.0
    v = g = peak = 0.0
    g = 1.0
    for _ in range(int(60 / dt)):
        v += dt * (g - v) / tm
        g -= dt * g / ts
        peak = max(peak, v)
    assert psp_peak_per_mv(tm, ts) == pytest.approx(peak, rel=1e-3)
    assert psp_peak_per_mv(10.0, 10.0) == pytest.approx(1 / math.e)


def test_direct_drive_estimate_uses_lif_equations(conn):
    config = LIFConfig()
    rows = direct_drive_estimate(conn, SIGNS, [0, 1], [5, 6], config)
    lf = {r["target_node"]: r for r in rows}
    assert lf[5]["direct_contacts"] == 3 and lf[5]["direct_sources"] == 1
    weight = 3 * config.weight_per_contact_mv
    assert lf[5]["direct_weight_mv"] == pytest.approx(weight)
    gap = config.v_threshold_mv - config.v_rest_mv
    rate = gap / (weight * config.tau_synapse_ms / 1000)
    assert lf[5]["uniform_source_rate_for_mean_threshold_hz"] == pytest.approx(rate)
    assert lf[6]["direct_contacts"] == 0
    assert lf[6]["uniform_source_rate_for_mean_threshold_hz"] is None


def test_cli_audit_writes_consistent_artifacts(built_graph, tmp_path):
    graph, sensory, motor = built_graph
    out = tmp_path / "audit"
    summary = run_audit(graph, sensory, motor, None, out, max_hops=4, enumerate_hops=3, edge_hops=3)
    assert sorted(p.name for p in out.iterdir()) == [
        "edges.parquet",
        "nodes.parquet",
        "paths.parquet",
        "pathway_summary.json",
        "report.json",
    ]
    paths = pq.read_table(out / "paths.parquet").to_pylist()
    expected = brute_force_walks(3, {0, 1}, {5, 6})
    assert len(paths) == len(expected)
    direct = [p for p in paths if p["hops"] == 1]
    assert [(p["source_body_id"], p["target_body_id"], p["contacts_path"]) for p in direct] == [
        (101, 301, [3])
    ]
    loop = next(p for p in paths if p["body_id_path"] == [101, 201, 201, 301])
    assert not loop["simple"] and loop["sign_class"] == "excitatory"
    assert loop["synaptic_delay_lower_bound_ms"] == pytest.approx(3 * 1.8)
    muted = next(p for p in paths if p["body_id_path"] == [102, 203, 302])
    assert muted["sign_class"] == "muted" and muted["nt_path"][1] is None

    nodes = pq.read_table(out / "nodes.parquet").to_pandas()
    assert 401 not in nodes.body_id.tolist()
    assert set(nodes[nodes.population == "intermediate"].body_id) == {201, 202, 203}

    walks = {r["hops"]: r["walks"] for r in summary["walks_by_hops"]}
    for hops in (1, 2, 3, 4):
        assert walks[hops] == len(
            [w for w in brute_force_walks(4, {0, 1}, {5, 6}) if len(w[0]) == hops + 1]
        )
    assert summary["reachability"]["shortest_hops_by_target_body_id"] == {"301": 1, "302": 2}
    report = json.loads((out / "report.json").read_text())
    assert report["diagnostic_only"] and report["rows"]["paths"] == len(paths)
    assert set(report["artifacts"]) == {
        "edges.parquet",
        "nodes.parquet",
        "paths.parquet",
        "pathway_summary.json",
    }
    with pytest.raises(FileExistsError):
        run_audit(graph, sensory, motor, None, out)


def test_cli_rejects_overlap_and_swapped_profiles(built_graph, tmp_path):
    graph, sensory, motor = built_graph
    with pytest.raises(ValueError, match="sensory bridge profile"):
        run_audit(graph, motor, motor, None, tmp_path / "a")
    overlap = tmp_path / "overlap.yaml"
    overlap.write_text(
        yaml.safe_dump(
            {
                "mapping_version": "motor-bridge-v1",
                "channels": [
                    {
                        "name": "joint",
                        "male_cns": {"flexor": {"body_ids": [101]}, "extensor": {"body_ids": []}},
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        run_audit(graph, sensory, overlap, None, tmp_path / "b")
    assert not any(p.name.startswith(".b-") for p in tmp_path.iterdir())


def test_all_counts_are_walks_not_simple_paths(conn):
    # The only cycle in the fixture is node 2's self-loop.
    counts = walk_counts(conn, SIGNS, [0, 1], [5, 6], 4)
    simple = [w for w in brute_force_walks(4, {0, 1}, {5, 6}) if len(set(w[0])) == len(w[0])]
    assert counts.sum() > len(simple)
