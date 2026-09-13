import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from malecns_sim.cns.census import audit_annotations
from malecns_sim.cns.download import FILES
from malecns_sim.cns.graph.build import build_graph
from malecns_sim.cns.graph.policy import exact_ids, load_policy, select_nodes
from malecns_sim.cns.graph.store import CNSGraph
from malecns_sim.cns.graph.validate import validate_graph

BIG_ID = 2**63 + 3


@pytest.fixture
def raw_graph(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    annotations = pd.DataFrame(
        {
            "bodyId": np.array([BIG_ID, 10, 20, 30, 40, 50], dtype=np.uint64),
            "status": ["Traced", "Traced", "Orphan", "Traced", "Glia", None],
            "superclass": ["vnc_motor", "cb_intrinsic", "ol_sensory", None, None, "cb_sensory"],
            "type": ["big", "small", None, "unresolved", None, "isolated"],
            "somaSide": ["L", "R", None, None, None, "M"],
        }
    )
    annotations.to_feather(raw / FILES["annotations"])
    pd.DataFrame(
        {
            "body": np.array([20, BIG_ID, 10], dtype=np.uint64),
            "consensus_nt": ["gaba", "unclear", "acetylcholine"],
            "predicted_nt_confidence": [0.8, None, 0.9],
        }
    ).to_feather(raw / FILES["neurotransmitters"])
    edges = pa.table(
        {
            "body_pre": pa.array([BIG_ID, 10, BIG_ID, 20, 30, 10, 30, 10], type=pa.uint64()),
            "body_post": pa.array([10, 20, 10, 20, 10, 40, 40, 20], type=pa.uint64()),
            "weight": pa.array([3, 7, 2, 4, 5, 6, 8, 1], type=pa.int64()),
            "unused": ["ignored"] * 8,
        }
    )
    feather.write_feather(edges, raw / FILES["connections"], chunksize=3)
    return raw


def build_fixture(raw, output, **kwargs):
    return build_graph(raw, output, load_policy("published"), **kwargs)


def test_census_exposes_policy_disagreement(raw_graph):
    frame = pd.read_feather(raw_graph / FILES["annotations"])
    audit = audit_annotations(frame)
    assert audit["total_annotations"] == 6
    assert audit["traced_count"] == 3
    assert audit["glia_count"] == 1
    assert audit["assigned_superclass_count"] == 4
    assert audit["traced_without_superclass"] == 1
    assert audit["assigned_superclass_not_traced"] == 2
    assert audit["policies"]["published-v1"]["difference_from_reference"] == 4 - 166691
    assert not audit["policies"]["published-v1"]["membership_verified"]


def test_graph_mapping_aggregation_metadata_and_queries(raw_graph, tmp_path):
    output = tmp_path / "graph"
    manifest = build_fixture(raw_graph, output, batch_rows=2, chunk_rows=1, partitions=2)
    assert (manifest["nodes"], manifest["edges"], manifest["contacts"]) == (4, 3, 17)
    accounting = manifest["contact_accounting"]
    assert accounting["raw_contacts"] == 36
    assert accounting["excluded_contacts"] == 19
    assert accounting["excluded_source_only_contacts"] == 5
    assert accounting["excluded_target_only_contacts"] == 6
    assert accounting["excluded_both_contacts"] == 8
    assert accounting["retained_duplicate_rows_aggregated"] == 2
    with CNSGraph.load(output) as graph:
        assert graph.body_ids.tolist() == [10, 20, 50, BIG_ID]
        assert graph.arrays["src"].tolist() == [0, 1, 3]
        assert graph.arrays["dst"].tolist() == [1, 1, 0]
        assert graph.arrays["contacts"].tolist() == [8, 4, 5]
        assert graph.node_index(BIG_ID) == 3
        assert graph.outgoing(2)[0].size == 0
        assert graph.outgoing(3)[1].tolist() == [5]
        assert graph.nodes["type"].to_pylist() == ["small", None, "isolated", "big"]
        assert graph.nodes["consensus_nt"].to_pylist() == ["acetylcholine", "gaba", None, "unclear"]
        assert "sign" not in graph.nodes.column_names
        with pytest.raises(KeyError):
            graph.node_index(30)
        with pytest.raises(IndexError):
            graph.outgoing(-1)
        with pytest.raises(TypeError):
            graph.node_index(float(BIG_ID))
        validate_graph(graph)


def test_same_inputs_byte_identical_across_chunk_plans(raw_graph, tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    build_fixture(raw_graph, left, batch_rows=1, chunk_rows=1, partitions=1)
    build_fixture(raw_graph, right, batch_rows=100, chunk_rows=100, partitions=8)
    assert {p.name for p in left.iterdir()} == {p.name for p in right.iterdir()}
    for path in left.iterdir():
        assert path.read_bytes() == (right / path.name).read_bytes(), path.name


def test_multilevel_external_merge(raw_graph, tmp_path):
    edges = pa.table({"body_pre": [10] * 100, "body_post": [20] * 100, "weight": [1] * 100})
    feather.write_feather(edges, raw_graph / FILES["connections"], chunksize=3)
    manifest = build_fixture(raw_graph, tmp_path / "graph", chunk_rows=1, partitions=1)
    assert manifest["edges"] == 1
    assert manifest["contacts"] == 100
    assert manifest["contact_accounting"]["raw_duplicate_rows"] == 99


def test_empty_retained_connectivity_and_isolated_nodes(raw_graph, tmp_path):
    edges = pa.table({"body_pre": [30], "body_post": [40], "weight": [1]})
    feather.write_feather(edges, raw_graph / FILES["connections"])
    manifest = build_fixture(raw_graph, tmp_path / "empty")
    assert manifest["edges"] == 0
    with CNSGraph.load(tmp_path / "empty") as graph:
        assert graph.arrays["csr_indptr"].tolist() == [0] * 5
        validate_graph(graph)


@pytest.mark.parametrize("weights", [[0], [-1], [1.5], [None], [2**32]])
def test_invalid_raw_contacts_rejected(raw_graph, tmp_path, weights):
    feather.write_feather(
        pa.table({"body_pre": [10], "body_post": [20], "weight": weights}),
        raw_graph / FILES["connections"],
    )
    with pytest.raises(ValueError):
        build_fixture(raw_graph, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_aggregation_overflow_is_not_wrapped(raw_graph, tmp_path):
    feather.write_feather(
        pa.table({"body_pre": [10, 10], "body_post": [20, 20], "weight": [2**32 - 1, 1]}),
        raw_graph / FILES["connections"],
    )
    with pytest.raises(OverflowError):
        build_fixture(raw_graph, tmp_path / "overflow", chunk_rows=1)
    assert not (tmp_path / "overflow").exists()


@pytest.mark.parametrize("table,column", [("annotations", "bodyId"), ("neurotransmitters", "body")])
def test_duplicate_metadata_ids_fail(raw_graph, tmp_path, table, column):
    path = raw_graph / FILES[table]
    frame = pd.read_feather(path)
    frame.loc[1, column] = frame.loc[0, column]
    frame.to_feather(path)
    with pytest.raises(ValueError, match="Duplicate"):
        build_fixture(raw_graph, tmp_path / "duplicate")


def test_existing_graph_is_not_overwritten(raw_graph, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        build_fixture(raw_graph, target)
    assert (target / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("field", ["dst", "csr_indptr", "csr_contacts"])
def test_corrupt_graph_rejected(raw_graph, tmp_path, field):
    output = tmp_path / "graph"
    build_fixture(raw_graph, output)
    path = output / f"{field}.npy"
    array = np.load(path)
    array[0] = 999
    np.save(path, array)
    with CNSGraph.load(output) as graph, pytest.raises(ValueError):
        validate_graph(graph)


def test_policy_is_explicit_and_rejects_ambiguous_rules():
    frame = pd.DataFrame({"superclass": [None, "", "  ", "vnc_tbc"], "status": [None] * 4})
    policy = load_policy("published")
    assert select_nodes(frame, policy["selection"]).tolist() == [False, False, False, True]
    with pytest.raises(ValueError):
        select_nodes(frame, {"column": "status", "in": ["Traced"], "typo": True})
    with pytest.raises(ValueError):
        select_nodes(frame, {"column": "missing", "assigned": True})
    with pytest.raises(ValueError):
        exact_ids(np.array([1.0]))


def test_policy_profiles_are_loadable():
    assert load_policy("traced")["id"] == "traced-v1"
    assert load_policy("all-annotated")["id"] == "all-annotated-v1"


def test_empty_node_policy_fails(raw_graph, tmp_path):
    policy = copy.deepcopy(load_policy("published"))
    policy["selection"] = {"column": "status", "in": ["does-not-exist"]}
    with pytest.raises(ValueError, match="retain"):
        build_graph(raw_graph, tmp_path / "no-nodes", policy)


def test_changing_input_during_build_is_rejected(raw_graph, tmp_path, monkeypatch):
    from malecns_sim.cns.graph import build

    original = build.scan_connections

    def changing_scan(*args, **kwargs):
        result = original(*args, **kwargs)
        path = raw_graph / FILES["neurotransmitters"]
        with path.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(build, "scan_connections", changing_scan)
    with pytest.raises(ValueError, match="Input changed"):
        build_fixture(raw_graph, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


@pytest.mark.graph
def test_real_graph_acceptance():
    directory = Path("data/processed/malecns-v1.0/published-v1")
    with CNSGraph.load(directory) as graph:
        report = validate_graph(graph)
        assert report["outgoing_queries_checked"] == 100
        assert report["nodes"] > 100_000
        assert report["edges"] > 1_000_000
        annotations = pd.read_feather(Path("data/malecns-v1.0") / FILES["annotations"])
        mask = select_nodes(annotations, graph.manifest["policy"]["selection"])
        expected = annotations.loc[mask].sort_values("bodyId").reset_index(drop=True)
        assert np.array_equal(expected.bodyId.to_numpy(), graph.body_ids)
        for column in ("status", "superclass", "class", "subclass", "type", "somaSide", "rootSide"):
            values = expected[column].astype(object).where(expected[column].notna(), None).tolist()
            assert graph.nodes[column].to_pylist() == values
