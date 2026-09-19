import numpy as np
import pytest

from malecns_sim.io.build import build_catalogue
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.roles import NeuronRole
from malecns_sim.io.validate import compare_catalogues, validate_catalogue


def test_build_preserves_metadata_and_records_rule_sources(graph_dir, tmp_path):
    output = tmp_path / "io-v1"
    manifest = build_catalogue(graph_dir, output)
    report = validate_catalogue(output, graph_dir)
    assert report["valid"]
    assert manifest["role_counts"]["roles"] == {
        "sensory": 1,
        "motor": 1,
        "descending": 1,
        "ascending": 1,
        "interneuron": 1,
        "endocrine": 1,
        "other": 1,
    }
    assert manifest["role_counts"]["unknown_role_rules"] == 1
    catalogue = IOCatalog.load(output)
    sensory = catalogue.sensory().to_pylist()[0]
    assert sensory["role_source_field"] == "superclass"
    assert sensory["role_source_value"] == "vnc_sensory"
    assert sensory["side"] == "R"
    assert sensory["side_source_field"] == "rootSide"
    assert sensory["nerve"] == "ProLN"
    assert sensory["nerve_source_field"] == "entryNerve"
    assert sensory["sensory_system"] == "mechanosensory"
    motor = catalogue.motor().to_pylist()[0]
    assert motor["side"] == "L" and motor["nerve"] == "ProLN"
    assert motor["body_region"] == "T1"


def test_query_api_returns_graph_indices_and_body_ids(graph_dir, tmp_path):
    output = tmp_path / "io-v1"
    build_catalogue(graph_dir, output)
    catalogue = IOCatalog.load(output)
    result = catalogue.filter(role=NeuronRole.MOTOR, side="L")
    np.testing.assert_array_equal(result.node_indices, [1])
    np.testing.assert_array_equal(result.body_ids, [101])
    assert catalogue.by_type("DNa01").node_indices.tolist() == [2]
    assert catalogue.by_side("R").node_indices.tolist() == [0, 2, 4]
    assert catalogue.descending().body_ids.tolist() == [102]
    assert catalogue.ascending().body_ids.tolist() == [103]
    with pytest.raises(KeyError):
        catalogue.filter(nonexistent="x")


def test_build_is_deterministic_and_never_overwrites(graph_dir, tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    build_catalogue(graph_dir, left)
    build_catalogue(graph_dir, right)
    for name in ("neurons.parquet", "role_counts.json", "manifest.json"):
        assert (left / name).read_bytes() == (right / name).read_bytes()
    compare_catalogues(left, right, graph_dir)
    with pytest.raises(FileExistsError):
        build_catalogue(graph_dir, left)


def test_validation_detects_corruption(graph_dir, tmp_path):
    output = tmp_path / "io-v1"
    build_catalogue(graph_dir, output)
    with (output / "role_counts.json").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_catalogue(output, graph_dir)


@pytest.mark.io_catalogue
def test_real_catalogue_has_full_graph_mapping():
    output = graph = None
    from pathlib import Path

    graph = Path("data/processed/malecns-v1.0/published-v1")
    output = graph / "io-v1"
    if not (output / "manifest.json").exists():
        pytest.skip("Build the io-v1 catalogue first")
    report = validate_catalogue(output, graph)
    assert report["neurons"] == 166700
    assert report["unknown_role_rules"] == 0
