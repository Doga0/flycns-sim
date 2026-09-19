"""Validate catalogue provenance, normalized derivations, and graph identity."""

import argparse
import json
from pathlib import Path

import numpy as np

from malecns_sim.cns.graph.store import CNSGraph, file_hashes
from malecns_sim.io.roles import NeuronRole
from malecns_sim.io.selectors import derive_catalogue
from malecns_sim.io.store import load_catalogue_artifacts


def validate_catalogue(directory: str | Path, graph_dir: str | Path | None = None) -> dict:
    directory = Path(directory)
    manifest, neurons = load_catalogue_artifacts(directory)
    required = {"neurons.parquet", "role_counts.json"}
    if set(manifest.get("artifacts", {})) != required:
        raise ValueError("Unexpected catalogue artifact manifest")
    for name in sorted(required):
        if file_hashes(directory / name) != manifest["artifacts"][name]:
            raise ValueError(f"Catalogue checksum mismatch: {name}")
    counts_file = json.loads((directory / "role_counts.json").read_text(encoding="utf-8"))
    if counts_file != manifest["role_counts"]:
        raise ValueError("Role counts differ between manifest and role_counts.json")
    if neurons.num_rows != manifest["neurons"]:
        raise ValueError("Catalogue row count differs from manifest")
    node_indices = neurons["node_index"].to_numpy()
    body_ids = neurons["body_id"].to_numpy()
    if node_indices.dtype != np.int32 or not np.array_equal(
        node_indices, np.arange(len(node_indices), dtype=np.int32)
    ):
        raise ValueError("Catalogue node indices are not contiguous int32")
    if body_ids.dtype != np.uint64 or len(np.unique(body_ids)) != len(body_ids):
        raise ValueError("Catalogue body IDs are not unique uint64 values")
    role_values = set(neurons["role"].to_pylist())
    if not role_values <= {role.value for role in NeuronRole}:
        raise ValueError("Catalogue contains unsupported roles")
    actual_counts = {
        role.value: neurons["role"].to_pylist().count(role.value) for role in NeuronRole
    }
    if actual_counts != counts_file["roles"]:
        raise ValueError("Role counts do not match catalogue rows")
    statuses = neurons["role_policy_status"].to_pylist()
    if statuses.count("unmapped") != counts_file["unknown_role_rules"]:
        raise ValueError("Unknown role count does not match catalogue rows")

    graph_dir = Path(graph_dir) if graph_dir else directory.parent
    with CNSGraph.load(graph_dir) as graph:
        if file_hashes(graph_dir / "manifest.json") != manifest["graph"]["manifest"]:
            raise ValueError("Catalogue was built against a different graph manifest")
        if file_hashes(graph_dir / "nodes.parquet") != manifest["graph"]["nodes"]:
            raise ValueError("Catalogue was built against different graph node metadata")
        if not np.array_equal(node_indices, graph.nodes["node_index"].to_numpy()):
            raise ValueError("Catalogue node indices differ from graph")
        if not np.array_equal(body_ids, graph.body_ids):
            raise ValueError("Catalogue body IDs differ from graph")
        expected = derive_catalogue(graph.nodes.to_pandas(), manifest["catalogue_policy"])
        for field in (
            "role",
            "role_source_field",
            "role_source_value",
            "role_policy_status",
            "side",
            "side_source_field",
            "side_source_value",
            "nerve",
            "nerve_source_field",
            "nerve_source_value",
            "body_region",
            "body_region_source_field",
            "body_region_source_value",
            "sensory_system",
            "sensory_system_source_field",
            "sensory_system_source_value",
        ):
            actual = neurons[field].to_pandas().astype("string")
            if not actual.equals(expected[field].astype("string")):
                raise ValueError(f"Derived catalogue field differs from policy: {field}")
        for field in graph.nodes.column_names:
            if neurons[field].to_pylist() != graph.nodes[field].to_pylist():
                raise ValueError(f"Raw graph metadata changed in catalogue: {field}")
    return {
        "valid": True,
        "neurons": neurons.num_rows,
        "roles": actual_counts,
        "unknown_role_rules": counts_file["unknown_role_rules"],
    }


def compare_catalogues(left: str | Path, right: str | Path, graph_dir=None) -> None:
    left, right = Path(left), Path(right)
    if validate_catalogue(left, graph_dir) != validate_catalogue(right, graph_dir):
        raise ValueError("Catalogue validation summaries differ")
    for name in ("neurons.parquet", "role_counts.json", "manifest.json"):
        if file_hashes(left / name) != file_hashes(right / name):
            raise ValueError(f"Catalogue artifacts differ: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    try:
        report = validate_catalogue(args.directory, args.graph)
        if args.compare:
            compare_catalogues(args.directory, args.compare, args.graph)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"I/O catalogue validation failed: {error}") from None
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.compare:
        print("Catalogue artifacts are byte-identical.")


if __name__ == "__main__":
    main()
