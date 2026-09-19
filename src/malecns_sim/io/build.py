"""Build a deterministic, provenance-rich I/O catalogue from graph node metadata."""

import argparse
import shutil
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from malecns_sim import __version__
from malecns_sim.cns.graph.store import CNSGraph, file_hashes, write_json
from malecns_sim.io.roles import NeuronRole
from malecns_sim.io.selectors import derive_catalogue, load_io_policy

DEFAULT_GRAPH = Path("data/processed/malecns-v1.0/published-v1")


def build_catalogue(graph_dir: str | Path, output: str | Path, policy_path=None) -> dict:
    graph_dir, output = Path(graph_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Catalogue directory exists: {output}")
    policy = load_io_policy(policy_path)
    with CNSGraph.load(graph_dir) as graph:
        frame = graph.nodes.to_pandas()
        derived = derive_catalogue(frame, policy)
        table = pa.Table.from_pandas(derived, preserve_index=False).replace_schema_metadata(None)
        if not table["node_index"].equals(graph.nodes["node_index"]):
            raise ValueError("I/O derivation changed graph node order")
        if not table["body_id"].equals(graph.nodes["body_id"]):
            raise ValueError("I/O derivation changed graph body IDs")
        counts = {role.value: int((derived["role"] == role.value).sum()) for role in NeuronRole}
        source_counts = {
            str(value): int(count)
            for value, count in derived[policy["source_field"]].value_counts(dropna=False).items()
        }
        role_counts = {
            "total_graph_neurons": graph.num_nodes,
            "roles": counts,
            "mapped_role_rules": int(derived["role_policy_status"].eq("mapped").sum()),
            "unknown_role_rules": int(derived["role_policy_status"].eq("unmapped").sum()),
            "source_field": policy["source_field"],
            "source_values": source_counts,
        }
        manifest = {
            "dataset": "MaleCNS v1.0",
            "catalogue_schema": 1,
            "catalogue_policy": policy,
            "builder_version": __version__,
            "graph": {
                "directory_name": graph_dir.name,
                "manifest": file_hashes(graph_dir / "manifest.json"),
                "nodes": file_hashes(graph_dir / "nodes.parquet"),
                "graph_schema": graph.manifest["graph_schema"],
                "node_policy": graph.manifest["node_policy"],
            },
            "neurons": graph.num_nodes,
            "role_counts": role_counts,
            "normalization": {
                "raw_values_preserved": True,
                "motor_to_actuator_mapping": None,
                "sensor_to_neuron_encoding": None,
                "role": "Exact released superclass values mapped by catalogue_policy.",
                "side": "First assigned raw side field in configured priority; source recorded.",
                "nerve": "Sensory entryNerve or motor exitNerve only; source recorded.",
                "sensory_system": "Exact released class mapping for primary sensory roles.",
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve()
    stage = parent / f".{output.name}-{uuid4().hex}"
    stage.mkdir()
    try:
        pq.write_table(table, stage / "neurons.parquet", compression="zstd")
        write_json(stage / "role_counts.json", role_counts)
        manifest["artifacts"] = {
            name: file_hashes(stage / name) for name in ("neurons.parquet", "role_counts.json")
        }
        write_json(stage / "manifest.json", manifest)
        stage.rename(output)
    finally:
        if stage.exists() and stage.resolve().parent == parent:
            shutil.rmtree(stage)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    output = args.output or args.graph / "io-v1"
    try:
        manifest = build_catalogue(args.graph, output, args.policy)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        raise SystemExit(f"I/O catalogue build failed: {error}") from None
    print(f"MaleCNS I/O catalogue built: {output}")
    print(f"Neurons: {manifest['neurons']:,}")
    for role, count in manifest["role_counts"]["roles"].items():
        print(f"{role}: {count:,}")


if __name__ == "__main__":
    main()
