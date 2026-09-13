"""Validate graph structure, metadata, contact conservation and file integrity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from malecns_sim.cns.graph.store import ARRAY_DTYPES, CNSGraph, file_hashes, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_graph(graph: CNSGraph, *, check_hashes: bool = True) -> dict:
    manifest, arrays = graph.manifest, graph.arrays
    n, e = manifest["nodes"], manifest["edges"]
    require(isinstance(n, int) and 0 < n <= np.iinfo(np.int32).max, "Invalid node count")
    require(isinstance(e, int) and e >= 0, "Invalid edge count")
    for name, dtype in ARRAY_DTYPES.items():
        expected_size = n if name == "body_ids" else n + 1 if name == "csr_indptr" else e
        require(arrays[name].dtype == np.dtype(dtype), f"Invalid dtype: {name}")
        require(arrays[name].shape == (expected_size,), f"Invalid shape: {name}")
    require(
        np.all(graph.body_ids[1:] > graph.body_ids[:-1]),
        "body_id mapping must be sorted and bijective",
    )
    require(graph.nodes.num_rows == n, "Metadata row count mismatch")
    require(graph.nodes["body_id"].type == pa.uint64(), "Metadata body_id must be uint64")
    require(graph.nodes["node_index"].type == pa.int32(), "Metadata node_index must be int32")
    require(
        np.array_equal(graph.nodes["body_id"].to_numpy(), graph.body_ids),
        "Metadata body_id mismatch",
    )
    require(
        np.array_equal(graph.nodes["node_index"].to_numpy(), np.arange(n)),
        "Metadata node_index mismatch",
    )
    pointers = arrays["csr_indptr"]
    require(pointers[0] == 0 and pointers[-1] == e, "Invalid CSR endpoints")
    require(np.all(pointers[1:] >= pointers[:-1]), "CSR indptr is not monotonic")
    degree = np.zeros(n, dtype="<i8")
    contacts, self_edges, previous = 0, 0, None
    for start in range(0, e, 1_000_000):
        end = min(start + 1_000_000, e)
        src, dst, weight = (arrays[name][start:end] for name in ("src", "dst", "contacts"))
        require(np.all((src >= 0) & (src < n) & (dst >= 0) & (dst < n)), "Unknown node endpoint")
        require(np.all(weight > 0), "Nonpositive contacts")
        keys = (src.astype("<u8") << np.uint64(32)) | dst.astype("<u8")
        require(np.all(keys[1:] > keys[:-1]), "COO edges are not sorted unique pairs")
        require(previous is None or int(keys[0]) > previous, "COO order fails across chunks")
        previous = int(keys[-1])
        require(
            np.array_equal(dst, arrays["csr_indices"][start:end]), "CSR targets differ from COO"
        )
        require(
            np.array_equal(weight, arrays["csr_contacts"][start:end]),
            "CSR contacts differ from COO",
        )
        degree += np.bincount(src, minlength=n)
        contacts += int(weight.sum(dtype=np.uint64))
        self_edges += int(np.count_nonzero(src == dst))
    require(np.array_equal(np.diff(pointers), degree), "CSR pointers disagree with source degrees")
    require(contacts == manifest["contacts"], "Contact total differs from manifest")
    require(self_edges == manifest["self_edges"], "Self-edge total differs from manifest")
    accounting = manifest["contact_accounting"]
    require(contacts == accounting["retained_contacts"], "Aggregation did not preserve contacts")
    require(
        accounting["retained_rows"] - e == accounting["retained_duplicate_rows_aggregated"],
        "Duplicate accounting mismatch",
    )
    for unit in ("rows", "contacts"):
        require(
            accounting[f"raw_{unit}"]
            == accounting[f"retained_{unit}"] + accounting[f"excluded_{unit}"],
            "Filtering conservation failed",
        )
        require(
            accounting[f"excluded_{unit}"]
            == sum(
                accounting[f"excluded_{reason}_{unit}"]
                for reason in ("source_only", "target_only", "both")
            ),
            "Exclusion reason accounting failed",
        )
    sampled = np.random.default_rng(0).choice(n, size=min(100, n), replace=False)
    for node in sampled:
        targets, weights = graph.outgoing(int(node))
        left = np.searchsorted(arrays["src"], node, side="left")
        right = np.searchsorted(arrays["src"], node, side="right")
        require(np.array_equal(targets, arrays["dst"][left:right]), "Outgoing targets mismatch")
        require(
            np.array_equal(weights, arrays["contacts"][left:right]), "Outgoing weights mismatch"
        )
        require(graph.node_index(int(graph.body_ids[node])) == node, "body_id lookup mismatch")
    artifact_names = {"nodes.parquet", "census.json", "audit.json"} | {
        f"{n}.npy" for n in ARRAY_DTYPES
    }
    require(set(manifest["artifacts"]) == artifact_names, "Unexpected artifact list")
    if check_hashes:
        for name in sorted(artifact_names):
            require(
                file_hashes(graph.directory / name) == manifest["artifacts"][name],
                f"Artifact checksum mismatch: {name}",
            )
    return {
        "graph_schema": 1,
        "node_policy": manifest["node_policy"],
        "nodes": n,
        "edges": e,
        "contacts": contacts,
        "self_edges": self_edges,
        "outgoing_queries_checked": len(sampled),
        "difference_from_reference": manifest["difference_from_reference"],
        "artifact_sha256": {name: data["sha256"] for name, data in manifest["artifacts"].items()},
        "manifest_sha256": file_hashes(graph.directory / "manifest.json")["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--compare", type=Path, help="Require byte-identical graph artifacts and manifest"
    )
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    with CNSGraph.load(args.directory) as graph:
        report = validate_graph(graph)
    if args.compare:
        with CNSGraph.load(args.compare) as other:
            other_report = validate_graph(other)
        require(report == other_report, "Graph summaries or artifact bytes differ")
        print("Byte-identical graph artifacts and manifest verified.")
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
