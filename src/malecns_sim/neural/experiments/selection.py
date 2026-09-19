"""Deterministic, bounded outgoing subgraphs with preserved anatomical indices."""

import numpy as np

from malecns_sim.cns.graph.store import CNSGraph, file_hashes
from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.config import LIFConfig
from malecns_sim.neural.signs import transmitter_signs


def choose_source(graph: CNSGraph, config: LIFConfig) -> int:
    """Strongest positive-to-positive non-self edge; ties use canonical COO order."""
    signs, _ = transmitter_signs(graph.nodes["consensus_nt"].to_pylist(), config)
    best_weight, best_source = 0, None
    for start in range(0, len(graph.arrays["src"]), 1_000_000):
        src, dst, contacts = (
            graph.arrays[key][start : start + 1_000_000] for key in ("src", "dst", "contacts")
        )
        eligible = (signs[src] > 0) & (signs[dst] > 0) & (src != dst)
        weights = np.where(eligible, contacts, 0)
        if len(weights):
            i = int(np.argmax(weights))
            if weights[i] > best_weight:
                best_weight, best_source = int(weights[i]), int(src[i])
    if best_source is None:
        raise ValueError("No positive-to-positive non-self edge; specify a source explicitly")
    return best_source


def extract_subgraph(
    graph: CNSGraph, source: int, *, max_nodes: int = 5000, hops: int = 2
) -> NeuralGraph:
    if not 0 <= source < graph.num_nodes or max_nodes < 1 or hops not in (1, 2):
        raise ValueError("Need a valid source, positive node cap and 1 or 2 hops")
    selected, frontier = {source}, [source]
    audit = []
    distances = {source: 0}
    for hop in range(1, hops + 1):
        scores = np.zeros(graph.num_nodes, dtype=np.uint64)
        for node in frontier:
            targets, contacts = graph.outgoing(node)
            np.add.at(scores, targets, contacts)
        scores[list(selected)] = 0
        candidates = np.flatnonzero(scores)
        # Strongest total contacts from previous frontier, then parent node index.
        ranking = np.lexsort((candidates, -scores[candidates].astype(np.int64)))
        keep = candidates[ranking[: max(0, max_nodes - len(selected))]]
        frontier = [int(i) for i in keep]
        selected.update(frontier)
        distances.update({node: hop for node in frontier})
        audit.append(
            {
                "hop": hop,
                "candidates": len(candidates),
                "retained": len(keep),
                "omitted_by_cap": len(candidates) - len(keep),
            }
        )
    parent = np.array(sorted(selected), dtype=np.int32)
    local = np.full(graph.num_nodes, -1, dtype=np.int32)
    local[parent] = np.arange(len(parent), dtype=np.int32)
    src_parts, dst_parts, contact_parts = [], [], []
    for i, node in enumerate(parent):
        targets, contacts = graph.outgoing(int(node))
        mapped = local[targets]
        keep = mapped >= 0
        src_parts.append(np.full(np.count_nonzero(keep), i, dtype=np.int32))
        dst_parts.append(mapped[keep])
        contact_parts.append(contacts[keep].copy())
    metadata = graph.nodes.take(parent)
    return NeuralGraph(
        graph.body_ids[parent].copy(),
        parent,
        np.concatenate(src_parts),
        np.concatenate(dst_parts),
        np.concatenate(contact_parts),
        metadata["consensus_nt"].to_pylist(),
        {
            "graph_manifest": graph.manifest,
            "graph_manifest_sha256": file_hashes(graph.directory / "manifest.json")["sha256"],
            "selection": {
                "type": "outgoing-induced-subgraph",
                "hops": hops,
                "source_node_index": source,
                "source_body_id": int(graph.body_ids[source]),
                "max_nodes": max_nodes,
                "hop_audit": audit,
                "local_node_hops": [distances[int(i)] for i in parent],
                "cap_rule": (
                    "Breadth first; summed incoming contacts from previous retained frontier "
                    "descending, parent index ascending on ties. All induced edges retained."
                ),
            },
        },
    )
