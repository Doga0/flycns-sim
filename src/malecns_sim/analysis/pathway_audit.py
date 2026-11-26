"""Audit anatomical sensory→motor paths and what the current sign policy does to them.

Diagnostic only: it reads the built graph, the reviewed bridge profiles and the LIF
config, and adds no biological assumption. Functional activation is a separate sweep.
"""

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

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
from malecns_sim.bridge.store import stage_output
from malecns_sim.cns.graph.store import CNSGraph, file_hashes, write_json
from malecns_sim.neural.config import load_config
from malecns_sim.neural.signs import transmitter_signs

DEFAULT_GRAPH = Path("data/processed/malecns-v1.0/published-v1")
MAX_HOPS_LIMIT = 6
METADATA = ("type", "superclass", "class", "somaSide", "rootSide", "consensus_nt")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def population_from_config(path: Path) -> tuple[dict, list[tuple[int, str]]]:
    """Read explicit body IDs from a sensory or motor bridge profile, never by type name."""
    policy = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    version = policy.get("mapping_version") if isinstance(policy, dict) else None
    members = []
    if version == "sensory-bridge-v1":
        for channel in policy["channels"]:
            members += [(int(b), channel["name"]) for b in channel["malecns"]["body_ids"]]
    elif version == "motor-bridge-v1":
        for channel in policy["channels"]:
            for group in ("flexor", "extensor"):
                ids = channel["male_cns"][group]["body_ids"]
                members += [(int(b), f"{channel['name']}:{group}") for b in ids]
    else:
        raise ValueError(f"Unsupported bridge profile: {path}")
    if len({b for b, _ in members}) != len(members):
        raise ValueError(f"Duplicate body IDs in {path}")
    return policy, members


def resolve(graph, members, name):
    try:
        return np.array([graph.node_index(b) for b, _ in members], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"{name} body ID {exc.args[0]} is not in the graph") from None


def node_metadata(graph, indices) -> dict[str, list]:
    taken = graph.nodes.take(pa.array(np.asarray(indices, dtype=np.int64)))
    # Columns absent from a graph's released metadata are reported as missing, not inferred.
    return {
        name: taken.column(name).to_pylist() if name in taken.column_names else [None] * len(taken)
        for name in METADATA
    }


def side_of(meta, i):
    return meta["somaSide"][i] or meta["rootSide"][i]


def count_int(value: float) -> int:
    if value >= 2**53:
        raise ValueError("Walk count exceeds exact float64 range; lower --max-hops")
    return int(value)


def run_audit(
    graph_dir,
    sensory_config,
    motor_config,
    lif_config,
    output,
    *,
    max_hops=5,
    enumerate_hops=3,
    edge_hops=4,
    path_limit=2_000_000,
):
    if not 1 <= max_hops <= MAX_HOPS_LIMIT:
        raise ValueError(f"--max-hops must be in [1, {MAX_HOPS_LIMIT}]")
    if not 1 <= enumerate_hops <= min(max_hops, 4):
        raise ValueError("--enumerate-hops must be in [1, min(max_hops, 4)]")
    if not 1 <= edge_hops <= max_hops:
        raise ValueError("--edge-hops must be in [1, max_hops]")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    sensory_policy, sensory_members = population_from_config(sensory_config)
    motor_policy, motor_members = population_from_config(motor_config)
    if sensory_policy["mapping_version"] != "sensory-bridge-v1":
        raise ValueError("--sensory-config must be a sensory bridge profile")
    if motor_policy["mapping_version"] != "motor-bridge-v1":
        raise ValueError("--motor-config must be a motor bridge profile")
    config = load_config(lif_config)

    with CNSGraph.load(graph_dir) as graph, stage_output(output) as stage:
        conn = Connectivity.from_graph(graph)
        sources = resolve(graph, sensory_members, "Sensory")
        targets = resolve(graph, motor_members, "Motor")
        if set(sources.tolist()) & set(targets.tolist()):
            raise ValueError("Sensory and motor populations overlap")
        labels = {int(n): lab for n, (_, lab) in zip(sources, sensory_members)}
        labels |= {int(n): lab for n, (_, lab) in zip(targets, motor_members)}
        nts = graph.nodes.column("consensus_nt").to_pylist()
        signs, sign_report = transmitter_signs(nts, config)
        body_ids = np.asarray(graph.body_ids)

        from_sensory = hop_distances(conn, sources, max_hops)
        to_motor = hop_distances(conn, targets, max_hops, reverse=True)
        through = from_sensory.astype(np.int32) + to_motor
        on_path = (from_sensory != UNREACHED) & (to_motor != UNREACHED) & (through <= max_hops)
        counts = walk_counts(conn, signs, sources, targets, max_hops)

        # nodes.parquet: every neuron on at least one path of <= max_hops synapses.
        node_idx = np.flatnonzero(on_path)
        meta = node_metadata(graph, node_idx)
        is_source = np.isin(node_idx, sources)
        is_target = np.isin(node_idx, targets)
        role = np.where(is_source, "sensory", np.where(is_target, "motor", "intermediate"))
        nodes_table = pa.table(
            {
                "node_index": pa.array(node_idx, pa.int32()),
                "body_id": pa.array(body_ids[node_idx], pa.uint64()),
                "population": pa.array(role.tolist(), pa.string()),
                "bridge_label": pa.array([labels.get(n) for n in node_idx.tolist()], pa.string()),
                "type": pa.array(meta["type"], pa.string()),
                "superclass": pa.array(meta["superclass"], pa.string()),
                "class": pa.array(meta["class"], pa.string()),
                "side": pa.array([side_of(meta, i) for i in range(len(node_idx))], pa.string()),
                "consensus_nt": pa.array(meta["consensus_nt"], pa.string()),
                "model_sign": pa.array(signs[node_idx], pa.int8()),
                "hops_from_sensory": pa.array(from_sensory[node_idx], pa.int16()),
                "hops_to_motor": pa.array(to_motor[node_idx], pa.int16()),
                "shortest_path_hops_through": pa.array(through[node_idx], pa.int16()),
            }
        )
        pq.write_table(nodes_table, stage / "nodes.parquet")

        # edges.parquet: synapses lying on some path of <= edge_hops synapses.
        src, dst = conn.src, conn.dst
        f, b = from_sensory[src], to_motor[dst]
        edge_mask = (f != UNREACHED) & (b != UNREACHED) & (f.astype(np.int32) + 1 + b <= edge_hops)
        e_src, e_dst = np.asarray(src[edge_mask]), np.asarray(dst[edge_mask])
        e_contacts = np.asarray(conn.contacts[edge_mask])
        e_signs = signs[e_src]
        edges_table = pa.table(
            {
                "src_node": pa.array(e_src, pa.int32()),
                "dst_node": pa.array(e_dst, pa.int32()),
                "src_body_id": pa.array(body_ids[e_src], pa.uint64()),
                "dst_body_id": pa.array(body_ids[e_dst], pa.uint64()),
                "contacts": pa.array(e_contacts, pa.uint32()),
                "src_model_sign": pa.array(e_signs, pa.int8()),
                "weight_mv": pa.array(e_contacts * e_signs * config.weight_per_contact_mv),
                "shortest_path_hops": pa.array(
                    from_sensory[e_src].astype(np.int16) + 1 + to_motor[e_dst], pa.int16()
                ),
            }
        )
        pq.write_table(edges_table, stage / "edges.parquet")

        # paths.parquet: every walk of <= enumerate_hops synapses, listed explicitly.
        rows = {k: [] for k in ("nodes", "contacts")}
        for path, contacts in enumerate_paths(
            conn, sources, targets, enumerate_hops, limit=path_limit
        ):
            rows["nodes"].append(path)
            rows["contacts"].append(contacts)
        path_nodes = rows["nodes"]
        used = sorted({n for p in path_nodes for n in p})
        used_meta = node_metadata(graph, used)
        position = {n: i for i, n in enumerate(used)}
        path_signs = [[int(signs[n]) for n in p] for p in path_nodes]
        classes = [sign_class(s[:-1]) for s in path_signs]
        path_nts = [[used_meta["consensus_nt"][position[n]] for n in p] for p in path_nodes]
        hops = [len(p) - 1 for p in path_nodes]
        paths_table = pa.table(
            {
                "source_node": pa.array([p[0] for p in path_nodes], pa.int32()),
                "target_node": pa.array([p[-1] for p in path_nodes], pa.int32()),
                "source_body_id": pa.array([int(body_ids[p[0]]) for p in path_nodes], pa.uint64()),
                "target_body_id": pa.array([int(body_ids[p[-1]]) for p in path_nodes], pa.uint64()),
                "target_label": pa.array([labels[p[-1]] for p in path_nodes], pa.string()),
                "hops": pa.array(hops, pa.int8()),
                "node_path": pa.array(path_nodes, pa.list_(pa.int32())),
                "body_id_path": pa.array(
                    [[int(body_ids[n]) for n in p] for p in path_nodes], pa.list_(pa.uint64())
                ),
                "type_path": pa.array(
                    [[used_meta["type"][position[n]] for n in p] for p in path_nodes],
                    pa.list_(pa.string()),
                ),
                "side_path": pa.array(
                    [[side_of(used_meta, position[n]) for n in p] for p in path_nodes],
                    pa.list_(pa.string()),
                ),
                "nt_path": pa.array(path_nts, pa.list_(pa.string())),
                "sign_path": pa.array(path_signs, pa.list_(pa.int8())),
                "contacts_path": pa.array(rows["contacts"], pa.list_(pa.uint32())),
                "weight_mv_path": pa.array(
                    [
                        [c * s * config.weight_per_contact_mv for c, s in zip(cs, ss)]
                        for cs, ss in zip(rows["contacts"], path_signs)
                    ],
                    pa.list_(pa.float64()),
                ),
                "bottleneck_contacts": pa.array([min(c) for c in rows["contacts"]], pa.uint32()),
                "sign_class": pa.array(classes, pa.string()),
                "simple": pa.array([len(set(p)) == len(p) for p in path_nodes]),
                "synaptic_delay_lower_bound_ms": pa.array(
                    [h * config.synaptic_delay_ms for h in hops], pa.float64()
                ),
            }
        )
        pq.write_table(paths_table, stage / "paths.parquet")

        # Cross-check: explicit enumeration must equal propagated walk counts.
        for h in range(1, enumerate_hops + 1):
            listed = sum(1 for x in hops if x == h)
            if listed != count_int(counts[h - 1].sum()):
                raise RuntimeError(f"Enumerated {listed} paths of {h} hops; counts disagree")

        summary = summarize(
            graph,
            conn,
            signs,
            sources,
            targets,
            labels,
            counts,
            nodes_table,
            paths_table,
            config,
            max_hops,
            enumerate_hops,
            from_sensory,
            to_motor,
        )
        write_json(stage / "pathway_summary.json", summary)

        artifacts = {
            name: file_hashes(stage / name)
            for name in ("nodes.parquet", "edges.parquet", "paths.parquet", "pathway_summary.json")
        }
        report = {
            "schema": 1,
            "kind": "sensory-motor-pathway-audit",
            "diagnostic_only": True,
            "parameters": {
                "max_hops": max_hops,
                "enumerate_hops": enumerate_hops,
                "edge_hops": edge_hops,
                "path_limit": path_limit,
            },
            "definitions": {
                "hops": "number of synapses; a path of h hops has h-1 intermediate neurons",
                "multiplicity": "walk counts: a neuron may repeat; see paths.simple",
                "sign": "presynaptic neurons only (source and intermediates), as in the LIF",
                "latency": "hops * synaptic_delay_ms; excludes membrane integration time",
            },
            "graph_manifest": file_hashes(Path(graph_dir) / "manifest.json"),
            "sensory_config": {
                "path": Path(sensory_config).as_posix(),
                "sha256": file_hashes(Path(sensory_config))["sha256"],
                "mapping_version": sensory_policy["mapping_version"],
            },
            "motor_config": {
                "path": Path(motor_config).as_posix(),
                "sha256": file_hashes(Path(motor_config))["sha256"],
                "mapping_version": motor_policy["mapping_version"],
            },
            "lif_config": config.to_dict(),
            "lif_config_sha256": digest(config.to_dict()),
            "sign_report": {k: v for k, v in sign_report.items() if k != "reason"},
            "versions": {
                n: importlib.metadata.version(n) for n in ("malecns-sim", "numpy", "pyarrow")
            },
            "rows": {
                "nodes": nodes_table.num_rows,
                "edges": edges_table.num_rows,
                "paths": paths_table.num_rows,
            },
            "artifacts": artifacts,
        }
        write_json(stage / "report.json", report)
    return summary


def summarize(
    graph,
    conn,
    signs,
    sources,
    targets,
    labels,
    counts,
    nodes_table,
    paths_table,
    config,
    max_hops,
    enumerate_hops,
    from_sensory,
    to_motor,
):
    body_ids = np.asarray(graph.body_ids)

    def population(indices):
        meta = node_metadata(graph, indices)
        return [
            {
                "node_index": int(n),
                "body_id": int(body_ids[n]),
                "label": labels[int(n)],
                "type": meta["type"][i],
                "side": side_of(meta, i),
                "consensus_nt": meta["consensus_nt"][i],
                "model_sign": int(signs[n]),
            }
            for i, n in enumerate(indices.tolist())
        ]

    per_hop = []
    for h in range(max_hops):
        by_class = {c: count_int(counts[h, k].sum()) for k, c in enumerate(SIGN_CLASSES)}
        per_target = {
            str(int(body_ids[t])): {
                c: count_int(counts[h, k, j]) for k, c in enumerate(SIGN_CLASSES)
            }
            for j, t in enumerate(targets.tolist())
        }
        per_hop.append(
            {
                "hops": h + 1,
                "synaptic_delay_lower_bound_ms": (h + 1) * config.synaptic_delay_ms,
                "walks": sum(by_class.values()),
                "walks_by_sign_class": by_class,
                "walks_by_target_body_id": per_target,
            }
        )

    # Intermediate neurons grouped by the shortest path they lie on.
    frame = nodes_table.to_pandas()
    mid = frame[frame.population == "intermediate"]
    layers = []
    for h, group in mid.groupby("shortest_path_hops_through"):
        layers.append(
            {
                "shortest_path_hops": int(h),
                "neurons": int(len(group)),
                "by_consensus_nt": _counts(group.consensus_nt),
                "by_model_sign": {
                    str(k): int(v) for k, v in group.model_sign.value_counts().items()
                },
                "by_superclass": _counts(group.superclass),
                "by_side": _counts(group.side),
                "top_types": _counts(group.type, 15),
            }
        )

    path_hops = paths_table.column("hops").to_pylist()
    path_classes = paths_table.column("sign_class").to_pylist()
    path_simple = paths_table.column("simple").to_pylist()
    path_nts = paths_table.column("nt_path").to_pylist()
    listed = []
    for h in range(1, enumerate_hops + 1):
        sel = [i for i, x in enumerate(path_hops) if x == h]
        classes = Counter(path_classes[i] for i in sel)
        # Presynaptic transmitters only: the target's own transmitter does not sign the path.
        nt_seqs = Counter(" > ".join(str(x) for x in path_nts[i][:-1]) for i in sel)
        listed.append(
            {
                "hops": h,
                "paths": len(sel),
                "simple_paths": sum(path_simple[i] for i in sel),
                "by_sign_class": dict(sorted(classes.items())),
                "top_presynaptic_nt_sequences": dict(nt_seqs.most_common(10)),
            }
        )

    drive = direct_drive_estimate(conn, signs, sources, targets, config)
    for row in drive:
        row["target_body_id"] = int(body_ids[row["target_node"]])
        row["target_label"] = labels[row["target_node"]]
    zero_mid = int((mid.model_sign == 0).sum())
    return {
        "question": (
            "Which anatomical paths connect the sensory population to the mapped motor "
            "population, and how does the current LIF sign policy classify them?"
        ),
        "diagnostic_only": True,
        "sensory_population": population(sources),
        "motor_population": population(targets),
        "reachability": {
            "motor_reached_from_sensory_within_max_hops": int(
                np.sum(from_sensory[targets] != UNREACHED)
            ),
            "shortest_hops_by_target_body_id": {
                str(int(body_ids[t])): int(from_sensory[t]) for t in targets.tolist()
            },
            "neurons_within_hops_from_sensory": {
                str(h): int(np.sum(from_sensory == h)) for h in range(max_hops + 1)
            },
            "neurons_within_hops_to_motor": {
                str(h): int(np.sum(to_motor == h)) for h in range(max_hops + 1)
            },
        },
        "walks_by_hops": per_hop,
        "intermediate_layers": layers,
        "intermediate_zero_sign_neurons": zero_mid,
        "intermediate_neurons": int(len(mid)),
        "enumerated_paths": listed,
        "direct_drive_estimate": {
            "assumption": (
                "Linear subthreshold estimate from the shiu-lif-v1 equations: mean "
                "depolarization = direct_weight_mv * rate * tau_synapse. Ignores indirect "
                "paths, inhibition from elsewhere, target refractoriness and fluctuations."
            ),
            "psp_peak_per_mv_of_g": psp_peak_per_mv(config.tau_membrane_ms, config.tau_synapse_ms),
            "max_source_rate_under_refractory_hz": 1000 / config.refractory_ms,
            "targets": drive,
        },
        "unknown_nt_policy": config.unknown_nt_policy,
    }


def _counts(series, top=None):
    values = series.fillna("<missing>").value_counts()
    if top is not None:
        values = values.head(top)
    return {str(k): int(v) for k, v in values.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument(
        "--sensory-config", type=Path, default=Path("configs/bridge/sensory_lf_tibia_v1.yaml")
    )
    parser.add_argument(
        "--motor-config", type=Path, default=Path("configs/bridge/lf_tibia_v1.yaml")
    )
    parser.add_argument("--lif-config", type=Path, default=Path("configs/neural/shiu_lif_v1.yaml"))
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--enumerate-hops", type=int, default=3)
    parser.add_argument("--edge-hops", type=int, default=4)
    parser.add_argument("--path-limit", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_audit(
        args.graph,
        args.sensory_config,
        args.motor_config,
        args.lif_config,
        args.output,
        max_hops=args.max_hops,
        enumerate_hops=args.enumerate_hops,
        edge_hops=args.edge_hops,
        path_limit=args.path_limit,
    )
    print(
        json.dumps(
            {
                "reachability": summary["reachability"]["shortest_hops_by_target_body_id"],
                "walks_by_hops": [
                    {"hops": r["hops"], **r["walks_by_sign_class"]}
                    for r in summary["walks_by_hops"]
                ],
                "direct_drive_estimate": summary["direct_drive_estimate"]["targets"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
