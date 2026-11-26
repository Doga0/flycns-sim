"""Combine a pathway audit and an activation sweep into one latency/connectivity report.

Read-only: it re-derives nothing from simulation, only joins saved spikes with the graph
and the LIF sign policy to show where each motor neuron's synaptic drive came from.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from malecns_sim.analysis.pathway_audit import DEFAULT_GRAPH
from malecns_sim.bridge.store import stage_output
from malecns_sim.cns.graph.store import CNSGraph, file_hashes, write_json
from malecns_sim.neural.config import load_config
from malecns_sim.neural.signs import transmitter_signs


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_inputs(audit_dir: Path, sweep_dir: Path, graph_dir: Path, config) -> tuple[dict, dict]:
    """Both inputs must come from the same graph and LIF config as this report."""
    audit, sweep = read_json(audit_dir / "report.json"), read_json(sweep_dir / "report.json")
    if audit.get("kind") != "sensory-motor-pathway-audit":
        raise ValueError(f"Not a pathway audit: {audit_dir}")
    if sweep.get("kind") != "sensory-motor-activation-sweep":
        raise ValueError(f"Not an activation sweep: {sweep_dir}")
    graph_hash = file_hashes(Path(graph_dir) / "manifest.json")["sha256"]
    lif_hash = digest(config.to_dict())
    for name, report in (("audit", audit), ("sweep", sweep)):
        if report["graph_manifest"]["sha256"] != graph_hash:
            raise ValueError(f"The {name} used a different graph")
        if report["lif_config_sha256"] != lif_hash:
            raise ValueError(f"The {name} used a different LIF config")
    for directory, report in ((audit_dir, audit), (sweep_dir, sweep)):
        for name, expected in report["artifacts"].items():
            if file_hashes(directory / name)["sha256"] != expected["sha256"]:
                raise ValueError(f"Artifact changed since it was written: {directory / name}")
    return audit, sweep


def incoming_drive(graph, signs, config, targets, spikes, layer_of) -> pd.DataFrame:
    """Per motor neuron: summed weight x presynaptic spikes, split by sign, NT and layer.

    Units are mV of g-increment summed over the run (sum of w over every presynaptic spike).
    A bookkeeping measure of delivered drive, not a membrane-potential prediction.
    """
    src, dst = np.asarray(graph.arrays["src"]), np.asarray(graph.arrays["dst"])
    mask = np.isin(dst, targets)
    pre, post = src[mask], dst[mask]
    contacts = np.asarray(graph.arrays["contacts"])[mask].astype(np.int64)
    counts = np.bincount(spikes, minlength=len(graph.body_ids))
    nts = np.asarray(graph.nodes.column("consensus_nt").to_pylist(), dtype=object)
    weight = contacts * signs[pre] * config.weight_per_contact_mv
    frame = pd.DataFrame(
        {
            "target_node": post,
            "pre_node": pre,
            "pre_nt": [x if x is not None else "<missing>" for x in nts[pre]],
            "pre_layer": [layer_of.get(int(n), "not_on_path") for n in pre],
            "weight_mv": weight,
            "pre_spikes": counts[pre],
        }
    )
    frame["delivered_mv"] = frame.weight_mv * frame.pre_spikes
    return frame


def run_report(audit_dir, sweep_dir, graph_dir, lif_config, output):
    audit_dir, sweep_dir, output = Path(audit_dir), Path(sweep_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = load_config(lif_config)
    audit, sweep_report = check_inputs(audit_dir, sweep_dir, Path(graph_dir), config)
    summary_a = read_json(audit_dir / "pathway_summary.json")
    summary_s = read_json(sweep_dir / "sweep_summary.json")
    nodes = pq.read_table(audit_dir / "nodes.parquet").to_pandas()
    targets = nodes[nodes.population == "motor"].node_index.to_numpy()
    sources = nodes[nodes.population == "sensory"].node_index.to_numpy()
    layer_of = {
        int(n): ("sensory" if p == "sensory" else f"hop_{h}")
        for n, p, h in zip(nodes.node_index, nodes.population, nodes.hops_from_sensory)
    }
    conditions = [c["condition"]["name"] for c in summary_s["conditions"]]

    with CNSGraph.load(graph_dir) as graph, stage_output(output) as stage:
        body_ids = np.asarray(graph.body_ids)
        nts = graph.nodes.column("consensus_nt").to_pylist()
        signs, _ = transmitter_signs(nts, config)
        types = graph.nodes.column("type").to_pylist()
        labels = {int(m["node_index"]): m["label"] for m in summary_a["motor_population"]}
        drive_rows, active_rows = [], []
        target_set = set(targets.tolist())
        source_set = set(sources.tolist())
        motor_edges = {}
        src, dst = np.asarray(graph.arrays["src"]), np.asarray(graph.arrays["dst"])
        to_motor = np.isin(dst, targets)
        for s, d in zip(src[to_motor].tolist(), dst[to_motor].tolist()):
            motor_edges.setdefault(s, set()).add(d)
        for name in conditions:
            spikes = pq.read_table(sweep_dir / "spikes" / f"{name}.parquet").to_pandas()
            frame = incoming_drive(
                graph, signs, config, targets, spikes.node_index.to_numpy(), layer_of
            )
            for target, group in frame.groupby("target_node"):
                active = group[group.pre_spikes > 0]
                row = {
                    "condition": name,
                    "target_node": int(target),
                    "target_body_id": int(body_ids[target]),
                    "target_label": labels[int(target)],
                    "presynaptic_partners": int(len(group)),
                    "active_presynaptic_partners": int(len(active)),
                    "excitatory_mv": float(active.delivered_mv.clip(lower=0).sum()),
                    "inhibitory_mv": float(active.delivered_mv.clip(upper=0).sum()),
                }
                for key, part in active.groupby("pre_layer"):
                    row[f"net_mv_from_{key}"] = float(part.delivered_mv.sum())
                for key, part in active[active.delivered_mv < 0].groupby("pre_nt"):
                    row[f"inhibitory_mv_from_{key}"] = float(part.delivered_mv.sum())
                drive_rows.append(row)
            counts = spikes[~spikes.node_index.isin(source_set | target_set)]
            counts = counts.node_index.value_counts()
            for node, n in counts.items():
                node = int(node)
                active_rows.append(
                    {
                        "condition": name,
                        "node_index": node,
                        "body_id": int(body_ids[node]),
                        "type": types[node],
                        "consensus_nt": nts[node],
                        "model_sign": int(signs[node]),
                        "layer": layer_of.get(node, "not_on_path"),
                        "spikes": int(n),
                        "synapses_onto_mapped_motor": sorted(
                            int(body_ids[t]) for t in motor_edges.get(node, ())
                        ),
                    }
                )
        drive = pd.DataFrame(drive_rows).fillna(0.0)
        active = pd.DataFrame(active_rows)
        pq.write_table(
            pa.Table.from_pandas(drive, preserve_index=False), stage / "incoming_drive.parquet"
        )
        pq.write_table(
            pa.Table.from_pandas(active, preserve_index=False), stage / "active_neurons.parquet"
        )

        benchmark = build_benchmark(summary_a, summary_s, drive, body_ids, targets, labels)
        write_json(stage / "pathway_report.json", benchmark)
        (stage / "pathway_report.md").write_text(
            render_markdown(benchmark, active, summary_s), encoding="utf-8", newline="\n"
        )
        artifacts = {
            name: file_hashes(stage / name)
            for name in (
                "incoming_drive.parquet",
                "active_neurons.parquet",
                "pathway_report.json",
                "pathway_report.md",
            )
        }
        write_json(
            stage / "report.json",
            {
                "schema": 1,
                "kind": "sensory-motor-pathway-report",
                "diagnostic_only": True,
                "inputs": {
                    "audit": {
                        "path": audit_dir.as_posix(),
                        "report": file_hashes(audit_dir / "report.json"),
                    },
                    "sweep": {
                        "path": sweep_dir.as_posix(),
                        "report": file_hashes(sweep_dir / "report.json"),
                    },
                },
                "graph_manifest": audit["graph_manifest"],
                "lif_config_sha256": sweep_report["lif_config_sha256"],
                "definitions": {
                    "delivered_mv": (
                        "sum over presynaptic spikes of contacts * model sign * "
                        "weight_per_contact_mv; bookkeeping of drive into g, not a voltage"
                    ),
                    "layer": "hops from the sensory population in the pathway audit",
                },
                "versions": {
                    n: importlib.metadata.version(n) for n in ("malecns-sim", "numpy", "pandas")
                },
                "artifacts": artifacts,
            },
        )
    return benchmark


def build_benchmark(summary_a, summary_s, drive, body_ids, targets, labels):
    walks = {r["hops"]: r for r in summary_a["walks_by_hops"]}
    direct = {r["target_body_id"]: r for r in summary_a["direct_drive_estimate"]["targets"]}
    strongest = [c for c in summary_s["conditions"] if c["condition"]["type"] == "poisson"]
    strongest = max(strongest, key=lambda c: c["condition"]["rate_hz"])["condition"]["name"]
    motors = []
    for target in targets.tolist():
        key = str(int(body_ids[target]))
        at_max = drive[(drive.condition == strongest) & (drive.target_node == target)].iloc[0]
        motors.append(
            {
                "body_id": int(key),
                "label": labels[int(target)],
                "direct_contacts": direct[int(key)]["direct_contacts"],
                "direct_sources": direct[int(key)]["direct_sources"],
                "shortest_path_hops": summary_a["reachability"]["shortest_hops_by_target_body_id"][
                    key
                ],
                "walks_by_hops": {
                    str(h): sum(walks[h]["walks_by_target_body_id"][key].values()) for h in walks
                },
                "minimum_tested_poisson_rate_hz": summary_s[
                    "minimum_tested_poisson_rate_hz_by_motor"
                ][key],
                "spikes_by_condition": {c: v[key] for c, v in summary_s["motor_spikes"].items()},
                "first_spike_latency_ms_by_condition": {
                    c: v[key] for c, v in summary_s["motor_first_spike_latency_ms"].items()
                },
                "drive_at_strongest_input": {
                    "condition": strongest,
                    "excitatory_mv": float(at_max.excitatory_mv),
                    "inhibitory_mv": float(at_max.inhibitory_mv),
                    "net_mv": float(at_max.excitatory_mv + at_max.inhibitory_mv),
                    "active_presynaptic_partners": int(at_max.active_presynaptic_partners),
                    "inhibitory_mv_by_presynaptic_nt": {
                        k.removeprefix("inhibitory_mv_from_"): float(v)
                        for k, v in at_max.items()
                        if k.startswith("inhibitory_mv_from_") and v
                    },
                    "net_mv_by_presynaptic_layer": {
                        k.removeprefix("net_mv_from_"): float(v)
                        for k, v in at_max.items()
                        if k.startswith("net_mv_from_") and v
                    },
                },
            }
        )
    layers = summary_s["layer_activity"][strongest]
    return {
        "question": (
            "Given this sensory population and this motor population, under the current "
            "MaleCNS graph and LIF model, what input is required to activate the motor "
            "population?"
        ),
        "diagnostic_only": True,
        "minimum_tested_poisson_rate_hz_any_motor": summary_s[
            "minimum_tested_poisson_rate_hz_any_motor"
        ],
        "active_motor_neurons_at_any_input": sorted(
            m["body_id"] for m in motors if m["minimum_tested_poisson_rate_hz"] is not None
        ),
        "deepest_active_layer_at_strongest_input": max(
            (r["hops_from_sensory"] for r in layers if r["spikes"] and r["hops_from_sensory"]),
            default=0,
        ),
        "strongest_condition": strongest,
        "motors": motors,
    }


def render_markdown(benchmark, active, summary_s):
    strongest = benchmark["strongest_condition"]
    active_motors = benchmark["active_motor_neurons_at_any_input"]
    depth = benchmark["deepest_active_layer_at_strongest_input"]
    lines = [
        "# LF tibia sensory→motor pathway report (generated)",
        "",
        "Diagnostic only; generated by `malecns_sim.analysis.pathway_report`.",
        "",
        f"- Minimum tested Poisson rate activating any mapped motor neuron: "
        f"{benchmark['minimum_tested_poisson_rate_hz_any_motor']} Hz (nominal).",
        f"- Motor neurons active at any tested input: {active_motors}.",
        f"- Deepest active layer at {strongest}: {depth} synapses from the sensory population.",
        "",
        "| Motor | Label | Direct contacts | Shortest hops | Min rate (Hz) | Spikes at max | "
        "Latency at max (ms) | Excitatory mV | Inhibitory mV | Net mV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in benchmark["motors"]:
        d = m["drive_at_strongest_input"]
        latency = m["first_spike_latency_ms_by_condition"][strongest]
        lines.append(
            f"| {m['body_id']} | {m['label']} | {m['direct_contacts']} | "
            f"{m['shortest_path_hops']} | {m['minimum_tested_poisson_rate_hz'] or '—'} | "
            f"{m['spikes_by_condition'][strongest]} | "
            f"{'—' if latency is None else f'{latency:.1f}'} | {d['excitatory_mv']:.0f} | "
            f"{d['inhibitory_mv']:.0f} | {d['net_mv']:.0f} |"
        )
    lines += ["", f"## Most active non-sensory, non-motor neurons at {strongest}", ""]
    lines += ["| Body ID | Type | NT | Sign | Layer | Spikes | Synapses onto mapped motor |"]
    lines += ["| --- | --- | --- | ---: | --- | ---: | --- |"]
    top = active[active.condition == strongest].sort_values(
        ["spikes", "body_id"], ascending=[False, True]
    )
    for _, r in top.head(20).iterrows():
        lines.append(
            f"| {r.body_id} | {r.type} | {r.consensus_nt} | {r.model_sign} | {r.layer} | "
            f"{r.spikes} | {', '.join(map(str, r.synapses_onto_mapped_motor)) or '—'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("outputs/analysis/lf_tibia_pathway"))
    parser.add_argument(
        "--sweep", type=Path, default=Path("outputs/analysis/lf_tibia_activation_sweep")
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--lif-config", type=Path, default=Path("configs/neural/shiu_lif_v1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    benchmark = run_report(args.audit, args.sweep, args.graph, args.lif_config, args.output)
    print((args.output / "pathway_report.md").read_text(encoding="utf-8"))
    print(json.dumps({"motors": len(benchmark["motors"])}))


if __name__ == "__main__":
    main()
