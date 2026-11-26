"""Drive the mapped sensory population directly and measure mapped motor activation.

Full LIF network, no body: answers what input the current graph and shiu-lif-v1 policy
need before the mapped motor population fires. Diagnostic only; inputs are engineering
probes and the results are not biological firing rates.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from malecns_sim.analysis.pathway_audit import DEFAULT_GRAPH, population_from_config, resolve
from malecns_sim.analysis.sensory_motor_paths import UNREACHED, Connectivity, hop_distances
from malecns_sim.bridge.store import stage_output
from malecns_sim.cns.graph.store import CNSGraph, file_hashes, write_json
from malecns_sim.cns.graph.validate import validate_graph
from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.config import load_config
from malecns_sim.neural.stimulus import Stimulus

DEFAULT_RATES = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
DEFAULT_VOLLEYS = (1, 10)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_conditions(rates, volleys, *, onset_ms, drive_ms, volley_interval_ms):
    """Name → stimulus parameters. Every condition shares the same timing and population."""
    conditions = [{"name": "no_input", "type": None}]
    for count in volleys:
        if count < 1 or (count - 1) * volley_interval_ms >= drive_ms:
            raise ValueError("Volleys must fit inside the drive window")
        conditions.append(
            {
                "name": f"volley_{count}",
                "type": "spikes",
                "volleys": count,
                "interval_ms": volley_interval_ms,
                "times_ms": [onset_ms + i * volley_interval_ms for i in range(count)],
            }
        )
    for rate in rates:
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Rates must be positive")
        conditions.append({"name": f"poisson_{rate:g}hz", "type": "poisson", "rate_hz": rate})
    if len({c["name"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate sweep conditions")
    return conditions


def select_subset(members, subset):
    """Keep only explicitly listed profile body IDs; None keeps the whole population."""
    if subset is None:
        return members
    subset = [int(b) for b in subset]
    if not subset or len(set(subset)) != len(subset):
        raise ValueError("Sensory subset must list distinct body IDs")
    known = {b for b, _ in members}
    missing = [b for b in subset if b not in known]
    if missing:
        raise ValueError(f"Sensory subset IDs not in the sensory profile: {missing}")
    wanted = set(subset)
    return [m for m in members if m[0] in wanted]


def stimulus_for(condition, sources, gain_mv, onset_ms, stop_ms):
    if condition["type"] is None:
        return None
    nodes = tuple(sorted(int(n) for n in sources))
    if condition["type"] == "spikes":
        return Stimulus(
            nodes,
            type="spikes",
            rate_hz=0.0,
            start_ms=onset_ms,
            stop_ms=stop_ms,
            gain_mv=gain_mv,
            times_ms=tuple(condition["times_ms"]),
        )
    return Stimulus(
        nodes,
        rate_hz=condition["rate_hz"],
        start_ms=onset_ms,
        stop_ms=stop_ms,
        gain_mv=gain_mv,
    )


def measure(times, nodes, *, sources, targets, target_labels, body_ids, layer, onset_ms, end_ms):
    """Condition metrics from one run's spikes (times in ms, parent node indices)."""
    after = times >= onset_ms
    window_s = (end_ms - onset_ms) / 1000
    is_source = np.isin(nodes, sources)
    motor = []
    for target in targets.tolist():
        hit = nodes == target
        post = times[hit & after]
        motor.append(
            {
                "target_node": int(target),
                "target_body_id": int(body_ids[target]),
                "target_label": target_labels[int(target)],
                "spikes": int(len(post)),
                "spikes_before_onset": int(np.sum(hit & ~after)),
                "rate_hz": len(post) / window_s,
                "first_spike_latency_ms": float(post.min() - onset_ms) if len(post) else None,
            }
        )
    layers = []
    for h in sorted(set(layer.tolist())):
        members = layer[nodes[after]] == h
        layers.append(
            {
                "hops_from_sensory": int(h) if h != UNREACHED else None,
                "neurons_in_layer": int(np.sum(layer == h)),
                "active_neurons": int(len(np.unique(nodes[after][members]))),
                "spikes": int(np.sum(members)),
            }
        )
    first_motor = [m["first_spike_latency_ms"] for m in motor if m["spikes"]]
    return {
        "sensory_spikes": int(np.sum(is_source & after)),
        "sensory_active_cells": int(len(np.unique(nodes[is_source & after]))),
        "sensory_mean_rate_per_cell_hz": float(np.sum(is_source & after)) / len(sources) / window_s,
        "total_spikes": int(len(times)),
        "spikes_before_onset": int(np.sum(~after)),
        "active_neurons": int(len(np.unique(nodes))),
        "mapped_motor_spikes": int(sum(m["spikes"] for m in motor)),
        "active_mapped_motor_neurons": int(sum(m["spikes"] > 0 for m in motor)),
        "first_motor_latency_ms": min(first_motor) if first_motor else None,
        "motor": motor,
        "layers": layers,
    }


def run_sweep(
    graph_dir,
    sensory_config,
    motor_config,
    lif_config,
    output,
    *,
    rates=DEFAULT_RATES,
    volleys=DEFAULT_VOLLEYS,
    volley_interval_ms=3.0,
    baseline_ms=50.0,
    drive_ms=200.0,
    tail_ms=50.0,
    seed=42,
    max_hops=5,
    validate=True,
    sensory_subset=None,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = load_config(lif_config)
    for name, value in (("baseline", baseline_ms), ("drive", drive_ms), ("tail", tail_ms)):
        if value < 0 or (name == "drive" and value <= 0):
            raise ValueError(f"Invalid {name} duration")
        config.steps(value)
    onset, stop = baseline_ms, baseline_ms + drive_ms
    total = stop + tail_ms
    conditions = build_conditions(
        rates, volleys, onset_ms=onset, drive_ms=drive_ms, volley_interval_ms=volley_interval_ms
    )
    sensory_policy, sensory_members = population_from_config(sensory_config)
    sensory_members = select_subset(sensory_members, sensory_subset)
    motor_policy, motor_members = population_from_config(motor_config)
    if sensory_policy["mapping_version"] != "sensory-bridge-v1":
        raise ValueError("--sensory-config must be a sensory bridge profile")
    if motor_policy["mapping_version"] != "motor-bridge-v1":
        raise ValueError("--motor-config must be a motor bridge profile")
    gains = {c["encoder"]["gain_mv"] for c in sensory_policy["channels"]}
    if len(gains) != 1:
        raise ValueError("Sweep needs one shared sensory input gain")
    gain_mv = float(gains.pop())

    with CNSGraph.load(graph_dir) as stored, stage_output(output) as stage:
        if validate:
            validate_graph(stored)
        sources = resolve(stored, sensory_members, "Sensory")
        targets = resolve(stored, motor_members, "Motor")
        if set(sources.tolist()) & set(targets.tolist()):
            raise ValueError("Sensory and motor populations overlap")
        labels = {int(n): lab for n, (_, lab) in zip(targets, motor_members)}
        layer = hop_distances(Connectivity.from_graph(stored), sources, max_hops)
        graph = NeuralGraph.from_graph(stored)
        body_ids = np.asarray(graph.body_ids)
        results, motor_rows, layer_rows = [], [], []
        (stage / "spikes").mkdir()
        for condition in conditions:
            began = time.perf_counter()
            print(f"Running {condition['name']} ({len(body_ids)} neurons)...", flush=True)
            backend = Brian2LIFBackend(graph, config, seed=seed)
            stimulus = stimulus_for(condition, sources, gain_mv, onset, stop)
            if stimulus is not None:
                backend.schedule(stimulus)
            backend.run(total)
            times, nodes = backend.get_spikes()
            nodes = graph.node_indices[nodes]
            pq.write_table(
                pa.table(
                    {
                        "t_ms": pa.array(times, pa.float64()),
                        "node_index": pa.array(nodes, pa.int32()),
                        "body_id": pa.array(body_ids[nodes], pa.uint64()),
                        "hops_from_sensory": pa.array(layer[nodes], pa.int16()),
                    }
                ),
                stage / "spikes" / f"{condition['name']}.parquet",
            )
            metrics = measure(
                times,
                nodes,
                sources=sources,
                targets=targets,
                target_labels=labels,
                body_ids=body_ids,
                layer=layer,
                onset_ms=onset,
                end_ms=total,
            )
            metrics["condition"] = condition
            metrics["wall_seconds"] = round(time.perf_counter() - began, 1)
            results.append(metrics)
            for m in metrics["motor"]:
                motor_rows.append({"condition": condition["name"], **m})
            for row in metrics["layers"]:
                layer_rows.append({"condition": condition["name"], **row})
            print(
                json.dumps(
                    {
                        k: metrics[k]
                        for k in (
                            "sensory_spikes",
                            "total_spikes",
                            "mapped_motor_spikes",
                            "active_mapped_motor_neurons",
                            "first_motor_latency_ms",
                            "wall_seconds",
                        )
                    }
                ),
                flush=True,
            )
            del backend
            gc.collect()

        pq.write_table(pa.Table.from_pylist(motor_rows), stage / "motor_matrix.parquet")
        pq.write_table(pa.Table.from_pylist(layer_rows), stage / "layer_activity.parquet")
        summary = summarize(results, targets, body_ids, labels)
        summary["timing_ms"] = {"onset": onset, "stop": stop, "total": total}
        write_json(stage / "sweep_summary.json", summary)
        artifacts = {
            p.relative_to(stage).as_posix(): file_hashes(p)
            for p in sorted(stage.rglob("*.parquet")) + [stage / "sweep_summary.json"]
        }
        write_json(
            stage / "report.json",
            {
                "schema": 1,
                "kind": "sensory-motor-activation-sweep",
                "diagnostic_only": True,
                "body_coupling": False,
                "parameters": {
                    "rates_hz": list(rates),
                    "volleys": list(volleys),
                    "volley_interval_ms": volley_interval_ms,
                    "baseline_ms": baseline_ms,
                    "drive_ms": drive_ms,
                    "tail_ms": tail_ms,
                    "seed": seed,
                    "sensory_gain_mv": gain_mv,
                    "max_hops": max_hops,
                    "sensory_subset_body_ids": (
                        None
                        if sensory_subset is None
                        else sorted(int(b) for b, _ in sensory_members)
                    ),
                },
                "definitions": {
                    "input": (
                        "external events added to v of every driven sensory cell (the whole "
                        "profile, or the explicit subset); hops are counted from driven cells"
                    ),
                    "poisson": "independent per-cell, per-step Poisson events in [onset, stop)",
                    "volley": "all mapped sensory cells receive one event at each volley time",
                    "latency": "first mapped-motor spike time minus onset",
                    "window": "metrics count spikes in [onset, total)",
                },
                "graph_manifest": file_hashes(Path(graph_dir) / "manifest.json"),
                "sensory_config_sha256": file_hashes(Path(sensory_config))["sha256"],
                "motor_config_sha256": file_hashes(Path(motor_config))["sha256"],
                "lif_config": config.to_dict(),
                "lif_config_sha256": digest(config.to_dict()),
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ("malecns-sim", "brian2", "numpy", "pyarrow")
                },
                "artifacts": artifacts,
            },
        )
    return summary


def summarize(results, targets, body_ids, labels):
    matrix = {
        r["condition"]["name"]: {str(m["target_body_id"]): m["spikes"] for m in r["motor"]}
        for r in results
    }
    latency = {
        r["condition"]["name"]: {
            str(m["target_body_id"]): m["first_spike_latency_ms"] for m in r["motor"]
        }
        for r in results
    }
    poisson = [r for r in results if r["condition"]["type"] == "poisson"]
    poisson.sort(key=lambda r: r["condition"]["rate_hz"])
    minimum = {}
    for target in targets.tolist():
        key = str(int(body_ids[target]))
        fired = [r["condition"]["rate_hz"] for r in poisson if matrix[r["condition"]["name"]][key]]
        minimum[key] = min(fired) if fired else None
    any_rate = [r["condition"]["rate_hz"] for r in poisson if r["mapped_motor_spikes"]]
    return {
        "question": (
            "What input to the mapped sensory population makes the mapped motor population "
            "fire in the full LIF network?"
        ),
        "diagnostic_only": True,
        "motor_labels": {str(int(body_ids[t])): labels[int(t)] for t in targets.tolist()},
        "conditions": [
            {k: v for k, v in r.items() if k not in ("motor", "layers")} for r in results
        ],
        "motor_spikes": matrix,
        "motor_first_spike_latency_ms": latency,
        "minimum_tested_poisson_rate_hz_by_motor": minimum,
        "minimum_tested_poisson_rate_hz_any_motor": min(any_rate) if any_rate else None,
        "layer_activity": {r["condition"]["name"]: r["layers"] for r in results},
    }


def _floats(text):
    return tuple(float(x) for x in text.split(",") if x.strip())


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
    parser.add_argument("--rates", type=_floats, default=DEFAULT_RATES, help="comma-separated Hz")
    parser.add_argument(
        "--volleys", type=lambda s: tuple(int(x) for x in s.split(",") if x), default=(1, 10)
    )
    parser.add_argument("--volley-interval-ms", type=float, default=3.0)
    parser.add_argument("--baseline-ms", type=float, default=50.0)
    parser.add_argument("--drive-ms", type=float, default=200.0)
    parser.add_argument("--tail-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sensory-subset",
        type=lambda s: tuple(int(x) for x in s.split(",") if x.strip()),
        default=None,
        help="comma-separated body IDs from the sensory profile to drive (default: all)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_sweep(
        args.graph,
        args.sensory_config,
        args.motor_config,
        args.lif_config,
        args.output,
        rates=args.rates,
        volleys=args.volleys,
        volley_interval_ms=args.volley_interval_ms,
        baseline_ms=args.baseline_ms,
        drive_ms=args.drive_ms,
        tail_ms=args.tail_ms,
        seed=args.seed,
        sensory_subset=args.sensory_subset,
    )
    print(json.dumps({"motor_spikes": summary["motor_spikes"]}, indent=2))
    print(
        json.dumps(
            {"minimum_rate_by_motor": summary["minimum_tested_poisson_rate_hz_by_motor"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
