"""Sparse spike/rate results with complete experiment provenance."""

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from malecns_sim.cns.graph.store import file_hashes, write_json


def save_result(
    backend,
    output: Path,
    *,
    experiment: str,
    extra: dict | None = None,
    extra_tables: dict[str, pa.Table] | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Result directory exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    times, local = backend.get_spikes()
    graph = backend.graph
    duration = backend.time_steps * backend.config.dt_ms
    if duration <= 0:
        raise ValueError("No completed simulation to save")
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(times >= duration):
        raise ValueError("Invalid spike times")
    counts = np.bincount(local, minlength=len(graph.body_ids)).astype(np.uint64)
    spike_table = pa.table(
        {
            "t_ms": pa.array(times, type=pa.float64()),
            "node_index": pa.array(graph.node_indices[local], type=pa.int32()),
            "body_id": pa.array(graph.body_ids[local], type=pa.uint64()),
        }
    )
    rate_table = pa.table(
        {
            "node_index": pa.array(graph.node_indices, type=pa.int32()),
            "body_id": pa.array(graph.body_ids, type=pa.uint64()),
            "spike_count": pa.array(counts, type=pa.uint64()),
            "rate_hz": pa.array(counts / (duration / 1000), type=pa.float64()),
        }
    )
    specification = {
        "experiment": experiment,
        "model": backend.config.model,
        "config": backend.config.to_dict(),
        "duration_ms": duration,
        "dt_ms": backend.config.dt_ms,
        "seed": backend.seed,
        "stimuli": [s.to_dict() for s in backend.stimuli],
        "silence_events": backend.silence_events,
        "stimulus_node_space": (
            "local neural index; see node_mapping.parquet for parent graph indices"
        ),
        "graph_provenance": graph.provenance,
        "backend": "Brian2LIFBackend",
        "codegen": "numpy",
        "integration": "exact",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("brian2", "numpy", "pyarrow", "malecns-sim")
        },
        "assumptions": {
            "source": "https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py",
            "paper": "https://doi.org/10.1038/s41586-024-07763-9",
            "refractory": (
                "Both v and g are clamped for all cells, including stimulated cells; "
                "Shiu's input helper instead removes refractory for stimulated targets."
            ),
            "external_drive": (
                "Direct v increments; gain is an optogenetic drive model parameter, "
                "separate from recurrent contacts × sign × weight_per_contact."
            ),
            "poisson": (
                "Independent Bernoulli(rate*dt) per target and timestep, generated with "
                "per-stimulus PCG64 streams; input events arrive in the synapses scheduling "
                "slot before next-step thresholding."
            ),
            "signs": backend.sign_report,
            "interpretation": (
                "Connectivity-driven neural activity, not validated MaleCNS physiology "
                "or fly behavior."
            ),
        },
        "extra": extra or {},
    }
    summary = {
        "neurons": len(graph.body_ids),
        "connections": len(graph.src),
        "active_neurons": int(np.count_nonzero(counts)),
        "total_spikes": len(times),
        "duration_ms": duration,
        "finite_state": True,
        "stimulated_neurons": len({node for s in backend.stimuli for node in s.nodes}),
        "sign_coverage": backend.sign_report,
    }
    driven = {node for s in backend.stimuli for node in s.nodes}
    summary["active_unstimulated_neurons"] = int(
        sum(count > 0 and i not in driven for i, count in enumerate(counts))
    )
    # TemporaryDirectory uses a private Windows ACL that survives rename.
    # A regular sibling directory inherits access from the result parent instead.
    parent = output.parent.resolve()
    stage = parent / f".{output.name}-{uuid4().hex}"
    stage.mkdir()
    try:
        pq.write_table(spike_table, stage / "spikes.parquet", compression="zstd")
        pq.write_table(rate_table, stage / "rates.parquet", compression="zstd")
        pq.write_table(
            pa.table(
                {
                    "local_index": np.arange(len(graph.body_ids), dtype=np.int32),
                    "node_index": graph.node_indices,
                    "body_id": graph.body_ids,
                }
            ),
            stage / "node_mapping.parquet",
            compression="zstd",
        )
        write_json(stage / "experiment.json", specification)
        artifacts = ["spikes.parquet", "rates.parquet", "node_mapping.parquet", "experiment.json"]
        if backend.record_nodes:
            trace = backend.get_state_trace()
            nodes = np.asarray(backend.record_nodes, dtype=np.int32)
            samples = len(trace["t_ms"])
            pq.write_table(
                pa.table(
                    {
                        "t_ms": np.tile(trace["t_ms"], len(nodes)),
                        "local_index": np.repeat(nodes, samples),
                        "node_index": np.repeat(graph.node_indices[nodes], samples),
                        "body_id": np.repeat(graph.body_ids[nodes], samples),
                        "v_mv": trace["v_mv"].reshape(-1),
                        "g_mv": trace["g_mv"].reshape(-1),
                    }
                ),
                stage / "state_trace.parquet",
                compression="zstd",
            )
            artifacts.append("state_trace.parquet")
        for name, table in sorted((extra_tables or {}).items()):
            if (
                not name.endswith(".parquet")
                or Path(name).name != name
                or name in artifacts
                or not isinstance(table, pa.Table)
            ):
                raise ValueError(f"Invalid extra result table: {name}")
            pq.write_table(table, stage / name, compression="zstd")
            artifacts.append(name)
        summary["artifacts"] = {name: file_hashes(stage / name) for name in artifacts}
        write_json(stage / "summary.json", summary)
        stage.rename(output)
    finally:
        if stage.exists() and stage.resolve().parent == parent:
            shutil.rmtree(stage)
    return summary


def validate_result(directory: str | Path) -> dict:
    """Check saved artifacts, mapping, spike counts, rates, and timestep alignment."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    required = {"spikes.parquet", "rates.parquet", "node_mapping.parquet", "experiment.json"}
    artifact_names = set(summary["artifacts"])
    if not required <= artifact_names:
        raise ValueError("Missing required result artifacts")
    for name in artifact_names - required - {"state_trace.parquet"}:
        if Path(name).name != name or not name.endswith(".parquet"):
            raise ValueError("Unexpected result artifact name")
    for name in sorted(summary["artifacts"]):
        if file_hashes(directory / name) != summary["artifacts"][name]:
            raise ValueError(f"Result checksum mismatch: {name}")
    specification = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    mapping = pq.read_table(directory / "node_mapping.parquet").to_pydict()
    spikes = pq.read_table(directory / "spikes.parquet").to_pydict()
    rates = pq.read_table(directory / "rates.parquet").to_pydict()
    n = summary["neurons"]
    if mapping["local_index"] != list(range(n)) or len(set(mapping["node_index"])) != n:
        raise ValueError("Non-bijective result node mapping")
    if len(set(mapping["body_id"])) != n:
        raise ValueError("Non-bijective body ID mapping")
    if rates["node_index"] != mapping["node_index"] or rates["body_id"] != mapping["body_id"]:
        raise ValueError("Rate metadata differs from node mapping")
    local_by_parent = {node: i for i, node in enumerate(mapping["node_index"])}
    counts = np.zeros(n, dtype=np.uint64)
    for node, body in zip(spikes["node_index"], spikes["body_id"]):
        if node not in local_by_parent or mapping["body_id"][local_by_parent[node]] != body:
            raise ValueError("Spike cannot be mapped to its anatomical body ID")
        counts[local_by_parent[node]] += 1
    times = np.array(spikes["t_ms"])
    duration, dt = specification["duration_ms"], specification["dt_ms"]
    if (
        duration <= 0
        or dt <= 0
        or not np.isfinite([duration, dt]).all()
        or not np.isfinite(times).all()
        or np.any(times < 0)
        or np.any(times >= duration)
        or not np.allclose(times / dt, np.rint(times / dt), rtol=0, atol=1e-8)
    ):
        raise ValueError("Invalid saved spike timing")
    if not np.array_equal(counts, rates["spike_count"]) or not np.array_equal(
        counts / (duration / 1000), rates["rate_hz"]
    ):
        raise ValueError("Saved spike counts/rates disagree")
    if (
        len(times) != summary["total_spikes"]
        or int(np.count_nonzero(counts)) != summary["active_neurons"]
        or duration != summary["duration_ms"]
    ):
        raise ValueError("Result summary disagrees with saved spikes")
    if "state_trace.parquet" in summary["artifacts"]:
        trace = pq.read_table(directory / "state_trace.parquet").to_pandas()
        nodes = trace["local_index"].unique()
        if not 1 <= len(nodes) <= 32 or any(node not in range(n) for node in nodes):
            raise ValueError("Invalid state trace nodes")
        expected = np.arange(round(duration / dt)) * dt
        for node, rows in trace.groupby("local_index", sort=False):
            if (
                len(rows) != len(expected)
                or not np.allclose(rows["t_ms"], expected, rtol=0, atol=1e-8)
                or not np.isfinite(rows[["v_mv", "g_mv"]].to_numpy()).all()
                or not (rows["node_index"] == mapping["node_index"][node]).all()
                or not (rows["body_id"] == mapping["body_id"][node]).all()
            ):
                raise ValueError("Invalid state trace timing, values or mapping")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or compare neural result artifacts")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    summary = validate_result(args.directory)
    if args.compare:
        if summary != validate_result(args.compare):
            raise ValueError("Results differ (including experiment provenance or artifact bytes)")
        print("All result artifacts and experiment provenance are byte-identical.")
    print(
        f"Validated {summary['total_spikes']} spikes; {summary['active_neurons']} active neurons."
    )


if __name__ == "__main__":
    main()
