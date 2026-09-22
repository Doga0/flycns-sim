"""Offline MuJoCo body-state → sensory events → full-CNS LIF replay."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from malecns_sim.bridge.sensory.body_state import BodyState, JointStateReader
from malecns_sim.bridge.sensory.build import DEFAULT_MAPPING
from malecns_sim.bridge.sensory.encoder import SensoryEncoder
from malecns_sim.bridge.sensory.mapping import SensoryMapping
from malecns_sim.bridge.sensory.validate import validate_mapping, validate_run
from malecns_sim.bridge.store import stage_output
from malecns_sim.cns.graph.store import CNSGraph, file_hashes, write_json
from malecns_sim.cns.graph.validate import validate_graph
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.inspect import DEFAULT_CATALOGUE
from malecns_sim.io.validate import validate_catalogue
from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.config import load_config
from malecns_sim.neural.stimulus import Stimulus
from malecns_sim.simulation.fly import build_fly
from malecns_sim.simulation.smoke_test import check_state


def _trajectory(policy, t_ms):
    rows = policy.get("trajectory", {}).get("keyframes", [])
    if len(rows) < 2 or rows[0].get("t_ms") != 0:
        raise ValueError("Sensory trajectory requires keyframes starting at 0 ms")
    times = np.asarray([row["t_ms"] for row in rows], dtype=float)
    values = np.asarray([row["target_offset_rad"] for row in rows], dtype=float)
    if not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Sensory trajectory keyframes must be finite and increasing")
    return float(np.interp(t_ms, times, values)), float(times[-1])


def body_trajectory(sim, mapping: SensoryMapping, *, zero_motion=False) -> list[BodyState]:
    """Generate controlled body samples at sensory_dt without reading raw arrays elsewhere."""
    if len(mapping.channels) != 1:
        raise ValueError("v0.6 body trajectory currently requires exactly one sensory channel")
    channel = mapping.channels[0]
    source = channel["source"]
    reader = JointStateReader(sim.model, source["joint"], source.get("actuator"))
    dt, physics_dt = mapping.sensory_dt_ms, sim.model.opt.timestep * 1000
    interval = round(dt / physics_dt)
    _, duration = _trajectory(mapping.policy, 0)
    steps = round(duration / physics_dt)
    if not np.isclose(steps * physics_dt, duration, rtol=0, atol=1e-9):
        raise ValueError("Trajectory duration must align to the physics clock")
    if zero_motion:
        neutral = reader.read(sim.data)
        return [
            BodyState(
                float(t),
                neutral.joint_name,
                neutral.joint_position_rad,
                0.0,
                neutral.actuator_target_rad,
            )
            for t in np.arange(0, duration + dt, dt)
        ]
    key = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_KEY, "neutral")
    if key < 0:
        raise ValueError("Body needs a neutral keyframe")
    mujoco.mj_resetDataKeyframe(sim.model, sim.data, key)
    mujoco.mj_forward(sim.model, sim.data)
    neutral_target = float(sim.data.ctrl[reader.actuator_index])
    states = []
    for step in range(steps + 1):
        t_ms = step * physics_dt
        offset, _ = _trajectory(mapping.policy, t_ms)
        sim.data.ctrl[reader.actuator_index] = neutral_target + offset
        if step % interval == 0:
            states.append(reader.read(sim.data))
        if step < steps:
            mujoco.mj_step(sim.model, sim.data)
            check_state(sim)
    return states


def encode_states(mapping, states, *, mode, seed):
    if len(states) < 2:
        raise ValueError("Need at least two body states")
    encoder = SensoryEncoder(mapping, mode=mode, seed=seed)
    activities, spikes = [], []
    for state, next_state in zip(states, states[1:]):
        output = encoder.encode(state, state.t_ms, next_state.t_ms)
        activities.append(output.activity)
        spikes.append(output.spikes)
    activity = pa.concat_tables(activities)
    events = pa.concat_tables(spikes) if spikes else pa.table({})
    body_by_node = {row["node_index"]: row["body_id"] for row in mapping.rows}
    if events.num_rows:
        events = events.append_column(
            "body_id",
            pa.array([body_by_node[int(node)] for node in events["node_index"]], type=pa.uint64()),
        ).select(["t_ms", "node_index", "body_id", "source_channel", "encoder"])
    else:
        events = pa.table(
            {
                "t_ms": pa.array([], type=pa.float64()),
                "node_index": pa.array([], type=pa.int32()),
                "body_id": pa.array([], type=pa.uint64()),
                "source_channel": pa.array([], type=pa.string()),
                "encoder": pa.array([], type=pa.string()),
            }
        )
    return activity, events, encoder


def run_lif(graph_dir, catalogue, mapping, events, duration_ms, *, seed, config_path=None):
    config = load_config(config_path)
    if config.steps(duration_ms) <= 0:
        raise ValueError("Positive neural run duration is required")
    gain_by_channel = {
        channel["name"]: channel["encoder"]["gain_mv"] for channel in mapping.channels
    }
    with CNSGraph.load(graph_dir) as stored:
        validate_graph(stored)
        graph = NeuralGraph.from_graph(stored)
        node_to_local = {int(node): index for index, node in enumerate(graph.node_indices)}
        backend = Brian2LIFBackend(graph, config, seed=seed)
        for node, rows in events.to_pandas().groupby("node_index", sort=True):
            times = tuple(float(value) for value in rows.t_ms)
            backend.schedule(
                Stimulus(
                    (node_to_local[int(node)],),
                    type="spikes",
                    rate_hz=0,
                    start_ms=0,
                    stop_ms=duration_ms,
                    gain_mv=gain_by_channel[rows.source_channel.iloc[0]],
                    times_ms=times,
                )
            )
        backend.run(duration_ms)
        times, local = backend.get_spikes()
        metadata = catalogue.neurons.select(
            ["node_index", "body_id", "role", "type", "side"]
        ).to_pandas()
        spikes = pa.table(
            {
                "t_ms": pa.array(times, type=pa.float64()),
                "node_index": pa.array(graph.node_indices[local], type=pa.int32()),
                "body_id": pa.array(graph.body_ids[local], type=pa.uint64()),
            }
        )
        frame = spikes.to_pandas().merge(
            metadata, on=["node_index", "body_id"], how="left", validate="many_to_one"
        )
        if frame.role.isna().any():
            raise ValueError("LIF spike cannot be mapped to catalogue metadata")
        return pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            None
        ), backend


def run_replay(
    graph_dir,
    catalogue_dir,
    mapping_dir,
    output,
    *,
    mode="poisson",
    seed=42,
    zero_motion=False,
    config_path=None,
):
    graph_dir, catalogue_dir, mapping_dir, output = map(
        Path, (graph_dir, catalogue_dir, mapping_dir, output)
    )
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    validate_catalogue(catalogue_dir, graph_dir)
    catalogue = IOCatalog.load(catalogue_dir)
    sim = build_fly()
    mapping = validate_mapping(mapping_dir, catalogue, sim)
    states = body_trajectory(sim, mapping, zero_motion=zero_motion)
    duration = round(states[-1].t_ms / mapping.sensory_dt_ms) * mapping.sensory_dt_ms
    activity, sensory_spikes, encoder = encode_states(mapping, states, mode=mode, seed=seed)
    neural_spikes, backend = run_lif(
        graph_dir, catalogue, mapping, sensory_spikes, duration, seed=seed, config_path=config_path
    )
    sensory_nodes = {row["node_index"] for row in mapping.rows}
    active = set(neural_spikes["node_index"].to_pylist())
    with stage_output(output) as stage:
        tables = {
            "body_states.parquet": pa.Table.from_pylist([state.to_dict() for state in states]),
            "sensory_activity.parquet": activity,
            "sensory_spikes.parquet": sensory_spikes,
            "neural_spikes.parquet": neural_spikes,
        }
        for name, table in tables.items():
            pq.write_table(table, stage / name, compression="zstd")
        manifest = json.loads((mapping_dir / "manifest.json").read_text(encoding="utf-8"))
        write_json(
            stage / "provenance.json",
            {
                "mapping": manifest,
                "mapping_manifest": file_hashes(mapping_dir / "manifest.json"),
                "graph_manifest": file_hashes(graph_dir / "manifest.json"),
                "encoder": {
                    "mode": encoder.mode,
                    "seed": encoder.seed,
                    "channels": [
                        {
                            "name": channel["name"],
                            "type": channel["encoder"]["type"],
                            "gain_mv": channel["encoder"]["gain_mv"],
                        }
                        for channel in mapping.channels
                    ],
                },
                "lif_config": backend.config.to_dict(),
                "lif_seed": seed,
                "clocks": {
                    "physics_dt_ms": sim.model.opt.timestep * 1000,
                    "sensory_dt_ms": mapping.sensory_dt_ms,
                    "neural_dt_ms": backend.config.dt_ms,
                },
                "trajectory": mapping.policy["trajectory"],
                "zero_motion_control": zero_motion,
                "motor_body_coupling": False,
                "body_versions": {
                    name: importlib.metadata.version(name)
                    for name in ("flygym", "mujoco", "numpy", "brian2", "malecns-sim")
                },
            },
        )
        report = {
            "duration_ms": duration,
            "sensory_dt_ms": mapping.sensory_dt_ms,
            "physics_dt_ms": sim.model.opt.timestep * 1000,
            "neural_dt_ms": backend.config.dt_ms,
            "mode": mode,
            "seed": seed,
            "zero_motion_control": zero_motion,
            "mapped_sensory_neurons": len(mapping.rows),
            "sensory_spikes": sensory_spikes.num_rows,
            "active_sensory_neurons": len(active & sensory_nodes),
            "active_downstream_neurons": len(active - sensory_nodes),
            "active_motor_neurons": len(
                set(
                    neural_spikes.filter(pc.equal(neural_spikes["role"], "motor"))[
                        "node_index"
                    ].to_pylist()
                )
            ),
            "total_neural_spikes": neural_spikes.num_rows,
            "finite_state": True,
            "mujoco_warnings": [],
            "motor_body_coupling": False,
            "artifacts": {
                str(path.relative_to(stage)).replace("\\", "/"): file_hashes(path)
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
        }
        write_json(stage / "report.json", report)
        validate_run(stage)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=Path("data/processed/malecns-v1.0/published-v1")
    )
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("explicit_spikes", "poisson"), default="poisson")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-motion", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    report = run_replay(
        args.graph,
        args.catalogue,
        args.mapping,
        args.output,
        mode=args.mode,
        seed=args.seed,
        zero_motion=args.zero_motion,
        config_path=args.config,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "artifacts"}, indent=2))
    print(f"Sensory bridge results: {args.output}")


if __name__ == "__main__":
    main()
