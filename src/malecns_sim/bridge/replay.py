"""Replay saved sensory-driven MaleCNS motor spikes through an explicit body bridge."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from malecns_sim.bridge.build import DEFAULT_MAPPING
from malecns_sim.bridge.controller import BodyController
from malecns_sim.bridge.decoder import MotorDecoder
from malecns_sim.bridge.store import stage_output
from malecns_sim.bridge.validate import validate_mapping, validate_run
from malecns_sim.cns.graph.store import file_hashes, write_json
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.inspect import DEFAULT_CATALOGUE
from malecns_sim.io.validate import validate_catalogue
from malecns_sim.neural.result import validate_result
from malecns_sim.simulation.fly import FlySimulation, build_fly
from malecns_sim.simulation.smoke_test import check_state


def decode_trace(mapping, motor_spikes: pa.Table, duration_ms: float):
    dt = mapping.control_dt_ms
    windows = duration_ms / dt
    if (
        not np.isfinite(windows)
        or windows <= 0
        or not np.isclose(windows, round(windows), rtol=0, atol=1e-9)
    ):
        raise ValueError("Neural duration must be a positive whole number of control windows")
    decoder = MotorDecoder(mapping)
    times = motor_spikes["t_ms"].to_numpy()
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(times >= duration_ms):
        raise ValueError("Invalid motor spike time")
    commands = []
    for i in range(round(windows)):
        window = motor_spikes.filter(pa.array((times >= i * dt) & (times < (i + 1) * dt)))
        commands.extend(decoder.decode(window, i * dt, (i + 1) * dt))
    return commands, decoder


def replay_physics(
    sim, mapping, commands, duration_ms, *, frame_dir=None, frame_count=6, on_step=None
):
    """Compare driven physics to a neutral replay with exactly the same initial state.

    One additional control interval holds the final decoded command. No neural
    event from outside the saved experiment is created during that interval.
    """
    controller = BodyController(mapping, sim.model)
    baseline_data = mujoco.MjData(sim.model)
    key = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_KEY, "neutral")
    if key < 0:
        raise ValueError("Body must contain a neutral initial keyframe")
    mujoco.mj_resetDataKeyframe(sim.model, baseline_data, key)
    mujoco.mj_forward(sim.model, baseline_data)
    if sim.data.time != 0 or any(
        not np.array_equal(getattr(sim.data, field), getattr(baseline_data, field))
        for field in ("qpos", "qvel", "ctrl", "act")
    ):
        raise ValueError("Replay must start from the neutral keyframe")
    baseline = FlySimulation(sim.model, baseline_data, sim.camera)
    physics_dt_ms = float(sim.model.opt.timestep * 1000)
    interval = round(mapping.control_dt_ms / physics_dt_ms)
    steps = round((duration_ms + mapping.control_dt_ms) / physics_dt_ms)
    windows = round(duration_ms / mapping.control_dt_ms)
    by_time = {}
    for command in commands:
        by_time.setdefault(round(command.t_ms / physics_dt_ms), []).append(command)
    expected = {c["body"]["actuator_name"] for c in mapping.channels}
    if set(by_time) != set(range(interval, (windows + 1) * interval, interval)):
        raise ValueError("Missing or off-grid motor commands")
    for batch in by_time.values():
        if len(batch) != len(expected) or {c.actuator_name for c in batch} != expected:
            raise ValueError("Missing or duplicate mapped actuator command")
    rows = []
    frame_steps = set(np.linspace(0, steps, frame_count, dtype=int)) if frame_count else set()
    renderer = None
    if frame_dir is not None and frame_count:
        frame_dir.mkdir()
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
    try:
        for step in range(steps + 1):
            if step in by_time:
                for command in by_time[step]:
                    controller.apply(command, sim.data)
            if step % interval == 0 or step == steps:
                for c in mapping.channels:
                    a = c["body"]
                    address = a["qpos_address"]
                    rows.append(
                        {
                            "t_ms": step * physics_dt_ms,
                            "actuator_name": a["actuator_name"],
                            "target": float(sim.data.ctrl[a["actuator_index"]]),
                            "joint_position": float(sim.data.qpos[address]),
                            "neutral_replay_position": float(baseline_data.qpos[address]),
                            "command_effect_rad": float(
                                sim.data.qpos[address] - baseline_data.qpos[address]
                            ),
                        }
                    )
            if renderer is not None and step in frame_steps:
                mujoco.mj_forward(sim.model, sim.data)
                renderer.update_scene(sim.data, camera=sim.camera)
                pixels = renderer.render()
                if np.ptp(pixels) == 0:
                    raise RuntimeError("Blank bridge render")
                Image.fromarray(pixels).save(frame_dir / f"frame_{step:06d}.png")
            if step < steps:
                if on_step is not None:
                    on_step(sim, False)
                mujoco.mj_step(sim.model, sim.data)
                mujoco.mj_step(sim.model, baseline_data)
                check_state(sim)
                check_state(baseline)
            elif on_step is not None:
                on_step(sim, True)
    finally:
        if renderer is not None:
            renderer.close()
    return pa.Table.from_pylist(rows)


def run_replay(
    neural_run: Path,
    mapping_dir: Path,
    catalogue_dir: Path,
    output: Path,
    *,
    frame_count: int = 6,
    viewer: bool = False,
    playback_speed: float = 0.1,
):
    if not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("Playback speed must be finite and positive")
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if not 0 <= frame_count <= 100:
        raise ValueError("Frame count must be in [0,100]")
    validate_catalogue(catalogue_dir)
    catalogue = IOCatalog.load(catalogue_dir)
    neural_summary = validate_result(neural_run)
    spec = json.loads((neural_run / "experiment.json").read_text(encoding="utf-8"))
    if (
        spec["graph_provenance"].get("graph_manifest_sha256")
        != catalogue.manifest["graph"]["manifest"]["sha256"]
    ):
        raise ValueError("Neural run and catalogue reference different graphs")
    identity = catalogue.neurons.select(
        ["node_index", "body_id", "role", "type", "side"]
    ).to_pandas()
    local = pq.read_table(neural_run / "node_mapping.parquet").to_pandas()
    joined = local.merge(identity, on=["node_index", "body_id"], validate="one_to_one", how="left")
    if joined.role.isna().any():
        raise ValueError("Neural node identities do not match the catalogue")
    stimulated = set()
    roles = joined.set_index("local_index")["role"].to_dict()
    for stimulus in spec["stimuli"]:
        for node in stimulus["nodes"]:
            if roles.get(node) != "sensory":
                raise ValueError("This demonstration requires sensory-only external stimulation")
            stimulated.add(node)
    spikes = pq.read_table(neural_run / "spikes.parquet").to_pandas()
    annotated = spikes.merge(
        identity, on=["node_index", "body_id"], how="left", validate="many_to_one"
    )
    if annotated.role.isna().any():
        raise ValueError("Unrecognized spike identity")
    motor = annotated[annotated.role.eq("motor")].copy()
    sim = build_fly()
    mapping = validate_mapping(mapping_dir, catalogue, sim)
    mapped_nodes = {r["node_index"] for r in mapping.rows}
    motor["is_mapped"] = motor.node_index.isin(mapped_nodes)
    motor_table = pa.Table.from_pandas(motor, preserve_index=False).replace_schema_metadata(None)
    commands, decoder = decode_trace(mapping, motor_table, spec["duration_ms"])
    command_table = pa.Table.from_pylist([c.to_dict() for c in commands])
    active_motors = set(motor.node_index)
    print(
        f"Validated sensory targets: {len(stimulated)}; active motors: {len(active_motors)}; "
        f"active mapped motors: {len(active_motors & mapped_nodes)}",
        flush=True,
    )
    with stage_output(output) as stage:
        states = replay_physics(
            sim,
            mapping,
            commands,
            spec["duration_ms"],
            frame_dir=stage / "frames",
            frame_count=frame_count,
        )
        for name, table in {
            "neural_activity.parquet": pa.Table.from_pandas(
                annotated, preserve_index=False
            ).replace_schema_metadata(None),
            "motor_activity.parquet": motor_table,
            "motor_commands.parquet": command_table,
            "joint_states.parquet": states,
        }.items():
            pq.write_table(table, stage / name, compression="zstd")
        manifest = json.loads((mapping_dir / "manifest.json").read_text(encoding="utf-8"))
        write_json(
            stage / "provenance.json",
            {
                "mapping": manifest,
                "mapping_manifest": file_hashes(mapping_dir / "manifest.json"),
                "neural_experiment": spec,
                "neural_summary": neural_summary,
                "neural_spikes": file_hashes(neural_run / "spikes.parquet"),
                "body_versions": {
                    name: importlib.metadata.version(name)
                    for name in ("flygym", "mujoco", "numpy", "malecns-sim")
                },
                "physics": {
                    "timestep_seconds": float(sim.model.opt.timestep),
                    "gravity": sim.model.opt.gravity.tolist(),
                    "integrator": int(sim.model.opt.integrator),
                },
                "clock_convention": (
                    "[start,end) spikes produce a command at end; "
                    "final command held for one control interval"
                ),
                "mechanics": (
                    "Position-target engineering decoder, not a muscle/tendon/torque model"
                ),
                "sensory_feedback": False,
            },
        )
        channels = {}
        frame = states.to_pandas()
        for c in mapping.channels:
            selected = frame[frame.actuator_name.eq(c["body"]["actuator_name"])]
            channels[c["name"]] = {
                "actuator": c["body"]["actuator_name"],
                "max_command_offset_rad": float(
                    abs(selected.target - c["body"]["neutral_position"]).max()
                ),
                "joint_displacement_rad": float(
                    abs(selected.joint_position - selected.joint_position.iloc[0]).max()
                ),
                "max_effect_vs_neutral_replay_rad": float(abs(selected.command_effect_rad).max()),
            }
        report = {
            "stimulated_sensory_neurons": len(stimulated),
            "active_motor_neurons": len(active_motors),
            "mapped_motor_neurons": len(mapped_nodes),
            "unmapped_motor_neurons": len(mapping.motor_identity) - len(mapped_nodes),
            "active_mapped_motor_neurons": len(active_motors & mapped_nodes),
            "active_unmapped_motor_neurons": len(decoder.unmapped_nodes),
            "unmapped_motor_spikes": decoder.unmapped_spikes,
            "mapped_motor_spikes": int(motor.is_mapped.sum()),
            "actuators_driven": sum(c["max_command_offset_rad"] > 0 for c in channels.values()),
            "channels": channels,
            "finite_state": True,
            "mujoco_warnings": [],
            "neural_dt_ms": spec["dt_ms"],
            "control_dt_ms": mapping.control_dt_ms,
            "physics_dt_ms": float(sim.model.opt.timestep * 1000),
            "neural_duration_ms": spec["duration_ms"],
            "physics_duration_ms": spec["duration_ms"] + mapping.control_dt_ms,
            "seed": spec["seed"],
            "sensory_feedback": False,
            "artifacts": {
                str(p.relative_to(stage)).replace("\\", "/"): file_hashes(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        write_json(stage / "report.json", report)
        validate_run(stage)
    if viewer:
        from malecns_sim.bridge.viewer import show_replay

        print(f"Results saved: {output}. Opening MuJoCo viewer...", flush=True)
        show_replay(sim, mapping, commands, spec["duration_ms"], speed=playback_speed)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-run", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--viewer", action="store_true", help="Open a local MuJoCo replay window")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.1,
        help="Viewer speed relative to real time (default: 0.1)",
    )
    args = parser.parse_args()
    report = run_replay(
        args.neural_run,
        args.mapping,
        args.catalogue,
        args.output,
        frame_count=args.frames,
        viewer=args.viewer,
        playback_speed=args.playback_speed,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts"}, indent=2))
    print(f"Bridge results: {args.output}")


if __name__ == "__main__":
    main()
