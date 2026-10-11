"""Short-run artifacts with explicit monitor coverage and partial-abort reports."""

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from malecns_sim.bridge.closed_loop.scheduler import SPIKE_SCHEMA
from malecns_sim.cns.graph.store import file_hashes, write_json


def records(rows, sample):
    """Keep the same Parquet schema for active, silent and aborted conditions."""
    return pa.Table.from_pylist(rows, schema=pa.Table.from_pylist([sample]).schema)


def save_run(directory, scheduler, provenance, catalogue, snapshots=()):
    directory.mkdir()
    spikes = pa.Table.from_pylist(scheduler.neural_spikes, schema=SPIKE_SCHEMA).to_pandas()
    metadata = catalogue.neurons.select(
        ["node_index", "body_id", "role", "type", "side"]
    ).to_pandas()
    annotated = spikes.merge(
        metadata, on=["node_index", "body_id"], how="left", validate="many_to_one"
    )
    tables = {
        "body_states": records(
            scheduler.body_states,
            {
                "t_ms": 0.0,
                "joint_name": "",
                "joint_position_rad": 0.0,
                "joint_velocity_rad_s": 0.0,
                "actuator_target_rad": 0.0,
            },
        ),
        "sensory_activity": records(
            scheduler.sensory_activity,
            {
                "t_start_ms": 0.0,
                "t_end_ms": 0.0,
                "channel": "",
                "joint": "",
                "signal": "",
                "body_value": 0.0,
                "normalized_value": 0.0,
                "rate_hz": 0.0,
                "encoder": "",
            },
        ),
        "sensory_events": pa.Table.from_pylist(
            scheduler.sensory_events,
            schema=pa.schema(
                [
                    ("t_ms", pa.float64()),
                    ("node_index", pa.int32()),
                    ("body_id", pa.uint64()),
                    ("source_channel", pa.string()),
                    ("encoder", pa.string()),
                    ("body_sample_t_ms", pa.float64()),
                    ("injected", pa.bool_()),
                ]
            ),
        ),
        "neural_spikes": pa.Table.from_pandas(annotated, preserve_index=False),
        "motor_activity": pa.Table.from_pylist(scheduler.motor_activity, schema=SPIKE_SCHEMA),
        "motor_commands": records(
            scheduler.motor_commands,
            {
                "t_ms": 0.0,
                "actuator_name": "",
                "target": 0.0,
                "source_activity": 0.0,
                "flexor_rate_hz": 0.0,
                "extensor_rate_hz": 0.0,
                "flexor_filtered_hz": 0.0,
                "extensor_filtered_hz": 0.0,
                "window_start_ms": 0.0,
                "window_end_ms": 0.0,
                "applied": False,
            },
        ),
        "timeline": records(
            scheduler.timeline,
            {
                "t_ms": 0.0,
                "end_ms": 0.0,
                "scheduler_time_ms": 0.0,
                "neural_time_ms": 0.0,
                "physics_time_ms": 0.0,
                "q": 0.0,
                "qvel": 0.0,
                "target": 0.0,
                "sensory_rate_hz": 0.0,
                "sensory_events": 0,
                "injected_events": 0,
                "cns_spikes": 0,
                "flexor_rate_hz": 0.0,
                "extensor_rate_hz": 0.0,
            },
        ),
    }
    for name, table in tables.items():
        pq.write_table(
            table.replace_schema_metadata(None), directory / f"{name}.parquet", compression="zstd"
        )
    write_json(directory / "provenance.json", provenance)
    if snapshots:
        frames = directory / "frames"
        frames.mkdir()
        sim = scheduler.body.sim
        replay_data = mujoco.MjData(sim.model)
        with mujoco.Renderer(sim.model, height=480, width=640) as renderer:
            for index, snapshot in enumerate(snapshots):
                for field, values in snapshot.items():
                    if field == "time":
                        replay_data.time = values
                    else:
                        getattr(replay_data, field)[:] = values
                mujoco.mj_forward(sim.model, replay_data)
                renderer.update_scene(replay_data, camera=sim.camera)
                Image.fromarray(renderer.render()).save(frames / f"frame_{index:03d}.png")
    mapped = {r["node_index"] for r in scheduler.decoder.mapping.rows}
    motors = [r for r in scheduler.motor_activity if r["node_index"] in mapped]
    inputs = set(scheduler.input_gains)
    activity = scheduler.sensory_activity
    commands = scheduler.motor_commands
    body = scheduler.body_states
    neutral = scheduler.body.neutral
    report = {
        "status": scheduler.status,
        "reason": scheduler.reason,
        "condition": scheduler.mode,
        "duration_ms": scheduler.time_ms,
        "requested_duration_ms": provenance["config"]["duration_ms"],
        "sensory_events": sum(r["injected"] for r in scheduler.sensory_events),
        "generated_sensory_events": len(scheduler.sensory_events),
        "active_sensory_neurons": len(
            {r["node_index"] for r in scheduler.neural_spikes if r["node_index"] in inputs}
        ),
        "total_cns_spikes": scheduler.neural.total_spikes,
        "recorded_cns_spikes": len(scheduler.neural_spikes),
        "mapped_motor_spikes": len(motors),
        "active_mapped_motor_neurons": len({r["node_index"] for r in motors}),
        "motor_commands": len(commands),
        "nonneutral_motor_commands": sum(
            abs(r["target"] - neutral) > 1e-12 for r in commands if r["applied"]
        ),
        "max_joint_offset_rad": max(
            (abs(r["joint_position_rad"] - neutral) for r in body), default=0
        ),
        "max_joint_velocity_rad_s": max((abs(r["joint_velocity_rad_s"]) for r in body), default=0),
        "command_saturation_fraction": sum(abs(r["source_activity"]) >= 1 for r in commands)
        / max(1, len(commands)),
        "sensory_rate_saturation_fraction": sum(r["normalized_value"] >= 1 for r in activity)
        / max(1, len(activity)),
        "clock_drift_max_ms": scheduler.drift_max,
        "mujoco_warnings": [int(w.number) for w in scheduler.body.sim.data.warning if w.number],
        "nonfinite_states": any(
            not np.isfinite(getattr(scheduler.body.sim.data, n)).all()
            for n in ("qpos", "qvel", "ctrl")
        ),
    }
    report["artifacts"] = {
        p.relative_to(directory).as_posix(): file_hashes(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    write_json(directory / "report.json", report)
    return report
