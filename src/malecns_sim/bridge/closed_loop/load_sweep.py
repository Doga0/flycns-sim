"""Physics-only external-load calibration; no CNS inputs or motor-derived targets."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from malecns_sim.bridge.closed_loop.perturbation import JointTorque
from malecns_sim.bridge.sensory.body_state import JointStateReader
from malecns_sim.bridge.store import stage_output
from malecns_sim.cns.graph.store import file_hashes, write_json
from malecns_sim.simulation.fly import build_fly
from malecns_sim.simulation.smoke_test import check_state


def sweep(config, amplitudes, output):
    """Measure the unchanged position actuator under a fixed time-only load."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    results = []
    with stage_output(output) as stage:
        artifacts = {}
        for index, amplitude in enumerate(amplitudes):
            sim = build_fly()
            joint = config["initial_condition"]["joint"]
            reader = JointStateReader(sim.model, joint)
            profile = {**config["perturbation"], "amplitude_nm": amplitude}
            torque = JointTorque.from_config(profile, sim.model)
            if torque.joint != joint or torque.stop_ms > config["duration_ms"]:
                raise ValueError("Load outside configured experiment")
            dof = torque.dof(sim.model)
            neutral = reader.read(sim.data).joint_position_rad
            values = [neutral]
            applied = []
            dt_ms = sim.model.opt.timestep * 1000
            count = round(config["duration_ms"] / dt_ms)
            if abs(count * dt_ms - config["duration_ms"]) > 1e-8:
                raise ValueError("Duration must align with physics clock")
            for _ in range(count):
                t_ms = sim.data.time * 1000
                load_nm = torque.torque_at(t_ms)
                sim.data.qfrc_applied[dof] = load_nm
                applied.append(load_nm)
                mujoco.mj_step(sim.model, sim.data)
                check_state(sim)
                q = reader.read(sim.data).joint_position_rad
                values.append(q)
                if (
                    not np.isfinite(q)
                    or abs(q - neutral) > config["safety"]["max_abs_joint_offset_rad"]
                ):
                    raise ValueError(f"Load {amplitude} exceeds joint safety bounds")
            arr = np.asarray(values)
            path = stage / f"trajectory_{index:02d}.parquet"
            pq.write_table(
                pa.table(
                    {
                        "t_ms": np.arange(len(arr)) * dt_ms,
                        "q_rad": arr,
                        "offset_rad": arr - neutral,
                        "external_torque_nm": np.r_[applied, 0.0],
                    }
                ),
                path,
                compression="zstd",
            )
            artifacts[path.name] = file_hashes(path)
            hold_start = torque.start_ms + torque.ramp_duration_ms
            hold_end = hold_start + torque.hold_duration_ms
            at = np.arange(len(arr)) * dt_ms
            held = arr[(at >= hold_start) & (at < hold_end)] - neutral
            result = {
                "amplitude_nm": amplitude,
                "peak_positive_offset_rad": float(np.max(arr - neutral)),
                "peak_negative_offset_rad": float(np.min(arr - neutral)),
                "mean_hold_offset_rad": float(np.mean(held)),
                "peak_nominal_sensory_rate_hz": float(
                    np.clip(np.max(arr - neutral) / 0.35, 0, 1) * 100
                ),
                "end_offset_rad": float(arr[-1] - neutral),
                "mujoco_warnings": [int(w.number) for w in sim.data.warning if w.number],
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        write_json(
            stage / "report.json",
            {
                "config": config,
                "config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "results": results,
                "artifact_hashes": artifacts,
            },
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/closed_loop/lf_tibia_load_v2.yaml")
    )
    parser.add_argument("--amplitudes-nm", nargs="+", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sweep(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.amplitudes_nm, args.output)


if __name__ == "__main__":
    main()
