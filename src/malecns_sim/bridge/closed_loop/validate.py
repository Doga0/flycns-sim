"""Validate causality, identity, provenance and matched-control effects."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from malecns_sim.cns.graph.store import file_hashes


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_run(directory):
    directory = Path(directory)
    report, provenance = (read_json(directory / f"{n}.json") for n in ("report", "provenance"))
    required = {
        f"{name}.parquet"
        for name in (
            "body_states",
            "sensory_activity",
            "sensory_events",
            "neural_spikes",
            "motor_activity",
            "motor_commands",
            "timeline",
        )
    } | {"provenance.json"}
    if not required <= set(report["artifacts"]):
        raise ValueError("Missing closed-loop artifacts")
    for name, hashes in report["artifacts"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe artifact path")
        if file_hashes(directory / name) != hashes:
            raise ValueError(f"Closed-loop checksum mismatch: {name}")
    for kind in ("motor_mapping", "sensory_mapping"):
        manifest = provenance[kind]
        if (
            manifest["graph_manifest"] != provenance["graph_manifest"]
            or manifest["catalogue_manifest"] != provenance["catalogue_manifest"]
        ):
            raise ValueError("Cross-bridge provenance mismatch")
    config = provenance["config"]

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    if (
        digest(config) != provenance["config_sha256"]
        or digest(provenance["lif_config"]) != provenance["lif_config_sha256"]
    ):
        raise ValueError("Embedded configuration checksum mismatch")
    if provenance["lif_config_sha256"] != config["expected_lif_config_sha256"]:
        raise ValueError("LIF configuration mismatch")
    tables = {
        name: pq.read_table(directory / f"{name}.parquet").to_pandas()
        for name in (
            "body_states",
            "sensory_activity",
            "sensory_events",
            "neural_spikes",
            "motor_activity",
            "motor_commands",
            "timeline",
        )
    }
    if report["status"] == "ABORTED_SAFETY_LIMIT":
        if not report["reason"]:
            raise ValueError("Safety abort needs a reason")
        return report
    if report["status"] != "COMPLETED" or report["mujoco_warnings"] or report["nonfinite_states"]:
        raise ValueError("Invalid completed run state")
    for name, table in tables.items():
        if not np.isfinite(table.select_dtypes(include="number").to_numpy()).all():
            raise ValueError(f"Nonfinite completed table: {name}")
    timeline, commands = tables["timeline"], tables["motor_commands"]
    dt, motor_dt = config["clocks"]["sensory_dt_ms"], config["clocks"]["motor_dt_ms"]
    expected = np.arange(round(config["duration_ms"] / dt)) * dt
    if len(timeline) != len(expected) or not np.allclose(
        timeline.t_ms, expected, rtol=0, atol=1e-8
    ):
        raise ValueError("Missing scheduler ticks")
    for name in ("scheduler_time_ms", "neural_time_ms", "physics_time_ms"):
        if not np.allclose(timeline[name], expected + dt, rtol=0, atol=1e-7):
            raise ValueError("Clock drift in saved timeline")
    channel = provenance["motor_mapping"]["resolved_channels"][0]
    sensory = provenance["sensory_mapping"]["resolved_channels"][0]
    allowed = set(sensory["nodes"])
    events = tables["sensory_events"]
    if len(events):
        if (
            not set(events.node_index) <= allowed
            or not (events.t_ms == events.body_sample_t_ms).all()
        ):
            raise ValueError("Unmapped input or future sensory body sample")
        if (events.t_ms < 0).any() or (events.t_ms >= config["duration_ms"]).any():
            raise ValueError("Sensory event outside timeline")
        if provenance["condition"] == "sensory_clamp" and events.injected.any():
            raise ValueError("Clamped input was injected")
        body_ids = provenance["sensory_mapping"]["policy"]["channels"][0]["malecns"]["body_ids"]
        expected_bodies = dict(zip(sensory["nodes"], body_ids))
        if any(expected_bodies[r.node_index] != r.body_id for r in events.itertuples()):
            raise ValueError("Sensory node/body identity mismatch")
    expected_motor = np.arange(1, round(config["duration_ms"] / motor_dt) + 1) * motor_dt
    if len(commands) != len(expected_motor) or not np.allclose(commands.t_ms, expected_motor):
        raise ValueError("Missing motor windows")
    if set(commands.actuator_name) != {channel["body"]["actuator_name"]}:
        raise ValueError("Unknown actuator")
    if not (
        (commands.window_end_ms == commands.t_ms)
        & (commands.window_start_ms == commands.t_ms - motor_dt)
    ).all():
        raise ValueError("Noncausal command windows")
    targets = commands.target.to_numpy()
    neutral = channel["body"]["neutral_position"]
    if (
        np.any(targets < channel["minimum"])
        or np.any(targets > channel["maximum"])
        or np.any(
            np.abs(np.diff(np.r_[neutral, targets])) > channel["decoder"]["max_delta_rad"] + 1e-12
        )
    ):
        raise ValueError("Command bounds/slew violated")
    # Read-back causality: target on interval t must be the last already applied command.
    expected_target = np.full(len(timeline), neutral)
    for row in commands.itertuples():
        if row.applied:
            expected_target[timeline.t_ms >= row.t_ms] = row.target
    if not np.allclose(timeline.target, expected_target, rtol=0, atol=1e-12):
        raise ValueError("Command affected an earlier physics interval")
    if provenance["condition"] in {"feedback_off", "sensory_clamp"} and (
        report["total_cns_spikes"] or report["sensory_events"] or commands.applied.any()
    ):
        raise ValueError("Control condition generated/applied neural feedback")
    if provenance["condition"] == "motor_clamp" and commands.applied.any():
        raise ValueError("Motor-clamped command was applied")
    if "external_torque_nm" in timeline:
        from malecns_sim.bridge.closed_loop.perturbation import JointTorque

        # The profile is already resolved in provenance; no MuJoCo model is needed here.
        perturbation = None
        policy = provenance.get("external_perturbation")
        if policy is not None:
            perturbation = JointTorque(**policy)
        expected_torque = [
            0.0 if perturbation is None else perturbation.torque_at(t) for t in timeline.t_ms
        ]
        if not np.allclose(timeline.external_torque_nm, expected_torque, rtol=0, atol=1e-12):
            raise ValueError("External torque does not match the time-only profile")
    return report


def compare_conditions(directory):
    directory = Path(directory)
    has_motor_clamp = (directory / "motor_clamp").exists()
    conditions = (
        ("feedback_off", "sensory_clamp", "motor_clamp", "feedback_on")
        if has_motor_clamp
        else ("feedback_off", "sensory_clamp", "feedback_on")
    )
    reports = {c: validate_run(directory / c) for c in conditions}
    provenance = {c: read_json(directory / c / "provenance.json") for c in conditions}
    for key in (
        "initial_state_sha256",
        "config_sha256",
        "graph_manifest",
        "motor_manifest_hash",
        "sensory_manifest_hash",
    ):
        if any(provenance[c][key] != provenance["feedback_off"][key] for c in conditions):
            raise ValueError(f"Control conditions differ in {key}")
    if any(r["status"] != "COMPLETED" for r in reports.values()):
        return {
            "acceptance_passed": False,
            "reason": "At least one condition aborted",
            "reports": reports,
        }
    states = {
        c: pq.read_table(directory / c / "body_states.parquet")
        .to_pandas()
        .joint_position_rad.to_numpy()
        for c in conditions
    }
    clamp_error = float(np.max(np.abs(states["feedback_off"] - states["sensory_clamp"])))
    effect = float(np.max(np.abs(states["feedback_on"] - states["feedback_off"])))
    on = reports["feedback_on"]
    if has_motor_clamp:
        motor_clamp_error = float(np.max(np.abs(states["feedback_off"] - states["motor_clamp"])))
        torque = {
            c: pq.read_table(directory / c / "timeline.parquet")
            .to_pandas()
            .external_torque_nm.to_numpy()
            for c in conditions
        }
        if any(not np.array_equal(torque[c], torque["feedback_off"]) for c in conditions):
            raise ValueError("External load differs between conditions")
        clamp = reports["motor_clamp"]
        sensory_on = pq.read_table(
            directory / "feedback_on" / "sensory_activity.parquet"
        ).to_pandas()
        sensory_clamped = pq.read_table(
            directory / "motor_clamp" / "sensory_activity.parquet"
        ).to_pandas()
        if len(sensory_on) != len(sensory_clamped):
            raise ValueError("Sensory activity ticks differ between motor conditions")
        sensory_delta = np.abs(sensory_on.rate_hz.to_numpy() - sensory_clamped.rate_hz.to_numpy())
        max_sensory_effect = float(np.max(sensory_delta))
        changed_sensory = np.flatnonzero(sensory_delta > 1e-6)
        first_sensory_change = (
            None
            if not len(changed_sensory)
            else float(sensory_on.t_start_ms.iloc[changed_sensory[0]])
        )
        metrics = {
            "motor_clamp_vs_baseline_rad": motor_clamp_error,
            "motor_clamp_mapped_motor_spikes": clamp["mapped_motor_spikes"],
            "motor_clamp_decoded_nonneutral_commands": clamp["decoded_nonneutral_motor_commands"],
            "functional_target_0p01_rad_met": effect >= 0.01,
            "max_sensory_rate_change_vs_motor_clamp_hz": max_sensory_effect,
            "first_feedback_sensory_rate_change_ms": first_sensory_change,
        }
    else:
        motor_clamp_error, max_sensory_effect, first_sensory_change, metrics = 0.0, 0.0, None, {}
    commands = pq.read_table(directory / "feedback_on" / "motor_commands.parquet").to_pandas()
    motors = pq.read_table(directory / "feedback_on" / "motor_activity.parquet").to_pandas()
    sensory_events = pq.read_table(directory / "feedback_on" / "sensory_events.parquet").to_pandas()
    neutral = provenance["feedback_on"]["motor_mapping"]["resolved_channels"][0]["body"][
        "neutral_position"
    ]
    driven = commands[commands.applied & (np.abs(commands.target - neutral) > 1e-12)]
    first_motor = None if motors.empty else float(motors.t_ms.min())
    first_command = None if driven.empty else float(driven.t_ms.min())
    dt_ms = provenance["feedback_on"]["config"]["clocks"]["sensory_dt_ms"]
    effect_tolerance_rad = 1e-6 if has_motor_clamp else 1e-10
    changed = np.flatnonzero(
        np.abs(states["feedback_on"] - states["feedback_off"]) > effect_tolerance_rad
    )
    first_physical = None if len(changed) == 0 else float(changed[0] * dt_ms)
    first_sensory = None if sensory_events.empty else float(sensory_events.t_ms.min())
    causal = (
        first_sensory is not None
        and first_motor is not None
        and first_command is not None
        and first_physical is not None
        and first_sensory <= first_motor < first_command <= first_physical
    )
    feedback_causal = not has_motor_clamp or (
        first_sensory_change is not None
        and first_physical is not None
        and first_sensory_change >= first_physical
    )
    passed = (
        effect > effect_tolerance_rad
        and clamp_error <= 1e-12
        and motor_clamp_error <= 1e-12
        and on["sensory_events"] > 0
        and on["mapped_motor_spikes"] > 0
        and on["nonneutral_motor_commands"] > 0
        and causal
        and feedback_causal
        and (not has_motor_clamp or max_sensory_effect > 1e-6)
    )
    diff = states["feedback_on"] - states["feedback_off"]
    dt_s = dt_ms / 1000
    load = provenance["feedback_on"].get("external_perturbation")
    recovery_ms = None
    if load is not None and effect > 0:
        release_end = sum(
            load[k]
            for k in ("start_ms", "ramp_duration_ms", "hold_duration_ms", "release_duration_ms")
        )
        after_release = np.arange(len(diff)) * dt_ms >= release_end
        recovered = np.flatnonzero(after_release & (np.abs(diff) <= 0.1 * effect))
        if len(recovered):
            recovery_ms = float(recovered[0] * dt_ms - release_end)
    return {
        "acceptance_passed": bool(passed),
        "max_effect_vs_open_loop_rad": effect,
        "engineering_effect_tolerance_rad": effect_tolerance_rad,
        "rms_effect_vs_open_loop_rad": float(np.sqrt(np.mean(diff**2))),
        "integrated_abs_effect_rad_s": float(np.sum(np.abs(diff[:-1])) * dt_s),
        "recovery_to_10pct_peak_after_load_ms": recovery_ms,
        "first_sensory_event_ms": first_sensory,
        "first_mapped_motor_spike_ms": first_motor,
        "first_nonneutral_command_ms": first_command,
        "first_physical_response_ms": first_physical,
        "causal_order_passed": bool(causal),
        "feedback_sensory_causality_passed": bool(feedback_causal),
        "peak_decoded_target_offset_rad": on.get("peak_decoded_target_offset_rad"),
        "sensory_clamp_vs_baseline_rad": clamp_error,
        **metrics,
        "feedback_on_mapped_motor_spikes": on["mapped_motor_spikes"],
        "reason": "Causal physical feedback observed"
        if passed
        else (
            "No measurable mapped-motor feedback under this unchanged model/profile; "
            "do not claim embodiment acceptance"
        ),
    }


def compare_reproductions(left, right):
    """Require equal discrete traces and tightly matching physical state across hosts."""
    if (Path(left) / "motor_clamp").exists() != (Path(right) / "motor_clamp").exists():
        raise ValueError("Reproductions have different condition sets")
    conditions = (
        ("feedback_off", "sensory_clamp", "motor_clamp", "feedback_on")
        if (Path(left) / "motor_clamp").exists() and (Path(right) / "motor_clamp").exists()
        else ("feedback_off", "sensory_clamp", "feedback_on")
    )
    for condition in conditions:
        a, b = Path(left) / condition, Path(right) / condition
        ra, rb = validate_run(a), validate_run(b)
        for key in (
            "status",
            "duration_ms",
            "sensory_events",
            "total_cns_spikes",
            "mapped_motor_spikes",
            "nonneutral_motor_commands",
        ):
            if ra[key] != rb[key]:
                raise ValueError(f"Different {condition} metric: {key}")
        pa, pb = read_json(a / "provenance.json"), read_json(b / "provenance.json")
        for key in (
            "graph_manifest",
            "catalogue_manifest",
            "motor_manifest_hash",
            "sensory_manifest_hash",
            "lif_config_sha256",
        ):
            if pa[key] != pb[key]:
                raise ValueError(f"Different reproduction provenance: {key}")
        for key in ("seed", "clocks", "initial_condition", "encoder_mode", "monitor"):
            if pa["config"][key] != pb["config"][key]:
                raise ValueError(f"Different reproduction config: {key}")
        for name in (
            "body_states",
            "sensory_activity",
            "sensory_events",
            "neural_spikes",
            "motor_activity",
            "motor_commands",
            "timeline",
        ):
            first, second = [
                pq.read_table(p / f"{name}.parquet").to_pandas().sort_index(axis=1) for p in (a, b)
            ]
            if first.empty and second.empty:
                continue
            exact = name in {"sensory_events", "neural_spikes", "motor_activity", "motor_commands"}
            pd.testing.assert_frame_equal(
                first, second, check_dtype=False, check_exact=exact, rtol=0, atol=1e-9
            )
    return {
        "reproduction_passed": True,
        "physical_tolerance": 1e-9,
        "discrete_events_and_motor_commands": "exact",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-effect", action="store_true")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if (args.directory / "comparison.json").exists():
        summary = read_json(args.directory / "comparison.json")
        if "conditions" in summary:
            result = {
                "conditions": {
                    name: validate_run(args.directory / name) for name in summary["conditions"]
                }
            }
        else:
            result = compare_conditions(args.directory)
    else:
        result = validate_run(args.directory)
    print(json.dumps(result, indent=2))
    if args.compare:
        print(json.dumps(compare_reproductions(args.directory, args.compare), indent=2))
    if args.require_effect and not result.get("acceptance_passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
