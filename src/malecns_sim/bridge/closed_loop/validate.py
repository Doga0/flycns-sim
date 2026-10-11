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
    for table in tables.values():
        if not np.isfinite(table.select_dtypes(include="number").to_numpy()).all():
            raise ValueError("Nonfinite completed timeline")
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
    return report


def compare_conditions(directory):
    directory = Path(directory)
    conditions = ("feedback_off", "sensory_clamp", "feedback_on")
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
    passed = (
        effect > 1e-10
        and clamp_error <= 1e-12
        and on["mapped_motor_spikes"] > 0
        and on["nonneutral_motor_commands"] > 0
    )
    return {
        "acceptance_passed": bool(passed),
        "max_effect_vs_open_loop_rad": effect,
        "sensory_clamp_vs_baseline_rad": clamp_error,
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
    for condition in ("feedback_off", "sensory_clamp", "feedback_on"):
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
