"""Validate bridge provenance, saved command safety and replay state integrity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from malecns_sim.bridge.mapping import MotorMapping, resolve_mapping
from malecns_sim.cns.graph.store import file_hashes


def validate_mapping(directory, catalogue, sim):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != 1 or manifest["sensory_feedback"] is not False:
        raise ValueError("Unsupported bridge manifest")
    if file_hashes(catalogue.directory / "manifest.json") != manifest["catalogue_manifest"]:
        raise ValueError("Mapping and catalogue provenance differ")
    if manifest["graph_manifest"] != catalogue.manifest["graph"]["manifest"]:
        raise ValueError("Mapping and graph provenance differ")
    if set(manifest["artifacts"]) != {"motor_mapping.parquet", "motor_audit.json"}:
        raise ValueError("Missing mapping artifacts")
    for name, hashes in manifest["artifacts"].items():
        if file_hashes(directory / name) != hashes:
            raise ValueError(f"Mapping checksum mismatch: {name}")
    mapping = resolve_mapping(manifest["policy"], catalogue, sim)
    if pq.read_table(directory / "motor_mapping.parquet").to_pylist() != list(mapping.rows):
        raise ValueError("Saved mapping does not reproduce from annotation evidence")
    if json.loads(json.dumps(mapping.channels)) != manifest["resolved_channels"]:
        raise ValueError("Compiled body or resolved mapping differs from provenance")
    if manifest["mapped_motor_neurons"] != len(mapping.rows) or manifest[
        "unmapped_motor_neurons"
    ] != len(mapping.motor_identity) - len(mapping.rows):
        raise ValueError("Mapping coverage counts differ from catalogue")
    return mapping


def validate_run(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    required = {
        "neural_activity.parquet",
        "motor_activity.parquet",
        "motor_commands.parquet",
        "joint_states.parquet",
        "provenance.json",
    }
    if not required <= set(report["artifacts"]):
        raise ValueError("Missing replay artifacts")
    for name, hashes in report["artifacts"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or file_hashes(directory / name) != hashes
        ):
            raise ValueError(f"Replay checksum mismatch: {name}")
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    commands = pq.read_table(directory / "motor_commands.parquet").to_pandas()
    states = pq.read_table(directory / "joint_states.parquet").to_pandas()
    if not np.isfinite(commands.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite command artifact")
    if not np.isfinite(states.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite state artifact")
    channels = provenance["mapping"]["resolved_channels"]
    if provenance["sensory_feedback"] is not False or report["sensory_feedback"] is not False:
        raise ValueError("This replay format does not support sensory feedback")
    expected_names = {c["body"]["actuator_name"] for c in channels}
    if set(commands.actuator_name) != expected_names or set(states.actuator_name) != expected_names:
        raise ValueError("Missing or unknown actuator in replay artifacts")
    dt = report["control_dt_ms"]
    expected_times = np.arange(1, round(report["neural_duration_ms"] / dt) + 1) * dt
    for c in channels:
        rows = commands[commands.actuator_name == c["body"]["actuator_name"]]
        if not np.array_equal(rows.t_ms.to_numpy(), expected_times):
            raise ValueError("Command timestamps differ from causal control grid")
        targets = rows.target.to_numpy()
        if (
            np.any(targets < c["minimum"])
            or np.any(targets > c["maximum"])
            or np.any(
                np.abs(np.diff(np.r_[c["body"]["neutral_position"], targets]))
                > c["decoder"]["max_delta_rad"] + 1e-12
            )
        ):
            raise ValueError("Saved command violates range/slew bounds")
    if report["mujoco_warnings"] or not report["finite_state"]:
        raise ValueError("Replay did not finish with finite warning-free physics")
    # Independently reconstruct commands from the saved motor spike table. Hash
    # integrity alone would not detect a consistently rehashed but invalid command.
    from malecns_sim.bridge.replay import decode_trace

    motor = pq.read_table(directory / "motor_activity.parquet")
    neural = pq.read_table(directory / "neural_activity.parquet").to_pandas()
    expected_motor = neural[neural.role.eq("motor")].reset_index(drop=True)
    actual_motor = motor.to_pandas().drop(columns="is_mapped").reset_index(drop=True)
    if not expected_motor.equals(actual_motor):
        raise ValueError("Motor activity differs from source neural activity")
    identity = dict(zip(motor["node_index"].to_pylist(), motor["body_id"].to_pylist()))
    mapped_rows = []
    for c in channels:
        for group in ("flexor", "extensor"):
            for node in c["groups"][group]:
                mapped_rows.append({"node_index": node})
    mapping = MotorMapping(
        provenance["mapping"]["policy"], tuple(channels), tuple(mapped_rows), identity
    )
    decoded, decoder = decode_trace(mapping, motor, report["neural_duration_ms"])
    if commands.to_dict("records") != [c.to_dict() for c in decoded]:
        raise ValueError("Saved commands do not reproduce from motor spikes")
    mapped_nodes = {r["node_index"] for r in mapped_rows}
    if motor["is_mapped"].to_pylist() != [
        n in mapped_nodes for n in motor["node_index"].to_pylist()
    ]:
        raise ValueError("Incorrect mapped/unmapped motor flags")
    if report["unmapped_motor_spikes"] != decoder.unmapped_spikes or report[
        "active_unmapped_motor_neurons"
    ] != len(decoder.unmapped_nodes):
        raise ValueError("Unmapped motor counts disagree with saved spikes")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--compare", type=Path, help="Compare deterministic decoded commands")
    args = parser.parse_args()
    report = validate_run(args.directory)
    if args.compare:
        other = validate_run(args.compare)
        key = "motor_commands.parquet"
        if report["artifacts"][key] != other["artifacts"][key]:
            raise ValueError("Motor command traces differ")
        print("Motor command artifacts are byte-identical.")
    print(f"Validated {report['actuators_driven']} driven actuators; finite state.")


if __name__ == "__main__":
    main()
