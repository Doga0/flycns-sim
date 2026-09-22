"""Validate sensory mapping provenance and one-way replay artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from malecns_sim.bridge.sensory.mapping import resolve_mapping
from malecns_sim.cns.graph.store import file_hashes


def validate_mapping(directory, catalogue, sim):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("motor_body_coupling") is not False:
        raise ValueError("Unsupported sensory bridge manifest")
    if file_hashes(catalogue.directory / "manifest.json") != manifest["catalogue_manifest"]:
        raise ValueError("Sensory mapping and catalogue provenance differ")
    if manifest["graph_manifest"] != catalogue.manifest["graph"]["manifest"]:
        raise ValueError("Sensory mapping and graph provenance differ")
    required = {"sensory_mapping.parquet", "sensory_audit.json"}
    if set(manifest.get("artifacts", {})) != required:
        raise ValueError("Missing sensory mapping artifacts")
    for name, hashes in manifest["artifacts"].items():
        if file_hashes(directory / name) != hashes:
            raise ValueError(f"Sensory mapping checksum mismatch: {name}")
    mapping = resolve_mapping(manifest["policy"], catalogue, sim)
    if pq.read_table(directory / "sensory_mapping.parquet").to_pylist() != list(mapping.rows):
        raise ValueError("Saved sensory mapping does not reproduce from annotation evidence")
    if json.loads(json.dumps(mapping.channels)) != manifest["resolved_channels"]:
        raise ValueError("Compiled body or resolved sensory mapping differs from provenance")
    if manifest["mapped_sensory_neurons"] != len(mapping.rows) or manifest[
        "unmapped_sensory_neurons"
    ] != len(mapping.sensory_identity) - len(mapping.rows):
        raise ValueError("Sensory mapping coverage counts differ from catalogue")
    return mapping


def validate_run(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    required = {
        "body_states.parquet",
        "sensory_activity.parquet",
        "sensory_spikes.parquet",
        "neural_spikes.parquet",
        "provenance.json",
    }
    if not required <= set(report.get("artifacts", {})):
        raise ValueError("Missing sensory replay artifacts")
    if report.get("motor_body_coupling") is not False:
        raise ValueError("v0.6 runs must not apply neural motor output to body")
    for name, hashes in report["artifacts"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe sensory artifact name")
        if file_hashes(directory / name) != hashes:
            raise ValueError(f"Sensory replay checksum mismatch: {name}")
    body = pq.read_table(directory / "body_states.parquet").to_pandas()
    activity = pq.read_table(directory / "sensory_activity.parquet").to_pandas()
    sensory = pq.read_table(directory / "sensory_spikes.parquet").to_pandas()
    neural = pq.read_table(directory / "neural_spikes.parquet").to_pandas()
    for table in (body, activity, sensory, neural):
        if not np.isfinite(table.select_dtypes(include="number").to_numpy()).all():
            raise ValueError("Nonfinite sensory replay artifact")
    dt = report["sensory_dt_ms"]
    expected = np.arange(0, report["duration_ms"] + dt, dt)
    if not np.allclose(body.t_ms, expected, rtol=0, atol=1e-8):
        raise ValueError("Body state timestamps differ from the sensory clock")
    if not np.allclose(activity.t_start_ms.unique(), expected[:-1], rtol=0, atol=1e-8):
        raise ValueError("Sensory activity timestamps differ from body-state windows")
    if not ((activity.normalized_value >= 0) & (activity.normalized_value <= 1)).all():
        raise ValueError("Sensory normalization escaped [0,1]")
    if (
        (activity.rate_hz < 0).any()
        or (sensory.t_ms < 0).any()
        or (sensory.t_ms >= report["duration_ms"] + dt).any()
    ):
        raise ValueError("Invalid sensory rates or event times")
    if report.get("mujoco_warnings") or not report.get("finite_state"):
        raise ValueError("Body trajectory did not finish with finite warning-free physics")
    return report


def _same_activity(left: pa.Table, right: pa.Table) -> bool:
    """Compare encoder activity semantically across floating-point platforms."""
    if left.column_names != right.column_names or left.num_rows != right.num_rows:
        return False
    for name in left.column_names:
        lhs, rhs = left[name], right[name]
        if pa.types.is_floating(lhs.type):
            if not np.allclose(lhs.to_numpy(), rhs.to_numpy(), rtol=0, atol=1e-10):
                return False
        elif not lhs.equals(rhs):
            return False
    return True


def compare_runs(directory, other_directory):
    """Check cross-platform deterministic inputs/spikes while allowing tiny q differences."""
    report, other = validate_run(directory), validate_run(other_directory)
    keys = (
        "duration_ms",
        "sensory_dt_ms",
        "physics_dt_ms",
        "neural_dt_ms",
        "mode",
        "seed",
        "zero_motion_control",
        "mapped_sensory_neurons",
        "sensory_spikes",
        "active_sensory_neurons",
        "active_downstream_neurons",
        "active_motor_neurons",
        "total_neural_spikes",
        "motor_body_coupling",
    )
    if any(report[key] != other[key] for key in keys):
        raise ValueError("Sensory replay summaries differ")
    left, right = Path(directory), Path(other_directory)
    if not _same_activity(
        pq.read_table(left / "sensory_activity.parquet"),
        pq.read_table(right / "sensory_activity.parquet"),
    ):
        raise ValueError("Sensory activity differs beyond floating-point tolerance")
    for name in ("sensory_spikes.parquet", "neural_spikes.parquet"):
        if not pq.read_table(left / name).equals(pq.read_table(right / name)):
            raise ValueError(f"Deterministic event artifact differs: {name}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    report = validate_run(args.directory)
    if args.compare:
        compare_runs(args.directory, args.compare)
        print("Sensory activity matches within 1e-10; input and neural spikes are identical.")
    print(f"Validated {report['sensory_spikes']} sensory events; "
          f"{report['active_downstream_neurons']} active downstream neurons.")


if __name__ == "__main__":
    main()
