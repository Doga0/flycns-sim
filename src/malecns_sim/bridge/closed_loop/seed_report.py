"""Aggregate matched full-CNS load experiments across declared random seeds."""

import argparse
import copy
from pathlib import Path

from malecns_sim.bridge.closed_loop.validate import compare_conditions, read_json
from malecns_sim.bridge.store import stage_output
from malecns_sim.cns.graph.store import write_json


def summarize(directories, expected_seeds):
    """Do not select successful seeds; require the whole declared set."""
    if len(directories) != len(expected_seeds) or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("One distinct run is required for every declared seed")
    rows = []
    reference = None
    for directory in directories:
        directory = Path(directory)
        comparison = compare_conditions(directory)
        if "motor_clamp_mapped_motor_spikes" not in comparison:
            raise ValueError("Seed report requires the four-condition v0.8.1 protocol")
        provenance = read_json(directory / "feedback_on" / "provenance.json")
        config = copy.deepcopy(provenance["config"])
        seed = config.pop("seed")
        if seed not in expected_seeds or any(row["seed"] == seed for row in rows):
            raise ValueError("Unexpected or repeated seed")
        identity = {
            "config_without_seed": config,
            "graph_manifest": provenance["graph_manifest"],
            "catalogue_manifest": provenance["catalogue_manifest"],
            "motor_manifest_hash": provenance["motor_manifest_hash"],
            "sensory_manifest_hash": provenance["sensory_manifest_hash"],
            "lif_config_sha256": provenance["lif_config_sha256"],
            "body_specification": provenance["body_specification"],
        }
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("Seed runs differ in configuration or provenance")
        rows.append({"seed": seed, "directory": str(directory), **comparison})
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != sorted(expected_seeds):
        raise ValueError("Declared seed set is incomplete")
    return {
        "expected_seeds": sorted(expected_seeds),
        "replication_passed": all(row["acceptance_passed"] for row in rows),
        "functional_target_0p01_rad_all_seeds": all(
            row["functional_target_0p01_rad_met"] for row in rows
        ),
        "runs": rows,
        "common_provenance": reference,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-effect", action="store_true")
    args = parser.parse_args()
    report = summarize(args.directories, args.seeds)
    with stage_output(args.output) as stage:
        write_json(stage / "report.json", report)
    print(f"Replication passed: {report['replication_passed']}")
    if args.require_effect and not report["replication_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
