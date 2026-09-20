"""Build a provenance-bound motor mapping without running physics or LIF."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from malecns_sim.bridge.mapping import load_policy, resolve_mapping
from malecns_sim.bridge.motor_catalogue import audit_motors
from malecns_sim.bridge.store import stage_output
from malecns_sim.cns.graph.store import file_hashes, write_json
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.inspect import DEFAULT_CATALOGUE
from malecns_sim.io.validate import validate_catalogue

DEFAULT_MAPPING = DEFAULT_CATALOGUE.parent / "bridge-v1"


def build_mapping(policy, catalogue_dir: Path, sim, output: Path):
    validate_catalogue(catalogue_dir)
    catalogue = IOCatalog.load(catalogue_dir)
    mapping = resolve_mapping(policy, catalogue, sim)
    with stage_output(output) as stage:
        pq.write_table(
            pa.Table.from_pylist(list(mapping.rows)),
            stage / "motor_mapping.parquet",
            compression="zstd",
        )
        write_json(stage / "motor_audit.json", audit_motors(catalogue))
        manifest = {
            "schema": 1,
            "mapping_version": policy["mapping_version"],
            "policy": policy,
            "resolved_channels": mapping.channels,
            "catalogue_manifest": file_hashes(catalogue_dir / "manifest.json"),
            "graph_manifest": catalogue.manifest["graph"]["manifest"],
            "mapped_motor_neurons": len(mapping.rows),
            "unmapped_motor_neurons": len(mapping.motor_identity) - len(mapping.rows),
            "sensory_feedback": False,
            "artifacts": {
                name: file_hashes(stage / name)
                for name in ("motor_mapping.parquet", "motor_audit.json")
            },
        }
        write_json(stage / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MAPPING)
    args = parser.parse_args()
    from malecns_sim.simulation.fly import build_fly

    result = build_mapping(load_policy(args.policy), args.catalogue, build_fly(), args.output)
    print(
        f"Mapped motors: {result['mapped_motor_neurons']}; "
        f"unmapped: {result['unmapped_motor_neurons']}; artifact: {args.output}"
    )


if __name__ == "__main__":
    main()
