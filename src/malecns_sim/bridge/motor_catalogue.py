"""Audit released motor annotations and inventory the actual compiled body."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from malecns_sim.cns.graph.store import write_json
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.inspect import DEFAULT_CATALOGUE


@dataclass(frozen=True)
class MaleCNSMotorTarget:
    body_id: int
    node_index: int
    side: str | None
    subclass: str | None
    type: str | None
    exit_nerve: str | None
    muscle_target: str | None


@dataclass(frozen=True)
class BodyActuator:
    actuator_name: str
    actuator_index: int
    joint_name: str
    joint_id: int
    qpos_address: int
    axis: tuple[float, ...]
    control_range: tuple[float, float] | None
    joint_range: tuple[float, float] | None
    neutral_position: float
    side: str | None
    leg: str | None


def body_actuators(sim) -> dict[str, BodyActuator]:
    """Read names/axes/limits from MuJoCo, without inferring CNS correspondence."""
    model = sim.model
    result = {}
    for i in range(model.nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = int(model.actuator_trnid[i, 0])
        name = model.actuator(i).name
        joint_name = model.joint(j).name
        # Explicit body naming convention only; never used to select CNS cells.
        leg = next(
            (
                code
                for code in ("LF", "LM", "LH", "RF", "RM", "RH")
                if joint_name.startswith(f"fly/{code.lower()}_")
            ),
            None,
        )
        result[name] = BodyActuator(
            name,
            i,
            model.joint(j).name,
            j,
            int(model.jnt_qposadr[j]),
            tuple(float(x) for x in model.jnt_axis[j]),
            tuple(model.actuator_ctrlrange[i]) if model.actuator_ctrllimited[i] else None,
            tuple(model.jnt_range[j]) if model.jnt_limited[j] else None,
            float(sim.data.ctrl[i]),
            leg[0] if leg else None,
            leg,
        )
    return result


def audit_motors(catalogue: IOCatalog) -> dict:
    frame = catalogue.motor().table.to_pandas()
    target_fields = [c for c in frame if "muscle" in c.lower() or "target" in c.lower()]
    coverage = {}
    for field in ["type", "subclass", "side", "exitNerve", "somaNeuromere", *target_fields]:
        if field not in frame:
            coverage[field] = {"present": False, "nonempty": 0, "values": {}}
            continue
        values = frame[field].dropna().astype(str)
        values = values[values.str.strip().ne("")]
        coverage[field] = {
            "present": True,
            "nonempty": len(values),
            "missing": len(frame) - len(values),
            "values": {str(k): int(v) for k, v in values.value_counts().sort_index().items()},
        }
    return {
        "motor_neurons": len(frame),
        "available_columns": list(frame),
        "coverage": coverage,
        "possible_target_fields": target_fields,
        "muscle_target_note": (
            "Only existing target/muscle columns are reported; type strings are never parsed "
            "to manufacture exact muscle innervation. Nonempty does not mean unambiguous."
        ),
        "known_left_or_right": int(frame.side.isin(["L", "R"]).sum()),
    }


def motor_targets(catalogue: IOCatalog) -> list[MaleCNSMotorTarget]:
    # This release has no exact muscle column. A future schema needs an explicit policy.
    return [
        MaleCNSMotorTarget(
            int(r["body_id"]),
            int(r["node_index"]),
            r.get("side"),
            r.get("subclass"),
            r.get("type"),
            r.get("exitNerve"),
            None,
        )
        for r in catalogue.motor().to_pylist()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output", type=Path, default=Path("outputs/bridge/motor-audit.json"))
    args = parser.parse_args()
    from malecns_sim.simulation.fly import build_fly

    report = audit_motors(IOCatalog.load(args.catalogue))
    sim = build_fly()
    report["body_actuators"] = [asdict(a) for a in body_actuators(sim).values()]
    report["physics_dt_ms"] = float(sim.model.opt.timestep * 1000)
    if not np.isfinite(sim.data.ctrl).all():
        raise ValueError("Nonfinite neutral body controls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"Motor neurons: {report['motor_neurons']}; audit: {args.output}")
    for field, coverage in report["coverage"].items():
        print(f"{field}: {coverage['nonempty']} nonempty")
    print(f"Exact target/muscle columns: {report['possible_target_fields']}")


if __name__ == "__main__":
    main()
