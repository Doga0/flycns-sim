"""Resolve reviewed body IDs against immutable annotations and real actuators."""

from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import mujoco
import numpy as np
import yaml

from malecns_sim.bridge.motor_catalogue import body_actuators


@dataclass(frozen=True)
class MotorMapping:
    policy: dict
    channels: tuple[dict, ...]
    rows: tuple[dict, ...]
    motor_identity: dict[int, int]

    @property
    def control_dt_ms(self):
        return float(self.policy["control_dt_ms"])


def load_policy(path: str | Path | None = None) -> dict:
    if path is None:
        local = Path("configs/bridge/lf_tibia_v1.yaml")
        path = (
            local
            if local.exists()
            else files("malecns_sim.bridge").joinpath("profiles/lf_tibia_v1.yaml")
        )
    return yaml.safe_load(
        path.read_text(encoding="utf-8")
        if not isinstance(path, str)
        else Path(path).read_text(encoding="utf-8")
    )


def resolve_mapping(policy: dict, catalogue, sim) -> MotorMapping:
    if policy.get("mapping_version") != "motor-bridge-v1" or not policy.get("channels"):
        raise ValueError("Missing mapping or unsupported version")
    dt = float(policy["control_dt_ms"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid control clock")
    ratio = dt / (sim.model.opt.timestep * 1000)
    if round(ratio) < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
        raise ValueError("Control interval must contain an integer number of physics steps")
    motors = {int(r["body_id"]): r for r in catalogue.motor().to_pylist()}
    inventory = body_actuators(sim)
    channels, rows, seen_ids, seen_names, seen_actuators = [], [], set(), set(), set()
    for source in policy["channels"]:
        name = source["name"]
        body = source["body"]
        actuator = inventory.get(body["actuator"])
        if actuator is None:
            raise ValueError(f"Unknown actuator: {body['actuator']}")
        if name in seen_names or actuator.actuator_name in seen_actuators:
            raise ValueError("Duplicate channel/actuator mapping")
        seen_names.add(name)
        seen_actuators.add(actuator.actuator_name)
        if actuator.joint_name != body["joint"]:
            raise ValueError("Actuator joint differs from explicit mapping")
        if actuator.side != body["side"] or actuator.leg != body["leg"]:
            raise ValueError("Body side/leg differs from compiled joint identity")
        i, j = actuator.actuator_index, actuator.joint_id
        model = sim.model
        if (
            model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE
            or model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE
            or model.actuator_gainprm[i, 0] <= 0
            or model.actuator_biasprm[i, 1] != -model.actuator_gainprm[i, 0]
            or not np.array_equal(model.actuator_gear[i], [1, 0, 0, 0, 0, 0])
        ):
            raise ValueError("Bridge requires direct hinge position actuators")
        decoder = source["decoder"]
        if decoder["mode"] != "antagonistic_rate":
            raise ValueError("Unsupported decoder")
        keys = (
            "gain_rad",
            "r0_hz",
            "rmax_hz",
            "tau_ms",
            "max_delta_rad",
            "min_offset_rad",
            "max_offset_rad",
            "extensor_direction",
        )
        if not np.isfinite([decoder[k] for k in keys]).all():
            raise ValueError("Nonfinite decoder parameter")
        if (
            decoder["gain_rad"] <= 0
            or decoder["r0_hz"] < 0
            or decoder["rmax_hz"] <= decoder["r0_hz"]
            or decoder["tau_ms"] <= 0
            or decoder["max_delta_rad"] <= 0
            or not decoder["min_offset_rad"] < 0 < decoder["max_offset_rad"]
            or decoder["extensor_direction"] not in (-1, 1)
        ):
            raise ValueError("Invalid decoder bounds")
        neutral = actuator.neutral_position
        low, high = neutral + decoder["min_offset_rad"], neutral + decoder["max_offset_rad"]
        for bounds in (actuator.control_range, actuator.joint_range):
            if bounds is not None:
                low, high = max(low, bounds[0]), min(high, bounds[1])
        if not np.isfinite([neutral, low, high]).all() or not low < neutral < high:
            raise ValueError("Neutral position is outside safe control bounds")
        evidence = source["evidence"]
        if (
            evidence.get("source") != "malecns_annotation"
            or not evidence.get("mechanical_assumption")
            or not evidence.get("confidence")
            or not evidence.get("reference")
        ):
            raise ValueError("Missing anatomical evidence or mechanical assumption")
        groups = {}
        for group in ("flexor", "extensor"):
            population = source["male_cns"][group]
            ids, expected = population["body_ids"], population["annotations"]
            if not ids or not {"type", "side", "subclass", "exitNerve", "somaNeuromere"} <= set(
                expected
            ):
                raise ValueError("Population needs explicit IDs and complete annotation evidence")
            groups[group] = []
            for body_id in ids:
                if type(body_id) is not int or body_id not in motors:
                    raise ValueError(f"Unknown motor neuron: {body_id}")
                if body_id in seen_ids:
                    raise ValueError("Motor neuron appears in multiple populations")
                seen_ids.add(body_id)
                r = motors[body_id]
                if any(r.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Annotation evidence mismatch for {body_id}")
                if expected["side"] != body["side"]:
                    raise ValueError("CNS/body side mismatch")
                node = int(r["node_index"])
                groups[group].append(node)
                rows.append(
                    {
                        "body_id": body_id,
                        "node_index": node,
                        "channel": name,
                        "population": group,
                        "actuator_name": actuator.actuator_name,
                        "side": r["side"],
                        "subclass": r["subclass"],
                        "type": r["type"],
                        "exit_nerve": r["exitNerve"],
                        "soma_neuromere": r["somaNeuromere"],
                        "muscle_target": None,
                        "evidence_source": evidence["source"],
                        "confidence": evidence["confidence"],
                    }
                )
        channels.append(
            {
                "name": name,
                "body": {**asdict(actuator), "side": body["side"], "leg": body["leg"]},
                "groups": groups,
                "decoder": dict(decoder),
                "minimum": float(low),
                "maximum": float(high),
                "evidence": dict(evidence),
            }
        )
    return MotorMapping(
        policy, tuple(channels), tuple(rows), {int(r["node_index"]): b for b, r in motors.items()}
    )
