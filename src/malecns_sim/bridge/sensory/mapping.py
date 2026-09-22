"""Resolve reviewed sensory IDs against immutable annotations and a named body joint."""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml

from malecns_sim.bridge.sensory.body_state import JointStateReader


@dataclass(frozen=True)
class SensoryMapping:
    policy: dict
    channels: tuple[dict, ...]
    rows: tuple[dict, ...]
    sensory_identity: dict[int, int]

    @property
    def sensory_dt_ms(self) -> float:
        return float(self.policy["sensory_dt_ms"])


def load_policy(path: str | Path | None = None) -> dict:
    if path is None:
        local = Path("configs/bridge/sensory_lf_tibia_v1.yaml")
        path = (
            local
            if local.exists()
            else files("malecns_sim.bridge.sensory").joinpath("profiles/sensory_lf_tibia_v1.yaml")
        )
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_mapping(policy: dict, catalogue, sim) -> SensoryMapping:
    if policy.get("mapping_version") != "sensory-bridge-v1" or not policy.get("channels"):
        raise ValueError("Missing sensory mapping or unsupported version")
    dt = float(policy["sensory_dt_ms"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid sensory clock")
    ratio = dt / (sim.model.opt.timestep * 1000)
    if round(ratio) < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
        raise ValueError("Sensory interval must contain an integer number of physics steps")
    sensory = {int(row["body_id"]): row for row in catalogue.sensory().to_pylist()}
    channels, rows, seen_ids, seen_names = [], [], set(), set()
    for source in policy["channels"]:
        name = source["name"]
        if name in seen_names:
            raise ValueError("Duplicate sensory channel")
        seen_names.add(name)
        body = source["source"]
        if body.get("signal") not in {"position", "velocity"}:
            raise ValueError("Unsupported body sensory signal")
        reader = JointStateReader(sim.model, body["joint"], body.get("actuator"))
        population = source["malecns"]
        ids, expected = population.get("body_ids"), population.get("annotations", {})
        required = {"class", "subclass", "side", "entryNerve"}
        if not ids or not required <= set(expected):
            raise ValueError("Sensory population needs explicit IDs and annotation evidence")
        encoder = source["encoder"]
        needed = {"type", "min_value", "max_value", "min_rate_hz", "max_rate_hz", "gain_mv"}
        if not needed <= set(encoder) or encoder["type"] not in {
            "linear_position_encoder_v1",
            "directional_velocity_encoder_v1",
        }:
            raise ValueError("Unsupported or incomplete sensory encoder")
        if encoder["type"].startswith("linear_position") and body["signal"] != "position":
            raise ValueError("Position encoder requires a position signal")
        if encoder["type"].startswith("directional_velocity") and body["signal"] != "velocity":
            raise ValueError("Velocity encoder requires a velocity signal")
        values = [
            encoder[key]
            for key in ("min_value", "max_value", "min_rate_hz", "max_rate_hz", "gain_mv")
        ]
        if not np.isfinite(values).all() or not (
            encoder["max_value"] > encoder["min_value"]
            and 0 <= encoder["min_rate_hz"] <= encoder["max_rate_hz"]
            and encoder["gain_mv"] >= 0
        ):
            raise ValueError("Invalid sensory encoder range/rate/gain")
        if encoder["type"] == "directional_velocity_encoder_v1" and encoder.get(
            "direction"
        ) not in {"positive", "negative"}:
            raise ValueError("Directional velocity encoder needs positive or negative direction")
        evidence = source["evidence"]
        if not all(
            evidence.get(key) for key in ("source", "confidence", "reference", "model_assumption")
        ):
            raise ValueError("Missing sensory annotation evidence or model assumption")
        nodes = []
        for body_id in ids:
            if type(body_id) is not int or body_id not in sensory:
                raise ValueError(f"Unknown/non-sensory neuron: {body_id}")
            if body_id in seen_ids:
                raise ValueError("Sensory neuron appears in multiple channels")
            seen_ids.add(body_id)
            row = sensory[body_id]
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Sensory annotation evidence mismatch for {body_id}")
            node = int(row["node_index"])
            nodes.append(node)
            rows.append(
                {
                    "body_id": body_id,
                    "node_index": node,
                    "channel": name,
                    "joint": body["joint"],
                    "signal": body["signal"],
                    "class": row["class"],
                    "subclass": row["subclass"],
                    "type": row["type"],
                    "entry_nerve": row["entryNerve"],
                    "side": row["side"],
                    "receptor_type": row["receptorType"],
                    "evidence_source": evidence["source"],
                    "confidence": evidence["confidence"],
                }
            )
        channels.append(
            {
                "name": name,
                "source": {
                    **body,
                    "joint_id": reader.joint_id,
                    "qpos_address": reader.qpos_address,
                    "qvel_address": reader.qvel_address,
                },
                "nodes": nodes,
                "encoder": dict(encoder),
                "evidence": dict(evidence),
            }
        )
    return SensoryMapping(
        policy,
        tuple(channels),
        tuple(rows),
        {int(row["node_index"]): body for body, row in sensory.items()},
    )
