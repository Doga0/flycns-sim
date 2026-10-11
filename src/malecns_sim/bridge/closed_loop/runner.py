"""Perturb-and-release LF tibia in full MaleCNS with matched feedback controls."""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from malecns_sim.bridge.closed_loop.clocks import Clocks, steps
from malecns_sim.bridge.closed_loop.recorder import save_run
from malecns_sim.bridge.closed_loop.scheduler import ClosedLoopScheduler, MuJoCoBody
from malecns_sim.bridge.closed_loop.validate import compare_conditions, validate_run
from malecns_sim.bridge.controller import BodyController
from malecns_sim.bridge.decoder import MotorDecoder
from malecns_sim.bridge.sensory.body_state import JointStateReader
from malecns_sim.bridge.sensory.encoder import SensoryEncoder
from malecns_sim.bridge.sensory.validate import validate_mapping as validate_sensory
from malecns_sim.bridge.store import stage_output
from malecns_sim.bridge.validate import validate_mapping as validate_motor
from malecns_sim.cns.graph.store import CNSGraph, file_hashes, write_json
from malecns_sim.cns.graph.validate import validate_graph
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.validate import validate_catalogue
from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.config import load_config
from malecns_sim.simulation.fly import build_fly


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def body_fingerprint(model):
    """Hash the complete compiled model before controller bounds are applied."""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def prepare(config):
    if config["version"] != "closed-loop-lf-tibia-v1":
        raise ValueError("Unknown closed-loop configuration")
    clocks = Clocks(**config["clocks"])
    steps(config["duration_ms"], clocks.motor_dt_ms)
    if type(config.get("frames", 0)) is not int or not 0 <= config.get("frames", 0) <= 20:
        raise ValueError("frames must be an integer in [0,20]")
    safety = config["safety"]
    required = {
        "max_duration_ms",
        "max_abs_joint_offset_rad",
        "max_abs_velocity_rad_s",
        "max_sensory_rate_hz",
        "max_motor_rate_hz",
        "max_total_spikes_per_ms",
    }
    if not required <= safety.keys() or any(
        not np.isfinite(safety[k]) or safety[k] <= 0 for k in required
    ):
        raise ValueError("Missing/nonpositive engineering safety limits")
    if config["duration_ms"] > safety["max_duration_ms"]:
        raise ValueError("Duration exceeds safety guard")
    lif = load_config(config["lif_config"])
    if digest(lif.to_dict()) != config["expected_lif_config_sha256"]:
        raise ValueError("LIF config differs from reviewed expected hash")
    catalogue_path, graph_path = Path(config["catalogue"]), Path(config["graph"])
    validate_catalogue(catalogue_path, graph_path)
    catalogue = IOCatalog.load(catalogue_path)
    sim = build_fly()
    motor = validate_motor(config["motor_mapping"], catalogue, sim)
    sensory = validate_sensory(config["sensory_mapping"], catalogue, sim)
    if len(motor.channels) != 1 or len(sensory.channels) != 1:
        raise ValueError("v0.7 profile requires exactly one LF joint channel")
    joint = config["initial_condition"]["joint"]
    if (
        motor.channels[0]["body"]["joint_name"] != joint
        or sensory.channels[0]["source"]["joint"] != joint
    ):
        raise ValueError("Motor/sensory/initial joint identities differ")
    if (
        lif.dt_ms != clocks.neural_dt_ms
        or sim.model.opt.timestep * 1000 != clocks.physics_dt_ms
        or motor.control_dt_ms != clocks.motor_dt_ms
        or sensory.sensory_dt_ms != clocks.sensory_dt_ms
    ):
        raise ValueError("Configured clocks differ from resolved bridge/backend clocks")
    offset = config["initial_condition"]["offset_rad"]
    if not np.isfinite(offset) or abs(offset) > safety["max_abs_joint_offset_rad"]:
        raise ValueError("Initial offset exceeds safety bounds")
    provenance = {
        "schema": 1,
        "config": copy.deepcopy(config),
        "config_sha256": digest(config),
        "lif_config": lif.to_dict(),
        "lif_config_sha256": digest(lif.to_dict()),
        "graph_manifest": file_hashes(graph_path / "manifest.json"),
        "catalogue_manifest": file_hashes(catalogue_path / "manifest.json"),
        "motor_mapping": json.loads((Path(config["motor_mapping"]) / "manifest.json").read_text()),
        "sensory_mapping": json.loads(
            (Path(config["sensory_mapping"]) / "manifest.json").read_text()
        ),
        "motor_manifest_hash": file_hashes(Path(config["motor_mapping"]) / "manifest.json"),
        "sensory_manifest_hash": file_hashes(Path(config["sensory_mapping"]) / "manifest.json"),
        "body_specification": {
            "compiled_model_sha256": body_fingerprint(sim.model),
            "builder": "malecns_sim.simulation.fly.build_fly",
            "nq": sim.model.nq,
            "nv": sim.model.nv,
            "nu": sim.model.nu,
            "joint": joint,
            "actuator": motor.channels[0]["body"]["actuator_name"],
        },
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("malecns-sim", "brian2", "mujoco", "flygym", "numpy")
        },
        "monitor": config["monitor"],
        "causal_order": (
            "read x(t), encode, neural [t,t+dt), body under old target [t,t+dt), "
            "decode/apply at boundary"
        ),
    }
    for manifest in (provenance["motor_mapping"], provenance["sensory_mapping"]):
        if (
            manifest["graph_manifest"] != provenance["graph_manifest"]
            or manifest["catalogue_manifest"] != provenance["catalogue_manifest"]
        ):
            raise ValueError("Cross-bridge graph/catalogue hash mismatch")
    return clocks, lif, catalogue, motor, sensory, provenance


def run_experiment(config, output, *, conditions=("feedback_off", "sensory_clamp", "feedback_on")):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    clocks, lif, catalogue, motor, sensory, provenance = prepare(config)
    with CNSGraph.load(config["graph"]) as stored, stage_output(output) as stage:
        validate_graph(stored)
        graph = NeuralGraph.from_graph(stored)
        identity = {int(n): int(b) for n, b in zip(graph.node_indices, graph.body_ids)}
        if not np.array_equal(graph.node_indices, np.arange(len(graph.body_ids))):
            raise ValueError("Closed loop currently requires the full graph identity mapping")
        gains = {n: c["encoder"]["gain_mv"] for c in sensory.channels for n in c["nodes"]}
        monitor = config["monitor"]
        if monitor not in {"full", "control_and_sensory"}:
            raise ValueError("Unknown spike monitor profile")
        recording = (
            None
            if monitor == "full"
            else tuple(sorted(set(gains) | {r["node_index"] for r in motor.rows}))
        )
        reports = {}
        for condition in conditions:
            print(
                f"Starting {condition}: {len(graph.body_ids)} neurons, {len(graph.src)} edges",
                flush=True,
            )
            sim = build_fly()
            if (
                body_fingerprint(sim.model)
                != provenance["body_specification"]["compiled_model_sha256"]
            ):
                raise ValueError("Compiled body changed after validating both mappings")
            reader = JointStateReader(
                sim.model,
                config["initial_condition"]["joint"],
                motor.channels[0]["body"]["actuator_name"],
            )
            neutral = motor.channels[0]["body"]["neutral_position"]
            sim.data.qpos[reader.qpos_address] += config["initial_condition"]["offset_rad"]
            mujoco.mj_forward(sim.model, sim.data)
            initial_hash = hashlib.sha256(
                b"".join(getattr(sim.data, n).tobytes() for n in ("qpos", "qvel", "ctrl", "act"))
            ).hexdigest()
            controller = BodyController(motor, sim.model)
            body = MuJoCoBody(sim, reader, controller, clocks, config["safety"], neutral)
            neural = Brian2LIFBackend(graph, lif, seed=config["seed"], spike_record_nodes=recording)
            neural.configure_event_input(list(gains), list(gains.values()))
            snapshots = []
            frame_ticks = set(
                np.rint(
                    np.linspace(
                        0, config["duration_ms"] / clocks.sensory_dt_ms, config.get("frames", 0)
                    )
                )
                * clocks.sensory_dt_ms
            )

            def capture(t_ms, sim=sim):
                if t_ms in frame_ticks:
                    snapshots.append(
                        {
                            **{
                                n: getattr(sim.data, n).copy()
                                for n in ("qpos", "qvel", "ctrl", "act")
                            },
                            "time": float(sim.data.time),
                        }
                    )

            scheduler = ClosedLoopScheduler(
                body=body,
                neural=neural,
                encoder=SensoryEncoder(sensory, mode=config["encoder_mode"], seed=config["seed"]),
                decoder=MotorDecoder(motor),
                clocks=clocks,
                safety=config["safety"],
                node_identity=identity,
                input_gains=gains,
                mode=condition,
                on_sample=capture,
            ).run(config["duration_ms"])
            p = {**provenance, "condition": condition, "initial_state_sha256": initial_hash}
            reports[condition] = save_run(stage / condition, scheduler, p, catalogue, snapshots)
            validate_run(stage / condition)
            print(
                json.dumps({k: v for k, v in reports[condition].items() if k != "artifacts"}),
                flush=True,
            )
            del scheduler, neural, body, controller, sim
            gc.collect()
        comparison = (
            compare_conditions(stage) if len(conditions) == 3 else {"conditions": list(conditions)}
        )
        write_json(stage / "comparison.json", comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/closed_loop/lf_tibia_v1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("all", "feedback_on", "feedback_off", "sensory_clamp"),
        default="all",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    conditions = (
        ("feedback_off", "sensory_clamp", "feedback_on")
        if args.condition == "all"
        else (args.condition,)
    )
    result = run_experiment(config, args.output, conditions=conditions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
