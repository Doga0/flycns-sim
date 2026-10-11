from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pytest

from malecns_sim.bridge.closed_loop.clocks import Clocks
from malecns_sim.bridge.closed_loop.recorder import save_run
from malecns_sim.bridge.closed_loop.scheduler import SPIKE_SCHEMA, ClosedLoopScheduler, MuJoCoBody
from malecns_sim.bridge.controller import BodyController
from malecns_sim.bridge.decoder import MotorDecoder
from malecns_sim.bridge.mapping import load_policy, resolve_mapping
from malecns_sim.bridge.replay import decode_trace
from malecns_sim.bridge.sensory.body_state import BodyState, JointStateReader
from malecns_sim.bridge.sensory.encoder import SensoryEncoder
from malecns_sim.bridge.sensory.mapping import SensoryMapping
from malecns_sim.bridge.sensory.replay import encode_states
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.simulation.fly import build_fly


@pytest.fixture
def make_scheduler():
    def make(mode="feedback_on", recorded=(), safety=None):
        sim = build_fly()
        policy = load_policy()
        channel = policy["channels"][0]
        channel["male_cns"]["flexor"]["body_ids"] = [11]
        channel["male_cns"]["extensor"]["body_ids"] = [12]
        rows = [
            dict(
                body_id=body,
                node_index=node,
                role="motor",
                side="L",
                subclass="fl",
                exitNerve="ProLN",
                somaNeuromere="T1",
                type=kind,
            )
            for node, body, kind in [(1, 11, "Ti flexor MN"), (2, 12, "Ti extensor MN")]
        ]
        motor = resolve_mapping(
            policy, IOCatalog(Path("unused"), {}, pa.Table.from_pylist(rows)), sim
        )
        joint = channel["body"]["joint"]
        reader = JointStateReader(sim.model, joint, channel["body"]["actuator"])
        neutral = motor.channels[0]["body"]["neutral_position"]
        sim.data.qpos[reader.qpos_address] += 0.2
        mujoco.mj_forward(sim.model, sim.data)
        sensory = SensoryMapping(
            {"sensory_dt_ms": 1.0},
            (
                {
                    "name": "synthetic_position",
                    "source": {"joint": joint, "signal": "position"},
                    "nodes": [0],
                    "encoder": {
                        "type": "linear_position_encoder_v1",
                        "min_value": neutral,
                        "max_value": neutral + 0.35,
                        "min_rate_hz": 0,
                        "max_rate_hz": 100,
                        "gain_mv": 68.75,
                    },
                },
            ),
            ({"node_index": 0, "body_id": 10},),
            {0: 10},
        )
        graph = NeuralGraph(
            np.array([10, 11, 12], dtype=np.uint64),
            np.arange(3, dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([1000], dtype=np.uint32),
            ["acetylcholine"] * 3,
            {},
        )
        neural = Brian2LIFBackend(graph)
        neural.configure_event_input([0], [68.75])
        clocks = Clocks(0.1, 0.1, 1, 10)
        limits = {
            "max_duration_ms": 500,
            "max_abs_joint_offset_rad": 0.4,
            "max_abs_velocity_rad_s": 100,
            "max_sensory_rate_hz": 100,
            "max_motor_rate_hz": 1000,
            "max_total_spikes_per_ms": 10000,
        }
        limits.update(safety or {})
        body = MuJoCoBody(sim, reader, BodyController(motor, sim.model), clocks, limits, neutral)
        return ClosedLoopScheduler(
            body=body,
            neural=neural,
            # Seed 34 emits at t=0 for this one-cell synthetic probe.
            encoder=SensoryEncoder(sensory, mode="poisson", seed=34),
            decoder=MotorDecoder(motor),
            clocks=clocks,
            safety=limits,
            node_identity={0: 10, 1: 11, 2: 12},
            input_gains={0: 68.75},
            mode=mode,
            recorded_motor=recorded,
        )

    return make


def test_motor_only_matches_v05_commands_and_causal_targets(make_scheduler):
    rows = [
        {"t_ms": 9.9, "node_index": 2, "body_id": 12},
        {"t_ms": 10.0, "node_index": 2, "body_id": 12},
    ]
    scheduler = make_scheduler("motor_only", rows).run(30)
    assert scheduler.status == "COMPLETED"
    offline, _ = decode_trace(
        scheduler.decoder.mapping, pa.Table.from_pylist(rows, schema=SPIKE_SCHEMA), 30
    )
    for actual, expected in zip(scheduler.motor_commands, offline):
        assert all(actual[k] == v for k, v in expected.to_dict().items())
    neutral = scheduler.body.neutral
    assert all(r["target"] == neutral for r in scheduler.timeline[:10])
    assert scheduler.timeline[10]["target"] == offline[0].target
    assert scheduler.drift_max < 1e-7


def test_sensory_only_matches_v06_offline_trace(make_scheduler):
    scheduler = make_scheduler("sensory_only").run(30)
    states = [BodyState(**row) for row in scheduler.body_states]
    activity, events, _ = encode_states(scheduler.encoder.mapping, states, mode="poisson", seed=34)
    for expected, actual in zip(activity.to_pylist(), scheduler.sensory_activity):
        for key, value in expected.items():
            assert actual[key] == (
                pytest.approx(value, abs=1e-8) if isinstance(value, float) else value
            )
    for expected, actual in zip(events.to_pylist(), scheduler.sensory_events):
        for key, value in expected.items():
            assert actual[key] == (
                pytest.approx(value, abs=1e-8) if isinstance(value, float) else value
            )
    assert len(events) == len(scheduler.sensory_events)
    assert not any(r["applied"] for r in scheduler.motor_commands)


def test_synthetic_feedback_changes_physics_clamp_removes_effect_and_seed_repeats(make_scheduler):
    on = make_scheduler().run(50)
    repeat = make_scheduler().run(50)
    off = make_scheduler("feedback_off").run(50)
    clamp = make_scheduler("sensory_clamp").run(50)
    assert on.status == "COMPLETED"
    assert on.sensory_events == repeat.sensory_events
    assert on.neural_spikes == repeat.neural_spikes
    assert on.motor_commands == repeat.motor_commands
    assert clamp.body_states == off.body_states
    assert not clamp.neural_spikes and not off.neural_spikes
    assert on.neural_spikes and any(r["node_index"] == 2 for r in on.motor_activity)
    assert (
        max(
            abs(a["joint_position_rad"] - b["joint_position_rad"])
            for a, b in zip(on.body_states, off.body_states)
        )
        > 1e-6
    )


@pytest.mark.parametrize(
    "guard,value",
    [
        ("max_abs_velocity_rad_s", 1e-10),
        ("max_sensory_rate_hz", 1),
        ("max_total_spikes_per_ms", 0.1),
        ("max_motor_rate_hz", 1),
    ],
)
def test_guard_stops_with_partial_recordings(make_scheduler, guard, value):
    scheduler = make_scheduler(safety={guard: value}).run(50)
    assert scheduler.status == "ABORTED_SAFETY_LIMIT"
    assert guard in scheduler.reason
    assert scheduler.body_states


def test_bad_clock_ratios_and_drift_fail():
    with pytest.raises(ValueError):
        Clocks(0.3, 0.1, 1, 10)
    with pytest.raises(RuntimeError, match="Clock drift"):
        Clocks(0.1, 0.1, 1, 10).check(1, 1, 1.01)


def test_aborted_run_preserves_artifacts(make_scheduler, tmp_path):
    scheduler = make_scheduler(safety={"max_sensory_rate_hz": 1}).run(20)
    catalogue = IOCatalog(
        Path("unused"),
        {},
        pa.Table.from_pylist(
            [
                {
                    "node_index": n,
                    "body_id": n + 10,
                    "role": "sensory" if n == 0 else "motor",
                    "type": "synthetic",
                    "side": "L",
                }
                for n in range(3)
            ]
        ),
    )
    report = save_run(tmp_path / "abort", scheduler, {"config": {"duration_ms": 20}}, catalogue)
    assert report["status"] == "ABORTED_SAFETY_LIMIT"
    assert (tmp_path / "abort/report.json").exists()
    assert len(report["artifacts"]) == 8


def test_single_condition_cli_validation(tmp_path, monkeypatch, capsys):
    from malecns_sim.bridge.closed_loop import validate

    (tmp_path / "comparison.json").write_text('{"conditions": ["feedback_on"]}')
    visited = []

    def check(path):
        visited.append(path)
        return {"status": "COMPLETED"}

    monkeypatch.setattr(validate, "validate_run", check)
    monkeypatch.setattr("sys.argv", ["validate", str(tmp_path)])
    validate.main()
    assert visited == [tmp_path / "feedback_on"]
    assert "COMPLETED" in capsys.readouterr().out
