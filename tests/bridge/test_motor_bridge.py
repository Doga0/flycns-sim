from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pytest

from malecns_sim.bridge.controller import BodyController
from malecns_sim.bridge.decoder import MotorDecoder
from malecns_sim.bridge.mapping import load_policy, resolve_mapping
from malecns_sim.bridge.motor_catalogue import audit_motors, body_actuators
from malecns_sim.bridge.replay import decode_trace, replay_physics
from malecns_sim.bridge.validate import validate_mapping, validate_run
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.stimulus import Stimulus
from malecns_sim.simulation.fly import FlySimulation, build_fly


def spike_table(rows=()):
    return pa.Table.from_pylist(
        list(rows),
        schema=pa.schema(
            [("t_ms", pa.float64()), ("node_index", pa.int32()), ("body_id", pa.uint64())]
        ),
    )


@pytest.fixture(scope="module")
def source_body():
    return build_fly()


@pytest.fixture
def setup(source_body):
    # Independent model/state per test; controller changes only this copy's control limits.
    model = deepcopy(source_body.model)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    sim = FlySimulation(model, data, source_body.camera)
    policy = load_policy()
    channel = policy["channels"][0]
    channel["male_cns"]["flexor"]["body_ids"] = [1]
    channel["male_cns"]["extensor"]["body_ids"] = [2]
    rows = [
        {
            "body_id": body,
            "node_index": body * 10,
            "role": "motor",
            "side": "L",
            "subclass": "fl",
            "exitNerve": "ProLN",
            "somaNeuromere": "T1",
            "type": kind,
        }
        for body, kind in [(1, "Ti flexor MN"), (2, "Ti extensor MN"), (3, "unmapped type")]
    ]
    catalogue = IOCatalog(Path("unused"), {}, pa.Table.from_pylist(rows))
    return policy, catalogue, sim


def test_no_spikes_means_neutral_commands_and_zero_physics_effect(setup):
    policy, catalogue, sim = setup
    mapping = resolve_mapping(policy, catalogue, sim)
    commands, decoder = decode_trace(mapping, spike_table(), 100)
    neutral = mapping.channels[0]["body"]["neutral_position"]
    assert all(c.target == neutral and c.source_activity == 0 for c in commands)
    assert decoder.unmapped_spikes == 0
    states = replay_physics(sim, mapping, commands, 100, frame_count=0)
    assert np.array_equal(states["command_effect_rad"].to_numpy(), np.zeros(states.num_rows))


def test_population_rate_filter_antagonism_and_causal_window(setup):
    policy, catalogue, sim = setup
    mapping = resolve_mapping(policy, catalogue, sim)
    decoder = MotorDecoder(mapping)
    row = {"t_ms": 9.9, "node_index": 20, "body_id": 2}
    assert decoder.alpha["lf_tibia"] == float(
        "0.39346934028736657639620046500881954655808186451281"
    )
    command = decoder.decode(spike_table([row]), 0, 10)[0]
    neutral = mapping.channels[0]["body"]["neutral_position"]
    assert command.t_ms == 10
    assert command.extensor_rate_hz == 100
    assert command.extensor_filtered_hz == pytest.approx(100 * (1 - np.exp(-0.5)))
    assert command.target == pytest.approx(neutral + 0.05)  # slew-limited
    decoder.reset()
    both = [row, {"t_ms": 9.9, "node_index": 10, "body_id": 1}]
    assert decoder.decode(spike_table(both), 0, 10)[0].target == neutral
    decoder.reset()
    flex = decoder.decode(spike_table([both[1]]), 0, 10)[0]
    assert flex.target < neutral
    decoder.reset()
    boundary = spike_table([{**row, "t_ms": 10}])
    with pytest.raises(ValueError, match="out-of-window"):
        decoder.decode(boundary, 0, 10)
    commands, _ = decode_trace(mapping, boundary, 20)
    assert commands[0].target == neutral
    assert commands[1].target > neutral


def test_saturation_decay_reset_and_real_joint_motion(setup):
    policy, catalogue, sim = setup
    mapping = resolve_mapping(policy, catalogue, sim)
    spikes = spike_table(
        [{"t_ms": t, "node_index": 20, "body_id": 2} for t in np.arange(0, 80, 0.5)]
    )
    commands, _ = decode_trace(mapping, spikes, 100)
    repeat, _ = decode_trace(mapping, spikes, 100)
    assert commands == repeat
    c = mapping.channels[0]
    targets = np.array([x.target for x in commands])
    assert targets.max() <= c["maximum"]
    assert np.abs(np.diff(np.r_[c["body"]["neutral_position"], targets])).max() <= 0.05 + 1e-12
    states = replay_physics(sim, mapping, commands, 100, frame_count=0)
    assert np.abs(states["command_effect_rad"].to_numpy()).max() > 0.001
    # Explicitly configured safety limits, not nonexistent anatomical ctrlrange [0,0].
    i = c["body"]["actuator_index"]
    assert sim.model.actuator_ctrllimited[i]
    np.testing.assert_allclose(sim.model.actuator_ctrlrange[i], [c["minimum"], c["maximum"]])
    decoder = MotorDecoder(mapping)
    decoder.decode(spikes.filter(pa.array(spikes["t_ms"].to_numpy() < 10)), 0, 10)
    previous = decoder.filtered[c["name"]][1]
    after = decoder.decode(spike_table(), 10, 20)[0]
    assert after.extensor_filtered_hz == pytest.approx(previous * np.exp(-0.5))
    decoder.reset()
    assert decoder.decode(spike_table(), 0, 10)[0].target == c["body"]["neutral_position"]


def test_unmapped_cells_are_reported_but_never_drive_body(setup):
    policy, catalogue, sim = setup
    mapping = resolve_mapping(policy, catalogue, sim)
    commands, decoder = decode_trace(
        mapping, spike_table([{"t_ms": 3, "node_index": 30, "body_id": 3}]), 10
    )
    assert decoder.unmapped_spikes == 1 and decoder.unmapped_nodes == {30}
    assert commands[0].source_activity == 0
    for row in [
        {"t_ms": 2, "node_index": 30, "body_id": 999},
        {"t_ms": float("nan"), "node_index": 20, "body_id": 2},
    ]:
        with pytest.raises(ValueError):
            decode_trace(mapping, spike_table([row]), 10)
    with pytest.raises(ValueError, match="control windows"):
        decode_trace(mapping, spike_table(), 10.5)


@pytest.mark.parametrize(
    "change",
    [
        "unknown_body",
        "unknown_actuator",
        "annotation",
        "empty",
        "nan",
        "duplicate",
        "clock",
        "evidence",
    ],
)
def test_bad_mapping_fails_before_physics(setup, change):
    policy, catalogue, sim = setup
    c = policy["channels"][0]
    if change == "unknown_body":
        c["male_cns"]["flexor"]["body_ids"] = [999]
    elif change == "unknown_actuator":
        c["body"]["actuator"] = "unknown"
    elif change == "annotation":
        c["male_cns"]["flexor"]["annotations"]["exitNerve"] = "wrong"
    elif change == "empty":
        policy["channels"] = []
    elif change == "nan":
        c["decoder"]["gain_rad"] = float("nan")
    elif change == "duplicate":
        policy["channels"].append(deepcopy(c))
    elif change == "clock":
        policy["control_dt_ms"] = 10.01
    else:
        c["evidence"]["mechanical_assumption"] = ""
    with pytest.raises(ValueError):
        resolve_mapping(policy, catalogue, sim)


def test_controller_rejects_unsafe_unknown_and_early_commands(setup):
    policy, catalogue, sim = setup
    mapping = resolve_mapping(policy, catalogue, sim)
    controller = BodyController(mapping, sim.model)
    command = MotorDecoder(mapping).decode(spike_table(), 0, 10)[0]
    with pytest.raises(ValueError, match="clock"):
        controller.apply(command, sim.data)
    sim.data.time = 0.01
    for bad in [
        replace(command, target=float("nan")),
        replace(command, target=float("inf")),
        replace(command, target=command.target + 1),
        replace(command, target=command.target + 0.06),
        replace(command, actuator_name="unknown"),
    ]:
        with pytest.raises(ValueError):
            controller.apply(bad, sim.data)
    controller.apply(command, sim.data)
    with pytest.raises(ValueError, match="Out-of-order"):
        controller.apply(command, sim.data)


def test_inventory_does_not_invent_limits_or_muscle_targets(setup):
    _, catalogue, sim = setup
    audit = audit_motors(catalogue)
    assert audit["possible_target_fields"] == []
    assert audit["motor_neurons"] == 3
    assert audit["known_left_or_right"] == 3
    inventory = body_actuators(sim)
    tibia = inventory["fly/lf_trochanterfemur-lf_tibia-pitch-position"]
    assert tibia.control_range is None and tibia.joint_range is None


def test_seeded_lif_to_decoder_is_reproducible(setup):
    policy, catalogue, sim = setup
    mapping = resolve_mapping(policy, catalogue, sim)
    # Sensory neuron 90 excites extensor motor 2; sparse parent indices preserved.
    graph = NeuralGraph(
        np.array([90, 2, 1], dtype=np.uint64),
        np.array([100, 20, 10], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([250], dtype=np.uint32),
        ["acetylcholine"] * 3,
        {},
    )
    backend = Brian2LIFBackend(graph, seed=42)
    backend.run(100)
    assert len(backend.get_spikes()[0]) == 0
    backend.reset(42)
    backend.schedule(Stimulus((0,), rate_hz=100, start_ms=0, stop_ms=100))
    results = []
    for _ in range(2):
        backend.reset(42)
        backend.run(100)
        times, nodes = backend.get_spikes()
        motor = nodes != 0
        assert motor.sum() > 0
        spikes = pa.table(
            {
                "t_ms": times[motor],
                "node_index": graph.node_indices[nodes[motor]],
                "body_id": graph.body_ids[nodes[motor]],
            }
        )
        results.append(decode_trace(mapping, spikes, 100)[0])
    assert results[0] == results[1]
    assert any(c.source_activity > 0 for c in results[0])


def test_missing_command_is_an_error(setup):
    policy, catalogue, sim = setup
    mapping = resolve_mapping(policy, catalogue, sim)
    commands, _ = decode_trace(mapping, spike_table(), 20)
    with pytest.raises(ValueError, match="Missing"):
        replay_physics(sim, mapping, commands[:-1], 20, frame_count=0)


@pytest.mark.io_catalogue
def test_real_mapping_provenance_and_replay_roundtrip(tmp_path):
    from malecns_sim.bridge.build import DEFAULT_MAPPING
    from malecns_sim.bridge.replay import run_replay
    from malecns_sim.io.inspect import DEFAULT_CATALOGUE

    source = Path("outputs/io/leg_proprio_seed42_100ms")
    if not source.exists() or not DEFAULT_MAPPING.exists():
        pytest.skip("Requires the saved full-CNS sensory experiment and bridge-v1")
    output = tmp_path / "replay"
    report = run_replay(source, DEFAULT_MAPPING, DEFAULT_CATALOGUE, output, frame_count=0)
    assert report["active_mapped_motor_neurons"] > 0
    assert report["channels"]["lf_tibia"]["max_effect_vs_neutral_replay_rad"] > 1e-4
    assert validate_run(output) == report
    catalogue = IOCatalog.load(DEFAULT_CATALOGUE)
    sim = build_fly()
    validate_mapping(DEFAULT_MAPPING, catalogue, sim)
    catalogue.directory = tmp_path
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        validate_mapping(DEFAULT_MAPPING, catalogue, sim)
    with (output / "motor_commands.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        validate_run(output)
