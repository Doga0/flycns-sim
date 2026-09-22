import pyarrow as pa
import pytest

from malecns_sim.bridge.sensory.body_state import BodyState, JointStateReader
from malecns_sim.bridge.sensory.encoder import SensoryEncoder
from malecns_sim.bridge.sensory.mapping import SensoryMapping
from malecns_sim.bridge.sensory.validate import _same_activity
from malecns_sim.simulation.fly import build_fly


def mapping(encoder=None):
    encoder = encoder or {
        "type": "linear_position_encoder_v1",
        "min_value": 1.0,
        "max_value": 2.0,
        "min_rate_hz": 0.0,
        "max_rate_hz": 100.0,
        "gain_mv": 10.0,
    }
    signal = "velocity" if encoder["type"] == "directional_velocity_encoder_v1" else "position"
    return SensoryMapping(
        {"sensory_dt_ms": 1.0},
        (
            {
                "name": "lf_position",
                "source": {"joint": "lf_tibia", "signal": signal},
                "nodes": [3, 5],
                "encoder": encoder,
            },
        ),
        (),
        {3: 30, 5: 50, 8: 80},
    )


def state(t, position=1.0, velocity=0.0):
    return BodyState(t, "lf_tibia", position, velocity, 1.0)


def test_zero_position_baseline_is_silent_and_explicit_encoding_is_deterministic():
    first, second = SensoryEncoder(mapping(), seed=42), SensoryEncoder(mapping(), seed=42)
    for t in range(10):
        a = first.encode(state(float(t), 1.0), t, t + 1)
        b = second.encode(state(float(t), 1.0), t, t + 1)
        assert a.activity["rate_hz"].to_pylist() == [0.0]
        assert a.spikes.equals(b.spikes) and a.spikes.num_rows == 0
    # At 100 Hz and a 1 ms sensory clock, each population has one shared event every 10 ms.
    for t in range(10, 20):
        output = first.encode(state(float(t), 2.0), t, t + 1)
    assert output.spikes["t_ms"].to_pylist() == [19.0, 19.0]
    assert output.spikes["node_index"].to_pylist() == [3, 5]
    assert output.activity["normalized_value"].to_pylist() == [1.0]


def test_position_is_clamped_and_velocity_direction_is_explicit():
    position = SensoryEncoder(mapping())
    low = position.encode(state(0, -10), 0, 1).activity.to_pydict()
    assert low["normalized_value"] == [0.0] and low["rate_hz"] == [0.0]
    velocity_mapping = mapping(
        {
            "type": "directional_velocity_encoder_v1",
            "direction": "negative",
            "min_value": 0.0,
            "max_value": 2.0,
            "min_rate_hz": 0.0,
            "max_rate_hz": 100.0,
            "gain_mv": 10.0,
        }
    )
    encoder = SensoryEncoder(velocity_mapping)
    negative = encoder.encode(state(0, velocity=-1), 0, 1).activity.to_pydict()
    positive = encoder.encode(state(1, velocity=1), 1, 2).activity.to_pydict()
    assert negative["rate_hz"] == [50.0]
    assert positive["rate_hz"] == [0.0]


def test_poisson_encoding_is_seeded_and_window_mismatches_are_rejected():
    def collect(seed):
        encoder = SensoryEncoder(mapping(), mode="poisson", seed=seed)
        return [
            encoder.encode(state(float(t), 2.0), t, t + 1).spikes.to_pylist() for t in range(100)
        ]

    assert collect(42) == collect(42)
    assert collect(42) != collect(43)
    encoder = SensoryEncoder(mapping())
    with pytest.raises(ValueError, match="consecutive"):
        encoder.encode(state(1), 1, 2)
    with pytest.raises(ValueError, match="matching body"):
        encoder.encode(state(0.5), 0, 1)


def test_joint_reader_hides_compiled_addresses_and_validates_actuator_identity():
    sim = build_fly()
    reader = JointStateReader(
        sim.model,
        "fly/lf_trochanterfemur-lf_tibia-pitch",
        "fly/lf_trochanterfemur-lf_tibia-pitch-position",
    )
    result = reader.read(sim.data)
    assert result.joint_name == "fly/lf_trochanterfemur-lf_tibia-pitch"
    assert result.joint_position_rad == pytest.approx(result.actuator_target_rad)
    with pytest.raises(ValueError, match="does not directly"):
        JointStateReader(sim.model, result.joint_name, "fly/lf_coxa-lf_coxa-pitch-position")


def test_cross_platform_activity_comparison_allows_only_tiny_float_differences():
    left = pa.table({"channel": ["lf"], "rate_hz": [50.0]})
    tiny_delta = pa.table({"channel": ["lf"], "rate_hz": [50.0 + 1e-12]})
    meaningful_delta = pa.table({"channel": ["lf"], "rate_hz": [50.0 + 1e-6]})
    assert _same_activity(left, tiny_delta)
    assert not _same_activity(left, meaningful_delta)


@pytest.mark.parametrize("mode", ["rate", "", None])
def test_invalid_mode_or_seed_is_rejected(mode):
    with pytest.raises(ValueError):
        SensoryEncoder(mapping(), mode=mode)
    with pytest.raises(ValueError):
        SensoryEncoder(mapping(), seed=-1)
