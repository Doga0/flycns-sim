import json
from dataclasses import replace

import numpy as np
import pytest

from malecns_sim.neural.config import LIFConfig, load_config
from malecns_sim.neural.signs import transmitter_signs
from malecns_sim.neural.stimulus import Stimulus


def test_config_and_sign_assumptions():
    config = load_config()
    assert config == LIFConfig()
    labels = [
        "acetylcholine",
        "gaba",
        "glutamate",
        "dopamine",
        "serotonin",
        "octopamine",
        None,
        "unclear",
        "histamine",
    ]
    signs, report = transmitter_signs(labels, config)
    np.testing.assert_array_equal(signs, [1, -1, -1, 1, 1, 1, 0, 0, 0])
    assert report["zero_sign_nodes"] == 3
    with pytest.raises(ValueError):
        transmitter_signs(labels, replace(config, unknown_nt_policy="error"))
    assert transmitter_signs(labels, replace(config, unknown_nt_policy="excitatory"))[0][-1] == 1


def test_numpy_node_indices_are_serializable():
    stimulus = Stimulus((np.int32(0), np.int64(1)))
    stimulus.validate(2, LIFConfig())
    assert json.loads(json.dumps(stimulus.to_dict()))["nodes"] == [0, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt_ms": 0},
        {"tau_synapse_ms": -1},
        {"v_rest_mv": float("nan")},
        {"synaptic_delay_ms": 1.85},
        {"unknown_nt_policy": "random"},
    ],
)
def test_bad_config_rejected(kwargs):
    with pytest.raises(ValueError):
        LIFConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nodes": (-1,)},
        {"nodes": (0, 0)},
        {"rate_hz": 10001},
        {"gain_mv": float("inf")},
        {"start_ms": 0.15},
        {"stop_ms": 0},
        {"type": "spikes", "times_ms": (2, 1)},
    ],
)
def test_bad_stimulus_rejected(kwargs):
    with pytest.raises(ValueError):
        Stimulus(**({"nodes": (0,)} | kwargs)).validate(2, LIFConfig())
