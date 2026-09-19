import numpy as np
import pytest

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.stimulus import Stimulus


def test_effect_arrives_after_transmission_delay(graph_factory):
    backend = Brian2LIFBackend(graph_factory(), record_nodes=(1,))
    backend.schedule(Stimulus((0,), type="spikes", times_ms=(10,)))
    backend.run(25)
    times, nodes = backend.get_spikes()
    source_spike = times[nodes == 0][0]
    trace = backend.get_state_trace()
    first_effect = np.flatnonzero(trace["g_mv"][0] > 0)[0]
    assert trace["t_ms"][first_effect] - source_spike == pytest.approx(1.8)
    assert np.allclose(trace["v_mv"][0, : first_effect + 1], -52)
    assert times[nodes == 1][0] > source_spike + 1.8
