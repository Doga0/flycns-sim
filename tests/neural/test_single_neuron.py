import numpy as np
import pytest
from brian2 import mV

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.stimulus import Stimulus


def test_no_stimulus_stable_and_exact_leak(graph_factory):
    backend = Brian2LIFBackend(graph_factory(n=1, edges=()))
    backend.run(10)
    assert len(backend.get_spikes()[0]) == 0
    assert float(backend.neurons.v[0] / mV) == pytest.approx(-52)
    backend.neurons.v = -55 * mV
    backend.run(10)
    assert float(backend.neurons.v[0] / mV) == pytest.approx(-52 - 3 * np.exp(-0.5))


def test_silence_and_nonfinite_guard(graph_factory):
    backend = Brian2LIFBackend(graph_factory(n=1, edges=()))
    backend.schedule(Stimulus((0,), rate_hz=10000))
    backend.silence([0])
    backend.run(10)
    assert len(backend.get_spikes()[0]) == 0
    backend.neurons.v = np.nan * mV
    with pytest.raises(FloatingPointError):
        backend.run(1)
