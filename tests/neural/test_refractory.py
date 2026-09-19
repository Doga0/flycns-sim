import numpy as np

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.stimulus import Stimulus


def test_strong_drive_respects_refractory(graph_factory):
    backend = Brian2LIFBackend(graph_factory(n=1, edges=()))
    backend.schedule(Stimulus((0,), rate_hz=10000))
    backend.run(50)
    times, _ = backend.get_spikes()
    assert len(times) > 10
    assert np.min(np.diff(times)) >= 2.2 - 1e-9
