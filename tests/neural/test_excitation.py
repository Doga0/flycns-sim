import numpy as np

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.stimulus import Stimulus


def test_excitation_is_connectivity_driven(graph_factory):
    counts = []
    for edges in ((), ((0, 1, 250),)):
        backend = Brian2LIFBackend(graph_factory(edges=edges))
        backend.schedule(Stimulus((0,), type="spikes", times_ms=(10, 30, 50)))
        backend.run(80)
        _, nodes = backend.get_spikes()
        assert np.count_nonzero(nodes == 0) == 3
        counts.append(np.count_nonzero(nodes == 1))
    assert counts[0] == 0
    assert counts[1] >= 3
