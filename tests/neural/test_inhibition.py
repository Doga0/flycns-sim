import numpy as np

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.stimulus import Stimulus


def test_inhibitory_input_reduces_target_spikes(graph_factory):
    counts = []
    for driven in ((0,), (0, 2)):
        graph = graph_factory(
            n=3, edges=((0, 1, 250), (2, 1, 300)), nts=["acetylcholine", "acetylcholine", "gaba"]
        )
        backend = Brian2LIFBackend(graph)
        backend.schedule(Stimulus(driven, type="spikes", times_ms=(10, 30, 50)))
        backend.run(80)
        counts.append(np.count_nonzero(backend.get_spikes()[1] == 1))
    assert counts[0] > 0
    assert counts[1] < counts[0]
