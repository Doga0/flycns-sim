import numpy as np
from brian2 import mV

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend


def test_weights_use_presynaptic_sign_and_keep_anatomical_contacts(graph_factory):
    graph = graph_factory(
        n=3, edges=((0, 1, 20), (1, 0, 12), (2, 0, 10)), nts=["acetylcholine", "gaba", None]
    )
    original = graph.contacts.copy()
    backend = Brian2LIFBackend(graph)
    np.testing.assert_allclose(backend.synapses.w[:] / mV, [5.5, -3.3, 0])
    np.testing.assert_array_equal(graph.contacts, original)
    assert backend.sign_report["zero_weight_edges"] == 1
    backend.run(1)
