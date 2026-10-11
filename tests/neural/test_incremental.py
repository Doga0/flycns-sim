import numpy as np
import pytest

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.stimulus import Stimulus


def test_incremental_events_preserve_membrane_synapses_refractory_and_delay(graph_factory):
    graph = graph_factory()
    whole = Brian2LIFBackend(graph, record_nodes=(0, 1))
    whole.schedule(
        Stimulus((0,), type="spikes", times_ms=(0.0, 4.0), start_ms=0, stop_ms=10, gain_mv=68.75)
    )
    whole.run(10)
    chunked = Brian2LIFBackend(graph, record_nodes=(0, 1))
    chunked.configure_event_input([0], [68.75])
    cursor, collected_t, collected_i = 0, [], []
    for t in range(10):
        chunked.inject_events([{"node_index": 0, "t_ms": t}] if t in (0, 4) else [])
        chunked.advance(1)
        (times, nodes), cursor = chunked.spikes_since(cursor)
        collected_t.extend(times)
        collected_i.extend(nodes)
        assert chunked.time_ms == pytest.approx(t + 1)
    assert np.array_equal(whole.get_spikes()[0], collected_t)
    assert np.array_equal(whole.get_spikes()[1], collected_i)
    for key in ("v_mv", "g_mv"):
        np.testing.assert_array_equal(whole.get_state_trace()[key], chunked.get_state_trace()[key])
    assert len(chunked.network.objects) < 10  # No new input groups per tick.
    chunked.reset()
    assert chunked.time_ms == 0 and chunked.total_spikes == 0


def test_selected_monitor_counts_all_but_retains_only_declared_nodes(graph_factory):
    b = Brian2LIFBackend(graph_factory(), spike_record_nodes=(1,))
    b.configure_event_input([0], [68.75])
    b.inject_events([{"node_index": 0, "t_ms": 0}])
    b.advance(10)
    assert b.total_spikes > len(b.get_spikes()[0]) > 0
    assert set(b.get_spikes()[1]) == {1}


def test_online_input_rejects_past_future_unknown_and_duplicate(graph_factory):
    b = Brian2LIFBackend(graph_factory())
    b.configure_event_input([0], [68.75])
    with pytest.raises(ValueError, match="Unmapped"):
        b.inject_events([{"node_index": 1, "t_ms": 0}])
    b.inject_events([{"node_index": 0, "t_ms": 1}])
    with pytest.raises(ValueError, match="Duplicate"):
        b.inject_events([{"node_index": 0, "t_ms": 1}])
    with pytest.raises(ValueError, match="next advance"):
        b.advance(1)
    b.advance(2)
    with pytest.raises(ValueError, match="past"):
        b.inject_events([{"node_index": 0, "t_ms": 0}])
