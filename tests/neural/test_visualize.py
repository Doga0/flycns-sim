import numpy as np
import pyarrow.parquet as pq
import pytest

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.result import save_result, validate_result
from malecns_sim.neural.stimulus import Stimulus
from malecns_sim.neural.visualize import next_spike_latencies, visualize


def test_trace_roundtrip_mapping_and_checksum(graph_factory, tmp_path):
    backend = Brian2LIFBackend(graph_factory(), record_nodes=(1, 0))
    backend.schedule(Stimulus((0,), type="spikes", times_ms=(1,)))
    backend.run(10)
    result = tmp_path / "result"
    save_result(backend, result, experiment="test")
    validate_result(result)
    saved = pq.read_table(result / "state_trace.parquet").to_pandas()
    actual = backend.get_state_trace()
    for row, local in enumerate((1, 0)):
        rows = saved[saved["local_index"] == local]
        assert (rows["body_id"] == backend.graph.body_ids[local]).all()
        assert (rows["node_index"] == backend.graph.node_indices[local]).all()
        np.testing.assert_array_equal(rows["v_mv"], actual["v_mv"][row])
        np.testing.assert_array_equal(rows["g_mv"], actual["g_mv"][row])
        np.testing.assert_array_equal(rows["t_ms"], actual["t_ms"])
    assert len(visualize(result)) == 3
    with (result / "state_trace.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        validate_result(result)


def test_silent_legacy_result_and_node_limit(graph_factory, tmp_path):
    backend = Brian2LIFBackend(graph_factory())
    backend.run(1)
    result = tmp_path / "result"
    save_result(backend, result, experiment="test")
    paths = visualize(result, max_nodes=1)
    assert {p.name for p in paths} == {"spike_raster.png", "firing_rates.png"}
    assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in paths)
    validate_result(result)
    with pytest.raises(ValueError, match="positive"):
        visualize(result, max_nodes=0)


def test_next_spike_pairing_handles_ties_reuse_and_missing_targets():
    times, delays = next_spike_latencies([8, 1, 2, 4], [4, 3])
    np.testing.assert_array_equal(times, [1, 2])
    np.testing.assert_array_equal(delays, [2, 1])
    assert len(next_spike_latencies([1], [])[0]) == 0
