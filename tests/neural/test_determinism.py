import numpy as np
import pyarrow.parquet as pq
import pytest

from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.result import save_result, validate_result
from malecns_sim.neural.stimulus import Stimulus


def test_seed_reset_and_split_run(graph_factory):
    backend = Brian2LIFBackend(graph_factory(), seed=42)
    backend.schedule(Stimulus((0,), start_ms=10, stop_ms=180))
    backend.run(200)
    first = backend.get_spikes()
    backend.reset(42)
    backend.run(75)
    backend.run(125)
    for actual, expected in zip(backend.get_spikes(), first):
        np.testing.assert_array_equal(actual, expected)
    backend.reset(43)
    backend.run(200)
    assert not np.array_equal(backend.get_spikes()[0], first[0])


def test_results_keep_parent_mapping_and_exact_counts(graph_factory, tmp_path):
    backend = Brian2LIFBackend(graph_factory(), seed=42)
    backend.stimulate([0], 100)
    backend.run(100)
    one = save_result(backend, tmp_path / "one", experiment="toy")
    two = save_result(backend, tmp_path / "two", experiment="toy")
    assert one == two
    assert validate_result(tmp_path / "one") == one
    for path in (tmp_path / "one").iterdir():
        assert path.read_bytes() == (tmp_path / "two" / path.name).read_bytes()
    spikes = pq.read_table(tmp_path / "one/spikes.parquet").to_pydict()
    rates = pq.read_table(tmp_path / "one/rates.parquet").to_pydict()
    assert sum(rates["spike_count"]) == one["total_spikes"]
    for node, body in zip(spikes["node_index"], spikes["body_id"]):
        assert body == int(backend.graph.body_ids[(node - 7) // 3])
    with pytest.raises(FileExistsError):
        save_result(backend, tmp_path / "one", experiment="toy")
    with (tmp_path / "two/spikes.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        validate_result(tmp_path / "two")
