import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from malecns_sim.analysis.activation_sweep import (
    build_conditions,
    measure,
    run_sweep,
    select_subset,
)
from malecns_sim.neural.config import LIFConfig


def sensory_with_gain(path, tmp_path):
    policy = yaml.safe_load(path.read_text())
    policy["channels"][0]["encoder"] = {"gain_mv": 68.75}
    out = tmp_path / "sensory_gain.yaml"
    out.write_text(yaml.safe_dump(policy))
    return out


def strong_lif(tmp_path):
    # Synthetic only: large per-contact weight so the 3-contact direct synapse can fire.
    values = LIFConfig().to_dict() | {"weight_per_contact_mv": 30.0}
    path = tmp_path / "strong.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_conditions_cover_control_volleys_and_rates():
    conditions = build_conditions((10, 100), (1, 3), onset_ms=5, drive_ms=20, volley_interval_ms=3)
    assert [c["name"] for c in conditions] == [
        "no_input",
        "volley_1",
        "volley_3",
        "poisson_10hz",
        "poisson_100hz",
    ]
    assert conditions[2]["times_ms"] == [5, 8, 11]
    with pytest.raises(ValueError, match="fit"):
        build_conditions((), (10,), onset_ms=0, drive_ms=20, volley_interval_ms=3)
    with pytest.raises(ValueError, match="positive"):
        build_conditions((0,), (), onset_ms=0, drive_ms=20, volley_interval_ms=3)


def test_measure_counts_windows_latency_and_layers():
    times = np.array([1.0, 6.0, 7.0, 9.5, 12.0])
    nodes = np.array([5, 0, 1, 5, 2])
    layer = np.array([0, 0, 1, 1, 1, 1, 2, -1], dtype=np.int16)
    result = measure(
        times,
        nodes,
        sources=np.array([0, 1]),
        targets=np.array([5, 6]),
        target_labels={5: "j:flexor", 6: "j:extensor"},
        body_ids=np.arange(100, 108),
        layer=layer,
        onset_ms=5.0,
        end_ms=15.0,
    )
    m5, m6 = result["motor"]
    assert m5["spikes"] == 1 and m5["spikes_before_onset"] == 1
    assert m5["first_spike_latency_ms"] == pytest.approx(4.5)
    assert m6["spikes"] == 0 and m6["first_spike_latency_ms"] is None
    assert result["sensory_spikes"] == 2 and result["spikes_before_onset"] == 1
    layers = {r["hops_from_sensory"]: r for r in result["layers"]}
    assert layers[0]["spikes"] == 2 and layers[1]["active_neurons"] == 2
    assert layers[None]["neurons_in_layer"] == 1


def test_sweep_runs_full_lif_without_body(built_graph, tmp_path):
    graph, sensory, motor = built_graph
    sensory = sensory_with_gain(sensory, tmp_path)
    out = tmp_path / "sweep"
    summary = run_sweep(
        graph,
        sensory,
        motor,
        strong_lif(tmp_path),
        out,
        rates=(200.0,),
        volleys=(1,),
        baseline_ms=5.0,
        drive_ms=20.0,
        tail_ms=10.0,
        validate=False,
    )
    conditions = {c["condition"]["name"]: c for c in summary["conditions"]}
    assert conditions["no_input"]["total_spikes"] == 0
    assert conditions["volley_1"]["sensory_spikes"] == 2
    # One volley crosses the direct 0→5 synapse; latency includes the 1.8 ms delay.
    assert summary["motor_spikes"]["volley_1"]["301"] >= 1
    latency = summary["motor_first_spike_latency_ms"]["volley_1"]["301"]
    assert latency >= 1.8
    assert summary["minimum_tested_poisson_rate_hz_by_motor"]["301"] == 200.0
    report = json.loads((out / "report.json").read_text())
    assert report["body_coupling"] is False and report["diagnostic_only"]
    assert "spikes/volley_1.parquet" in report["artifacts"]
    matrix = pq.read_table(out / "motor_matrix.parquet").to_pandas()
    assert set(matrix.condition) == {"no_input", "volley_1", "poisson_200hz"}
    spikes = pq.read_table(out / "spikes" / "volley_1.parquet").to_pandas()
    assert spikes.t_ms.min() >= 5.0
    with pytest.raises(FileExistsError):
        run_sweep(graph, sensory, motor, strong_lif(tmp_path), out, validate=False)


def test_select_subset_requires_explicit_profile_ids():
    members = [(101, "probe"), (102, "probe")]
    assert select_subset(members, None) == members
    assert select_subset(members, (102,)) == [(102, "probe")]
    with pytest.raises(ValueError, match="not in the sensory profile"):
        select_subset(members, (999,))
    with pytest.raises(ValueError, match="distinct"):
        select_subset(members, (101, 101))
    with pytest.raises(ValueError, match="distinct"):
        select_subset(members, ())


def test_sweep_drives_only_the_subset(built_graph, tmp_path):
    graph, sensory, motor = built_graph
    sensory = sensory_with_gain(sensory, tmp_path)
    out = tmp_path / "subset"
    summary = run_sweep(
        graph,
        sensory,
        motor,
        strong_lif(tmp_path),
        out,
        rates=(),
        volleys=(1,),
        baseline_ms=5.0,
        drive_ms=20.0,
        tail_ms=10.0,
        validate=False,
        sensory_subset=(102,),
    )
    volley = {c["condition"]["name"]: c for c in summary["conditions"]}["volley_1"]
    assert volley["sensory_spikes"] == 1
    spikes = pq.read_table(out / "spikes" / "volley_1.parquet").to_pandas()
    assert 101 not in set(spikes.body_id.tolist())
    report = json.loads((out / "report.json").read_text())
    assert report["parameters"]["sensory_subset_body_ids"] == [102]


def test_sweep_is_repeatable_for_a_seed(built_graph, tmp_path):
    graph, sensory, motor = built_graph
    sensory = sensory_with_gain(sensory, tmp_path)
    runs = [
        run_sweep(
            graph,
            sensory,
            motor,
            strong_lif(tmp_path),
            tmp_path / f"r{i}",
            rates=(500.0,),
            volleys=(),
            baseline_ms=0.0,
            drive_ms=20.0,
            tail_ms=0.0,
            validate=False,
        )
        for i in range(2)
    ]
    strip = [{k: v for k, v in c.items() if k != "wall_seconds"} for c in runs[0]["conditions"]]
    again = [{k: v for k, v in c.items() if k != "wall_seconds"} for c in runs[1]["conditions"]]
    assert strip == again
