import json

import pyarrow.parquet as pq
import pytest
import yaml

from malecns_sim.analysis.activation_sweep import run_sweep
from malecns_sim.analysis.pathway_audit import run_audit
from malecns_sim.analysis.pathway_report import run_report
from malecns_sim.neural.config import LIFConfig

# Same topology and signs as conftest.py: (src, dst, contacts) by body ID.
EDGES = [
    (101, 301, 3),
    (101, 201, 2),
    (201, 201, 1),
    (201, 301, 4),
    (201, 202, 6),
    (202, 301, 2),
    (102, 202, 5),
    (202, 302, 1),
    (102, 203, 7),
    (203, 302, 9),
]
SIGN = {101: 1, 102: 1, 201: 1, 202: -1, 203: 0}


@pytest.fixture
def analysed(built_graph, tmp_path):
    graph, sensory, motor = built_graph
    policy = yaml.safe_load(sensory.read_text())
    policy["channels"][0]["encoder"] = {"gain_mv": 68.75}
    sensory.write_text(yaml.safe_dump(policy))
    lif = tmp_path / "strong.yaml"
    lif.write_text(yaml.safe_dump(LIFConfig().to_dict() | {"weight_per_contact_mv": 30.0}))
    audit, sweep = tmp_path / "audit", tmp_path / "sweep"
    run_audit(graph, sensory, motor, lif, audit, max_hops=3, enumerate_hops=3, edge_hops=3)
    run_sweep(
        graph,
        sensory,
        motor,
        lif,
        sweep,
        rates=(300.0,),
        volleys=(1,),
        baseline_ms=5.0,
        drive_ms=20.0,
        tail_ms=10.0,
        validate=False,
    )
    return graph, lif, audit, sweep


def test_report_attributes_delivered_drive_to_presynaptic_spikes(analysed, tmp_path):
    graph, lif, audit, sweep = analysed
    out = tmp_path / "report"
    benchmark = run_report(audit, sweep, graph, lif, out)
    drive = pq.read_table(out / "incoming_drive.parquet").to_pandas()
    for condition in ("volley_1", "poisson_300hz"):
        spikes = pq.read_table(sweep / "spikes" / f"{condition}.parquet").to_pandas()
        counts = spikes.body_id.value_counts()
        for target in (301, 302):
            exc = inh = 0.0
            for pre, post, contacts in EDGES:
                if post == target:
                    value = contacts * SIGN[pre] * 30.0 * counts.get(pre, 0)
                    exc, inh = exc + max(value, 0), inh + min(value, 0)
            row = drive[(drive.condition == condition) & (drive.target_body_id == target)]
            assert row.excitatory_mv.item() == pytest.approx(exc)
            assert row.inhibitory_mv.item() == pytest.approx(inh)
    motors = {m["body_id"]: m for m in benchmark["motors"]}
    assert motors[301]["direct_contacts"] == 3 and motors[302]["shortest_path_hops"] == 2
    assert benchmark["strongest_condition"] == "poisson_300hz"
    assert (out / "pathway_report.md").read_text().startswith("# LF tibia")
    report = json.loads((out / "report.json").read_text())
    assert report["diagnostic_only"] and "pathway_report.json" in report["artifacts"]


def test_report_rejects_changed_inputs_and_other_configs(analysed, tmp_path):
    graph, lif, audit, sweep = analysed
    with pytest.raises(ValueError, match="different LIF config"):
        run_report(audit, sweep, graph, None, tmp_path / "a")
    summary = sweep / "sweep_summary.json"
    summary.write_text(summary.read_text() + " ")
    with pytest.raises(ValueError, match="Artifact changed"):
        run_report(audit, sweep, graph, lif, tmp_path / "b")
    with pytest.raises(ValueError, match="Not a pathway audit"):
        run_report(sweep, sweep, graph, lif, tmp_path / "c")
