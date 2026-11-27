from pathlib import Path

import pytest

from malecns_sim.bridge.closed_loop import seed_report


def test_seed_report_rejects_cherry_picking_and_profile_drift(monkeypatch):
    def comparison(path):
        return {
            "acceptance_passed": Path(path).name != "seed43",
            "functional_target_0p01_rad_met": True,
            "motor_clamp_mapped_motor_spikes": 1,
        }

    def provenance(path):
        seed = int(Path(path).parent.parent.name.removeprefix("seed"))
        return {
            "config": {"seed": seed, "version": "v2"},
            "graph_manifest": {"sha256": "graph"},
            "catalogue_manifest": {"sha256": "catalogue"},
            "motor_manifest_hash": {"sha256": "motor"},
            "sensory_manifest_hash": {"sha256": "sensory"},
            "lif_config_sha256": "lif",
            "body_specification": {"compiled_model_sha256": "body"},
        }

    monkeypatch.setattr(seed_report, "compare_conditions", comparison)
    monkeypatch.setattr(seed_report, "read_json", provenance)
    report = seed_report.summarize([Path("seed42"), Path("seed43")], [42, 43])
    assert not report["replication_passed"]
    with pytest.raises(ValueError, match="One distinct run"):
        seed_report.summarize([Path("seed42")], [42, 43])
    with pytest.raises(ValueError, match="Unexpected or repeated seed"):
        seed_report.summarize([Path("seed42"), Path("seed42")], [42, 43])

    def drifted(path):
        value = provenance(path)
        if Path(path).parent.parent.name == "seed43":
            value["lif_config_sha256"] = "changed"
        return value

    monkeypatch.setattr(seed_report, "read_json", drifted)
    with pytest.raises(ValueError, match="provenance"):
        seed_report.summarize([Path("seed42"), Path("seed43")], [42, 43])
