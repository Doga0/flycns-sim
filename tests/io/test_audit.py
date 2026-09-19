import numpy as np
import pandas as pd

from malecns_sim.io.audit import audit_annotations


def test_audit_reports_actual_columns_values_and_nulls(tmp_path):
    path = tmp_path / "annotations.feather"
    pd.DataFrame(
        {
            "bodyId": [1, 2],
            "superclass": ["vnc_motor", None],
            "class": [None, "visual"],
            "type": ["MN", "photoreceptor"],
            "somaLocation": [np.array([1, 2, 3]), None],
        }
    ).to_feather(path)
    report = audit_annotations(path)
    assert report["available_annotation_columns"] == [
        "bodyId",
        "superclass",
        "class",
        "type",
        "somaLocation",
    ]
    assert report["column_schema"][1]["nulls"] == 1
    assert report["classification_fields"]["superclass"] == [
        {"value": None, "count": 1},
        {"value": "vnc_motor", "count": 1},
    ]
