import base64
import hashlib
from unittest.mock import MagicMock

import pandas as pd
import pytest

from malecns_sim.cns.dataset import CNSDataset
from malecns_sim.cns.download import DEFAULT_DATA_DIR, FILES, download_file
from malecns_sim.cns.inspect import describe


@pytest.fixture
def raw_tables(tmp_path):
    tables = {
        "annotations": pd.DataFrame(
            {
                "bodyId": [1, 2],
                "status": ["Traced", "Untraced"],
                "statusLabel": ["Traced", "Orphan"],
            }
        ),
        "neurotransmitters": pd.DataFrame(
            {
                "body": [1, 2],
                "predicted_nt": ["gaba", None],
                "celltype_predicted_nt": [None, "acetylcholine"],
                "consensus_nt": ["gaba", "acetylcholine"],
            }
        ),
        "connections": pd.DataFrame({"body_pre": [1], "body_post": [2], "weight": [1]}),
    }
    for name, table in tables.items():
        table.to_feather(tmp_path / FILES[name])
    return tmp_path, tables


def test_load_preserves_raw_data(raw_tables):
    directory, tables = raw_tables
    dataset = CNSDataset.load(directory)
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(getattr(dataset, name), expected)


def test_inspection_includes_unfiltered_labels(raw_tables, capsys):
    directory, _ = raw_tables
    describe(CNSDataset.load(directory))
    output = capsys.readouterr().out
    assert "Untraced" in output
    assert "predicted_nt values" in output
    assert "gaba" in output


def test_missing_dataset_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="cns.download"):
        CNSDataset.load(tmp_path)


def test_corrupt_feather_rejected(raw_tables):
    directory, _ = raw_tables
    (directory / FILES["connections"]).write_bytes(b"not feather")
    with pytest.raises(ValueError, match="Cannot read Feather"):
        CNSDataset.load(directory)


def test_wrong_table_schema_rejected(raw_tables):
    directory, tables = raw_tables
    tables["connections"].drop(columns="weight").to_feather(directory / FILES["connections"])
    with pytest.raises(ValueError, match="Missing columns.*weight"):
        CNSDataset.load(directory)


def mock_session(payload, *, expected=None):
    expected = payload if expected is None else expected
    checksum = base64.b64encode(hashlib.md5(expected).digest()).decode()
    session = MagicMock()
    session.head.return_value.__enter__.return_value.headers = {
        "Content-Length": str(len(expected)),
        "x-goog-hash": f"crc32c=unused,md5={checksum}",
        "x-goog-generation": "123",
    }
    session.get.return_value.__enter__.return_value.iter_content.return_value = [payload]
    return session


def test_download_verified_and_reused(tmp_path):
    session = mock_session(b"payload")
    path = download_file(session, FILES["annotations"], tmp_path)
    assert path.read_bytes() == b"payload"
    assert not list(tmp_path.glob("*.part"))
    download_file(session, FILES["annotations"], tmp_path)
    assert session.get.call_count == 1


@pytest.mark.parametrize("payload", [b"short", b"corrupt"])
def test_bad_download_preserves_existing_file(tmp_path, payload):
    path = tmp_path / FILES["annotations"]
    path.write_bytes(b"old")
    with pytest.raises(ValueError, match="verification failed"):
        download_file(mock_session(payload, expected=b"correct"), path.name, tmp_path)
    assert path.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.dataset
def test_1_three_official_files_exist():
    for filename in FILES.values():
        path = DEFAULT_DATA_DIR / filename
        assert path.is_file(), "Run python -m malecns_sim.cns.download first"
        assert path.stat().st_size > 0


@pytest.mark.dataset
def test_2_official_feathers_load():
    dataset = CNSDataset.load()
    assert len(dataset.annotations) > 0
    assert len(dataset.neurotransmitters) > 0
    assert len(dataset.connections) > 0
