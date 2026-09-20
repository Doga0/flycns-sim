from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest

from malecns_sim.bridge.mapping import load_policy, resolve_mapping
from malecns_sim.bridge.replay import decode_trace
from malecns_sim.bridge.viewer import show_replay
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.simulation.fly import build_fly


def test_viewer_replays_motor_commands_without_mutating_source(tmp_path):
    sim = build_fly()
    policy = load_policy()
    rows = []
    for group in policy["channels"][0]["male_cns"].values():
        for body_id in group["body_ids"]:
            rows.append(
                {
                    **group["annotations"],
                    "body_id": body_id,
                    "node_index": len(rows),
                    "role": "motor",
                }
            )
    catalogue = IOCatalog(tmp_path, {}, pa.Table.from_pylist(rows))
    mapping = resolve_mapping(policy, catalogue, sim)
    motor = rows[-1]
    spikes = pa.table(
        {"t_ms": [1.0], "node_index": [motor["node_index"]], "body_id": [motor["body_id"]]}
    )
    commands, _ = decode_trace(mapping, spikes, 20)
    original_qpos, original_ctrl = sim.data.qpos.copy(), sim.data.ctrl.copy()
    samples = []

    class Handle:
        def __init__(self, model, data):
            self.model, self.data = model, data
            self.calls = 0
            self.cam = SimpleNamespace(lookat=np.zeros(3))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def lock(self):
            return nullcontext()

        def is_running(self):
            self.calls += 1
            return self.calls < 800

        def sync(self):
            samples.append((self.data.time, self.data.ctrl.copy(), self.data.qpos.copy()))
            # Simulate UI edits on its separate display model/data.
            self.data.ctrl[:] = 99
            self.model.opt.gravity[:] = 0

    clock = iter(np.arange(0, 10000, 0.1))
    with (
        patch("mujoco.viewer.launch_passive", side_effect=lambda m, d, **kw: Handle(m, d)),
        patch("malecns_sim.bridge.viewer.time.monotonic", side_effect=lambda: next(clock)),
        patch("malecns_sim.bridge.viewer.time.sleep"),
    ):
        show_replay(sim, mapping, commands, 20)
    np.testing.assert_array_equal(sim.data.qpos, original_qpos)
    np.testing.assert_array_equal(sim.data.ctrl, original_ctrl)
    assert sim.model.opt.gravity[2] != 0
    valid = [(t, ctrl, q) for t, ctrl, q in samples if np.all(ctrl != 99)]
    assert any(t >= 0.01 and ctrl[5] > original_ctrl[5] for t, ctrl, q in valid)
    assert any(not np.array_equal(q, original_qpos) for t, ctrl, q in valid)
    # A second loop returns to the same initial state, never keeps integrating.
    assert sum(t == 0 for t, ctrl, q in valid) >= 2


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf")])
def test_invalid_viewer_speed_is_rejected_before_opening(speed):
    with pytest.raises(ValueError, match="speed"):
        show_replay(None, None, None, 100, speed=speed)


def test_cli_dispatches_viewer_and_speed():
    from malecns_sim.bridge.replay import main

    with (
        patch(
            "sys.argv",
            [
                "replay",
                "--neural-run",
                "input",
                "--output",
                "output",
                "--viewer",
                "--frames",
                "0",
                "--playback-speed",
                "0.2",
            ],
        ),
        patch("malecns_sim.bridge.replay.run_replay", return_value={}) as run,
    ):
        main()
    assert run.call_args.kwargs["viewer"] is True
    assert run.call_args.kwargs["playback_speed"] == 0.2
    assert run.call_args.kwargs["frame_count"] == 0
