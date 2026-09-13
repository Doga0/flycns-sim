import json
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from PIL import Image

from malecns_sim.simulation.fly import build_fly
from malecns_sim.simulation.smoke_test import check_state, main, run_headless


@pytest.fixture(scope="module")
def sim():
    return build_fly()


def test_3_flygym_compile(sim):
    assert isinstance(sim.model, mujoco.MjModel)
    assert isinstance(sim.data, mujoco.MjData)
    assert sim.model.nu == 42
    assert sim.model.nq > sim.model.nu
    assert sim.model.ngeom > 0
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, sim.camera) >= 0


def test_4_thousand_steps_and_render(sim, tmp_path):
    start = float(sim.data.time)
    report = run_headless(sim, tmp_path, steps=1000, frame_count=3)
    assert report["steps"] == 1000
    assert sim.data.time - start == pytest.approx(1000 * sim.model.opt.timestep)
    assert report["finite_state"]
    assert report["warning_count"] == 0
    for name in report["frames"]:
        with Image.open(tmp_path / name) as image:
            assert image.size == (640, 480)
            assert np.asarray(image).std() > 1
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_nan_is_rejected(sim):
    previous = sim.data.qpos[0]
    try:
        sim.data.qpos[0] = np.nan
        with pytest.raises(RuntimeError, match="Non-finite"):
            check_state(sim)
    finally:
        sim.data.qpos[0] = previous


def test_viewer_dispatch(sim):
    with (
        patch("sys.argv", ["smoke_test", "--viewer"]),
        patch("malecns_sim.simulation.smoke_test.build_fly", return_value=sim),
        patch("flygym.rendering.launch_interactive_viewer") as viewer,
    ):
        main()
    viewer.assert_called_once_with(sim.model, sim.data)
