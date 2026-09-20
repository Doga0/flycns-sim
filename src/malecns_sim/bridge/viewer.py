"""Local MuJoCo window for repeated, paced replay of the actual motor commands."""

import time
from copy import deepcopy
from threading import Event

import mujoco
import numpy as np

from malecns_sim.simulation.fly import FlySimulation


class _ViewerClosed(Exception):
    pass


def show_replay(source, mapping, commands, duration_ms, *, speed=0.1):
    """Re-run the same physics with a separate display state until the window closes.

    UI edits/perturbations affect only the display copy, never the replay or
    saved experiment. There is no new neural input and no sensory feedback.
    """
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be finite and positive")
    from mujoco.viewer import launch_passive

    from malecns_sim.bridge.replay import replay_physics

    model = deepcopy(source.model)
    sim = FlySimulation(model, mujoco.MjData(model), source.camera)
    display_model = deepcopy(model)
    display_data = mujoco.MjData(display_model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral")
    mujoco.mj_resetDataKeyframe(display_model, display_data, key)
    mujoco.mj_forward(display_model, display_data)
    paused = Event()

    def key_callback(keycode):
        if keycode == 32:  # Space
            if paused.is_set():
                paused.clear()
            else:
                paused.set()

    print(
        f"MuJoCo motor replay at {speed:g}x speed. Loops until you close the window.\n"
        "Space: pause/resume. Mouse: rotate/zoom/pan. No sensory feedback.",
        flush=True,
    )
    with launch_passive(
        display_model,
        display_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as handle:
        print("MuJoCo viewer is open. Close its window to finish.", flush=True)
        with handle.lock():
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            handle.cam.lookat[:] = display_data.qpos[:3]
            handle.cam.distance = 4.0
            handle.cam.azimuth = 90
            handle.cam.elevation = -20
        while handle.is_running():
            mujoco.mj_resetDataKeyframe(model, sim.data, key)
            mujoco.mj_forward(model, sim.data)
            started = time.monotonic()
            next_frame = 0.0

            def present(current, final):
                nonlocal started, next_frame
                if not handle.is_running():
                    raise _ViewerClosed
                if current.data.time < next_frame and not final:
                    return
                while True:
                    if not handle.is_running():
                        raise _ViewerClosed
                    if paused.is_set():
                        before = time.monotonic()
                        handle.sync()
                        time.sleep(1 / 60)
                        started += time.monotonic() - before
                        continue
                    remaining = started + current.data.time / speed - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 1 / 60))
                with handle.lock():
                    for field in ("qpos", "qvel", "ctrl", "act"):
                        getattr(display_data, field)[:] = getattr(current.data, field)
                    display_data.time = current.data.time
                    mujoco.mj_forward(display_model, display_data)
                handle.sync()
                next_frame = current.data.time + speed / 60

            try:
                replay_physics(sim, mapping, commands, duration_ms, frame_count=0, on_step=present)
                # Show the final position before resetting for the next repetition.
                deadline = time.monotonic() + 0.5
                while handle.is_running() and time.monotonic() < deadline:
                    handle.sync()
                    time.sleep(1 / 60)
            except _ViewerClosed:
                break
