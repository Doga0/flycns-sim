"""Run the independent body with a local viewer or offscreen PNG rendering."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from malecns_sim.simulation.fly import FlySimulation, build_fly


def check_state(sim: FlySimulation) -> None:
    for name in ("qpos", "qvel", "qacc", "ctrl", "act"):
        if not np.isfinite(getattr(sim.data, name)).all():
            raise RuntimeError(f"Non-finite MuJoCo {name} at t={sim.data.time}")
    warnings = [int(warning.number) for warning in sim.data.warning]
    if any(warnings):
        raise RuntimeError(f"MuJoCo warnings at t={sim.data.time}: {warnings}")


def run_headless(
    sim: FlySimulation,
    output_dir: Path,
    *,
    steps: int,
    frame_count: int = 8,
) -> dict:
    if steps < 1 or frame_count < 1:
        raise ValueError("steps and frame_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_steps = set(np.linspace(1, steps, min(frame_count, steps), dtype=int).tolist())
    start_time = float(sim.data.time)
    check_state(sim)
    frames = []
    with mujoco.Renderer(sim.model, height=480, width=640) as renderer:
        for step in range(1, steps + 1):
            previous_time = float(sim.data.time)
            mujoco.mj_step(sim.model, sim.data)
            check_state(sim)
            if not np.isfinite(sim.data.time) or sim.data.time <= previous_time:
                raise RuntimeError("MuJoCo time reset or stopped advancing")
            if step in capture_steps:
                # Update derived body transforms for the current integration state.
                mujoco.mj_forward(sim.model, sim.data)
                check_state(sim)
                renderer.update_scene(sim.data, camera=sim.camera)
                pixels = renderer.render()
                if np.ptp(pixels) == 0:
                    raise RuntimeError("Headless renderer produced a blank image")
                filename = f"frame_{len(frames):03d}.png"
                Image.fromarray(pixels).save(output_dir / filename)
                frames.append(filename)
    report = {
        "steps": steps,
        "timestep": float(sim.model.opt.timestep),
        "simulated_seconds": float(sim.data.time) - start_time,
        "finite_state": True,
        "warning_count": 0,
        "camera": sim.camera,
        "frames": frames,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--viewer", action="store_true")
    modes.add_argument("--headless", action="store_true")
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--steps", type=int, help="Exact physics steps (smoke gate: 1000)")
    timing.add_argument(
        "--duration", type=float, default=2.0, help="Simulated seconds (default: 2)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke_test"))
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    sim = build_fly()
    print(f"Compiled NeuroMechFly: nq={sim.model.nq}, nv={sim.model.nv}, nu={sim.model.nu}")
    if args.viewer:
        from flygym.rendering import launch_interactive_viewer

        launch_interactive_viewer(sim.model, sim.data)
    else:
        steps = args.steps or math.ceil(args.duration / sim.model.opt.timestep)
        print(json.dumps(run_headless(sim, args.output_dir, steps=steps), indent=2))


if __name__ == "__main__":
    main()
