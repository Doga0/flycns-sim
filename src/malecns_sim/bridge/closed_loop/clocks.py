"""Integer-ratio clocks; floating times are checked, never used for scheduling."""

from dataclasses import dataclass

import numpy as np


def steps(duration, dt):
    if not np.isfinite([duration, dt]).all() or duration <= 0 or dt <= 0:
        raise ValueError("Clock durations must be finite and positive")
    count = round(duration / dt)
    if count < 1 or not np.isclose(count * dt, duration, rtol=0, atol=1e-9):
        raise ValueError("Clock intervals must be integer multiples")
    return count


@dataclass(frozen=True)
class Clocks:
    neural_dt_ms: float
    physics_dt_ms: float
    sensory_dt_ms: float
    motor_dt_ms: float
    tolerance_ms: float = 1e-7

    def __post_init__(self):
        steps(self.sensory_dt_ms, self.neural_dt_ms)
        steps(self.sensory_dt_ms, self.physics_dt_ms)
        steps(self.motor_dt_ms, self.sensory_dt_ms)
        if not np.isfinite(self.tolerance_ms) or not 0 < self.tolerance_ms < self.neural_dt_ms:
            raise ValueError("Invalid clock tolerance")

    def check(self, scheduler_time, neural_time, body_time):
        values = np.asarray([scheduler_time, neural_time, body_time])
        drift = float(np.ptp(values))
        if not np.isfinite(values).all() or drift > self.tolerance_ms:
            raise RuntimeError(f"Clock drift: scheduler/neural/body = {values.tolist()}")
        return drift
