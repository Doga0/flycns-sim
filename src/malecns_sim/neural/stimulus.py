"""Scheduled external drive with deterministic, timestep-discretized Poisson input."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from malecns_sim.neural.config import LIFConfig


@dataclass(frozen=True)
class Stimulus:
    nodes: tuple[int, ...]
    type: str = "poisson"
    rate_hz: float = 100.0
    start_ms: float = 0.0
    stop_ms: float | None = None
    gain_mv: float = 68.75
    times_ms: tuple[float, ...] = ()

    def validate(self, n: int, config: LIFConfig) -> None:
        if not self.nodes or any(
            isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < n
            for i in self.nodes
        ):
            raise ValueError("Stimulus nodes must be valid local node indices")
        if tuple(sorted(set(self.nodes))) != tuple(self.nodes):
            raise ValueError("Stimulus nodes must be sorted and unique")
        if self.type not in {"poisson", "spikes"}:
            raise ValueError("Stimulus type must be poisson or spikes")
        if not math.isfinite(self.rate_hz) or not 0 <= self.rate_hz * config.dt_ms / 1000 <= 1:
            raise ValueError("Poisson rate must be finite with 0 <= rate_hz * dt_seconds <= 1")
        if not math.isfinite(self.gain_mv) or self.gain_mv < 0:
            raise ValueError("External gain must be finite and nonnegative")
        config.steps(self.start_ms)
        if self.stop_ms is not None:
            config.steps(self.stop_ms)
            if self.stop_ms <= self.start_ms:
                raise ValueError("Stimulus stop must be after start")
        if self.type == "poisson" and self.times_ms:
            raise ValueError("Poisson stimulus cannot specify spike times")
        steps = [config.steps(t) for t in self.times_ms]
        if steps != sorted(set(steps)):
            raise ValueError("Explicit spike times must be sorted and unique")
        if any(
            t < self.start_ms or (self.stop_ms is not None and t >= self.stop_ms)
            for t in self.times_ms
        ):
            raise ValueError("Explicit spike times must lie within the stimulus window")

    def to_dict(self) -> dict:
        values = asdict(self)
        # NumPy node indices are accepted by the API and must remain JSON-safe.
        values["nodes"] = [int(node) for node in self.nodes]
        return values


def events(
    stimulus: Stimulus, config: LIFConfig, rng: np.random.Generator, first_step: int, end_step: int
) -> tuple[np.ndarray, np.ndarray]:
    start = max(first_step, config.steps(stimulus.start_ms))
    stop = (
        min(end_step, config.steps(stimulus.stop_ms)) if stimulus.stop_ms is not None else end_step
    )
    if stop <= start:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    if stimulus.type == "spikes":
        times = np.array([config.steps(t) for t in stimulus.times_ms], dtype=np.int64)
        times = times[(times >= start) & (times < stop)]
        return np.tile(np.arange(len(stimulus.nodes), dtype=np.int32), len(times)), np.repeat(
            times, len(stimulus.nodes)
        )
    indices, times = [], []
    for offset in range(start, stop, 4096):
        length = min(4096, stop - offset)
        rows, cols = np.nonzero(
            rng.random((length, len(stimulus.nodes))) < stimulus.rate_hz * config.dt_ms / 1000
        )
        indices.append(cols.astype(np.int32))
        times.append(rows + offset)
    return np.concatenate(indices), np.concatenate(times)
