"""Validated, explicit parameters for the Shiu-derived reference model."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LIFConfig:
    model: str = "shiu-lif-v1"
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_threshold_mv: float = -45.0
    tau_membrane_ms: float = 20.0
    tau_synapse_ms: float = 5.0
    refractory_ms: float = 2.2
    synaptic_delay_ms: float = 1.8
    weight_per_contact_mv: float = 0.275
    dt_ms: float = 0.1
    transmitter_sign: dict[str, int] = field(
        default_factory=lambda: {
            "acetylcholine": 1,
            "gaba": -1,
            "glutamate": -1,
            "dopamine": 1,
            "serotonin": 1,
            "octopamine": 1,
        }
    )
    unknown_nt_policy: str = "zero"
    stimulus: dict = field(
        default_factory=lambda: {"type": "poisson", "rate_hz": 100.0, "gain_mv": 68.75}
    )

    def __post_init__(self):
        if self.model != "shiu-lif-v1":
            raise ValueError("Unsupported neural model")
        for name, value in asdict(self).items():
            if name.endswith(("_mv", "_ms")) and (
                not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.v_threshold_mv <= max(self.v_rest_mv, self.v_reset_mv):
            raise ValueError("Threshold must exceed resting/reset potential")
        for name in (
            "tau_membrane_ms",
            "tau_synapse_ms",
            "refractory_ms",
            "dt_ms",
            "weight_per_contact_mv",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.synaptic_delay_ms < 0:
            raise ValueError("Synaptic delay cannot be negative")
        for name in ("refractory_ms", "synaptic_delay_ms"):
            self.steps(getattr(self, name))
        if self.unknown_nt_policy not in {"zero", "error", "excitatory"}:
            raise ValueError("unknown_nt_policy must be error, zero or excitatory")
        if not self.transmitter_sign or any(
            not isinstance(k, str) or v not in {-1, 1} for k, v in self.transmitter_sign.items()
        ):
            raise ValueError("Transmitter signs must map label strings to -1 or +1")
        if (
            set(self.stimulus) != {"type", "rate_hz", "gain_mv"}
            or self.stimulus["type"] != "poisson"
        ):
            raise ValueError("Stimulus defaults require type=poisson, rate_hz and gain_mv")
        for key in ("rate_hz", "gain_mv"):
            if not math.isfinite(self.stimulus[key]) or self.stimulus[key] < 0:
                raise ValueError(f"Stimulus {key} must be finite and nonnegative")

    def steps(self, time_ms: float) -> int:
        if not math.isfinite(time_ms) or time_ms < 0:
            raise ValueError("Time must be finite and nonnegative")
        steps = int(round(time_ms / self.dt_ms))
        if not math.isclose(steps * self.dt_ms, time_ms, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Time must be an integer multiple of dt_ms")
        return steps

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> LIFConfig:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "configs/neural/shiu_lif_v1.yaml"
        if not path.is_file():
            path = Path(__file__).parent / "profiles/shiu_lif_v1.yaml"
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Neural config must be a YAML mapping")
    return LIFConfig(**values)
