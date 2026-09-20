"""Causal, population-normalized rate decoder with independent control clock."""

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext

import numpy as np

from malecns_sim.bridge.mapping import MotorMapping


@dataclass(frozen=True)
class MotorCommand:
    t_ms: float
    actuator_name: str
    target: float
    source_activity: float
    flexor_rate_hz: float
    extensor_rate_hz: float
    flexor_filtered_hz: float
    extensor_filtered_hz: float

    def to_dict(self):
        return asdict(self)


class MotorDecoder:
    def __init__(self, mapping: MotorMapping):
        self.mapping = mapping
        # Host libm exp/expm1 may differ by one ulp between Windows and Linux.
        # Decimal.exp is correctly rounded; compute each scalar once at 50 digits
        # then convert to the same IEEE binary64 coefficient on both platforms.
        with localcontext() as context:
            context.prec = 50
            self.alpha = {
                c["name"]: float(
                    Decimal(1)
                    - (
                        -Decimal(str(mapping.control_dt_ms)) / Decimal(str(c["decoder"]["tau_ms"]))
                    ).exp()
                )
                for c in mapping.channels
            }
        self.reset()

    def reset(self):
        self.time_ms = 0.0
        self.filtered = {c["name"]: np.zeros(2) for c in self.mapping.channels}
        self.previous = {c["name"]: c["body"]["neutral_position"] for c in self.mapping.channels}
        self.unmapped_spikes = 0
        self.unmapped_nodes = set()

    def decode(self, spikes, t_start_ms: float, t_end_ms: float) -> list[MotorCommand]:
        """Consume only motor spikes in [start,end); output applies at end, never earlier.

        Known unmapped motor cells are counted, never assigned a default actuator.
        Spikes must use parent graph indices and contain their corresponding body IDs.
        """
        dt = self.mapping.control_dt_ms
        if (
            not np.isfinite([t_start_ms, t_end_ms]).all()
            or not np.isclose(t_start_ms, self.time_ms, rtol=0, atol=1e-9)
            or not np.isclose(t_end_ms - t_start_ms, dt, rtol=0, atol=1e-9)
        ):
            raise ValueError("Decoder needs consecutive complete control windows")
        counts = {}
        mapped = {r["node_index"] for r in self.mapping.rows}
        unmapped = []
        for r in spikes.to_pylist():
            t, node, body_id = r["t_ms"], r["node_index"], r["body_id"]
            if not np.isfinite(t) or not t_start_ms <= t < t_end_ms:
                raise ValueError("Nonfinite/out-of-window spike (future spikes are forbidden)")
            if self.mapping.motor_identity.get(node) != body_id:
                raise ValueError("Unknown motor neuron or mismatched node/body identity")
            counts[node] = counts.get(node, 0) + 1
            if node not in mapped:
                unmapped.append(node)
        commands = []
        for c in self.mapping.channels:
            d = c["decoder"]
            rate = np.array(
                [
                    sum(counts.get(n, 0) for n in c["groups"][g])
                    / len(c["groups"][g])
                    / (dt / 1000)
                    for g in ("flexor", "extensor")
                ]
            )
            filtered = self.filtered[c["name"]] + self.alpha[c["name"]] * (
                rate - self.filtered[c["name"]]
            )
            activation = np.clip((filtered - d["r0_hz"]) / (d["rmax_hz"] - d["r0_hz"]), 0, 1)
            activity = float(activation[1] - activation[0])
            target = (
                c["body"]["neutral_position"] + d["extensor_direction"] * d["gain_rad"] * activity
            )
            previous = self.previous[c["name"]]
            target = float(
                np.clip(
                    target,
                    max(c["minimum"], previous - d["max_delta_rad"]),
                    min(c["maximum"], previous + d["max_delta_rad"]),
                )
            )
            commands.append(
                MotorCommand(
                    float(t_end_ms),
                    c["body"]["actuator_name"],
                    target,
                    activity,
                    *map(float, rate),
                    *map(float, filtered),
                )
            )
            self.filtered[c["name"]] = filtered
            self.previous[c["name"]] = target
        self.unmapped_spikes += len(unmapped)
        self.unmapped_nodes.update(unmapped)
        self.time_ms = t_end_ms
        return commands
