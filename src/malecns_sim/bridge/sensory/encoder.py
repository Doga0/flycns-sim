"""Deterministic/Poisson body-state encoders independent of the neural backend."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from malecns_sim.bridge.sensory.body_state import BodyState


@dataclass(frozen=True)
class SensoryInput:
    activity: pa.Table
    spikes: pa.Table
    mode: str
    seed: int

    def to_dict(self):
        return {
            "mode": self.mode,
            "seed": self.seed,
            "activity_rows": self.activity.num_rows,
            "spike_rows": self.spikes.num_rows,
        }


class SensoryEncoder:
    """Body variable to explicit external events, with no knowledge of Brian2/LIF."""

    def __init__(self, mapping, *, mode="explicit_spikes", seed=12345):
        if mode not in {"explicit_spikes", "poisson"}:
            raise ValueError("Sensory mode must be explicit_spikes or poisson")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Sensory seed must be an integer in [0, 2**32)")
        self.mapping, self.mode, self.seed = mapping, mode, seed
        self.reset()

    def reset(self):
        self.time_ms = 0.0
        self.phase = {channel["name"]: 0.0 for channel in self.mapping.channels}
        self.rng = np.random.Generator(np.random.PCG64(self.seed))

    def _rate(self, channel, state: BodyState):
        source, encoder = channel["source"], channel["encoder"]
        if state.joint_name != source["joint"]:
            raise ValueError("Body state joint differs from sensory mapping")
        value = (
            state.joint_position_rad
            if source["signal"] == "position"
            else state.joint_velocity_rad_s
        )
        if encoder["type"] == "directional_velocity_encoder_v1":
            value = value if encoder["direction"] == "positive" else -value
        normalized = float(
            np.clip(
                (value - encoder["min_value"]) / (encoder["max_value"] - encoder["min_value"]), 0, 1
            )
        )
        rate = encoder["min_rate_hz"] + normalized * (
            encoder["max_rate_hz"] - encoder["min_rate_hz"]
        )
        return value, normalized, float(rate)

    def encode(self, body_state: BodyState, t_start_ms: float, t_end_ms: float) -> SensoryInput:
        dt = self.mapping.sensory_dt_ms
        if (
            not np.isfinite([t_start_ms, t_end_ms]).all()
            or not np.isclose(t_start_ms, self.time_ms, rtol=0, atol=1e-9)
            or not np.isclose(t_end_ms - t_start_ms, dt, rtol=0, atol=1e-9)
            or not np.isclose(body_state.t_ms, t_start_ms, rtol=0, atol=1e-7)
        ):
            raise ValueError("Encoder needs consecutive sensory windows and matching body state")
        activity, spikes = [], []
        for channel in self.mapping.channels:
            value, normalized, rate = self._rate(channel, body_state)
            activity.append(
                {
                    "t_start_ms": t_start_ms,
                    "t_end_ms": t_end_ms,
                    "channel": channel["name"],
                    "joint": body_state.joint_name,
                    "signal": channel["source"]["signal"],
                    "body_value": value,
                    "normalized_value": normalized,
                    "rate_hz": rate,
                    "encoder": channel["encoder"]["type"],
                }
            )
            if self.mode == "explicit_spikes":
                self.phase[channel["name"]] += rate * dt / 1000
                emit = self.phase[channel["name"]] + 1e-12 >= 1
                if emit:
                    self.phase[channel["name"]] -= np.floor(self.phase[channel["name"]])
                emitted_nodes = channel["nodes"] if emit else ()
            else:
                emitted_nodes = [
                    node for node in channel["nodes"] if self.rng.random() < rate * dt / 1000
                ]
            for row in emitted_nodes:
                spikes.append(
                    {
                        "t_ms": t_start_ms,
                        "node_index": row,
                        "source_channel": channel["name"],
                        "encoder": channel["encoder"]["type"],
                    }
                )
        self.time_ms = t_end_ms
        activity_table = pa.Table.from_pylist(activity)
        spikes_table = pa.Table.from_pylist(
            spikes,
            schema=pa.schema(
                [
                    ("t_ms", pa.float64()),
                    ("node_index", pa.int32()),
                    ("source_channel", pa.string()),
                    ("encoder", pa.string()),
                ]
            ),
        )
        return SensoryInput(activity_table, spikes_table, self.mode, self.seed)
