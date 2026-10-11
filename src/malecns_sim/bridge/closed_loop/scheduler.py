"""Causal ticks: observe, encode, advance both systems, then apply next-window command."""

import mujoco
import numpy as np
import pyarrow as pa

from malecns_sim.bridge.closed_loop.clocks import steps
from malecns_sim.simulation.smoke_test import check_state

SPIKE_SCHEMA = pa.schema(
    [
        ("t_ms", pa.float64()),
        ("node_index", pa.int32()),
        ("body_id", pa.uint64()),
    ]
)


class SafetyLimit(RuntimeError):
    """An engineering guard ended an experiment; partial artifacts must be retained."""


class MuJoCoBody:
    """The scheduler sees a body boundary, not MuJoCo address arithmetic."""

    def __init__(self, sim, reader, controller, clocks, safety, neutral):
        self.sim, self.reader, self.controller = sim, reader, controller
        self.clocks, self.safety, self.neutral = clocks, safety, neutral

    @property
    def time_ms(self):
        return float(self.sim.data.time * 1000)

    def read(self):
        state = self.reader.read(self.sim.data)
        if abs(state.joint_position_rad - self.neutral) > self.safety["max_abs_joint_offset_rad"]:
            raise SafetyLimit("max_abs_joint_offset_rad")
        if abs(state.joint_velocity_rad_s) > self.safety["max_abs_velocity_rad_s"]:
            raise SafetyLimit("max_abs_velocity_rad_s")
        return state

    def advance(self, duration_ms):
        for _ in range(steps(duration_ms, self.clocks.physics_dt_ms)):
            mujoco.mj_step(self.sim.model, self.sim.data)
            check_state(self.sim)
            self.read()

    def apply(self, command):
        self.controller.apply(command, self.sim.data)


class ClosedLoopScheduler:
    """Injectable interfaces keep orchestration independent of neural/encoder internals.

    modes: feedback_on, feedback_off, sensory_clamp; compatibility probes:
    motor_only accepts recorded motor events, sensory_only accepts a driven body.
    """

    def __init__(
        self,
        *,
        body,
        neural,
        encoder,
        decoder,
        clocks,
        safety,
        node_identity,
        input_gains,
        mode="feedback_on",
        recorded_motor=(),
        on_sample=None,
    ):
        if mode not in {
            "feedback_on",
            "feedback_off",
            "sensory_clamp",
            "motor_only",
            "sensory_only",
        }:
            raise ValueError("Unknown closed-loop condition")
        self.body, self.neural, self.encoder, self.decoder = body, neural, encoder, decoder
        self.clocks, self.safety, self.mode = clocks, safety, mode
        self.node_identity, self.input_gains = node_identity, input_gains
        self.recorded_motor = list(recorded_motor)
        self.on_sample = on_sample
        self.body_states, self.sensory_activity, self.sensory_events = [], [], []
        self.neural_spikes, self.motor_activity, self.motor_commands, self.timeline = [], [], [], []
        self.status, self.reason = "COMPLETED", None
        self.time_ms, self.drift_max = 0.0, 0.0

    def run(self, duration_ms):
        count = steps(duration_ms, self.clocks.sensory_dt_ms)
        steps(duration_ms, self.clocks.motor_dt_ms)
        if duration_ms > self.safety["max_duration_ms"]:
            raise ValueError("Requested duration exceeds engineering safety limit")
        interval = steps(self.clocks.motor_dt_ms, self.clocks.sensory_dt_ms)
        cursor, accumulator, last_rates = 0, [], (0.0, 0.0)
        try:
            self.clocks.check(0, self.neural.time_ms, self.body.time_ms)
            for tick in range(count):
                start = tick * self.clocks.sensory_dt_ms
                end = (tick + 1) * self.clocks.sensory_dt_ms
                state = self.body.read()
                self.body_states.append(state.to_dict())
                if self.on_sample is not None:
                    self.on_sample(start)
                activity, events = [], []
                if self.mode not in {"feedback_off", "motor_only"}:
                    sensory = self.encoder.encode(state, start, end)
                    activity, events = sensory.activity.to_pylist(), sensory.spikes.to_pylist()
                    for row in activity:
                        if (
                            not np.isfinite(row["rate_hz"])
                            or not 0 <= row["rate_hz"] <= self.safety["max_sensory_rate_hz"]
                        ):
                            raise SafetyLimit("max_sensory_rate_hz")
                    for event in events:
                        if (
                            event["node_index"] not in self.input_gains
                            or not start <= event["t_ms"] < end
                        ):
                            raise ValueError("Unmapped/future sensory event")
                        event["body_id"] = self.node_identity[event["node_index"]]
                        event["body_sample_t_ms"] = start
                        event["injected"] = self.mode != "sensory_clamp"
                    self.sensory_activity.extend(activity)
                    self.sensory_events.extend(events)
                    if self.mode != "sensory_clamp":
                        self.neural.inject_events(events)
                before = self.neural.total_spikes
                self.neural.advance(self.clocks.sensory_dt_ms)
                total = self.neural.total_spikes - before
                (times, nodes), cursor = self.neural.spikes_since(cursor)
                new = [
                    {"t_ms": float(t), "node_index": int(n), "body_id": self.node_identity[int(n)]}
                    for t, n in zip(times, nodes)
                ]
                if any(not start <= row["t_ms"] < end for row in new):
                    raise ValueError("Neural cursor returned past/future spikes")
                self.neural_spikes.extend(new)
                if total > self.safety["max_total_spikes_per_ms"] * self.clocks.sensory_dt_ms:
                    raise SafetyLimit("max_total_spikes_per_ms")
                motors = [r for r in new if r["node_index"] in self.decoder.mapping.motor_identity]
                if self.mode == "motor_only":
                    motors = [r for r in self.recorded_motor if start <= r["t_ms"] < end]
                self.motor_activity.extend(motors)
                accumulator.extend(motors)
                # Physics must consume the OLD target throughout [start,end).
                self.body.advance(self.clocks.sensory_dt_ms)
                drift = self.clocks.check(end, self.neural.time_ms, self.body.time_ms)
                self.drift_max = max(self.drift_max, drift)
                self.time_ms = end
                if (tick + 1) % interval == 0:
                    window_start = end - self.clocks.motor_dt_ms
                    counts = {}
                    for row in accumulator:
                        counts[row["node_index"]] = counts.get(row["node_index"], 0) + 1
                    if any(
                        n * 1000 / self.clocks.motor_dt_ms > self.safety["max_motor_rate_hz"]
                        for n in counts.values()
                    ):
                        raise SafetyLimit("max_motor_rate_hz")
                    commands = self.decoder.decode(
                        pa.Table.from_pylist(accumulator, schema=SPIKE_SCHEMA), window_start, end
                    )
                    accumulator = []
                    for command in commands:
                        applied = self.mode not in {"feedback_off", "sensory_clamp", "sensory_only"}
                        if applied:
                            self.body.apply(command)
                        self.motor_commands.append(
                            {
                                **command.to_dict(),
                                "window_start_ms": window_start,
                                "window_end_ms": end,
                                "applied": applied,
                            }
                        )
                        last_rates = (command.flexor_rate_hz, command.extensor_rate_hz)
                self.timeline.append(
                    {
                        "t_ms": start,
                        "end_ms": end,
                        "scheduler_time_ms": end,
                        "neural_time_ms": self.neural.time_ms,
                        "physics_time_ms": self.body.time_ms,
                        "q": state.joint_position_rad,
                        "qvel": state.joint_velocity_rad_s,
                        "target": state.actuator_target_rad,
                        "sensory_rate_hz": sum(r["rate_hz"] for r in activity),
                        "sensory_events": len(events),
                        "injected_events": sum(r["injected"] for r in events),
                        "cns_spikes": total,
                        "flexor_rate_hz": last_rates[0],
                        "extensor_rate_hz": last_rates[1],
                    }
                )
            self.body_states.append(self.body.read().to_dict())
            if self.on_sample is not None:
                self.on_sample(self.time_ms)
        except (SafetyLimit, FloatingPointError, RuntimeError) as exc:
            self.status = "ABORTED_SAFETY_LIMIT"
            self.reason = f"{type(exc).__name__}: {exc}"
        return self
