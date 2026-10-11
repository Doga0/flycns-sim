"""Brian2/NumPy reference LIF with explicit clocks, schedules, and no body coupling."""

import numpy as np
from brian2 import (
    Clock,
    Network,
    NetworkOperation,
    NeuronGroup,
    SpikeGeneratorGroup,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    ms,
    mV,
)
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject

from malecns_sim.neural.backends.base import NeuralBackend, NeuralGraph
from malecns_sim.neural.config import LIFConfig
from malecns_sim.neural.signs import transmitter_signs
from malecns_sim.neural.stimulus import Stimulus, events


class Brian2LIFBackend(NeuralBackend):
    """Reference engine; reset rebuilds state/queues rather than copying 25M synapses.

    Both v and g are clamped during refractory, as in the reference equations.
    All cells keep refractory, including externally driven cells (a documented
    departure from Shiu's optogenetic input helper). Record only spikes by default.
    """

    def __init__(
        self,
        graph: NeuralGraph,
        config: LIFConfig | None = None,
        *,
        seed: int = 12345,
        record_nodes: tuple[int, ...] = (),
        spike_record_nodes: tuple[int, ...] | None = None,
    ):
        graph.validate()
        self.graph = graph
        self.config = config or LIFConfig()
        self.node_signs, self.sign_report = transmitter_signs(graph.neurotransmitters, self.config)
        self.record_nodes = tuple(record_nodes)
        self.spike_record_nodes = spike_record_nodes
        if spike_record_nodes is not None and any(
            not 0 <= i < len(graph.body_ids) for i in spike_record_nodes
        ):
            raise ValueError("Invalid spike recording nodes")
        self.event_input_spec = None
        if len(record_nodes) > 32 or any(not 0 <= i < len(graph.body_ids) for i in record_nodes):
            raise ValueError("Optional state recording is limited to 32 valid nodes")
        self.stimuli: list[Stimulus] = []
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("seed must be an integer in [0, 2**32)")
            self.seed = seed
        config = self.config
        self.time_steps = 0
        self.silence_events = []
        self.clock = Clock(dt=config.dt_ms * ms)
        self.neurons = NeuronGroup(
            len(self.graph.body_ids),
            """
            dv/dt = (v_rest - v + g) / tau_m : volt (unless refractory)
            dg/dt = -g / tau_s : volt (unless refractory)
            silenced : boolean
            """,
            threshold="(v > v_threshold) and not silenced",
            reset="v = v_reset; g = 0*mV",
            refractory=config.refractory_ms * ms,
            method="exact",
            clock=self.clock,
            codeobj_class=NumpyCodeObject,
            namespace={
                "v_rest": config.v_rest_mv * mV,
                "v_reset": config.v_reset_mv * mV,
                "v_threshold": config.v_threshold_mv * mV,
                "tau_m": config.tau_membrane_ms * ms,
                "tau_s": config.tau_synapse_ms * ms,
            },
        )
        self.neurons.v = config.v_rest_mv * mV
        self.neurons.g = 0 * mV
        self.neurons.silenced = False
        self.synapses = Synapses(
            self.neurons,
            self.neurons,
            "w : volt",
            on_pre="g_post += w * int(not silenced_post)",
            delay=config.synaptic_delay_ms * ms,
            clock=self.clock,
            codeobj_class=NumpyCodeObject,
        )
        if len(self.graph.src):
            self.synapses.connect(i=self.graph.src, j=self.graph.dst)
            zero_edges = 0
            for start in range(0, len(self.graph.src), 1_000_000):
                end = min(start + 1_000_000, len(self.graph.src))
                signs = self.node_signs[self.graph.src[start:end]]
                weight = (
                    self.graph.contacts[start:end].astype(np.float64)
                    * signs
                    * config.weight_per_contact_mv
                )
                if not np.isfinite(weight).all():
                    raise ValueError("Non-finite model weights")
                self.synapses.w[start:end] = weight * mV
                zero_edges += int(np.count_nonzero(signs == 0))
            self.sign_report["zero_weight_edges"] = zero_edges
        else:
            self.synapses.active = False
            self.sign_report["zero_weight_edges"] = 0
        self.spike_monitor = SpikeMonitor(
            self.neurons, record=self.spike_record_nodes is None, codeobj_class=NumpyCodeObject
        )
        self.selected_times, self.selected_indices = [], []
        self.record_mask = np.zeros(len(self.graph.body_ids), dtype=bool)
        if self.spike_record_nodes is not None:
            self.record_mask[list(self.spike_record_nodes)] = True
        self.state_monitor = (
            StateMonitor(
                self.neurons,
                ["v", "g"],
                record=list(self.record_nodes),
                when="end",
                clock=self.clock,
                codeobj_class=NumpyCodeObject,
            )
            if self.record_nodes
            else None
        )
        # Detect divergence without retaining per-neuron/per-timestep histories.
        self.state_guard = NetworkOperation(
            self._check_finite, clock=self.clock, when="end", order=100
        )
        objects = [self.neurons, self.synapses, self.spike_monitor, self.state_guard]
        if self.state_monitor is not None:
            objects.append(self.state_monitor)
        if self.spike_record_nodes is not None:
            objects.append(NetworkOperation(self._record_selected, clock=self.clock, when="end"))
        self.network = Network(*objects)
        self.inputs = []
        for number, stimulus in enumerate(self.stimuli):
            self._add_input(number, stimulus)
        self.event_generator = None
        self.pending_events = []
        if self.event_input_spec is not None:
            self.configure_event_input(*self.event_input_spec)

    @property
    def time_ms(self):
        return float(self.network.t / ms)

    @property
    def total_spikes(self):
        return int(self.spike_monitor.num_spikes)

    def _record_selected(self):
        indices = self.neurons.spikes
        selected = indices[self.record_mask[indices]]
        self.selected_indices.extend(selected.tolist())
        self.selected_times.extend([float(self.clock.t / ms)] * len(selected))

    def configure_event_input(self, nodes, gains_mv):
        if self.time_steps or self.event_generator is not None:
            raise ValueError("Declare event inputs exactly once before advancing")
        nodes, gains = tuple(nodes), np.asarray(gains_mv, dtype=float)
        if (
            not nodes
            or len(set(nodes)) != len(nodes)
            or len(gains) != len(nodes)
            or not np.isfinite(gains).all()
            or np.any(gains < 0)
            or any(type(i) is not int or not 0 <= i < len(self.graph.body_ids) for i in nodes)
        ):
            raise ValueError("Invalid event input nodes/gains")
        self.event_input_spec = (nodes, gains.copy())
        self.event_node_lookup = {node: i for i, node in enumerate(nodes)}
        self.event_generator = SpikeGeneratorGroup(
            len(nodes), [], [] * ms, clock=self.clock, codeobj_class=NumpyCodeObject
        )
        drive = Synapses(
            self.event_generator,
            self.neurons,
            "gain : volt",
            on_pre="v_post += gain * int(not silenced_post)",
            clock=self.clock,
            codeobj_class=NumpyCodeObject,
        )
        drive.connect(i=np.arange(len(nodes)), j=np.asarray(nodes))
        drive.gain = gains * mV
        self.network.add(self.event_generator, drive)

    def inject_events(self, events):
        if self.event_generator is None:
            raise ValueError("Event inputs have not been declared")
        pending = []
        for event in events:
            node, time = event["node_index"], float(event["t_ms"])
            if node not in self.event_node_lookup:
                raise ValueError("Unmapped external input node")
            tick = self.config.steps(time)
            if tick < self.time_steps:
                raise ValueError("Cannot inject an event in the past")
            pending.append((tick, self.event_node_lookup[node]))
        combined = self.pending_events + pending
        if len(set(combined)) != len(combined):
            raise ValueError("Duplicate event for a neuron in the same timestep")
        self.pending_events = combined

    def spikes_since(self, cursor):
        size = (
            len(self.spike_monitor.i)
            if self.spike_record_nodes is None
            else len(self.selected_indices)
        )
        if type(cursor) is not int or not 0 <= cursor <= size:
            raise ValueError("Invalid spike cursor")
        if self.spike_record_nodes is None:
            size = len(self.spike_monitor.i)
            times = np.asarray(self.spike_monitor.t[cursor:] / ms)
            indices = np.asarray(self.spike_monitor.i[cursor:], dtype=np.int32)
        else:
            size = len(self.selected_indices)
            times = np.asarray(self.selected_times[cursor:])
            indices = np.asarray(self.selected_indices[cursor:], dtype=np.int32)
        ticks = np.rint(times / self.config.dt_ms).astype(np.int64)
        order = np.lexsort((indices, ticks))
        return (ticks[order] * self.config.dt_ms, indices[order]), size

    def _check_finite(self):
        if not (np.isfinite(self.neurons.v_[:]).all() and np.isfinite(self.neurons.g_[:]).all()):
            raise FloatingPointError("Neural state contains NaN/Inf")

    def _add_input(self, number: int, stimulus: Stimulus) -> None:
        generator = SpikeGeneratorGroup(
            len(stimulus.nodes),
            np.empty(0, dtype=np.int32),
            np.empty(0) * ms,
            clock=self.clock,
            codeobj_class=NumpyCodeObject,
        )
        drive = Synapses(
            generator,
            self.neurons,
            on_pre="v_post += gain * int(not silenced_post)",
            clock=self.clock,
            codeobj_class=NumpyCodeObject,
            namespace={"gain": stimulus.gain_mv * mV},
        )
        drive.connect(i=np.arange(len(stimulus.nodes)), j=np.array(stimulus.nodes))
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, number])))
        self.inputs.append((generator, drive, rng))
        self.network.add(generator, drive)

    def schedule(self, stimulus: Stimulus) -> None:
        stimulus.validate(len(self.graph.body_ids), self.config)
        if self.config.steps(stimulus.start_ms) < self.time_steps:
            raise ValueError("Cannot schedule a stimulus in the past")
        self.stimuli.append(stimulus)
        self._add_input(len(self.stimuli) - 1, stimulus)

    def stimulate(self, node_indices, rate_hz: float) -> None:
        self.schedule(
            Stimulus(
                tuple(sorted(set(node_indices))),
                rate_hz=rate_hz,
                start_ms=self.time_steps * self.config.dt_ms,
                gain_mv=self.config.stimulus["gain_mv"],
            )
        )

    def silence(self, node_indices) -> None:
        indices = np.asarray(node_indices)
        if (
            indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= len(self.graph.body_ids))
        ):
            raise ValueError("Invalid silence node indices")
        self.neurons.silenced[indices] = True
        self.neurons.v[indices] = self.config.v_rest_mv * mV
        self.neurons.g[indices] = 0 * mV
        self.silence_events.append(
            {
                "t_ms": self.time_steps * self.config.dt_ms,
                "nodes": [int(i) for i in sorted(set(indices))],
            }
        )

    def run(self, duration_ms: float) -> None:
        steps = self.config.steps(duration_ms)
        if steps <= 0:
            raise ValueError("Run duration must be positive")
        end_step = self.time_steps + steps
        if self.event_generator is not None:
            if any(tick >= end_step for tick, _ in self.pending_events):
                raise ValueError("Injected events must belong to the next advance window")
            pending = sorted(self.pending_events)
            self.event_generator.set_spikes(
                np.asarray([i for _, i in pending], dtype=np.int32),
                np.asarray([t for t, _ in pending]) * self.config.dt_ms * ms,
                sorted=True,
            )
        for stimulus, (generator, _, rng) in zip(self.stimuli, self.inputs):
            indices, event_steps = events(stimulus, self.config, rng, self.time_steps, end_step)
            generator.set_spikes(indices, event_steps * self.config.dt_ms * ms, sorted=True)
        self._check_finite()
        self.network.run(duration_ms * ms, namespace={})
        self.time_steps = end_step
        self.pending_events = []
        self._check_finite()

    def get_spikes(self) -> tuple[np.ndarray, np.ndarray]:
        if self.spike_record_nodes is not None:
            return self.spikes_since(0)[0]
        # Normalize to the timestep grid, sort simultaneous spikes by local index.
        ticks = np.rint(np.asarray(self.spike_monitor.t / ms) / self.config.dt_ms).astype(np.int64)
        indices = np.asarray(self.spike_monitor.i, dtype=np.int32)
        order = np.lexsort((indices, ticks))
        return ticks[order] * self.config.dt_ms, indices[order]

    def get_state_trace(self) -> dict:
        if self.state_monitor is None:
            raise ValueError("No optional state recording was requested")
        return {
            "t_ms": np.asarray(self.state_monitor.t / ms),
            "v_mv": np.asarray(self.state_monitor.v / mV),
            "g_mv": np.asarray(self.state_monitor.g / mV),
        }
