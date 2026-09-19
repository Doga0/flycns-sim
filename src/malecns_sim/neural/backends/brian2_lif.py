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
    ):
        graph.validate()
        self.graph = graph
        self.config = config or LIFConfig()
        self.node_signs, self.sign_report = transmitter_signs(graph.neurotransmitters, self.config)
        self.record_nodes = tuple(record_nodes)
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
        self.spike_monitor = SpikeMonitor(self.neurons, codeobj_class=NumpyCodeObject)
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
        self.network = Network(*objects)
        self.inputs = []
        for number, stimulus in enumerate(self.stimuli):
            self._add_input(number, stimulus)

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
        for stimulus, (generator, _, rng) in zip(self.stimuli, self.inputs):
            indices, event_steps = events(stimulus, self.config, rng, self.time_steps, end_step)
            generator.set_spikes(indices, event_steps * self.config.dt_ms * ms, sorted=True)
        self._check_finite()
        self.network.run(duration_ms * ms, namespace={})
        self.time_steps = end_step
        self._check_finite()

    def get_spikes(self) -> tuple[np.ndarray, np.ndarray]:
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
