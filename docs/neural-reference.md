# v0.3: Shiu-derived LIF reference engine

This release tests whether activity can propagate through MaleCNS connectivity.
It does not connect spikes to FlyGym actuators, simulate sensory feedback, or
establish biological behavior. The anatomical graph is unchanged: contacts stay
positive integer counts. Transmitter signs, gain, dynamics and stimulation live
in `src/malecns_sim/neural/`.

## 1. Prepare the environment

Run commands from the repository root. Install uv for the local Windows workflow
or Docker Desktop with its WSL2 backend for the container workflow, following the
[README](../README.md). Python 3.12 and exact resolved package versions are in
`uv.lock`. Brian2 2.10.1 uses its NumPy CPU runtime; no compiler or GPU is needed.

```powershell
uv sync --locked
uv run pytest tests/neural -m "not graph" -q
uv run python -m malecns_sim.neural.experiments.toy --seed 42 --output outputs/neural/toy_demo
```

The toy is a three-cell excitatory chain with 250 contacts per edge and a 100 Hz
input from 100 to 600 ms, within a 1,000 ms experiment. Contacts were deliberately
chosen to demonstrate propagation. A single weak connection (for example, ten
contacts) need not make its target spike at the reference gain. The tests also
cover inhibition, delay, refractory behavior, resting stability, seed/reset
reproducibility, and output mapping.

To use real data, download and build once:

```powershell
uv run python -m malecns_sim.cns.download
uv run python -m malecns_sim.cns.census
uv run python -m malecns_sim.cns.graph.build --policy published
```

If `data/processed/malecns-v1.0/published-v1/manifest.json` already exists, use
that graph; the builder deliberately refuses to overwrite it. See the
[Graph Core guide](graph-core.md) for other policies and full provenance.
The current provisional profile contains 166,700 nodes, nine more than the
published 166,691 census. This discrepancy is carried into neural experiment
provenance; no nine cells are arbitrarily removed to force agreement.

## 2. Run a bounded MaleCNS experiment first

```powershell
uv run python -m malecns_sim.neural.experiments.subgraph `
    --stimulate-body-id 11755 `
    --max-nodes 5000 `
    --hops 2 `
    --duration-ms 1000 `
    --start-ms 100 `
    --stop-ms 600 `
    --rate-hz 100 `
    --seed 42 `
    --output outputs/neural/subgraph_demo
```

The subgraph follows outgoing anatomical edges breadth first. If a hop exceeds
the node cap, it ranks candidates by summed contacts from the previous retained
frontier, then parent node index to break ties. The resulting graph includes
**all induced edges**, including recurrent and self edges. Boundary connections
are absent; an induced subgraph is not equivalent to running those cells inside
the full CNS. Candidate counts, omitted nodes, selection rule and hop membership
are saved in `experiment.json`.

Omitting `--stimulate-body-id` chooses the source of the strongest positive-to-
positive non-self edge under the configured sign policy, breaking ties by
canonical edge order. On this dataset the source is body ID **11755**. This is
an engineering test source, not a claim about a sensory or motor role.
`python -m malecns_sim.neural.experiments.stimulate --seed 42` is an alias for
the default subgraph experiment.

The default schedule is 0–100 ms baseline, 100–600 ms stimulus, and 600–1,000 ms
recovery. Recovery describes the input schedule; recurrent activity need not
stop when the external drive stops. If changing duration, adjust the input
window to fit. Windows PowerShell uses a backtick at the end of each continued
line; one-line commands work in either Windows or a Linux shell.

## 3. Run only 100 ms on the full graph initially

```powershell
uv run python -m malecns_sim.neural.experiments.full_cns `
    --duration-ms 100 `
    --stimulate-body-id 11755 `
    --rate-hz 100 `
    --seed 42 `
    --output outputs/neural/full_demo
```

The full experiment defaults to a **0–100 ms** stimulation window for its
100 ms duration. It does not inherit the subgraph's 100 ms baseline, which
would otherwise leave this short run unstimulated. It validates graph artifact
checksums before constructing Brian2 neurons and synapses. Larger runs can take
considerably more time and RAM: Brian2 holds its own synapse structures in
addition to the memory-mapped graph. No claim is made that the engine fits in
the few hundred MB used by the canonical graph itself.

To run a no-input control:

```powershell
uv run python -m malecns_sim.neural.experiments.spontaneous --duration-ms 100 --seed 42 --output outputs/neural/rest_demo
```

The resting model has no background drive or intrinsic noise. Starting at rest,
it should produce zero spikes. `--no-stimulus` also works on `full_cns` and
`subgraph`. Later, explicitly request longer durations with new output folders;
the CLI never launches a sweep or multiple trials automatically.

## 4. Understand the reference model

Parameters are in `configs/neural/shiu_lif_v1.yaml`. Use `--config path/to.yaml`
to run a different documented policy. The full resolved config is saved with
every run. The starting equations and scale come from
[Shiu et al., Nature 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/) and
their [reference implementation](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py).

| Parameter | Value |
| --- | ---: |
| Rest / reset | −52 mV |
| Spike threshold | strictly greater than −45 mV |
| Membrane time constant | 20 ms |
| Synaptic decay | 5 ms |
| Refractory | 2.2 ms |
| Recurrent transmission delay | 1.8 ms |
| Recurrent scale per contact | 0.275 mV |
| Timestep | 0.1 ms |
| External Poisson input rate | 100 Hz |
| External input gain | 68.75 mV |

```text
dv/dt = (Vrest - v + g) / tau_membrane
dg/dt = -g / tau_synapse

spike when v > Vthreshold
reset v = Vreset, g = 0
delayed recurrent event: g_post += contacts * sign_pre * 0.275 mV
external input event:    v_target += gain_mv
```

Brian2 integrates the linear equations with `method="exact"` on an explicit
clock. Both `v` and `g` use `unless refractory`, including rejection of incoming
updates while refractory, following the source equations and
[Brian2 refractory semantics](https://brian2.readthedocs.io/en/stable/user/refractoriness.html).
This implementation preserves the 2.2 ms refractory period for **all** cells.
Shiu's optogenetic helper removes it for stimulated cells; that is a deliberate
documented difference, so this engine is called **Shiu-derived**, not an exact
reproduction of the publication's stimulation protocol.

External events directly increment membrane voltage, as in the source's
`PoissonInput(target_var="v")`. They do not use recurrent contact weights or
the recurrent 1.8 ms delay. External gain is an optogenetic drive model
parameter, not a universal biological constant. Brian2 processes events in
its synapses scheduling slot: an external event at 10.0 ms first becomes
eligible for neuron threshold detection on the next 0.1 ms step. Recurrent
delay is measured from the **actual presynaptic spike**, not from the external
stimulus event. The tests check this distinction.

## 5. Transmitter signs and missing labels

| Released transmitter label | Model sign |
| --- | ---: |
| acetylcholine | +1 |
| gaba | −1 |
| glutamate | −1 |
| dopamine | +1 |
| serotonin | +1 |
| octopamine | +1 |

These are configurable model assumptions. Case and surrounding whitespace are
normalized; other labels are not silently translated into one of these six.
`unknown_nt_policy` supports:

- `zero` (default): outgoing model weights are zero; neurons and anatomical
  contacts remain present. Such a neuron can still receive input and spike.
- `error`: stop before constructing the engine if any label is unmapped.
- `excitatory`: explicitly assign +1 to unmapped labels for a comparison run.

On the current full graph, 155,632 nodes match the six labels. **11,068 are
unmapped:** 7,891 histamine, 2,999 `unclear`, and 178 missing. Histamine is a known
transmitter but is outside this six-transmitter model policy; its treatment as
unmapped does not mean the dataset lacks its identity. The default disables
677,985 outgoing model edges. Coverage and the reason are included in every
experiment and summary. Raw `consensus_nt` and contact counts are never edited.

## 6. Backend and stimulus API

`NeuralBackend` defines `reset`, `schedule`, `stimulate`, `silence`, `run`, and
`get_spikes`. `NeuralGraph` is a backend-independent view: local contiguous
indices address neurons, while `node_indices` and `body_ids` retain parent graph
identity. Full-graph edge arrays remain borrowed memory-map views, so keep
`CNSGraph` open for the lifetime of the neural graph/backend.

```python
from malecns_sim.cns.graph.store import CNSGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.config import load_config
from malecns_sim.neural.experiments.selection import extract_subgraph
from malecns_sim.neural.stimulus import Stimulus

with CNSGraph.load("data/processed/malecns-v1.0/published-v1") as stored:
    graph = extract_subgraph(stored, stored.node_index(11755), max_nodes=5000)
    source = int((graph.body_ids == 11755).nonzero()[0][0])
    engine = Brian2LIFBackend(graph, load_config(), seed=42)
    engine.schedule(Stimulus(nodes=(source,), type="poisson", rate_hz=100,
                             start_ms=100, stop_ms=600, gain_mv=68.75))
    engine.run(1000)
    t_ms, local_nodes = engine.get_spikes()
    original_body_ids = graph.body_ids[local_nodes]
```

Schedules use local neural indices and half-open `[start, stop)` windows. Times
must align to the configured timestep. `stimulate(nodes, rate_hz)` schedules
ongoing input starting at the current time. `silence(nodes)` prevents future
spikes and incoming drive, resets those cells to rest, and records the action;
already transmitted presynaptic events are not retracted. `reset(seed)` rebuilds
state, queues and RNG streams, retains scheduled stimuli, and clears silencing.

For deterministic tests, `Stimulus(type="spikes", times_ms=(10, 30, 50), ...)`
injects explicit external event times into all listed targets. Production
Poisson input uses independent Bernoulli trials with probability `rate_hz *
dt_ms / 1000` per target per step, matching a single-input timestep approximation.
At most one event per target per timestep is allowed. Rates above this bound
are rejected. Each stimulus has a separate PCG64 stream derived from the seed
and scheduling order. Splitting a run into adjacent durations preserves events.

No membrane histories are stored by default in the backend. For diagnostics,
`record_nodes=(...)` enables `get_state_trace()` for at most 32 local nodes.
The toy CLI records all three cells. `save_result()` persists any enabled recording
as `state_trace.parquet`, with `t_ms`, `local_index`, parent `node_index`, uint64
`body_id`, `v_mv`, and `g_mv`. The in-memory voltage arrays are shaped
`(recorded_nodes, timesteps)`. Samples are taken at the end of each timestep,
after reset, so spike peaks need not appear in the voltage trace.

## 7. Read and validate results

Every new output directory contains:

| File | Content |
| --- | --- |
| `spikes.parquet` | `t_ms`, parent `node_index`, original uint64 `body_id` |
| `rates.parquet` | all retained nodes, `body_id`, `spike_count`, whole-run `rate_hz` |
| `node_mapping.parquet` | local index → parent index → original body ID |
| `experiment.json` | seed, resolved parameters, stimuli, silencing, graph manifest/hash, selection, model assumptions, package versions |
| `summary.json` | counts, active unstimulated cells, NT coverage, artifact hashes |

Spike records are ordered by timestep, then local node index. Rate denominators
use the **entire experiment duration**, including baseline and recovery; these
are not stimulus-window rates. Subgraph `experiment.json` also reports activity
by selection hop. Use the parent node index/body ID to join spikes back to the
original `nodes.parquet` metadata.

```powershell
Get-Content outputs/neural/full_demo/summary.json
uv run python -m malecns_sim.neural.result outputs/neural/full_demo
```

The validator checks hashes, mapping bijection, spike/body correspondence,
timestep alignment, counts and rates. The state guard separately checks all
membrane and synaptic states for NaN/Inf on every timestep without retaining
their histories. Output is staged and only published after a successful run;
existing directories are never overwritten.

### Plot results

```powershell
uv run python -m malecns_sim.neural.visualize outputs/neural/toy_demo
```

This validates the result and writes `plots/spike_raster.png`,
`plots/firing_rates.png`, and, when recorded, `plots/membrane_potential.png`.
No GUI is required. Stimulus windows and driven neurons are labeled from the
saved experiment and local node mapping; silent neurons remain visible.
Rates use the full experiment duration. `--output PATH` selects a plot directory;
rerunning replaces the generated plots. `--max-nodes 32` limits plotted cells,
prioritizing driven cells then spike count (local index breaks ties). Validation
still reads the full result, so its memory cost scales with the saved data.

Old results without voltage recording still produce the raster and rate plots.
To obtain voltage traces, run the updated toy into a **new** directory:

```powershell
uv run python -m malecns_sim.neural.experiments.toy --seed 42 --output outputs/neural/toy_visualized
uv run python -m malecns_sim.neural.visualize outputs/neural/toy_visualized --toy-latency
```

`--toy-latency` adds a histogram and CSV for N0 → N1, N1 → N2, and N0 → N2.
Each source spike is paired with the strictly next target spike; target spikes
may be reused and sources without a later target are omitted. These descriptive
latencies are not causal estimates or measurements of the configured synaptic
delay: threshold integration and other inputs also affect spike timing.
Optional state traces are covered by artifact hashes and validated for node
mapping, finite values and complete timestep coverage. Older results remain valid.

## 8. Docker and reproducibility

```powershell
docker compose build
docker compose run --rm sim pytest tests/neural -q
docker compose run --rm sim python -m malecns_sim.neural.experiments.full_cns --duration-ms 100 --stimulate-body-id 11755 --rate-hz 100 --seed 42 --output outputs/neural/full_docker
uv run python -m malecns_sim.neural.result outputs/neural/full_demo --compare outputs/neural/full_docker
```

Use identical config, graph, stimuli, duration, seed and locked dependencies.
The comparison requires identical result bytes and experiment provenance,
including versions, rather than comparing only spike totals. The image contains
the neural configuration and tests; graph files and results stay in host volumes.
These commands do not launch the physical fly or a GUI. Running just the default
Compose service command still launches the existing v0.1 headless body, so use
the explicit neural commands above for this milestone.

The NumPy runtime and explicit RNG avoid platform-dependent random generators.
Cross-platform behavior is checked for this release's Windows/Docker environments;
identical floating-point results on every future dependency or CPU combination
are not assumed. Re-run the comparison after environment changes.

## 9. Acceptance observations

The local seed-42 experiment on the provisional published-v1 graph produced:

| Experiment | Nodes | Edges | Duration | Active neurons | Total spikes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bounded two-hop subgraph | 5,000 | 616,735 | 1,000 ms | 1,321 | 55,850 |
| Full CNS, body 11755, 100 Hz | 166,700 | 25,582,938 | 100 ms | 4,354 | 10,791 |
| Full CNS, no external input | 166,700 | 25,582,938 | 100 ms | 0 | 0 |

The stimulated full-CNS experiment also completed in Docker with **byte-identical
spikes, rates, node mapping and experiment provenance** compared with Windows.
All 25 neural tests passed in both environments. The graph and raw-data tests
also passed locally. The only test warnings came from Brian2's use of a
deprecated Pyparsing argument; they did not affect the results.

In the subgraph, the source, 170 first-hop cells and 1,150 second-hop cells fired.
In the full experiment, 4,353 active neurons were not externally stimulated.
These are measured results, not target counts enforced by the model. Results
can change under a different node/sign policy, stimulus, seed or model parameter.

Run the neural, graph and raw-data checks without launching body simulations:

```powershell
uv run pytest tests/neural tests/test_graph.py tests/test_cns.py -q
uv run ruff check src/malecns_sim/neural tests/neural
```

The `graph` marker skips real-data checks if no processed graph is available;
use `-m "not graph"` for only small fixtures. v0.4 can add a sensory/motor
catalogue while preserving the present graph/model/body separation.
