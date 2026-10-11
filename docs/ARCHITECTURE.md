# Architecture

Package: `src/malecns_sim/` (about 9,100 lines of Python including tests). Every runnable
tool is a module with a `main()` invoked as `python -m malecns_sim.<module>`; there is no
console-script entry point. Paths below are relative to `src/malecns_sim/` unless noted.

## Layering

```text
            raw Feather tables (data/malecns-v1.0/, unchanged)
                               │
  cns/ ── census, node policy ─┴─> graph build ──> data/processed/malecns-v1.0/published-v1/
                                                     (COO + CSR .npy memmaps, nodes.parquet)
                 ┌───────────────────────────────────────┼───────────────────────┐
                 v                                       v                       │
  neural/  NeuralGraph + signs + Brian2 LIF        io/  role catalogue (io-v1)   │
                 │                                       │                       │
                 │          bridge/ (motor mapping, sensory mapping) <───────────┤
                 │                    │                                          │
                 └──────> bridge/closed_loop/ scheduler <──── simulation/ (FlyGym/MuJoCo body)
```

Dependency rules (verified from imports): `cns` depends on nothing internal; `neural`
depends on `cns.graph.store`; `io` depends on `cns`; `simulation` imports nothing from
`cns` except `cns.graph.store.write_json` in `body_io.py`; `bridge` depends on all of them.
The body is never driven by anything except `bridge/controller.py`.

## Layer by layer

### `cns/`: raw data and anatomical graph (v0.1–v0.2)

| File | Responsibility |
| --- | --- |
| `cns/download.py` | Download the three MaleCNS v1.0 Feather files from GCS with size + MD5 checks, atomic `.part` rename |
| `cns/dataset.py` | `CNSDataset.load()`: read tables unchanged into pandas |
| `cns/inspect.py`, `cns/census.py` | Print schemas/labels; audit membership under each node policy |
| `cns/graph/policy.py` | Declarative node selection from `configs/graph/{published,all-annotated,traced}.yaml` (no Python expressions) |
| `cns/graph/build.py` | Stream connections in batches, external sort/merge into unique directed edges; write COO (`src/dst/contacts.npy`), outgoing CSR, `body_ids.npy`, `nodes.parquet`, `manifest.json`, `audit.json` |
| `cns/graph/store.py` | `CNSGraph.load()`: read-only memory-mapped access; `file_hashes`, `write_json` helpers used everywhere |
| `cns/graph/validate.py` | Structural, contact-conservation and hash validation |
| `cns/graph/query.py` | Neighborhood inspection and plotting CLI |

Node index = rank of body ID (ascending). Edges are sorted `(src, dst)`; self edges kept
(101 in published-v1).

### `neural/`: LIF dynamics (v0.3, extended in v0.7)

| File | Responsibility |
| --- | --- |
| `neural/config.py` | `LIFConfig` (frozen dataclass, validated) loaded from `configs/neural/shiu_lif_v1.yaml` |
| `neural/signs.py` | Transmitter → ±1 sign; unknown labels → 0 by default (`unknown_nt_policy`) |
| `neural/backends/base.py` | `NeuralGraph` (backend-neutral arrays + provenance) and `NeuralBackend` ABC |
| `neural/backends/brian2_lif.py` | `Brian2LIFBackend`, the only backend |
| `neural/stimulus.py` | `Stimulus` schedules; deterministic per-step Poisson events |
| `neural/result.py` | Save/validate sparse spikes, rates, node mapping, provenance |
| `neural/experiments/*` | CLIs: `toy` (3 cells, no data), `subgraph`, `full_cns`, `spontaneous`, `stimulate` |
| `neural/visualize.py` | Raster/rate/trace plots |

Model (per neuron): `dv/dt = (v_rest − v + g)/τ_m`, `dg/dt = −g/τ_s`, threshold −45 mV,
reset/rest −52 mV, refractory 2.2 ms, delay 1.8 ms, dt 0.1 ms. Recurrent synapses add
`w = contacts × sign × 0.275 mV` to `g`. External inputs (stimuli and closed-loop events)
add a fixed gain directly to `v` (68.75 mV by default, enough to force a spike from rest).
Everything uses Brian2 `NumpyCodeObject` (no compiled code generation).

Backend contract (`NeuralBackend`): `reset`, `schedule`, `stimulate`, `silence`, `run`,
`get_spikes` (abstract, v0.3), plus the v0.7 incremental API, which is **uncommitted**:
`time_ms`, `total_spikes`, `configure_event_input(nodes, gains_mv)`,
`inject_events(events)`, `advance(duration_ms)`, `spikes_since(cursor)`. The Brian2
implementation keeps one persistent `SpikeGeneratorGroup` for event input, refuses past,
unknown, duplicate or out-of-window events, and supports a selective spike monitor
(`spike_record_nodes`) while still counting all spikes.

### `io/`: functional role catalogue (v0.4)

`io/selectors.py` applies exact-value rules from `configs/io/malecns_io_v1.yaml` to the
released `superclass` (and preserved metadata) to give each node one primary role
(`io/roles.py:NeuronRole`). `io/build.py` writes `published-v1/io-v1/neurons.parquet`.
`io/catalogue.py:IOCatalog` gives exact-match queries (`sensory()`, `motor()`, `by_type()`
…). `io/experiment.py` stimulates an annotated population in the full LIF graph and
classifies activity by role. Role counts in io-v1: sensory 17,937, motor 815, descending
1,316, ascending 1,846, interneuron 144,494, endocrine 94, other 198.

### `simulation/`: body (v0.1)

`simulation/fly.py:build_fly()` composes FlyGym's NeuroMechFly (all biological joints,
position actuators on active leg DOFs with `kp=50`, neutral pose keyframe) on
`FlatGroundWorld` and returns a `FlySimulation(model, data, camera)`. MuJoCo timestep is
0.1 ms. `smoke_test.py` runs it headless (PNG frames + report) or in a viewer and provides
`check_state()` (finite state, no warnings), which the bridges reuse. `body_io.py`
inventories actuators/joints/sensors.

### `bridge/`: one-way bridges (v0.5, v0.6)

Motor direction (CNS → body):

| File | Responsibility |
| --- | --- |
| `bridge/motor_catalogue.py` | Audit motor annotations and the compiled body's actuators |
| `bridge/mapping.py` | `resolve_mapping()`: check explicit body IDs in `configs/bridge/lf_tibia_v1.yaml` (or `six_leg_tibia_v1.yaml`) against catalogue annotations and real actuators → `MotorMapping` |
| `bridge/build.py` | Write `published-v1/bridge-v1/` (or `bridge-six-leg-v1/`) |
| `bridge/decoder.py` | `MotorDecoder.decode(spikes, start, end)`: per-window flexor/extensor population rates → first-order low-pass (τ 20 ms, exact `Decimal` alpha for cross-platform equality) → normalized antagonistic activity → position target, clamped to range and slew |
| `bridge/controller.py` | `BodyController.apply()`: writes `data.ctrl` only on the exact clock boundary; sets engineering `ctrlrange` |
| `bridge/replay.py` | Offline: saved neural run → `decode_trace` → `replay_physics` (+ matched neutral replay) → artifacts |
| `bridge/viewer.py` | Paced local MuJoCo viewer of a replay |
| `bridge/validate.py` | Mapping provenance and replay artifact checks |
| `bridge/store.py` | `stage_output()`: atomic staging directory, refuses existing outputs |

Sensory direction (body → CNS), `bridge/sensory/`:

| File | Responsibility |
| --- | --- |
| `sensory/body_state.py` | `JointStateReader` → `BodyState(t_ms, joint q, qdot, actuator target)`; hides MuJoCo addresses |
| `sensory/catalogue.py` | Audit sensory annotations (`bridge/sensory_catalogue.py` is a compatibility alias) |
| `sensory/mapping.py`, `sensory/build.py` | Resolve `configs/bridge/sensory_lf_tibia_v1.yaml` (23 LF/ProLN chordotonal cells) → `sensory-bridge-v1/` |
| `sensory/encoder.py` | `SensoryEncoder.encode(state, start, end)`: linear position (or directional velocity) → rate → events (`explicit_spikes` phase accumulator or seeded per-node `poisson`) |
| `sensory/replay.py` | Offline: drive the body through a keyframed trajectory, encode, run full-CNS LIF |
| `sensory/validate.py` | Mapping/run validation, Windows/Docker comparison |

### `bridge/closed_loop/`: causal orchestration (v0.7, uncommitted)

| File | Responsibility |
| --- | --- |
| `closed_loop/clocks.py` | `Clocks` with integer-ratio checks; `check()` asserts scheduler/neural/body times agree within 1e-7 ms |
| `closed_loop/scheduler.py` | `MuJoCoBody` (read/advance/apply plus joint offset and velocity guards) and `ClosedLoopScheduler.run()` |
| `closed_loop/runner.py` | `prepare()` (config, LIF hash, catalogue, both mappings, clock consistency, body fingerprint) and `run_experiment()` over conditions |
| `closed_loop/recorder.py` | `save_run()`: parquet tables, provenance, report, optional frames, partial artifacts on abort |
| `closed_loop/validate.py` | `validate_run`, `compare_conditions` (acceptance), `compare_reproductions` (Windows vs Docker) |

Config: `configs/closed_loop/lf_tibia_v1.yaml` (200 ms, seed 42, Poisson encoder,
`control_and_sensory` monitor, +0.20 rad initial tibia offset, safety limits).

## Closed-loop data flow (one 1 ms tick)

From `scheduler.py:ClosedLoopScheduler.run`:

1. `body.read()` gives the state at `t` (guards: joint offset, velocity).
2. Unless the condition is `feedback_off`: `encoder.encode(state, t, t+1)` gives events,
   stamped with `body_sample_t_ms`; they are injected only in `feedback_on`.
3. `neural.advance(1 ms)` (10 Brian2 steps); `spikes_since(cursor)` returns the new
   recorded spikes; `total_spikes` feeds the storm guard.
4. Mapped motor spikes accumulate.
5. `body.advance(1 ms)` (10 MuJoCo steps) under the target already in effect.
6. `clocks.check(t+1)`.
7. Every 10 ms: `decoder.decode(window)`, and `body.apply(command)` only in `feedback_on`
   (and `motor_only`), so spikes in `[t−10, t)` affect physics from `t` onward.

Any `SafetyLimit`, `FloatingPointError` or `RuntimeError` sets
`status = ABORTED_SAFETY_LIMIT` and keeps partial records.

Test-only modes `motor_only` (replay recorded motor spikes) and `sensory_only` (driven
body, no motor application) check equivalence with the v0.5/v0.6 offline paths.

## Artifacts and storage

All under git-ignored `data/` and `outputs/`:

- `data/malecns-v1.0/*.feather`: raw tables.
- `data/processed/malecns-v1.0/published-v1/`: graph, plus `io-v1/`, `bridge-v1/`,
  `bridge-six-leg-v1/`, `sensory-bridge-v1/`. `*-docker` siblings are Docker rebuilds
  used for comparisons.
- `outputs/{neural,io,bridge,sensory_bridge,closed_loop,smoke_test}/<run>/`: experiment
  results; each has `provenance.json` and `report.json` or `summary.json` with artifact
  hashes.

Wheel packaging copies `configs/*` into the package as `…/profiles/` fallbacks
(`pyproject.toml` `force-include`); `load_config()` prefers the repo copy.

## External dependencies and environment

- No environment variables are required locally. Docker sets `MUJOCO_GL=osmesa` and
  `PYOPENGL_PLATFORM=osmesa`; do not set these for the Windows viewer.
- Network: only `cns.download` (public GCS bucket) and package installation.
- FlyGym is installed from a git tag (`v2.1.0`), so installs need GitHub access.
- Docker image: `uv` builder stage → `python:3.12.11-slim-bookworm` with OSMesa; mounts
  `data/` and `outputs/`; default command is the headless smoke test.
