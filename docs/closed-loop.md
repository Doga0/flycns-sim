# v0.7 development: causal CNS ↔ body orchestration

This implementation joins the v0.5 MotorDecoder/BodyController and v0.6
SensoryEncoder under a stateful scheduler. No new biological transfer function,
motor mapping, artificial motor stimulation, learning, or locomotion controller
has been introduced.

## Acceptance status: physical feedback not yet demonstrated

The first full MaleCNS experiment uses 166,700 neurons, 25,582,938 connections,
23 explicitly mapped LF/ProLN chordotonal cells, seven mapped LF tibia motor
cells, a +0.20 rad initial tibia displacement, 200 ms duration and seed 42.
The perturbation changes initial qpos only; the actuator begins at its original
neutral target and no external trajectory is imposed after time zero.

Measured Windows results:

| Metric | Feedback off | Sensory clamp | Feedback on |
| --- | ---: | ---: | ---: |
| Generated sensory events | 0 | 8 | 8 |
| Injected sensory events | 0 | 0 | 8 |
| Total CNS spikes | 0 | 0 | 9 |
| Active mapped sensory cells | 0 | 0 | 7 |
| Spikes in mapped LF motors | 0 | 0 | 0 |
| Non-neutral applied commands | 0 | 0 | 0 |
| Maximum trajectory difference from feedback off | 0 | 0 | 0 rad |

All conditions completed without MuJoCo warnings or nonfinite states. Maximum
clock drift was approximately `5.85e-12 ms`. The default selective monitor kept
eight neural spikes; the global spike counter recorded nine. Unselected spikes
still propagate through the complete CNS normally.

The neural input works, but these inputs did not drive the mapped LF motor
population. The previously observed v0.6 motor activity was in body IDs
`800061` and `801173`; neither is in the v0.5 seven-cell LF tibia mapping.
That earlier observation alone did not establish a functional return path to
this actuator. General CNS activity must not be counted as successful physical
feedback. `comparison.json` records `acceptance_passed: false`, and
`validate --require-effect` exits with status 1.

A further graph audit found all 23 mapped sensory cells labeled acetylcholine,
so this population is not muted by the unknown-transmitter policy. There are
11 direct sensory-to-mapped-motor pairs. Summing contacts over the 23 sensory
sources gives `{807165: 39, 809912: 21, 818057: 1, 819384: 0, 909831: 4,
800636: 4, 815344: 5}`. This establishes some anatomical connectivity, not
sufficient drive to cross the LIF threshold in this experiment. Indirect paths
may also matter; the result is not evidence that no anatomical return path exists.

The package remains `0.7.0.dev0`. A later scientifically reviewed experiment
or mapping/encoder policy may establish this return path; this implementation
does not change weights, select a lucky seed, inject motor spikes into the
real-data run, or reinterpret other motor cells as LF tibia neurons to force
that result.

## Clocks and causal order

The versioned profile is `configs/closed_loop/lf_tibia_v1.yaml`:

| Clock | Interval |
| --- | ---: |
| Neural | 0.1 ms |
| Physics | 0.1 ms |
| Sensory / orchestration tick | 1 ms |
| Motor decoder | 10 ms |

For each interval `[t,t+1 ms)`:

1. Read the actual body position and velocity at `t`.
2. Encode that state, recording the body sample timestamp for every event.
3. Inject only mapped input events and advance the persistent CNS by 1 ms.
4. Advance MuJoCo by ten steps under the target already effective at `t`.
5. Assert that CNS, physics and scheduler clocks all equal `t+1 ms`.
6. At each motor boundary, decode only new motor spikes from the preceding
   10 ms window and apply the target for subsequent physics steps.

Thus spikes in `[0,10)` can affect physics starting at 10 ms, never earlier.
The scheduler uses integer tick counters and rejects incompatible clock ratios.
`timeline.parquet` records the target actually used during each interval, with
separate end-of-interval clock values. Motor rates reflect the latest decoder
update available at the end of that interval.

The backend uses a persistent input SpikeGeneratorGroup with fixed reviewed
input nodes and gains. `inject_events()` queues events for the next `advance()`;
past, unknown, duplicate and out-of-window events are rejected. No neural state
or delay queue is reset between windows. This follows Brian2's documented
[set_spikes between runs](https://brian2.readthedocs.io/en/2.10.0/user/input.html)
and MuJoCo's [control-before-step model](https://mujoco.readthedocs.io/en/latest/programming/simulation.html).

## Controls, monitors and safety

`feedback_off` omits sensory encoding/injection and motor application.
`sensory_clamp` computes and records sensory activity/events, but disables their
injection and motor application. `feedback_on` connects both bridges. All three
conditions begin with identical physical state hashes, seed, graph, LIF config
and bridge policies.

The default `control_and_sensory` monitor retains only mapped motor and sensory
spikes. A global counter still measures **all** CNS spikes for reports and storm
guards. `monitor: full` optionally retains all neural spike history for these
short diagnostic runs. Spike cursors return only the new records.

Engineering guards bound duration, joint offset/velocity, sensory rate, per-cell
motor rate and total spikes per sensory window. The unchanged motor bridge
continues to clamp command range and slew. These are simulation guards, not
physiological limits. A guard or clock failure ends with
`ABORTED_SAFETY_LIMIT`, a reason and partial artifacts. Invalid mapping/config
provenance fails before the run starts.

Preflight verifies both mapping artifacts against the same graph, catalogue
and freshly compiled body. It records all artifact hashes, the exact resolved
LIF config and expected hash, package versions, and a full compiled-body hash.
Each condition rechecks that body hash before applying controller bounds.

## Commands and outputs

The graph, `io-v1`, `bridge-v1` and `sensory-bridge-v1` must already exist. Run
from the repository root. Existing output directories are never overwritten.

```powershell
uv sync --locked
uv run python -m malecns_sim.bridge.closed_loop.runner --output outputs/closed_loop/lf_tibia_demo
uv run python -m malecns_sim.bridge.closed_loop.validate outputs/closed_loop/lf_tibia_demo

# Run just one condition when investigating a specific failure.
uv run python -m malecns_sim.bridge.closed_loop.runner --condition feedback_on --output outputs/closed_loop/lf_tibia_on

# Strict release acceptance gate, currently expected to return exit code 1.
uv run python -m malecns_sim.bridge.closed_loop.validate outputs/closed_loop/lf_tibia_demo --require-effect
```

The three-condition run writes `comparison.json` plus a directory for each
condition containing:

```text
body_states.parquet     sensory_activity.parquet    sensory_events.parquet
neural_spikes.parquet   motor_activity.parquet      motor_commands.parquet
timeline.parquet        provenance.json             report.json
frames/                 # optional snapshots; frames: 0 disables rendering
```

Neural spikes retain node/body IDs and role/type/side metadata. Sensory events
also identify source channel, body sample time and whether injection occurred.
Motor commands include source window start/end and whether they were applied.
Snapshots are rendered from copied completed states and cannot alter the
simulated trajectory. This runner does not open an interactive viewer.

```powershell
docker compose build
docker compose run --rm sim python -m malecns_sim.bridge.closed_loop.runner --output outputs/closed_loop/lf_tibia_docker
docker compose run --rm sim python -m malecns_sim.bridge.closed_loop.validate outputs/closed_loop/lf_tibia_docker
docker compose run --rm sim python -m malecns_sim.bridge.closed_loop.validate outputs/closed_loop/lf_tibia_demo --compare outputs/closed_loop/lf_tibia_docker
docker compose run --rm sim pytest tests/neural/test_incremental.py tests/bridge/test_closed_loop.py -q
```

## Regression and engineering validation

Tests compare chunked versus continuous Brian2 state/spikes, causal command
boundaries, selective versus global spike recording, clock errors, storm/rate/
velocity guards, partial-abort artifacts, and repeatability. Motor-only scheduler
tests reproduce v0.5 `decode_trace()` commands. Sensory-only scheduler tests
reproduce v0.6 `encode_states()` events for the same recorded body trajectory,
allowing only clock roundoff in timestamps.

A clearly labeled **synthetic three-neuron graph** on the real MuJoCo body
demonstrates motor feedback changing the trajectory, with clamp removing the
effect. This validates the software mechanism and does not substitute for the
currently failing real MaleCNS mapped-motor acceptance.

The cross-platform comparison requires identical sensory/neural events and
motor commands, with an absolute `1e-9` tolerance for continuous body values.
Matching Windows/Docker results establish reproducibility; they do not turn a
zero-effect result into a passed physical-feedback acceptance.

## v0.8.1: external-load physical feedback probe

The owner selected a **physical validation** experiment after the v0.7.x pathway
investigation. The v1 graph, Shiu LIF, 23-cell sensory mapping and encoder, seven-cell
motor mapping and decoder remain unchanged. The new
`configs/closed_loop/lf_tibia_load_v2.yaml` applies an engineering external torque
to the named LF tibia hinge: ramp 100–200 ms, hold 200–700 ms, release 700–800 ms,
and observe through 1000 ms. Torque is a function of simulation time only, applied
through MuJoCo `qfrc_applied` at every physics step, with the same profile in each
condition. It is not a muscle or environmental contact model.

`motor_clamp` is the fourth condition. It generates and injects sensory events,
runs the full CNS and decodes mapped motor spikes, but never applies the decoded
target to the body. This lets the experiment distinguish a missing neural motor
response from a failed mechanical effect. The four modes are:

| Mode | Encode | Inject | Decode | Apply |
| --- | --- | --- | --- | --- |
| `feedback_off` | no | no | neutral | no |
| `sensory_clamp` | yes | no | neutral | no |
| `motor_clamp` | yes | yes | yes | no |
| `feedback_on` | yes | yes | yes | yes |

Physics-only calibration, before running the full graph:

```powershell
uv run --frozen python -m malecns_sim.bridge.closed_loop.load_sweep `
  --amplitudes-nm 10 15 20 `
  --output outputs/closed_loop/my_new_load_sweep
```

On 2026-11-26, Windows physics-only sweeps measured mean hold offsets of
0.1547, 0.2385 and 0.3236 rad for 10, 15 and 20 N·m respectively, with no
MuJoCo warnings. The last corresponds to about 93 Hz **nominal** v1 encoder
rate; actual event rate is stochastic and measured in `motor_clamp`.
25 N·m crossed the 0.40 rad engineering guard and was rejected. The profile
uses 20 N·m based on this physical calibration, not a neuronal outcome.

```powershell
# Always choose an output name that does not already exist.
uv run --frozen python -m malecns_sim.bridge.closed_loop.runner `
  --config configs/closed_loop/lf_tibia_load_v2.yaml `
  --condition motor_clamp `
  --output outputs/closed_loop/my_new_motor_clamp

# All four modes for seed 42; --seed 43 through 46 repeat the experiment.
uv run --frozen python -m malecns_sim.bridge.closed_loop.runner `
  --config configs/closed_loop/lf_tibia_load_v2.yaml `
  --seed 42 `
  --output outputs/closed_loop/my_new_load_seed42

uv run --frozen python -m malecns_sim.bridge.closed_loop.validate `
  outputs/closed_loop/my_new_load_seed42 --require-effect

docker compose build
docker compose run --rm sim python -m malecns_sim.bridge.closed_loop.runner `
  --config configs/closed_loop/lf_tibia_load_v2.yaml `
  --seed 42 `
  --output outputs/closed_loop/my_new_docker_load_seed42
docker compose run --rm sim python -m malecns_sim.bridge.closed_loop.validate `
  outputs/closed_loop/my_new_load_seed42 `
  --compare outputs/closed_loop/my_new_docker_load_seed42
```

`comparison.json` separates the engineering gate from the provisional
`functional_target_0p01_rad_met`. The gate requires sensory input, mapped motor
spikes, a nonneutral applied command, a trajectory difference over `1e-6` rad
(well above the `1e-9` Windows/Docker comparison tolerance),
causal onset, and both clamps matching baseline. It reports maximum, RMS and
integrated absolute trajectory differences, and first sensory/motor/command/body
effect times. A single passing seed does not establish seed replication.

After all five new output directories exist, aggregate the declared seeds:

```powershell
uv run --frozen python -m malecns_sim.bridge.closed_loop.seed_report `
  outputs/closed_loop/my_new_load_seed42 `
  outputs/closed_loop/my_new_load_seed43 `
  outputs/closed_loop/my_new_load_seed44 `
  outputs/closed_loop/my_new_load_seed45 `
  outputs/closed_loop/my_new_load_seed46 `
  --output outputs/closed_loop/my_new_seed_report --require-effect
```

The aggregator refuses duplicate/missing seeds and graph, bridge, LIF, body or
non-seed configuration drift. Every declared seed must pass; it does not choose
the best response.

### First measured full-CNS result (seed 42, 2026-11-27)

`outputs/closed_loop/lf_tibia_load_v2_seed42_2026-11-26/` passed
`closed_loop.validate --require-effect` under the current validator. All four
conditions completed with zero MuJoCo warnings and finite state. In
`motor_clamp`, 1,228 sensory events were injected and five spikes occurred in
two mapped flexors; 68 nonneutral commands were decoded but none applied.
`feedback_on` had 1,207 injected sensory events, three mapped motor spikes and
51 nonneutral applied commands. The first sensory event was at 118 ms, first
mapped motor spike at 325.7 ms, first nonneutral command at 330 ms and first
physical trajectory difference over `1e-6` rad at 331 ms. The maximum
`|q_on - q_off|` was 0.04249 rad (RMS 0.00821 rad; integrated absolute
difference 0.00407 rad·s). `sensory_clamp` and `motor_clamp` matched
`feedback_off` physically. The changed body state first changed the encoder
rate at 331 ms, by up to 12.14 Hz versus `motor_clamp`. The sign of the peak
joint difference was negative, consistent with the v1 flexor decoder convention.

This is **one seed's engineering feedback validation** with a deliberately
large external load and a position-target decoder. It does not establish a
biologically correct reflex, locomotion, or robustness across seeds. The
seed 43–46 checks are recorded below. A full seed-42 Docker run at
`outputs/closed_loop/lf_tibia_load_v2_docker_seed42_2026-11-27/` also passed.
`closed_loop.validate <Windows run> --compare <Docker run>` returned
`reproduction_passed: true`: discrete sensory/neural events and motor commands
are exact; continuous body values agree within `1e-9`.

### Declared five-seed replication (2026-11-27)

`closed_loop.seed_report` revalidated all four conditions for seeds 42–46 and
wrote `outputs/closed_loop/lf_tibia_load_v2_seed_report_2026-11-27/report.json`.
`replication_passed: true`; the provisional 0.01 rad target was met in all five.

| Seed | Mapped motor spikes, feedback on | First motor / command / body (ms) | Peak `|q_on − q_off|` (rad) |
| ---: | ---: | --- | ---: |
| 42 | 3 | 325.7 / 330 / 331 | 0.04249 |
| 43 | 10 | 211.5 / 220 / 221 | 0.04830 |
| 44 | 7 | 265.8 / 270 / 271 | 0.04240 |
| 45 | 7 | 290.4 / 300 / 301 | 0.03462 |
| 46 | 8 | 297.3 / 300 / 301 | 0.04346 |

Every run had a 20 N·m external load, 1,000 ms duration, the same unchanged
v1 graph/LIF/encoder/decoder/mappings, and both clamps matching baseline.
This establishes a reproducible **engineering causal physical effect** under
the specified load. It does not establish biological reflex fidelity or gait.
