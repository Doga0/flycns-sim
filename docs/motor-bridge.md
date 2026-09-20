# v0.5: one-way motor bridge

The first body coupling is offline and causal:

```text
annotated leg proprioceptive cells → external Poisson drive → full MaleCNS LIF
  → saved motor spikes → population rate decoder → position targets → MuJoCo
```

Nothing in this release sends body state back to the CNS. The graph, neural
model, I/O catalogue and original body composition remain separate. `bridge/`
owns correspondence, decoding, safety and replay. The default profile drives
one tibia joint; it does not make a walking fly or validate biological behavior.

## What the annotations actually support

Run `uv run python -m malecns_sim.bridge.motor_catalogue`. The JSON audit
includes all distinct motor annotation values, missing counts, actual body
actuator names/axes/neutral targets, and enabled ranges. For the current 815
catalogued motor cells, nonempty coverage is:


| Field         | Nonempty rows |
| --------------- | --------------: |
| side          |           815 |
| subclass      |           815 |
| exitNerve     |           814 |
| somaNeuromere |           727 |
| type          |           805 |

Nonempty coverage does not imply unambiguous biological identity; the audit
also reports known `L`/`R` counts and actual values. No separate target/muscle
column exists in this released Feather. `muscle_target` therefore remains null.
The [MaleCNS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/) describes
hierarchical motor and innervation annotations, but this implementation uses
only the fields actually present in the local release.

`configs/bridge/lf_tibia_v1.yaml` explicitly lists:

- Flexor: 807165, 809912, 818057, 819384, 909831.
- Extensor: 800636, 815344.
- Required annotations: exact curated `Ti flexor MN` / `Ti extensor MN`, side
  `L`, subclass `fl`, exit nerve `ProLN`, soma neuromere `T1`.
- Actuator: `fly/lf_trochanterfemur-lf_tibia-pitch-position`.

All body IDs, roles and annotation evidence are validated on build and replay.
There is no substring parser, nearest-match fallback, or mapping of unknown
cells. Known unmapped motors are explicitly counted and ignored by the decoder.
An unknown motor, mismatched body/index, missing channel or unknown actuator is
an error. The compiled joint must match the YAML and be a direct hinge position
actuator.

The **population annotations are explicit evidence**; translating those
populations to a position target is a **manual mechanical hypothesis**. Each
channel records both. Positive pitch is assigned to extensor drive as an
engineering convention; physiological polarity, muscle moment arms, muscle
force and tendon mechanics have not been calibrated. The female-derived
NeuroMechFly body is also not a male-specific muscle reconstruction. This demo
establishes connectivity-driven movement, not a validated muscle mapping.

## Decoder, clocks and bounds

For each population, count spikes in `[start,end)` and divide by the total
mapped population size and window duration in seconds. Silent mapped cells
remain in the denominator. Default control windows are 10 ms.

```text
rate = spike_count / population_size / window_seconds
filtered += (1 - exp(-control_dt / tau)) * (rate - filtered)
activation = clip((filtered - r0) / (rmax - r0), 0, 1)
target = neutral + extensor_direction * gain * (extensor - flexor)
```

Defaults: `tau=20 ms`, `r0=0 Hz`, `rmax=100 Hz`, `gain=0.35 rad`, neutral
offset bounds `[-0.35,+0.35] rad`, maximum change `0.05 rad/control step`.
All are decoder parameters, not measured biological constants. The controller
checks finite values, actuator identity, range, slew, clock and command order.

The exponential filter coefficient is calculated once using the standard
library's correctly rounded Decimal exponential at 50-digit precision, then
converted to binary64. This avoids Windows/Linux libm differences in saved
filtered activity while preserving the specified exponential model.

The current model has neither enabled tibia control limits nor enabled tibia
joint limits. Its raw `[0,0]` range arrays therefore **do not mean a locked
joint**. The inventory reports absent limits as null. Explicit decoder bounds
are intersected with any enabled model ranges and installed as MuJoCo control
limits for selected actuators. This bounds the *target*, not a hard mechanical
joint stop. The existing FlyGym force limits are retained.

Neural dt is 0.1 ms; physics dt is read from MuJoCo (currently 0.1 ms); control
dt is 10 ms. A `[0,10)` spike window generates a command at 10 ms, never at
0 ms. A spike exactly at 10 ms belongs to `[10,20)`. Physics begins with neutral
controls and holds the last decoded command for one extra control interval:
100 ms neural data → 110 ms replay. Missing/partial windows are errors.

`MotorDecoder.reset()` and `decode(spikes, t_start_ms, t_end_ms)` operate on
parent graph node indices and original body IDs. `decode` returns one
`MotorCommand` per channel. `BodyController.apply(command, mj_data)` applies it
on the physics clock boundary. These interfaces can later consume live spike
windows without changing the decoder; no live CNS coupling is implemented here.

## Reproduce the single-joint demonstration

From the repository root, after building the graph and I/O catalogue:

```powershell
uv sync --locked
uv run python -m malecns_sim.bridge.build
uv run python -m malecns_sim.io.experiment --duration-ms 100 --rate-hz 100 --seed 42 --output outputs/io/bridge_sensory_seed42
uv run python -m malecns_sim.bridge.replay --neural-run outputs/io/bridge_sensory_seed42 --output outputs/bridge/lf_demo
uv run python -m malecns_sim.bridge.validate outputs/bridge/lf_demo
```

If a saved v0.4 sensory experiment exists, use its directory as `--neural-run`;
no new full-CNS run is necessary. Checksums, graph identity, all spike
body/index identities and sensory-only stimulus targets are checked before
decoding. The saved experiment, seed, clocks, source hashes and full mapping
are copied into `provenance.json`. PNG frames are written without launching an
interactive viewer; `--frames 0` skips rendering.

For a local interactive MuJoCo window, use:

```powershell
uv run python -m malecns_sim.bridge.replay --neural-run outputs/io/bridge_sensory_seed42 --output outputs/bridge/lf_viewer --viewer --frames 0 --playback-speed 0.1
```

After saving the verified experiment, this mode repeatedly runs the same motor
command trace through MuJoCo at 0.1× real-time speed. Space pauses/resumes;
mouse controls rotate, pan and zoom. Closing the window exits. The model and
data displayed by the native [MuJoCo passive viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)
are separate copies: GUI perturbations cannot change the replay physics or
saved results. Playback repeats the 110 ms experiment with a brief hold at its
end; it does not extend the neural experiment or generate a walking behavior.
Use this mode on the local desktop, not in the headless Docker container.

The measured seed-42, 100 Hz, 100 ms run stimulates 190 sensory cells. It has
193 active motor cells; two of the seven mapped LF cells contribute three
spikes. The maximum LF target offset is 0.08918 rad; maximum joint difference
from the neutral replay is **0.06011 rad (3.44 degrees)**. No MuJoCo warnings
or nonfinite state occurred. These numbers describe this specific experiment.

The neutral physics replay shares the exact model and initial keyframe and
holds all controls neutral. Both replays include gravity, ground contact and
mechanical coupling. `joint_displacement_rad` is motion from the initial
angle; `max_effect_vs_neutral_replay_rad` isolates the effect of changed
commands. Other legs can move mechanically without receiving neural commands.

## No-stimulus control and determinism

```powershell
uv run python -m malecns_sim.neural.experiments.spontaneous --duration-ms 100 --seed 42 --output outputs/neural/bridge_rest
uv run python -m malecns_sim.bridge.replay --neural-run outputs/neural/bridge_rest --output outputs/bridge/rest_demo --frames 0

# Repeat saved spike decoding in a new output directory
uv run python -m malecns_sim.bridge.replay --neural-run outputs/io/bridge_sensory_seed42 --output outputs/bridge/lf_repeat --frames 0
uv run python -m malecns_sim.bridge.validate outputs/bridge/lf_demo --compare outputs/bridge/lf_repeat

uv run pytest tests/bridge -q
```

The no-input LIF control must produce no spikes, neutral motor targets, and
zero joint difference from the neutral replay. Neural same-seed determinism
is tested in Brian2; bridge tests also run the same seeded sensory→motor toy
network twice and compare commands. `--compare` checks command artifact bytes;
it does not claim pixel-identical renderers or bitwise-identical physics across
platforms.

## Extend to six tibia channels

```powershell
uv run python -m malecns_sim.bridge.build --policy configs/bridge/six_leg_tibia_v1.yaml --output data/processed/malecns-v1.0/published-v1/bridge-six-leg-v1
uv run python -m malecns_sim.bridge.replay --mapping data/processed/malecns-v1.0/published-v1/bridge-six-leg-v1 --neural-run outputs/io/bridge_sensory_seed42 --output outputs/bridge/six_leg_demo
```

This second explicit profile includes 49 cells, six pairs of antagonist
populations and six tibia actuators (LF, LM, LH, RF, RM, RH). It does not bind all
42 actuators. The same measured neural run activates five mapped cells and
drives LF, LM and RF only. The other three channels remain neutral; no activity
is fabricated to make all legs move. Wing, haltere, neck and abdomen are unmapped.

## Docker

```powershell
docker compose build
# If the mapping was already built on the host, reuse the mounted artifact.
# Otherwise build it inside the container once:
docker compose run --rm sim python -m malecns_sim.bridge.build
docker compose run --rm sim python -m malecns_sim.bridge.replay --neural-run outputs/io/bridge_sensory_seed42 --output outputs/bridge/lf_docker
docker compose run --rm sim python -m malecns_sim.bridge.validate outputs/bridge/lf_demo --compare outputs/bridge/lf_docker
docker compose run --rm sim pytest tests/bridge -q
```

The same mounted `data/` and `outputs/` are used; no CNS data enters the image.
Rendering uses the existing CPU OSMesa configuration. Do not set `MUJOCO_GL=osmesa`
in a native Windows terminal; it is a container setting.

## Artifacts and acceptance scope

`bridge-v1/` contains `motor_mapping.parquet`, `motor_audit.json`, `manifest.json`.
The manifest contains the complete reviewed YAML policy, resolved model
channels, catalogue/graph hashes, mapped/unmapped counts and artifact hashes.

Each replay produces:


| Artifact                  | Meaning                                                                      |
| --------------------------- | ------------------------------------------------------------------------------ |
| `neural_activity.parquet` | All source spikes with body/index, role, type, side                          |
| `motor_activity.parquet`  | Motor spikes with explicit`is_mapped` flag                                   |
| `motor_commands.parquet`  | Causal command times, targets, population rates and filtered rates           |
| `joint_states.parquet`    | Joint targets/positions and matched neutral replay                           |
| `provenance.json`         | Input experiment, mapping, assumptions and hashes                            |
| `report.json`             | Coverage, active/unmapped cells, command/motion metrics, warnings and hashes |
| `frames/`                 | Optional real MuJoCo PNG renders                                             |

Tests cover rate normalization, antagonism, low-pass decay, causal boundaries,
reset, deterministic LIF→decoder output, empty input, saturation, slew limits,
identity/provenance rejection, unknown actuators/cells, missing commands,
nonfinite values and real joint displacement. The next milestone is a separate
body→CNS sensory bridge; closed-loop operation follows afterward.
