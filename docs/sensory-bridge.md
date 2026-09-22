# v0.6: one-way body → CNS sensory bridge

v0.6 is an offline, one-way sensory validation. It does not apply the LIF
output to FlyGym. The path is:

```text
controlled LF tibia trajectory → sampled body state → sensory encoder
  → explicit MaleCNS sensory events → full-CNS LIF → recorded activity
```

This complements v0.5's separately validated motor direction. It is not a
closed-loop simulation; that remains v0.7.

## Explicit anatomy, separate model assumptions

Run:

```powershell
uv run python -m malecns_sim.bridge.sensory_catalogue
uv run python -m malecns_sim.bridge.sensory.build
```

The current release contains 508 left `ProLN` sensory cells, including 45
`mechanosensory_proprioceptive` cells. Within this local release, 23 are
explicitly annotated as `chordotonal organ`, one as `hair plate`, and two as
`campaniform sensilla`. `somaNeuromere` is absent for this population.

`configs/bridge/sensory_lf_tibia_v1.yaml` maps only the 23 explicit chordotonal
IDs selected by the exact released fields:

```text
class      = mechanosensory_proprioceptive
subclass   = chordotonal organ
side       = L
entryNerve = ProLN
```

This supports an LF/ProLN chordotonal **population** selection. It does not
support classifying these particular cells as claw, hook or club: those labels
are not present in the local Feather, and `SNpp56` is not a member of this LF
population. The [proprioception literature](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/)
shows why those distinctions matter: claw cells encode position, hook cells
directional movement, and club cells movement/vibration.

The mapping therefore labels its transfer function
`linear_position_encoder_v1`, not a FeCO model. It maps the compiled neutral
LF tibia angle (1.309 rad) to 0 Hz and neutral + 0.35 rad to 100 Hz. Negative
direction positions clamp to 0 Hz. This one-sided signal is an engineering
probe with a zero baseline, not a measured tuning curve. The generic encoder
also supports `directional_velocity_encoder_v1`, but v0.6 deliberately has no
velocity mapping because no released annotation identifies a reviewed LF hook
population. Hair-plate and campaniform mappings are likewise deferred rather
than guessed.

## Run the demonstrations

```powershell
uv sync --locked

# One-time, immutable mapping artifact
uv run python -m malecns_sim.bridge.sensory.build

# Deterministic encoding acceptance: state remains neutral and static.
uv run python -m malecns_sim.bridge.sensory.replay `
    --output outputs/sensory_bridge/lf_tibia_zero `
    --zero-motion --mode explicit_spikes --seed 42

# Real demo: MuJoCo drives the body trajectory; per-neuron Poisson events use the seed.
uv run python -m malecns_sim.bridge.sensory.replay `
    --output outputs/sensory_bridge/lf_tibia_poisson `
    --mode poisson --seed 42

# Deterministic, rate-coded alternative useful for debugging exact event timing.
uv run python -m malecns_sim.bridge.sensory.replay `
    --output outputs/sensory_bridge/lf_tibia_explicit `
    --mode explicit_spikes --seed 42

uv run python -m malecns_sim.bridge.sensory.validate outputs/sensory_bridge/lf_tibia_explicit
```

The controlled trajectory is 800 ms: neutral (0–100), negative target direction
(100–300), neutral (300–500), positive target direction (500–700), then neutral
(700–800). “Negative” and “positive” refer to the MuJoCo hinge axis only; they
are not an uncalibrated flexion/extension claim. Body state is sampled every
1 ms, physics advances at 0.1 ms, and LIF advances at 0.1 ms. The encoder uses
the state at the start of `[t,t+1 ms)` and emits its event at `t`; no future
body state is used.

`explicit_spikes` shares deterministic rate-code events across the reviewed
population. `poisson` samples each mapped cell independently using PCG64 and a
saved seed. Both modes create `Stimulus(type="spikes")` records through the
existing backend API; the encoder never imports Brian2.

## Results and scope checks

Each run writes:

| Artifact | Contents |
| --- | --- |
| `body_states.parquet` | LF joint position, velocity and position target |
| `sensory_activity.parquet` | body value, normalized value and rate each 1 ms |
| `sensory_spikes.parquet` | input events with original node/body IDs and encoder channel |
| `neural_spikes.parquet` | LIF spikes mapped back to role, type and side |
| `provenance.json` | mapping, graph, clocks, encoder, LIF config and seed |
| `report.json` | coverage, activity counts, finite state and artifact hashes |

The zero-motion control uses repeated neutral `BodyState` samples without
advancing physics. It must yield zero sensory inputs and zero downstream LIF
spikes. It tests the sensory boundary itself, independent of gravity/contact.
The moving demo advances the real MuJoCo model and stores actual `q` and
`qdot`; it has no contact/load channel in v0.6.

`report.json` always records `motor_body_coupling: false`. Neural motor spikes
may be observed in `neural_spikes.parquet`, but they are not decoded or written
to MuJoCo controls. There is no sensory feedback to the motor replay, no online
scheduler, no locomotion claim and no live viewer for this one-way experiment.

## Docker

```powershell
docker compose build
docker compose run --rm sim python -m malecns_sim.bridge.sensory.build
docker compose run --rm sim python -m malecns_sim.bridge.sensory.replay --output outputs/sensory_bridge/lf_tibia_docker --mode explicit_spikes --seed 42
docker compose run --rm sim python -m malecns_sim.bridge.sensory.validate outputs/sensory_bridge/lf_tibia_explicit --compare outputs/sensory_bridge/lf_tibia_docker
```

The Docker image receives no CNS data. It uses mounted `data/` and `outputs/`.
The comparison checks sensory activity within an absolute `1e-10` floating-point
tolerance and requires exact input and neural spike tables. It intentionally
does not require byte-identical body-state arrays across operating systems.
