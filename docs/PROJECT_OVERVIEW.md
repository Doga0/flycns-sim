# Project overview

## Purpose

`flycns-sim` builds a computational model of the male *Drosophila melanogaster* central
nervous system from the **MaleCNS v1.0** connectome and couples it to a physics-based fly
body (**FlyGym / NeuroMechFly on MuJoCo**). The long-term research question, stated by the
project owner, is how far behavior that interacts with the environment can be reproduced
when a biological wiring diagram and its dynamics are combined with sensory feedback and
body mechanics. Movement should emerge from neural dynamics rather than from scripted
animation or an external controller.

Owner-stated goals (from the project description, not all implemented):
connectome-based neural simulation; brain–body integration (motor neurons drive joints);
closed sensorimotor loop; behavior analysis (walking, orientation, obstacle avoidance,
stimulus responses); later, plasticity and reinforcement learning; interactive
visualization (Three.js is mentioned). Locomotion, RL, GPU backends and Three.js are
explicitly outside the work done so far (README "Version boundary").

## Core requirements evident in the code

- Raw released tables are read **unchanged**; every scientific choice (node policy,
  transmitter signs, I/O roles, sensory/motor mappings) is an explicit, versioned YAML
  profile with provenance hashes.
- Artifacts are deterministic, hash-checked, and never overwritten.
- Each coupling direction is validated separately before being combined.
- Results must be reproducible on Windows (uv) and Linux (Docker), with exact discrete
  events.
- Acceptance gates are explicit and may fail honestly (the current v0.7 gate fails).

## Technology stack

| Area | Technology | Pinned in |
| --- | --- | --- |
| Language / env | Python 3.12 (`>=3.12,<3.13`), uv, hatchling | `pyproject.toml`, `uv.lock`, `.python-version` |
| Body / physics | FlyGym 2.1.0 (git tag), MuJoCo 3.9 | `pyproject.toml` |
| Neural dynamics | Brian2 2.10 (NumPy code objects, no C++ codegen) | `neural/backends/brian2_lif.py` |
| Data | pandas, pyarrow (Feather/Parquet), NumPy memmaps | `cns/`, `io/` |
| Config | PyYAML profiles under `configs/` | |
| Plotting | matplotlib | `neural/visualize.py`, `cns/graph/query.py` |
| Quality | pytest, ruff (E, F, I, UP; line length 100) | `pyproject.toml` |
| Container | Docker + Compose, OSMesa headless rendering | `Dockerfile`, `docker-compose.yml` |

The data source is the public MaleCNS v1.0 distribution (three Feather files, about
1.1 GB, CC-BY). No credentials or private services are used.

## Major components

| Package | Role | Guide |
| --- | --- | --- |
| `malecns_sim.cns` | Download/inspect raw tables, census, sparse graph build (COO/CSR) | [graph-core.md](graph-core.md) |
| `malecns_sim.neural` | Shiu-derived LIF model on Brian2, transmitter signs, stimuli, results | [neural-reference.md](neural-reference.md) |
| `malecns_sim.io` | Functional role catalogue (sensory/motor/…) and anatomy-aware experiments | [io-catalogue.md](io-catalogue.md) |
| `malecns_sim.simulation` | Standalone NeuroMechFly body on flat ground, smoke test, channel inventory | README §2–3 |
| `malecns_sim.bridge` | Motor decoder (CNS→body), sensory encoder (body→CNS) | [motor-bridge.md](motor-bridge.md), [sensory-bridge.md](sensory-bridge.md) |
| `malecns_sim.bridge.closed_loop` | Causal scheduler joining both bridges (v0.7, in progress) | [closed-loop.md](closed-loop.md) |

See [ARCHITECTURE.md](ARCHITECTURE.md) for module-level detail.

## Version history (from git and README)

| Version | Adds |
| --- | --- |
| v0.1 | Raw data access, independent physical body, smoke test |
| v0.2 | Node policies, census audit, sparse graph |
| v0.3 | Brian2 LIF engine, transmitter policy, stimuli, results |
| v0.4 | I/O catalogue, exact queries, body channel inventory |
| v0.5 | One-way motor bridge: saved spikes → decoder → MuJoCo replay |
| v0.6 | One-way sensory bridge: body state → encoder → full-CNS LIF replay (last commit, `0.6.0`) |
| v0.7.0.dev0 | Closed-loop scheduler (uncommitted; acceptance not met) |

## Terminology

- **MaleCNS v1.0**: the male fly CNS connectome release (brain + ventral nerve cord).
- **Body ID**: a MaleCNS neuron identifier. **Node index**: its position in the built graph
  (body IDs sorted ascending).
- **published-v1**: the default node policy; 166,700 nodes vs. 166,691 in the publication
  (the +9 difference is unresolved and documented).
- **Contacts**: anatomical synapse counts per directed edge; **weight** = contacts ×
  0.275 mV × transmitter sign (neural layer only).
- **Unknown-NT policy**: how neurons without a mapped transmitter are signed (`zero` by
  default, which mutes their output).
- **I/O catalogue (`io-v1`)**: per-node primary role derived from the released `superclass`.
- **LF tibia**: left-front leg, femur–tibia pitch joint (`fly/lf_trochanterfemur-lf_tibia-pitch`),
  the single joint used by the v0.5–v0.7 bridges.
- **ProLN**: prothoracic leg nerve; **FeCO / chordotonal organ**: the leg proprioceptors used
  as sensory inputs.
- **feedback_on / feedback_off / sensory_clamp**: the three matched closed-loop conditions.
- **Acceptance gate**: a validator flag (`--require-effect`) that must pass before a milestone
  is declared complete.
