# v0.4: MaleCNS I/O Catalogue

v0.4 adds functional interface annotations without changing anatomical graph
edges, LIF equations, or the physical fly. Its boundary is deliberate:

```text
MaleCNS annotations -> I/O catalogue -> node/body IDs -> LIF activity reports
                                                X
                                      FlyGym sensors/actuators
```

The `X` remains empty. A label such as `Ti flexor MN` is never converted into a
FlyGym joint command in this release.

## Build and inspect

```powershell
uv sync --locked
uv run python -m malecns_sim.io.audit
uv run python -m malecns_sim.io.build
uv run python -m malecns_sim.io.validate data/processed/malecns-v1.0/published-v1/io-v1
uv run python -m malecns_sim.io.inspect
```

`io.audit` reads the released annotation Feather file and writes
`outputs/io-annotation-audit.json`. It reports all 36 column names, types,
null counts, and complete distinct-value counts for classification fields. It
does not infer roles. The official download page describes this file as curated
neuron annotations containing classes, types, sides, and related properties:
[MaleCNS downloads](https://male-cns.janelia.org/download/).

`io.build` reads the existing graph's `nodes.parquet`, so the catalogue has the
same membership and contiguous indices as the neural engine. Its default output
is:

```text
data/processed/malecns-v1.0/published-v1/io-v1/
├── neurons.parquet
├── role_counts.json
└── manifest.json
```

The build is deterministic, staged before publication, and refuses to overwrite
an existing directory. `manifest.json` includes graph/node checksums, the entire
role policy, normalization descriptions, output checksums, counts, and package
version. Validation recomputes derived fields and confirms every raw metadata
value against the graph.

## Role policy

`configs/io/malecns_io_v1.yaml` maps **exact** released `superclass` values. It
does not use substring or regular-expression guesses. Every row keeps:

- `role`, the exclusive normalized primary role;
- `role_source_field=superclass`;
- `role_source_value`, the unmodified released value;
- `role_policy_status`, either `mapped` or `unmapped`.

The seven normalized roles are `sensory`, `motor`, `descending`, `ascending`,
`interneuron`, `endocrine`, and `other`. Combined released categories retain
their exact meaning in `role_source_value`. For example, `sensory_ascending` is
assigned the primary interface role `sensory`; it is not silently relabelled as
the curated `ascending_neuron` superclass. `efferent_ascending` and
`efferent_descending` remain `other`, rather than being conflated with motor,
AN, or DN populations. `*_tbc` values are explicitly listed, so their inclusion
is visible and versioned.

The real published-v1 graph produced:

| Primary role | Neurons |
| --- | ---: |
| sensory | 17,937 |
| motor | 815 |
| descending | 1,316 |
| ascending | 1,846 |
| interneuron | 144,494 |
| endocrine | 94 |
| other | 198 |
| **total** | **166,700** |

All 166,700 rows matched an explicit rule. This does not resolve the graph
policy's +9 difference from the published 166,691-neuron census; the catalogue
inherits graph membership and records its graph manifest checksum.

## Preserved and normalized fields

Every graph metadata column remains in `neurons.parquet`. Normalized convenience
fields sit beside the raw values and record their source:

| Normalized field | Rule | Provenance fields |
| --- | --- | --- |
| `side` | first assigned value from `rootSide`, then `somaSide` | `side_source_field`, `side_source_value` |
| `nerve` | sensory `entryNerve`; motor `exitNerve`; otherwise null | `nerve_source_field`, `nerve_source_value` |
| `body_region` | exact `somaNeuromere` value | `body_region_source_field`, `body_region_source_value` |
| `sensory_system` | exact class mapping only for primary sensory rows | `sensory_system_source_field`, `sensory_system_source_value` |

`side` is a query convenience, not a claim that root and soma laterality are
biologically interchangeable. The provenance columns let a later mapping policy
choose the appropriate raw field. Motor `subclass`, `type`, `somaNeuromere`, and
`exitNerve` remain available. No parser guesses a leg, muscle, or joint from a
type string. For example, `Ti flexor MN` can be queried, but is not assigned to
any FlyGym tibia actuator.

Sensory systems are normalized only from observed class values such as `visual`,
`olfactory`, `gustatory`, `hygrosensory`, `thermosensory`, and the released
mechanosensory classes. More detailed anatomy stays in raw `class`, `subclass`,
`type`, `entryNerve`, and `receptorType` fields. MaleCNS does not provide a joint
angle-to-firing-rate encoder. v0.6 supplies and tests one explicitly labeled
LF/ProLN chordotonal engineering encoder; see [the sensory bridge guide](sensory-bridge.md).

## Query API and CLI

```powershell
uv run python -m malecns_sim.io.query --role motor --side L
uv run python -m malecns_sim.io.query --role motor --type "Ti flexor MN"
uv run python -m malecns_sim.io.query --role sensory --class mechanosensory_proprioceptive --subclass leg
uv run python -m malecns_sim.io.query --role descending --type DNa01
uv run python -m malecns_sim.io.query --role ascending --limit 10
```

Queries use exact, case-sensitive values and print match counts plus node/body
IDs and relevant metadata. `--limit` limits display only. Use `--output` to save
the displayed JSON report.

```python
from malecns_sim.io import IOCatalog

catalogue = IOCatalog.load(
    "data/processed/malecns-v1.0/published-v1/io-v1"
)

left_motor = catalogue.motor(side="L")
dna01 = catalogue.by_type("DNa01")
leg_proprio = catalogue.sensory(
    **{"class": "mechanosensory_proprioceptive", "subclass": "leg"}
)

engine.stimulate(leg_proprio.node_indices, rate_hz=100)
original_ids = leg_proprio.body_ids
```

`sensory()`, `motor()`, `descending()`, `ascending()`, `by_type()`, `by_side()`,
and `filter()` return `IOQueryResult`. Its `node_indices` feed the v0.3 backend
directly; `body_ids` preserve MaleCNS identity. The example's `engine` assumes an
already constructed full-graph backend. Catalogue queries do not run dynamics.

The official MaleCNS site demonstrates type-based neuPrint queries such as
`DNge104`, consistent with keeping curated cell type separate from normalized
role: [programmatic access](https://male-cns.janelia.org/download/).

## Anatomy-aware neural experiment

```powershell
uv run python -m malecns_sim.io.experiment `
    --class mechanosensory_proprioceptive `
    --subclass leg `
    --duration-ms 100 `
    --rate-hz 100 `
    --seed 42 `
    --output outputs/io/leg_proprio_demo
```

The defaults select 190 real rows with primary role `sensory`, released
`class=mechanosensory_proprioceptive`, and `subclass=leg`. All are stimulated
from 0–100 ms using the v0.3 external-drive model. This is a population-level
optogenetic-style input, not an encoding of actual joint position or load.

In addition to standard v0.3 files, the experiment writes:

| File | Content |
| --- | --- |
| `active_neurons.parquet` | every active node with body ID, role, source superclass, class, subclass, type, side, nerve, body region, sensory system, spike count/rate, and stimulated flag |
| `role_activity.parquet` | catalogue size, stimulated cells, active cells, and spike count for every normalized role |

Both files are hashed in `summary.json`. The command validates graph and
catalogue provenance before simulation, then validates active-neuron metadata
and role/spike conservation after saving.

The seed-42, 100 ms acceptance run produced:

| Role | Active neurons | Spikes |
| --- | ---: | ---: |
| sensory | 191 | 1,642 |
| motor | 193 | 369 |
| descending | 217 | 411 |
| ascending | 172 | 425 |
| interneuron | 4,010 | 7,343 |
| endocrine | 4 | 4 |
| other | 29 | 54 |
| **total** | **4,816** | **10,248** |

These measured counts show connectivity-driven activity reaching curated DN,
AN, and motor populations. They do not prove biological response magnitude,
muscle recruitment, or behavior. Counts can change with model, NT policy,
selection, duration, gain, rate, or seed.

## FlyGym body I/O inventory

```powershell
uv run python -m malecns_sim.simulation.body_io
```

The command composes the existing neutral NeuroMechFly and writes
`outputs/body-io.json`. It does not step physics. The current body exposes:

- 42 position actuators on active leg joint degrees of freedom;
- 42 corresponding joint-position and joint-velocity state channels from
  `MjData.qpos` and `MjData.qvel`;
- 42 actuator-force channels from `MjData.actuator_force`;
- 6 configured ground-contact sensors in `MjData.sensordata`;
- runtime MuJoCo collision contacts;
- 1 currently configured site (the fly root); anatomical joint sites have not
  been added to this v0.1 body composition.

Each actuator and configured sensor is listed by exact compiled MuJoCo name,
index, target, range, data address, and dimension where applicable. The manifest
sets both `male_cns_to_actuator` and `body_sensor_to_male_cns` to null. FlyGym's
composition API supports adding actuator types and joint sites, while joint
angles, actuator forces, contact forces, and anatomical sites can be exposed for
mechanosensory feedback; see the [FlyGym body API](https://neuromechfly.org/api_reference/flygym/compose/fly/base_fly/)
and [basic composition tutorial](https://neuromechfly.org/tutorials/1a_basic_model_composition/).

## Docker and tests

```powershell
docker compose build
docker compose run --rm sim python -m malecns_sim.io.inspect
docker compose run --rm sim python -m malecns_sim.io.query --role motor --limit 5
docker compose run --rm sim pytest tests/io -q
```

To rebuild the catalogue independently inside the container and compare every
artifact byte:

```powershell
docker compose run --rm sim python -m malecns_sim.io.build --output data/processed/malecns-v1.0/published-v1/io-v1-docker
uv run python -m malecns_sim.io.validate data/processed/malecns-v1.0/published-v1/io-v1 --compare data/processed/malecns-v1.0/published-v1/io-v1-docker
```

The graph and `io-v1` catalogue are host data mounted at `/app/data`; neither is
baked into the image. To repeat the anatomy-aware experiment in Docker, choose a
new mounted output directory and compare all result artifacts:

```powershell
docker compose run --rm sim python -m malecns_sim.io.experiment --duration-ms 100 --rate-hz 100 --seed 42 --output outputs/io/leg_proprio_docker
uv run python -m malecns_sim.neural.result outputs/io/leg_proprio_demo --compare outputs/io/leg_proprio_docker
```

The I/O tests cover exact role rules, raw metadata preservation, graph identity,
deterministic artifacts, query mapping, corruption detection, list-valued audit
columns, body inventory boundaries, and hashed anatomy result tables. No v0.4
test maps a MaleCNS neuron to an actuator or a FlyGym state to a neuron.
