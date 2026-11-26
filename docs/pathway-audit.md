# v0.7.x: sensory→motor pathway audit

Step 1 of the v0.7.x investigation (ROADMAP §2.1). It answers an **anatomical** question:
which synaptic paths connect the 23 LF/ProLN chordotonal cells (`sensory-bridge-v1`) to
the 7 mapped LF tibia motor neurons (`motor-bridge-v1`), and how does the current LIF sign
policy classify them? It runs no dynamics. Whether these paths **fire** in the LIF model
is the separate activation sweep (ROADMAP §2.2).

Code: [`analysis/sensory_motor_paths.py`](../src/malecns_sim/analysis/sensory_motor_paths.py)
(computation) and [`analysis/pathway_audit.py`](../src/malecns_sim/analysis/pathway_audit.py)
(CLI, artifacts). Tests: `tests/analysis/test_pathway_audit.py` (synthetic graphs only).

## Run

```powershell
uv run python -m malecns_sim.analysis.pathway_audit --output outputs/analysis/lf_tibia_pathway
```

Defaults: `--graph data/processed/malecns-v1.0/published-v1`,
`--sensory-config configs/bridge/sensory_lf_tibia_v1.yaml`,
`--motor-config configs/bridge/lf_tibia_v1.yaml`,
`--lif-config configs/neural/shiu_lif_v1.yaml`, `--max-hops 5`, `--enumerate-hops 3`,
`--edge-hops 4`. Populations come only from the explicit body IDs in the two profiles.
About 40 s on the development machine; the output directory must not exist.

## Definitions

- **hops** = number of synapses. A 1-hop path is a direct synapse; an h-hop path has
  h − 1 intermediate neurons.
- **Sign class** uses the presynaptic neurons only (source and intermediates), as the LIF
  signs each synapse by its source: `excitatory` (all +), `net_inhibitory` (odd number of
  −), `net_excitatory_via_inhibition` (even number ≥ 2 of −), `muted` (any sign 0 under
  `unknown_nt_policy: zero`). Signs come from `neural/signs.py`; glutamate is −1 in
  shiu-lif-v1.
- **Multiplicity** counts walks (a neuron may repeat). Paths of ≤ `--enumerate-hops`
  synapses are also listed one by one, with `simple = false` where a neuron repeats; the
  tool checks that this list matches the propagated counts.
- **Latency** = hops × 1.8 ms synaptic delay, a lower bound that excludes membrane
  integration time.
- **Direct-drive estimate**: a linear subthreshold reading of the shiu-lif-v1 equations.
  One presynaptic spike adds w = contacts × 0.275 mV to `g`, giving a peak depolarization of
  0.158 × w (τm 20 ms, τs 5 ms). With every direct source firing at rate r, the mean
  depolarization is w × r × τs. It ignores indirect paths, inhibition, refractoriness and
  fluctuations.

## Outputs

| File | Content |
| --- | --- |
| `nodes.parquet` | Every neuron on some path of ≤ max-hops synapses: ids, population, type, superclass, class, side, transmitter, model sign, hops from sensory / to motor |
| `edges.parquet` | Synapses on some path of ≤ edge-hops synapses: contacts, presynaptic sign, weight (mV) |
| `paths.parquet` | Each path of ≤ enumerate-hops synapses: node/body/type/side/transmitter/sign/contact/weight sequences, bottleneck contacts, sign class, simple flag, latency bound |
| `pathway_summary.json` | Reachability, walk counts per hop and sign class (overall and per motor neuron), intermediate-layer composition, enumerated-path breakdown, direct-drive estimate |
| `report.json` | Parameters, definitions, input hashes, LIF config and hash, sign report, versions, artifact hashes |

## First result (2026-10-11, `outputs/analysis/lf_tibia_pathway`)

Facts from the audit (current graph and shiu-lif-v1 sign policy):

- All 7 motor neurons are reachable. Six receive a **direct** synapse from the sensory
  population; `819384` needs 2 hops.
- **Direct paths:** 11 sensory→motor pairs, all acetylcholine → excitatory. Contacts per
  motor neuron: 807165: 39 (3 sources), 809912: 21 (3), 815344: 5 (1), 909831: 4 (2),
  800636: 4 (1), 818057: 1 (1), 819384: 0. These agree with `docs/closed-loop.md`.

| Hops | Walks | Excitatory | Net excitatory via inhibition | Net inhibitory | Muted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11 | 11 | 0 | 0 | 0 |
| 2 | 1,052 | 538 | 0 | 494 | 20 |
| 3 | 166,452 | 33,338 | 38,079 | 87,717 | 7,318 |
| 4 | 33.1 M | 2.96 M | 12.06 M | 15.01 M | 3.10 M |
| 5 | 7.67 B | 0.34 B | 2.89 B | 3.18 B | 1.27 B |

- 2-hop intermediates: 230 neurons (108 ACh, 57 GABA, 58 glutamate, 7 unclear), mostly left
  side (177 L, 51 R).
- Intermediates on paths of ≤ 5 hops: 42,250 neurons, of which 1,273 (3.0%) have sign 0.
  Graph-wide, the zero policy mutes 11,068 neurons (7,891 histamine, 2,999 unclear,
  178 missing).
- Direct-drive estimate (linear, direct paths only): a synchronous single volley of all
  direct sources peaks at 1.69 mV in 807165 and ≤ 0.91 mV elsewhere, against a 7 mV
  threshold gap. Reaching threshold on average needs every direct source of 807165 at
  about 131 Hz, of 809912 at about 242 Hz, and above 1,000 Hz for the others. Input-driven
  sensory cells cannot exceed about 455 Hz (2.2 ms refractory).

Inferences, not yet tested: anatomical paths exist and short ones are mostly excitatory, so
the zero-sign policy is not the main block. The v0.7 run injected 8 events into 7 cells over
200 ms, orders of magnitude below the estimated direct drive. Beyond 2 hops,
net-inhibitory walks outnumber purely excitatory ones. The activation sweep must test
these readings; the linear estimate is not a prediction of spiking.

## Step 2: functional activation sweep (2026-10-12)

Code: [`analysis/activation_sweep.py`](../src/malecns_sim/analysis/activation_sweep.py), tests
`tests/analysis/test_activation_sweep.py`. It uses the full LIF network with no body. Each
condition builds a fresh network (seed 42), stays silent for 50 ms, drives all 23 mapped
sensory cells for 200 ms, then runs a 50 ms tail. Input events add the sensory profile's
68.75 mV gain to `v`, the same gain as the closed loop. `volley_N` gives every sensory cell
one event at N times spaced 3 ms apart. `poisson_R` gives independent per-cell Poisson
events at nominal R Hz.

```powershell
uv run python -m malecns_sim.analysis.activation_sweep --output outputs/analysis/lf_tibia_activation_sweep
```

About 6 minutes for the 10 default conditions. Outputs: `sweep_summary.json`,
`motor_matrix.parquet` (condition × motor neuron: spikes, first-spike latency),
`layer_activity.parquet` (active neurons and spikes by hops from the sensory population),
`spikes/<condition>.parquet`, and `report.json` (parameters, hashes, versions).

### Result (`outputs/analysis/lf_tibia_activation_sweep`, seed 42)

"Actual Hz" is the measured sensory spikes per cell in the 200 ms drive window. It falls
below the nominal rate because the 2.2 ms refractory period drops events.

| Condition | Actual sensory Hz per cell | 807165 | 809912 | Other 5 motor neurons | First motor latency (ms) | Active neurons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no input | 0 | 0 | 0 | 0 | | 0 |
| 1 volley | (1 spike each) | 0 | 0 | 0 | | 23 |
| 10 volleys / 3 ms | (10 spikes each) | 2 | 2 | 0 | 13.1 | 190 |
| Poisson 10 Hz | 10 | 0 | 0 | 0 | | 21 |
| Poisson 25 Hz | 24 | 0 | 0 | 0 | | 27 |
| Poisson 50 Hz | 51 | 0 | 0 | 0 | | 44 |
| Poisson 100 Hz | 87 | 5 | 3 | 0 | 51.0 | 142 |
| Poisson 200 Hz | 138 | 7 | 4 | 0 | 39.4 | 183 |
| Poisson 500 Hz | 240 | 14 | 11 | 0 | 15.5 | 252 |
| Poisson 1000 Hz | 312 | 18 | 13 | 0 | 13.7 | 268 |

Facts:
- There is no spontaneous activity. The lowest tested Poisson rate that drives a mapped motor
  neuron is **100 Hz nominal (about 87 Hz actual)**; 50 Hz does not. A 10-volley burst also
  activates both.
- Only **807165 and 809912** ever fire. These are the two flexors with the most direct
  sensory contacts (39 and 21). The other three flexors and **both extensors never fire**
  at any tested input.
- Activity never travels more than 2 synapses from the sensory cells. No neuron 3 or more hops
  away fires in any condition; the 2-hop layer peaks at 108 active neurons.
- Motor first-spike latency falls from 51 ms (100 Hz) to about 14 ms (1000 Hz, volleys). The
  synaptic-delay lower bound for a direct path is 1.8 ms, so integration time dominates.
- At the strongest input, 807165 fires 18 spikes in 250 ms (about 72 Hz).

Inferences, not yet tested:
- The pathway works under the current model only with sustained, near-synchronous sensory
  drive of about 87 Hz or more per cell across all 23 cells. In the v0.7 closed loop the
  encoder gave about 57 Hz per cell for only a few milliseconds before the joint returned,
  which explains the 0 motor spikes there. This answers the owner's question in the form
  "the pathway exists under the current model, but the current sensory encoder does not
  provide sufficient drive".
- The motor output is flexor-only. The antagonistic decoder would therefore drive only one
  direction, and the decoder's 100 Hz `rmax` would be reached only partly (population mean
  over 5 flexors).
- Caveats: one seed; rates between 50 and 100 Hz and longer drives were not tested; the
  linear estimate in step 1 (131 Hz for 807165) assumed direct input only and no
  fluctuations, so it is conservative.

The combined latency and connectivity report (step 3 of the investigation) is in
[lf-tibia-pathway-report.md](lf-tibia-pathway-report.md).

## Robustness sweep (2026-10-12, `outputs/analysis/lf_tibia_sweep_robustness/seed42..46`)

Same tool: 5 seeds (42–46), Poisson rates 50/60/70/80/90/100/200 Hz nominal, **500 ms**
drive, no volleys:

```powershell
uv run python -m malecns_sim.analysis.activation_sweep --rates 50,60,70,80,90,100,200 --volleys "" --drive-ms 500 --seed 42 --output outputs/analysis/lf_tibia_sweep_robustness/seed42
```

Total mapped-motor spikes (first motor latency in ms) per seed:

| Seed | 50 | 60 | 70 | 80 | 90 | 100 | 200 Hz |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0 | 2 (226) | 2 (225) | 2 (220) | 2 (189) | 8 (51) | 25 (39) |
| 43 | 0 | 0 | 1 (498) | 1 (497) | 3 (288) | 14 (128) | 33 (24) |
| 44 | 0 | 2 (481) | 2 (480) | 7 (309) | 12 (77) | 9 (74) | 34 (38) |
| 45 | 0 | 3 (296) | 3 (296) | 4 (296) | 4 (294) | 9 (208) | 29 (33) |
| 46 | 2 (434) | 2 (427) | 2 (415) | 2 (49) | 2 (43) | 5 (52) | 38 (22) |

Facts: the threshold is **soft**. From 50 to 90 Hz the motor neurons spike only sporadically
(0–12 spikes in 550 ms, mostly hundreds of ms after onset). At 100 Hz all seeds give 5–14
spikes with first-spike latency of 51–208 ms. At 200 Hz all seeds give 25–38 spikes within
22–39 ms. In all 35 driven runs only 807165 and 809912 fire, the extensors stay at 0, and
activity never goes beyond 2 synapses.

Implication (inference): the v1 encoder saturates at 100 Hz nominal (at neutral + 0.35 rad).
Even a fully held displacement therefore sits in the slow, sparse regime. Fast, reliable
activation needs about 200 Hz nominal, which is outside the v1 encoder's range. See
[v0.8-proposal.md](v0.8-proposal.md).

## Why 5 of 7 motor neurons stay silent (2026-11-09)

Two read-only checks of the silent neurons: a full-CNS input census of each mapped motor neuron
and a sweep that drives sensory subsets. Both use the current graph and shiu-lif-v1, with no
changes to weights, signs or profiles.

**Input census** (all presynaptic partners in the graph, contacts; "active" means the
partner fired in `poisson_1000hz` of `lf_tibia_activation_sweep`):

| Motor | Excitatory contacts | Inhibitory | From the 23 sensory cells | Excitatory from active partners |
| --- | ---: | ---: | ---: | ---: |
| 807165 flexor | 1,897 | 1,211 | 39 | 446 (23.5%) |
| 809912 flexor | 1,215 | 683 | 21 | 298 (24.5%) |
| 818057 flexor | 395 | 339 | 1 | 96 (24.3%) |
| 819384 flexor | 251 | 288 | 0 | 56 (22.3%) |
| 909831 flexor | 320 | 200 | 4 | 68 (21.2%) |
| 800636 extensor | 2,432 | 1,808 | 4 | 325 (13.4%) |
| 815344 extensor | 6,240 | 3,947 | 5 | 717 (11.5%) |

Facts: 75–89% of each motor neuron's excitatory input comes from partners that the
sensory drive never activates. Those partners are mostly VNC premotor interneurons (for
example IN04B031, IN12A001, IN01A063_a onto the extensors), plus descending neurons. The
silent flexors have about 1/5 to 1/8 as many excitatory contacts as 807165, while the active
share is the same (about 22–25%). Under shiu-lif-v1 every neuron has the same threshold and
the same weight per contact, so fewer contacts means less drive.

**Subset sweep** (`--sensory-subset`, explicit body IDs from `sensory_lf_tibia_v1.yaml`,
seed 42, 200 ms drive, Poisson 200 and 1000 Hz nominal; output
`outputs/analysis/lf_tibia_subset_sweep/<group>/`):

```powershell
uv run python -m malecns_sim.analysis.activation_sweep --rates 200,1000 --volleys "" --sensory-subset 815843,817680,935383 --output outputs/analysis/lf_tibia_subset_sweep/snpp51
```

The groups use the released `type` of each listed ID (looked up, not used to select cells):
SNpp51 = 815843, 817680, 935383; SNpp50 = 912317; SNpp39 = 815569, 903843, 935541;
`all_but_snpp51` = the other 20 cells. Motor spikes at 1000 Hz nominal (200 Hz in brackets):

| Group | Actual Hz per cell | 807165 | 809912 | 818057 | 909831 | 819384 | Extensors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All 23 (earlier sweep) | 312 (138) | 18 (7) | 13 (4) | 0 | 0 | 0 | 0 |
| SNpp51 (3 cells) | 244 (101) | 20 (7) | 14 (5) | 1 (0) | 1 (0) | 0 | 0 |
| SNpp50 (1 cell) | 268 (96) | 0 | 0 | 0 | 0 | 0 | 0 |
| SNpp39 (3 cells) | 244 (101) | 0 | 0 | 0 | 0 | 0 | 0 |
| All but SNpp51 (20) | 252 (113) | 0 | 0 | 0 | 0 | 0 | 0 |

Delivered net drive at 1000 Hz (mV of `g` increments, as in the pathway report): SNpp51
gives +2,045 to 807165, +207 / +221 to 818057 / 909831, −64 to 819384, and −1,022 / −3,079 to
the extensors. SNpp50 is the only group with net excitation to the extensors (+72 / +92),
far too little to fire them, and it inhibits the flexors (−309 to 807165). The other 20
cells together are net inhibitory to every motor neuron that has non-trivial input
(−375 to 807165, −425 / −844 to the extensors).

Inferences (not tested further):
- The whole flexor response comes from 3 SNpp51 cells. Driving the other 20 cells as well
  slightly *reduces* it, so the one-population encoder of `sensory-bridge-v1` mixes
  channels with opposite effects.
- No mapped sensory subset excites the extensors enough to matter. In this graph, extensor
  excitation comes from premotor circuits that LF chordotonal input does not recruit in an
  otherwise silent network (no descending or central-pattern drive).
- The small flexors (818057, 909831) can reach threshold with SNpp51 alone at high rates.
  Their silence is a matter of degree (few contacts × uniform LIF), not of missing pathways.
  819384 gets net inhibition from SNpp51.
- So the silence is not a pipeline bug. Making these neurons fire would need either more
  input to the premotor network (outside this sensory population) or a neuron-specific LIF
  calibration. Both are v0.8 scope decisions for the owner, not v0.7 fixes.
