# Project state

Last verified: **2026-10-11**, by reconstructing state from the repository (no handoff from
the previous agent existed). Verify against `git status` and the code before relying on it.

Legend: **[verified]** checked during this audit; **[recorded]** stated in repo docs or
saved artifacts but not re-run; **[inferred]** reasoned from evidence; **[unknown]**.

## Current stage

v0.7 (closed loop) is committed on `main` (HEAD `4a49f3c`); **its scientific acceptance
gate fails.** The package version in the
working tree is `0.7.0.dev0`.

**Owner decision (Berke, 2026-10-11):** v0.7 is closed as a *successful infrastructure
release with a failed biological/functional acceptance*. It stays `0.7.0.dev0` and must
**not** be tagged `0.7.0`. Do not keep debugging v0.7 to force motor spikes, and do not add
new simulator features in v0.7.x. The current phase is **v0.7.x pathway investigation**
(ROADMAP §2): pathway audit, then a MuJoCo-free sensory rate sweep, then a latency and
connectivity report. That result decides what v0.8 contains (DECISIONS.md D17).

v0.7 boundary as stated by the owner: done are the persistent LIF, incremental spikes,
causal scheduler, body→CNS, the CNS→body mechanism, clock sync, safety guards, provenance,
Docker reproducibility, v0.5/v0.6 regressions, the synthetic closed loop and the real
MaleCNS experiment. Not done: real MaleCNS physical feedback.

## Implemented and verified

| Capability | Status | Evidence |
| --- | --- | --- |
| Raw MaleCNS download/inspect (v0.1) | done | tests `test_cns.py` pass with local data [verified] |
| Standalone NeuroMechFly body, headless + viewer (v0.1) | done | `test_simulation.py` pass [verified]; viewer launch needs a display, see Unknowns |
| Graph build, policies, validation, queries (v0.2) | done | `test_graph.py` incl. `test_real_graph_acceptance` pass [verified]; 166,700 nodes, 25,582,938 edges, 124,177,617 contacts in `published-v1/manifest.json` [verified] |
| Brian2 LIF engine, signs, stimuli, results (v0.3) | done | `tests/neural/*` pass [verified]; full-CNS numbers in `docs/neural-reference.md` §9 [recorded] |
| I/O catalogue + anatomy-aware experiments (v0.4) | done | `tests/io/*` pass [verified]; role counts in `io-v1/role_counts.json` [verified] |
| One-way motor bridge + replay + viewer (v0.5) | done | `tests/bridge/test_motor_bridge.py`, `test_viewer.py` pass [verified]; LF profile maps 7 cells, six-leg profile 49 [verified from manifests] |
| One-way sensory bridge + replay (v0.6) | done | `tests/bridge/sensory/test_encoder.py` pass [verified]; 23 mapped cells [verified from manifest] |
| Incremental neural API (v0.7) | done | `tests/neural/test_incremental.py` 3 tests pass [verified] |
| Closed-loop scheduler, runner, recorder, validator (v0.7) | software done | `tests/bridge/test_closed_loop.py` pass, including a synthetic 3-neuron graph where feedback changes the trajectory [verified] |
| Windows ↔ Docker reproduction of the v0.7 run | passes on saved artifacts | `closed_loop.validate outputs/closed_loop/lf_tibia_seed42 --compare outputs/closed_loop/lf_tibia_docker_seed42` → `reproduction_passed: true` [verified, 2026-10-11] |

## In progress / incomplete

**v0.7 real-data physical-feedback acceptance: FAILING.**
The run (full MaleCNS, LF tibia +0.20 rad perturb-and-release, 200 ms, seed 42) gives
[recorded in `outputs/closed_loop/lf_tibia_seed42/`, `docs/closed-loop.md`]:
8 sensory events injected, 9 total CNS spikes, **0 spikes in the 7 mapped LF tibia motor
neurons**, 0 non-neutral motor commands, and identical trajectories across feedback_on,
feedback_off and sensory_clamp. `comparison.json`: `acceptance_passed: false`.
`validate --require-effect` exits 1 by design.

Re-run on the current working-tree code (2026-10-11): identical result (see "Recheck" below).

Observations from the saved timeline [verified from `timeline.parquet`]: the joint starts
0.20 rad from neutral and the position actuator pulls it back quickly (q falls from 1.509 to
1.369 rad in 10 ms; per-cell sensory rate drops from 57 Hz to 17 Hz), so sensory drive lasts
only a few milliseconds. Every injected event produces about one spike in the target sensory cell and
almost nothing downstream.

[inferred] Likely contributors, none tested: (a) brief stimulus because `kp=50` actuators
restore the joint quickly; (b) sparse input (8 events over 7 cells) versus the 7 mV
threshold gap, given direct sensory→motor contact sums of at most 39 contacts
(×0.275 mV ≈ 10.7 mV if *all* 23 sources fired together, and `g` decays with τ 5 ms);
(c) indirect paths muted by the unknown-NT zero policy or by inhibition. The closed-loop guide
already notes that all 23 sensory cells are cholinergic and that 11 direct sensory→mapped-motor
pairs exist.

## Known bugs, limitations and risks

- **Recorded Windows v0.7 result predates the final code.** `outputs/closed_loop/lf_tibia_seed42/`
  was written at 21:02 on 2026-09-22; `brian2_lif.py`, `runner.py`, `recorder.py` and
  `validate.py` were modified afterwards (21:05–21:20) [verified from file mtimes]. The
  2026-10-11 recheck below reproduced it exactly with the current code, so this is resolved.
- Scheduler treats any `RuntimeError` as `ABORTED_SAFETY_LIMIT` (DECISIONS.md D15).
- Only one joint (LF tibia) is closed-loop capable; the closed-loop runner rejects profiles
  with more than one channel (`runner.py:prepare`).
- The closed-loop runner requires the full graph (identity node mapping); subgraphs are
  rejected.
- No contact/load sensing; no muscle model; decoder and encoder are engineering hypotheses
  (DECISIONS.md D7, D8).
- `published-v1` has a +9 node discrepancy versus the publication (D2).
- `MUJOCO_LOG.TXT` (git-ignored, 2026-09-13) records one historical "Nan, Inf or huge value
  in QACC" warning at t = 65.3 s; its source run is unknown.
- Lint: 1 ruff I001 error and 5 files not ruff-formatted (ROADMAP 3.2) [verified].

## Test and build status (2026-10-11, Windows, local data present)

| Command | Result |
| --- | --- |
| `uv run --frozen pytest -q -rA -p no:cacheprovider` | **116 passed, 0 failed, 0 skipped, 20 warnings** in 246 s [verified]. Warnings: Brian2 `PyparsingDeprecationWarning` only. Slowest: synthetic closed-loop test 92 s, real graph acceptance 33 s |
| `uv run --frozen ruff check .` | 1 error (I001, `cns/graph/query.py`) [verified] |
| `uv run --frozen ruff format --check .` | 5 files would be reformatted [verified] |
| Docker build / Docker tests | not run in this audit; Docker outputs from 2026-09-22 exist [recorded] |
| `uv sync --locked` | not run (the environment already existed; `--frozen` was used to avoid touching the modified `uv.lock`) |

## Recheck of the v0.7 experiment on current code

Command: `uv run --frozen python -m malecns_sim.bridge.closed_loop.runner --output outputs/closed_loop/lf_tibia_seed42_recheck_2026-10-11`

Result [verified, 2026-10-11, 8 min 16 s wall time on Windows]: all three conditions
`COMPLETED`. feedback_off: 0 events, 0 spikes. sensory_clamp: 8 generated, 0 injected,
0 spikes. feedback_on: 8 injected, 9 total CNS spikes (8 recorded), 7 active sensory cells,
**0 mapped motor spikes, 0 non-neutral commands**. `acceptance_passed: false`.
`closed_loop.validate <recheck> --compare outputs/closed_loop/lf_tibia_seed42` gives
`reproduction_passed: true` (exact events/commands, 1e-9 body tolerance). So the current
working-tree code reproduces the recorded 2026-09-22 result.

## Git state (2026-10-12)

- Branch `main`, in sync with `origin/main` (`https://github.com/Doga0/flycns-sim`).
  HEAD `4a49f3c` "project docs for agents".
- The owner committed the v0.7 work directly to `main` and pushed it on 2026-10-11
  (`73f9f8c`..`4230a40`: closed-loop package, config, guide, incremental neural API,
  tests, version `0.7.0.dev0`), plus the bootstrap docs in `4a49f3c`. No tag.
- All earlier `feat/*` branches and `init` are merged. No stashes.
- **Uncommitted (v0.7.x pathway audit, preserve):**
  - Modified: `AGENTS.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`,
    `docs/PROJECT_STATE.md`, `docs/ROADMAP.md`.
  - Untracked: `src/malecns_sim/analysis/` (pathway audit, activation sweep, pathway
    report), `tests/analysis/`, `docs/pathway-audit.md`, `docs/lf-tibia-pathway-report.md`.
- Git-ignored outputs from this work: `outputs/closed_loop/lf_tibia_seed42_recheck_2026-10-11/`,
  `outputs/analysis/lf_tibia_pathway/`, `outputs/analysis/lf_tibia_activation_sweep/`,
  `outputs/analysis/lf_tibia_pathway_report/`, `outputs/analysis/lf_tibia_sweep_robustness/`,
  `outputs/analysis/lf_tibia_subset_sweep/` (2026-11-09).
- 2026-11-09: `tests/analysis` 18 passed (`uv run --frozen pytest tests/analysis -q`);
  `ruff check` / `ruff format --check` clean on `src/malecns_sim/analysis` and
  `tests/analysis`. Full suite not re-run.

## Current task: v0.7.x pathway investigation (owner plan, 2026-10-11)

The owner set this sequence. It is diagnostic work: report what the current graph and LIF
policy do, do not add biological assumptions, do not change v1 profiles, and do not read the
sweep results as biological fact.

1. **Pathway audit** (`malecns_sim.analysis.pathway_audit`): anatomical paths of 1 to 5
   synapses from the 23 LF/ProLN chordotonal cells to the 7 mapped LF tibia motor neurons.
   For each path: source, target, hop count, contacts, transmitter and model sign per
   neuron, type, side, intermediate population, multiplicity and synaptic-delay latency.
   Done; see "Pathway audit status" below.
2. **Functional activation sweep** without MuJoCo: drive the 23 sensory cells directly
   (single spikes, volleys, and 10–1000 Hz rates) in the full LIF model and record the
   spikes, first-spike latency and spike count of each mapped motor neuron. Done.
3. **Latency and connectivity report** combining 1 and 2: path depth, transmitter mix,
   latency and minimum input for motor activation. Done.
4. Only then design v0.8: "physical sensory-motor validation" if the pathway works under the
   current model, otherwise "reviewed interface/model calibration".

Acceptance for steps 1–3 (owner's question): *given this sensory population and this motor
population, under the current MaleCNS graph and LIF model, what input is required to
activate the motor population?* Outputs: minimum input rate, motor activation latency,
active motor neurons, spike count, path depth, path connectivity. All runs go to new output
directories with provenance; no tracked v1 config, weight, sign policy or mapping changes;
the existing tests keep passing.

### Pathway audit status

Step 1 done (2026-10-11), uncommitted. Tool: `src/malecns_sim/analysis/` (new package),
tests `tests/analysis/` (16 passed; full suite 132 passed on 2026-10-12). Guide and full numbers:
[pathway-audit.md](pathway-audit.md). Output: `outputs/analysis/lf_tibia_pathway/`.

Headline facts: all 7 motor neurons are reachable, 6 by a direct synapse (11 excitatory
ACh pairs, at most 39 contacts onto one neuron) and 819384 at 2 hops. 2-hop paths: 538
excitatory, 494 net inhibitory, 20 muted. Only 3% of on-path intermediates are muted by the
zero-sign policy. A linear direct-drive estimate puts the best-connected motor neuron
(807165) at threshold only when its 3 direct sources fire at about 131 Hz together.
Inference to test in step 2: the pathway exists anatomically, and the v0.7 input (8 events
in 200 ms) was far too weak for it.

Step 2 done (2026-10-12), uncommitted: `python -m malecns_sim.analysis.activation_sweep`,
output `outputs/analysis/lf_tibia_activation_sweep/`, details in
[pathway-audit.md](pathway-audit.md#step-2-functional-activation-sweep-2026-10-12).
Facts: with no input nothing fires. Poisson 10–50 Hz to all 23 sensory cells drives no
mapped motor neuron. 100 Hz nominal (87 Hz actual) and above, or a 10-volley burst, drives
**only flexors 807165 and 809912**; the other 3 flexors and both extensors never fire up to
1000 Hz. Activity never passes beyond 2 synapses from the sensory cells. First motor latency
ranges from 51 ms (100 Hz) to about 14 ms.

Step 3 done (2026-10-12), uncommitted: `python -m malecns_sim.analysis.pathway_report`,
output `outputs/analysis/lf_tibia_pathway_report/`, written up in
[lf-tibia-pathway-report.md](lf-tibia-pathway-report.md). New fact: at the strongest input
**both extensors receive net inhibition** (mostly from GABA-labeled 1-hop interneurons), and
the two active flexors get more excitation via 1-hop interneurons than directly from the
sensory cells.

**Next: the owner decides the v0.8 scope (ROADMAP 2.4).** The evidence supports
"physical sensory-motor validation" (the pathway works without weight or sign changes)
but raises three interface questions: how to reach about 87 Hz per sensory cell, whether a
flexor-only response is acceptable, and how the decoder should normalize when 2 of 5
flexors fire. See the report's last section and the proposal in
[v0.8-proposal.md](v0.8-proposal.md). A robustness sweep (seeds 42–46, rates 50–200 Hz, 500 ms
drive, `outputs/analysis/lf_tibia_sweep_robustness/`) showed a soft threshold: sporadic motor
spikes at 50–90 Hz, 5–14 spikes at 100 Hz (latency 51–208 ms), 25–38 at 200 Hz (22–39 ms);
extensors silent in all 35 runs. The v1 encoder saturates at 100 Hz, which bounds what a
v1-based v0.8 protocol can reach.

**Silent motor neurons explained (2026-11-09, uncommitted):** `activation_sweep` gained
`--sensory-subset` (explicit profile body IDs). A full-CNS input census and a subset sweep
(`outputs/analysis/lf_tibia_subset_sweep/{snpp51,snpp50,snpp39,all_but_snpp51}`) show:
the whole flexor response comes from the 3 SNpp51 cells (alone they give 20/14 spikes in
807165/809912 and 1 spike each in 818057 and 909831 at 1000 Hz); the other 20 cells are net
inhibitory and drive nothing; no subset excites the extensors meaningfully; and 75–89% of
each motor neuron's excitatory input comes from premotor partners this input never
activates. This is not a pipeline bug. Details in
[pathway-audit.md](pathway-audit.md#why-5-of-7-motor-neurons-stay-silent-2026-11-09). It adds
to the v0.8 decision: the one-population encoder mixes opposing sensory channels.

## Unknowns that need human input

1. (Resolved 2026-10-11: the owner committed v0.7 to `main`.)
2. v0.8 scope: physical sensory-motor validation or interface/model calibration? The
   investigation has reported (lf-tibia-pathway-report.md); this is now the open decision,
   including whether flexor-only activation is an acceptable first criterion.
3. Has the Windows MuJoCo viewer been verified manually? README says no, but
   `outputs/bridge/lf_viewer_live` and `six_leg_viewer_live` exist.
4. Were the 2026-09-22 Docker outputs produced from the final code?
5. Priority among long-term goals (six-leg loop, locomotion, visualization, learning).
6. Identity of the 9 extra nodes in `published-v1` (data-provider question).
