# Project state

Last verified: **2026-10-11**, by reconstructing state from the repository (no handoff from
the previous agent existed). Verify against `git status` and the code before relying on it.

Legend: **[verified]** checked during this audit; **[recorded]** stated in repo docs or
saved artifacts but not re-run; **[inferred]** reasoned from evidence; **[unknown]**.

## Current stage

v0.6.0 is committed and merged (`main` at `0d41abb`). **v0.7 (closed loop) is implemented
but uncommitted, and its scientific acceptance gate fails.** The package version in the
working tree is `0.7.0.dev0`.

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
| Incremental neural API (v0.7) | done, uncommitted | `tests/neural/test_incremental.py` 3 tests pass [verified] |
| Closed-loop scheduler, runner, recorder, validator (v0.7) | software done, uncommitted | `tests/bridge/test_closed_loop.py` pass, including a synthetic 3-neuron graph where feedback changes the trajectory [verified] |
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

## Git state (2026-10-11)

- Branch `main`, up to date with `origin/main` (`https://github.com/Doga0/flycns-sim`).
  HEAD `0d41abb` "Merge feat/sensory-bridge branch into main" (2026-09-22). 55 commits.
- All `feat/*` branches (`graph-core`, `lf-neural-dynamics`, `io-catalogue`,
  `one-way-motor-bridge`, `sensory-bridge`) and `init` are merged into `main`.
- No stashes.
- **Uncommitted, must be preserved** (all v0.7 work):
  - Modified: `README.md` (v0.7 header section), `pyproject.toml` (version
    `0.7.0.dev0`, description, closed_loop profile packaging), `uv.lock` (version only),
    `src/malecns_sim/__init__.py` (version), `neural/backends/base.py` (incremental API),
    `neural/backends/brian2_lif.py` (event input, selective monitor, cursors).
  - Untracked: `configs/closed_loop/`, `docs/closed-loop.md`,
    `src/malecns_sim/bridge/closed_loop/`, `tests/bridge/test_closed_loop.py`,
    `tests/neural/test_incremental.py`.
  - Added by this audit (untracked): `AGENTS.md`, `CLAUDE.md`, `docs/PROJECT_OVERVIEW.md`,
    `docs/ARCHITECTURE.md`, `docs/PROJECT_STATE.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`.
  - New git-ignored output from this audit: `outputs/closed_loop/lf_tibia_seed42_recheck_2026-10-11/`.

## Recommended next task

**Diagnose why LF chordotonal input does not drive the mapped LF tibia motor neurons**
(ROADMAP 2.1). This is a recommendation from the evidence; the previous developer's
intended next step is unknown. Do step 0 first, or get the owner's go-ahead to skip it.

0. (Owner decision) Commit the uncommitted v0.7 work to a `feat/closed-loop` branch
   (ROADMAP 1.1).
1. Static path analysis on `published-v1` with the shiu-lif-v1 signs: direct and 2-hop
   signed effective weights from the 23 sensory cells to the 7 motor cells; how many
   intermediate nodes have sign 0 (unknown NT); maximum depolarization reachable in `g`
   from realistic input patterns.
2. Diagnostic simulations in **new output directories** with **temporary diagnostic
   configs** outside `configs/`, never editing v1 profiles: sweep sensory drive (all 23
   cells at fixed rates, longer duration), perturbation offset/duration, and `monitor: full`
   to see where activity dies out. Report rates of the mapped motor cells.
3. Summarize which factor(s) block the path and propose candidate v2 policy changes with
   justification for the owner (ROADMAP 2.2).

Files likely affected: new diagnostic script(s), for example `src/malecns_sim/bridge/closed_loop/diagnose.py`
or `scripts/` (choose one and document it); `docs/closed-loop.md` (findings section);
`docs/PROJECT_STATE.md`, `docs/DECISIONS.md`, `docs/ROADMAP.md`. Read-only use of
`neural/signs.py`, `cns/graph/store.py`, `bridge/sensory/*`, `bridge/mapping.py`.

Acceptance criteria:
- A reproducible command (documented in `docs/closed-loop.md`) produces the path analysis
  and the sweep results, with provenance, in new output directories.
- The report quantifies, for the default v1 setup, the drive reaching each of the 7 mapped
  motor cells (direct and via ≤2 hops) and identifies at least one factor whose change
  produces mapped-motor spikes, or shows that none of the tested factors does.
- No tracked v1 config, weight, sign policy or mapping is modified; the existing 116 tests
  still pass; `ruff check` reports no new errors.
- Findings are recorded as facts vs. inferences in `docs/closed-loop.md` and summarized in
  this file, with proposed v2 options listed in ROADMAP 2.2 for the owner to decide.

## Unknowns that need human input

1. Should the uncommitted v0.7 work be committed, and on which branch?
2. Which v2 changes (encoder, gain, populations, protocol, actuator stiffness, NT policy)
   are scientifically acceptable for passing the gate? What counts as a pass beyond
   "any mapped-motor effect"?
3. Has the Windows MuJoCo viewer been verified manually? README says no, but
   `outputs/bridge/lf_viewer_live` and `six_leg_viewer_live` exist.
4. Were the 2026-09-22 Docker outputs produced from the final code?
5. Priority among long-term goals (six-leg loop, locomotion, visualization, learning).
6. Identity of the 9 extra nodes in `published-v1` (data-provider question).
