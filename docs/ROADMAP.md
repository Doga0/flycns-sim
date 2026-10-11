# Roadmap

Ordered by priority and dependency. Items in sections 1–3 follow from the current code and
the failed v0.7 gate. Section 4 lists owner-stated goals that have **no** concrete
requirements in the repo yet; do not treat their wording as a spec. Keep this file in sync
with [PROJECT_STATE.md](PROJECT_STATE.md).

## 1. Secure the v0.7 work (needs the owner)

**1.1 Decide how to persist the uncommitted closed-loop implementation.** About 180 changed
lines plus new modules, configs, tests and docs exist only in the working tree (see
PROJECT_STATE "Git state"). Previous milestones each used a `feat/<name>` branch merged
into `main`. Suggested: `feat/closed-loop` with the work committed as-is (`0.7.0.dev0`).
- Acceptance: the work is on a branch on `origin`; `main` is unchanged until the owner merges.

## 2. Make the v0.7 physical-feedback gate meaningful

**2.1 Diagnose why sensory input does not reach the mapped LF motor neurons** (recommended
next task; full criteria in PROJECT_STATE). Diagnostics only: no default profile changes.
Depends on nothing; can run before 1.1, but its outputs must not modify tracked configs.

**2.2 Scientific decision on a v2 sensory/motor policy** (needs the owner).
Using 2.1's evidence, decide whether to change any of: encoder transfer function or rate
range, input gain, the mapped sensory population, the mapped motor population, perturbation
protocol (offset, duration, active trajectory instead of release), the unknown-NT policy,
or the body actuator stiffness. Every change becomes a new versioned profile (`_v2`), never
an edit of `_v1`.
- Acceptance: a written decision (DECISIONS.md entry) naming which parameters change, the
  biological or engineering justification, and what would count as a pass.

**2.3 Implement the v2 profile and rerun the three-condition experiment.** Depends on 2.2.
- Acceptance: `bridge.closed_loop.validate <run> --require-effect` exits 0 without
  violating the rules in DECISIONS.md D11 (no weight edits, seed picking, motor injection or
  relabeling); Windows/Docker `--compare` passes; v0.5/v0.6 compatibility tests still pass;
  closed-loop guide and README updated; version bumped from `0.7.0.dev0` to `0.7.0`.
- If the gate still fails, record the negative result instead and return to 2.2.

## 3. Housekeeping (independent, small)

- **3.1 README contradictions.** README line ~289 says "The CNS and body are not connected
  yet" and §3 says the Windows viewer "has not yet been manually verified", though v0.5+
  bridges and `outputs/bridge/*viewer_live*` runs exist. README §5 still describes only the
  v0.1 test set ("14 passed"). Acceptance: README matches PROJECT_STATE; owner confirms the
  viewer status.
- **3.2 Lint.** `ruff check .` reports one I001 (unsorted imports) in
  `src/malecns_sim/cns/graph/query.py`; `ruff format --check .` would reformat 5 files
  (`cns/graph/query.py`, `bridge/sensory/validate.py`, `__init__.py`,
  `docs/io-catalogue.md`, `docs/neural-reference.md`). Acceptance: both commands clean.
- **3.3 Artifact naming.** The v0.6 sensory replay writes `sensory_spikes.parquet`; the v0.7
  closed loop writes `sensory_events.parquet` for the same concept. Decide whether to align
  (would need validator updates; old outputs keep their names).
- **3.4 Exception scope in the scheduler.** Consider narrowing the `RuntimeError` catch in
  `closed_loop/scheduler.py` so non-safety errors are not labeled `ABORTED_SAFETY_LIMIT`
  (see DECISIONS.md D15).
- **3.5 Third-party warning.** Brian2 2.10 triggers `PyparsingDeprecationWarning`
  (20 warnings in the suite). Harmless; track upstream.

## 4. Owner-stated long-term goals (no specs yet)

From the project description. Each needs requirements from the owner before it can be
planned. Listed in the dependency order the current architecture suggests:

1. Extend the closed loop beyond one joint (the six-leg motor profile
   `configs/bridge/six_leg_tibia_v1.yaml` already exists for the motor direction only).
2. Add sensory channels beyond joint position (contact/load, velocity is partly supported
   by `directional_velocity_encoder_v1`).
3. Behaviors: walking, orientation, obstacle avoidance, stimulus responses (needs world and
   task definitions).
4. Visualization of neural activity and body (Three.js mentioned by the owner).
5. Plasticity and reinforcement learning.
6. Faster neural backends (the `NeuralBackend` ABC allows it; README lists GPU backends and
   GNN dynamics as out of scope so far).
