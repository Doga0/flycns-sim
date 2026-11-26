# Roadmap

Ordered by priority and dependency. The version sequence below is the owner's plan
(Berke, 2026-10-11). Later stages are deliberately left open: the outcome of the v0.7.x
investigation decides what v0.8 contains. Keep this file in sync with
[PROJECT_STATE.md](PROJECT_STATE.md).

```text
v0.7    closed-loop infrastructure        (closed; stays 0.7.0.dev0, never tagged 0.7.0)
v0.7.x  functional pathway investigation  (current; audit, sweep and report done; v0.8 scope decision next)
v0.8    validated sensory-motor interface (content decided by v0.7.x)
v0.9    multi-leg embodiment
v1.0    locomotion benchmark
v1.1    RL environment
```

## 1. v0.7: closed

Infrastructure is done and the real-data physical-feedback acceptance failed (see
PROJECT_STATE). Owner rules: keep `0.7.0.dev0`, do not tag `0.7.0`, and do not keep
debugging v0.7 to produce motor spikes.

**1.1 Persist the v0.7 work:** done. The owner committed it to `main` on 2026-10-11 (no tag).

## 2. v0.7.x: functional pathway investigation (current)

The question: *given the 23 LF/ProLN chordotonal cells and the 7 mapped LF tibia motor
neurons, under the current MaleCNS graph and shiu-lif-v1 model, what input is required to
activate the motor population?* This must keep anatomical connectivity (does a path exist)
separate from functional connectivity (does it fire in this LIF model). No new biological
assumptions, no v1 profile edits, and results are diagnostics, not biology.

**2.1 Pathway audit** (done 2026-10-11, uncommitted; results in [pathway-audit.md](pathway-audit.md)). `python -m malecns_sim.analysis.pathway_audit`.
- Paths of 1 to 5 synapses from the sensory to the motor population, with source, target,
  hops, contacts, transmitter and model sign per neuron, type, side, intermediate
  population, multiplicity and synaptic-delay latency.
- Outputs: `pathway_summary.json`, `nodes.parquet`, `edges.parquet`, `paths.parquet`,
  `report.json`, in a new output directory with provenance.
- Acceptance: reproducible from the CLI; tested on a synthetic graph; it reports what the
  current sign policy does to each path class without changing it.

**2.2 Functional activation sweep** (done 2026-10-12, uncommitted; `analysis.activation_sweep`, results in pathway-audit.md).
Full LIF model, no MuJoCo. Drive the sensory cells directly: a single spike, volleys
(for example 10 spikes) and rates such as 10, 25, 50, 100, 200, 500 and 1000 Hz. Record each
mapped motor neuron's spike count and first-spike latency, plus activity per hop layer from
2.1.
- Acceptance: a rate × motor-neuron matrix and a latency table with provenance; one of
  "activates at some tested input" (report the minimum) or "no activation at any tested
  input" is stated explicitly.

**2.3 Latency and connectivity report** (done 2026-10-12, uncommitted; [lf-tibia-pathway-report.md](lf-tibia-pathway-report.md)). Direct contacts, path
counts per depth, transmitter mix, latency and motor activation in one document, added to
`docs/` and summarized in PROJECT_STATE.

**2.4 Decide v0.8 scope** (needs the owner; next). The report's last section lists the evidence and open interface questions. A concrete plan with tests is in [v0.8-proposal.md](v0.8-proposal.md) (proposal only). A robustness sweep (5 seeds, 50–200 Hz, 500 ms) finished; results are in pathway-audit.md. If the pathway can work under the current model,
v0.8 = physical sensory-motor validation. If not, v0.8 = reviewed interface/model
calibration.

## 3. Later stages (owner's direction, no specs yet)

- **v0.8 validated sensory-motor interface:** LF proprioception plus contact/load sensing,
  LF tibia motor, MaleCNS with a validated pathway, FlyGym/MuJoCo. No RL. New sensory
  populations only when literature supports them; never pick neurons ad hoc. `BodyState`
  may grow contact and force fields.
- **v0.9 multi-leg embodiment:** six legs (LF, LM, LR, RF, RM, RR) under simultaneous neural
  control. A six-leg motor profile (`configs/bridge/six_leg_tibia_v1.yaml`) already exists
  for the motor direction only.
- **v1.0 locomotion benchmark:** standing, single-leg movement, alternating stepping,
  forward stepping, walking. Each is a separate acceptance test judged on repeatability,
  finite state and closed-loop response, not biological correctness.
- **v1.1 RL environment**, only after the interfaces above are validated.
- Not yet placed by the owner: Three.js visualization, plasticity, faster neural backends.

## 4. Housekeeping (independent, small; touch only with the owner's go-ahead while v0.7.x is diagnostic)

- **4.1 README contradictions.** Line ~289 says "The CNS and body are not connected yet";
  §3 says the Windows viewer "has not yet been manually verified", though
  `outputs/bridge/*viewer_live*` runs exist; §5 describes only the v0.1 test set
  ("14 passed"). Acceptance: README matches PROJECT_STATE; owner confirms the viewer status.
- **4.2 Lint.** `ruff check .` reports one I001 in `src/malecns_sim/cns/graph/query.py`;
  `ruff format --check .` would reformat 5 files. Acceptance: both commands clean.
- **4.3 Artifact naming.** v0.6 writes `sensory_spikes.parquet`, v0.7 writes
  `sensory_events.parquet` for the same concept.
- **4.4 Scheduler exception scope.** Any `RuntimeError` becomes `ABORTED_SAFETY_LIMIT`
  (DECISIONS.md D15).
- **4.5 Third-party warning.** Brian2 2.10 triggers `PyparsingDeprecationWarning`.
