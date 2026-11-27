# Roadmap

Ordered by priority and dependency. The v0.8 scope was set by the owner on
2026-11-26 after the v0.7.x pathway investigation. Keep this file in sync with
[PROJECT_STATE.md](PROJECT_STATE.md).

```text
v0.7    closed-loop infrastructure        (closed; stays 0.7.0.dev0, never tagged 0.7.0)
v0.7.x  functional pathway investigation  (complete; findings in pathway-audit.md)
v0.8.1  physical feedback validation       (complete; five seeds, Docker, full tests)
v0.8.2  sensory interface characterization (anatomy and tuning evidence first)
v0.9    multi-joint and contact/load bridge
v0.10   six-leg embodiment
v0.11   functional behavior benchmarks
v1.0    general experimental platform
v1.x    learning and performance work
```

## 1. v0.7: closed

Infrastructure is done and the real-data physical-feedback acceptance failed (see
PROJECT_STATE). Owner rules: keep `0.7.0.dev0`, do not tag `0.7.0`, and do not keep
debugging v0.7 to produce motor spikes.

**1.1 Persist the v0.7 work:** done. The owner committed it to `main` on 2026-10-11 (no tag).

## 2. v0.7.x: functional pathway investigation (complete)

The question: *given the 23 LF/ProLN chordotonal cells and the 7 mapped LF tibia motor
neurons, under the current MaleCNS graph and shiu-lif-v1 model, what input is required to
activate the motor population?* This must keep anatomical connectivity (does a path exist)
separate from functional connectivity (does it fire in this LIF model). No new biological
assumptions, no v1 profile edits, and results are diagnostics, not biology.

**2.1 Pathway audit** (done 2026-10-11, committed in `f0ed454`; results in [pathway-audit.md](pathway-audit.md)). `python -m malecns_sim.analysis.pathway_audit`.
- Paths of 1 to 5 synapses from the sensory to the motor population, with source, target,
  hops, contacts, transmitter and model sign per neuron, type, side, intermediate
  population, multiplicity and synaptic-delay latency.
- Outputs: `pathway_summary.json`, `nodes.parquet`, `edges.parquet`, `paths.parquet`,
  `report.json`, in a new output directory with provenance.
- Acceptance: reproducible from the CLI; tested on a synthetic graph; it reports what the
  current sign policy does to each path class without changing it.

**2.2 Functional activation sweep** (done 2026-10-12, committed in `f0ed454`; `analysis.activation_sweep`, results in pathway-audit.md).
Full LIF model, no MuJoCo. Drive the sensory cells directly: a single spike, volleys
(for example 10 spikes) and rates such as 10, 25, 50, 100, 200, 500 and 1000 Hz. Record each
mapped motor neuron's spike count and first-spike latency, plus activity per hop layer from
2.1.
- Acceptance: a rate × motor-neuron matrix and a latency table with provenance; one of
  "activates at some tested input" (report the minimum) or "no activation at any tested
  input" is stated explicitly.

**2.3 Latency and connectivity report** (done 2026-10-12, committed in `56be07e`; [lf-tibia-pathway-report.md](lf-tibia-pathway-report.md)). Direct contacts, path
counts per depth, transmitter mix, latency and motor activation in one document, added to
`docs/` and summarized in PROJECT_STATE.

**2.4 Decide v0.8 scope** (done 2026-11-26). The owner chose physical validation under
the existing v1 neural and bridge profiles, followed by sensory characterization.
The robustness sweep is documented in [pathway-audit.md](pathway-audit.md).

## 3. v0.8.1: physical feedback validation (complete 2026-11-27)

- Add time-only step and ramp-hold-release joint torque profiles, resolved against the
  named MuJoCo hinge. Keep the same load across all four conditions.
- Add `motor_clamp`: sensory encode and CNS injection, motor decoding, no actuation.
  `feedback_off` and `sensory_clamp` retain their original meanings.
- Calibrate a safe load with physics-only sweeps. Run the full MaleCNS with seeds 42–46;
  measure mapped motor spikes, decoded commands, causal body effect and recovery.
- Engineering gate requires a numeric effect above validator tolerance and repeatability.
  An effect of 0.01 rad is a provisional functional target, recorded separately.
- Compare Windows and Docker traces. Failure must be reported as a result, never
  repaired with seed selection or a changed v1 encoder, decoder, graph or LIF profile.

The five-seed report is under `outputs/closed_loop/lf_tibia_load_v2_seed_report_2026-11-27/`:
all five engineering gates and provisional 0.01 rad targets passed. Windows/Docker
seed-42 discrete events and commands matched exactly; body tolerance `1e-9` passed.
Windows and Docker full suites each passed 139 tests. See [closed-loop.md](closed-loop.md).

## 4. v0.8.2 and later stages

- **v0.8.2 sensory characterization:** audit the 23 mapped cells, especially SNpp51,
  SNpp50 and SNpp39. Literature and annotations must support subtype or tuning claims;
  functional motor effects alone do not establish position selectivity. New encoders
  require separate versioned profiles.
- **v0.9 multi-joint and load/contact sensing:** after single-joint validation and
  reviewed sensory populations.
- **v0.10 six-leg embodiment:** simultaneous neural and body channels.
- **v0.11 behavior benchmarks:** resistance, posture and coordination; no assumed gait.
- **v1.0 experimental platform:** validated interfaces and benchmark API.
- **v1.x learning/performance:** only after interfaces and behaviors are interpretable.

## 5. Housekeeping (independent of v0.8 acceptance)

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
