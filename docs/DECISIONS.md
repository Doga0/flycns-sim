# Decisions

Architectural and scientific-policy decisions that the repository supports with evidence.
These were reconstructed on 2026-10-11 from code, configs, docs and git history. No
conversation history from the original development exists, so the **rationale is marked
"stated"** only when a file in the repo states it. Otherwise it is "inferred" or "unknown".

New decisions: append an entry with date, evidence, rationale, trade-offs.

---

### D1. Raw data is read unchanged; every selection is a declarative profile
- **Evidence:** `cns/dataset.py` docstring ("No joins, thresholds, or neuron selection");
  `cns/graph/policy.py` ("No Python expressions or implicit scientific filters");
  `configs/graph/*.yaml`.
- **Rationale (stated):** keep scientific choices auditable and separate from data access.
- **Trade-off:** more artifacts and validation code; nothing can be "quickly filtered".

### D2. `published-v1` is a provisional node set (166,700 vs 166,691)
- **Evidence:** `configs/graph/published.yaml` (`reference_membership_verified: false`,
  "Do not remove nine nodes merely to match the publication's count");
  `published-v1/manifest.json` (`difference_from_reference: 9`).
- **Rationale (stated):** no evidence identifies the nine extra nodes; matching the count by
  deletion would be fabrication.
- **Open:** the +9 discrepancy is unresolved.

### D3. Sparse COO + outgoing CSR in memory-mapped `.npy`, built by external sort
- **Evidence:** `cns/graph/build.py` (batch scan, partition sort, run merge),
  `cns/graph/store.py`; `docs/graph-core.md` "Memory and disk behavior".
- **Rationale (stated in guide):** the 151.9 M-row connection table must be aggregated into
  25.6 M edges deterministically, within limited RAM, and byte-identically across chunk
  plans (`tests/test_graph.py::test_same_inputs_byte_identical_across_chunk_plans`).

### D4. Shiu et al. LIF model on Brian2 with NumPy code objects
- **Evidence:** `neural/config.py`, `configs/neural/shiu_lif_v1.yaml`, README references
  to Shiu et al. and their `model.py`; `codeobj_class=NumpyCodeObject` throughout
  `brian2_lif.py`.
- **Rationale:** the model choice follows the published whole-brain model (stated by the
  references). Choosing NumPy over Brian2's C++/Cython codegen is **inferred** to avoid a
  compiler dependency on Windows and to keep results identical between Windows and Docker
  (byte-identical results are reported in `docs/neural-reference.md` §9). Not stated
  explicitly.
- **Trade-off:** slower full-CNS runs (minutes for 100–200 ms).
- **Backend abstraction:** `NeuralBackend` ABC exists so other backends can be added;
  only Brian2 is implemented. GPU backends are explicitly out of scope so far.

### D5. Unknown transmitters get sign 0 by default
- **Evidence:** `neural/signs.py` reason string; `unknown_nt_policy: zero`; alternatives
  `error` and `excitatory` are supported.
- **Rationale (stated):** do not assign a biological sign without evidence; histamine is
  unmapped under the six-transmitter policy. Nodes and contacts stay in the graph.
- **Trade-off:** mutes the outputs of every unlabeled neuron, which can cut paths.

### D6. Sensory and motor mappings use explicit, reviewed body IDs only
- **Evidence:** `configs/bridge/*.yaml` ("no runtime type parsing or auto-mapping");
  `bridge/mapping.py`, `bridge/sensory/mapping.py` validate each ID against annotations
  (type, side, subclass, nerve, neuromere) and the compiled body.
- **Rationale (stated, README):** "It never guesses a mapping from a type-name substring."
- **Scope:** motor-bridge-v1 maps 7 LF tibia motor neurons (5 Ti flexor, 2 Ti extensor) out of
  815 motor cells; the six-leg profile maps 49. sensory-bridge-v1 maps 23 LF/ProLN
  chordotonal cells out of 17,937 sensory cells.

### D7. Motor decoding is a labeled engineering hypothesis
- **Evidence:** `configs/bridge/lf_tibia_v1.yaml` `mechanical_assumption`;
  `bridge/decoder.py`; `bridge/controller.py` comment ("do not label these anatomical
  limits").
- **Decision:** antagonistic population-rate decoder → low-pass (τ 20 ms) → position
  target = neutral ± 0.35 rad × activity; slew ≤ 0.05 rad per 10 ms; positive pitch is
  extensor by convention. MuJoCo `ctrlrange` is set by the controller because the FlyGym
  body has no enabled tibia limits.
- **Rationale (stated):** the release has no muscle-target column and the body has no
  muscle model; bounds keep physics stable.

### D8. Sensory encoding is a labeled engineering probe
- **Evidence:** `configs/bridge/sensory_lf_tibia_v1.yaml` `model_assumption`.
- **Decision:** one shared linear positive-position → rate encoder (0–100 Hz over neutral
  to neutral + 0.35 rad); events added to `v` with 68.75 mV gain; Poisson or deterministic
  phase-accumulator modes.
- **Rationale (stated):** annotations identify the cells as chordotonal but not their
  claw/hook/club subtype, so tuning is not inferred.
- **Gain value:** 68.75 mV equals the LIF stimulus default; it is **inferred** to come from
  Shiu's Poisson input weight (250 × 0.275 mV).

### D9. Build each coupling direction separately, then combine
- **Evidence:** v0.5 (motor replay, `sensory_feedback: false` in manifest), v0.6 (sensory
  replay, `motor_body_coupling: false`), v0.7 scheduler compatibility tests
  (`motor_only`, `sensory_only` modes) proving equivalence with the offline paths.
- **Rationale (stated in guides):** validate each bridge in isolation so closed-loop
  failures can be localized.

### D10. Causal, integer-clock scheduling for the closed loop
- **Evidence:** `bridge/closed_loop/clocks.py`, `scheduler.py`, `docs/closed-loop.md`.
- **Decision:** neural and physics dt 0.1 ms, sensory/orchestration tick 1 ms, motor window
  10 ms. Physics runs under the old target during each interval; spikes in `[t−10, t)`
  take effect at `t`. Clock agreement is asserted every tick.
- **Rationale (stated):** no information may travel backward in time; follows Brian2's
  `set_spikes`-between-runs and MuJoCo's control-before-step models.
- **Trade-off:** one 1 ms sensory delay plus up to 10 ms motor latency, both imposed by the
  scheduler rather than by biology.

### D11. Acceptance gates may fail; versions stay `.dev` until they pass
- **Evidence:** `closed_loop/validate.py --require-effect`; `comparison.json`
  `acceptance_passed: false`; version `0.7.0.dev0`; README and closed-loop guide forbid
  weight changes, seed picking, motor-spike injection or relabeling motor cells.
- **Rationale (stated):** general CNS activity must not count as physical feedback.

### D12. Determinism across Windows and Docker is a requirement
- **Evidence:** `--compare` validators in every layer; `bridge/decoder.py` computes the
  filter coefficient with `Decimal` because libm `exp` can differ by one ulp between
  platforms (stated in a comment).
- **Trade-off:** continuous body values are compared with `1e-9` (closed loop) or `1e-10`
  (sensory activity) tolerance; discrete events must match exactly.

### D13. Outputs are immutable and written atomically
- **Evidence:** `bridge/store.py:stage_output`; `FileExistsError` checks in builders and
  runners; README "Output directories are never overwritten".
- **Rationale (inferred):** recorded hashes and cross-run comparisons depend on artifacts
  never changing in place.
- **Exception:** `simulation.smoke_test --headless` overwrites files in its output directory
  (documented in README §2.4).

### D14. Closed loop retains only mapped-cell spikes by default
- **Evidence:** `monitor: control_and_sensory`; `Brian2LIFBackend(spike_record_nodes=…)`
  with `total_spikes` still counting all spikes.
- **Rationale (stated):** keep memory bounded while the storm guard sees global activity.
  `monitor: full` exists for diagnostics.

### D15. Broad exception capture marks runs as safety aborts
- **Evidence:** `scheduler.py` catches `SafetyLimit`, `FloatingPointError` and **any**
  `RuntimeError` (clock drift, MuJoCo state checks) → `ABORTED_SAFETY_LIMIT` with partial
  artifacts.
- **Rationale (inferred):** keep partial evidence for any runtime failure.
- **Risk:** an unrelated `RuntimeError` (for example from Brian2) would also be reported as a
  safety abort; the `reason` string keeps the original type and message.

### D16. Dual workflow: Windows uv for development/viewer, Docker for headless reproducibility
- **Evidence:** README §2–3, `Dockerfile` (OSMesa), `docker-compose.yml` volume mounts.

### Unknown or undocumented
- Why `kp=50` position actuators and `ActuatedDOFPreset.LEGS_ACTIVE_ONLY` were chosen
  (likely FlyGym's basic example; `simulation/fly.py` docstring says it composes "the FlyGym
  basic example"). The influence of kp=50 on the v0.7 result is not analyzed.
- Why the LF tibia was chosen as the first joint (inferred: it has explicitly annotated
  Ti flexor/extensor populations and FeCO inputs on the same side and nerve).
- Why 200 ms and +0.20 rad for the v0.7 experiment.
- Whether the previous developer intended a v2 encoder/mapping policy to fix the v0.7 gate.

---

### D17. Close v0.7 as infrastructure; investigate the pathway before v0.8 (2026-10-11)
- **Decided by:** the owner (Berke), in the project chat on 2026-10-11.
- **Decision:** v0.7 is a successful closed-loop infrastructure release with a failed
  biological/functional acceptance. Version stays `0.7.0.dev0` and is never tagged `0.7.0`
  for this result. No more debugging of v0.7 to make motor spikes appear, and no new
  simulator features in v0.7.x. Next comes a diagnostic investigation (pathway audit,
  MuJoCo-free rate sweep, latency/connectivity report) whose result decides v0.8:
  physical sensory-motor validation if the pathway works under the current model, otherwise
  reviewed interface/model calibration. Walking and RL come after a validated single-leg
  interface and multi-leg embodiment.
- **Rationale (stated by the owner):** keep "does the software coupling work" (yes) separate
  from "does this sensory population + encoder + LIF parameters + motor population form a
  functional pathway" (not shown). Forcing the 0-spike result away would harm the
  project's scientific credibility; RL on top of unknown mappings would be uninterpretable.
- **Constraints:** diagnostics must not add biological assumptions; report what the current
  sign policy does; do not interpret sweep results as biology.

### D18. Pathway audit counts walks for multiplicity and enumerates short paths explicitly (2026-10-11)
- **Evidence:** `src/malecns_sim/analysis/sensory_motor_paths.py`.
- **Decision:** path multiplicity for 1–5 synapses is computed by propagating counts along
  all graph edges (walks, so a node may repeat), split into sign classes under the current
  LIF sign policy. Individual paths are listed exhaustively only up to a smaller hop count
  (default 3), because the LF tibia case has about 167 thousand walks of ≤3 synapses but
  33 million of 4 and 7.7 billion of 5. Latency is the synaptic-delay lower bound
  (hops × 1.8 ms); measured latency belongs to the activation sweep.
- **Trade-off:** walk counts overstate simple-path counts where cycles exist; the enumerated
  table flags non-simple paths so the difference is visible for short paths.
