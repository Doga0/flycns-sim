# LF tibia sensory→motor pathway: latency and connectivity report

Step 4 of the v0.7.x investigation (ROADMAP §2.3), 2026-10-12. It combines the pathway audit
and the activation sweep described in [pathway-audit.md](pathway-audit.md). All numbers come
from the current `published-v1` graph, the shiu-lif-v1 model and sign policy, the 23 cells
in `sensory_lf_tibia_v1.yaml` and the 7 cells in `lf_tibia_v1.yaml`, with seed 42. This is
a diagnostic of the model, not a biological claim.

Reproduce (the first two commands write the inputs to the third):

```powershell
uv run python -m malecns_sim.analysis.pathway_audit    --output outputs/analysis/lf_tibia_pathway
uv run python -m malecns_sim.analysis.activation_sweep --output outputs/analysis/lf_tibia_activation_sweep
uv run python -m malecns_sim.analysis.pathway_report   --output outputs/analysis/lf_tibia_pathway_report
```

The report tool checks that both inputs used the same graph and LIF config and that their
artifacts are unchanged. It writes `incoming_drive.parquet` (drive delivered to each motor
neuron per condition, split by sign, presynaptic transmitter and layer),
`active_neurons.parquet`, `pathway_report.json` and a generated `pathway_report.md`.

## The owner's question

*Given this sensory population and this motor population, under the current MaleCNS graph
and LIF model, what input is required to activate the motor population?*

| Measure | Result |
| --- | --- |
| Minimum input rate | 100 Hz nominal Poisson to all 23 cells (about 87 Hz actual per cell); 50 Hz fails. A 10-volley burst at 3 ms intervals also works |
| Active motor neurons | 2 of 7: flexors 807165 and 809912. The other 3 flexors and both extensors never fire up to 1000 Hz nominal (312 Hz actual) |
| Motor spike count | 807165 / 809912: 5 / 3 at 100 Hz, 18 / 13 at 1000 Hz (in 250 ms) |
| Motor activation latency | 51 ms at 100 Hz, 39 ms at 200 Hz, 15.5 ms at 500 Hz, 13.7 ms at 1000 Hz, 13.1 ms for the 10-volley burst (synaptic-delay lower bound 1.8 ms) |
| Path depth | Anatomically 6 of 7 are direct (1 synapse) and 819384 is 2 synapses away. Functionally, activity never travels more than 2 synapses from the sensory cells |
| Path connectivity | 11 direct pairs (all excitatory ACh); 1,052 two-synapse paths (538 excitatory, 494 net inhibitory, 20 muted); 166,452 three-synapse walks |

## Per motor neuron (strongest input, Poisson 1000 Hz)

"Delivered drive" is the sum over presynaptic spikes of contacts × model sign × 0.275 mV:
bookkeeping of what reached each neuron's synaptic variable `g`, not a voltage.

| Motor | Role | Direct contacts | Fires from | Spikes | Excitatory | Inhibitory | Net | Inhibition from GABA / glutamate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 807165 | flexor | 39 | 100 Hz | 18 | 1,969 | −194 | **+1,774** | −152 / −42 |
| 809912 | flexor | 21 | 100 Hz | 13 | 1,230 | −209 | **+1,021** | −52 / −157 |
| 818057 | flexor | 1 | never | 0 | 324 | −117 | +207 | −24 / −94 |
| 909831 | flexor | 4 | never | 0 | 263 | −63 | +201 | −5 / −57 |
| 819384 | flexor | 0 | never | 0 | 161 | −211 | −50 | −8 / −203 |
| 800636 | extensor | 4 | never | 0 | 931 | −2,393 | **−1,462** | −2,104 / −289 |
| 815344 | extensor | 5 | never | 0 | 1,562 | −5,428 | **−3,866** | −4,239 / −1,189 |

Facts from `incoming_drive.parquet`:
- **The extensors receive net inhibition**, delivered mostly by GABA-labeled neurons one
  synapse from the sensory cells (net −1,549 mV and −4,245 mV from that layer). The
  extensors' inhibition exceeds their excitation from 25 Hz upward.
- In the two active flexors, **more excitation comes through 1-hop interneurons than directly
  from the sensory cells**: for 807165, +1,222 mV via interneurons versus +650 mV direct.
- The silent flexors 818057 and 909831 receive net excitation, but about one-tenth of what
  807165 receives.
- The most active intermediate neurons are VNC interneurons one synapse from the sensory
  cells (types such as IN13A006, IN21A006, IN21A004, IN03A004, IN19A005, IN08A005). Several
  synapse directly onto the mapped motor neurons; see the generated `pathway_report.md`.

## Interpretation (inferences, not tested)

1. Under the current model the chordotonal → LF tibia pathway is **functional but
   selective**. Sensory drive excites two flexors and actively inhibits both extensors
   through local GABAergic interneurons. That gives a direction-specific pattern rather
   than co-activation. The current motor bridge's antagonistic decoder would therefore
   only ever push the joint toward flexion; that direction is a decoder convention (DECISIONS.md D7).
2. The **v0.7 closed-loop zero result** follows from input strength. The encoder gave about
   57 Hz per cell for a few milliseconds while the joint returned to neutral; the sweep
   needs about 87 Hz per cell sustained over tens of milliseconds (first spike at 51 ms).
3. The inhibition that silences the extensors is mostly GABA, so it does not hinge on the
   glutamate = −1 assumption in shiu-lif-v1. The zero-sign policy mutes only 3% of on-path
   intermediates.
4. Limits: one seed; rates between 50 and 100 Hz untested; 200 ms drives only; one
   mapped sensory population (all 23 cells driven together, no subtype tuning).

Follow-up (2026-11-09): a subset sweep and full-CNS input census explain the 5 silent motor
neurons. The flexor response comes entirely from 3 SNpp51 cells, the other 20 mapped cells
are net inhibitory, no subset excites the extensors, and most motor input comes from premotor
neurons this drive never recruits. See
[pathway-audit.md](pathway-audit.md#why-5-of-7-motor-neurons-stay-silent-2026-11-09).

## What this means for v0.8 (owner's decision)

The owner's plan sets two outcomes. The evidence supports the first, with caveats:

- **v0.8 = physical sensory-motor validation** is possible: the pathway activates under the
  current model without changing weights or signs. The open design questions are input and
  output, not the network:
  - how body state should map to sensory drive strong enough to reach about 87 Hz per cell
    (encoder range, perturbation size and duration, or literature-supported tuning);
  - whether a flexor-only response is an acceptable first physical-feedback criterion;
  - whether the decoder's population normalization (mean over 5 flexors) and the 100 Hz
    `rmax` are appropriate when only 2 of 5 flexors fire.
- **v0.8 = reviewed interface/model calibration** would be needed instead if the owner
  requires extensor activity, or a response at the v0.6 encoder's rates.

Any change becomes a new versioned profile; the v1 profiles and recorded results stay as
they are.
