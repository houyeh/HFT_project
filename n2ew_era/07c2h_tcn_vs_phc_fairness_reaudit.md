status: resolved_correction_required

# TCN vs PHC Comparison Fairness Re-Audit

## Scope

Triggered by a direct challenge: is the "PHC beats TCN" framing used in prior discussion actually
correct, or does it compare numbers at different verification tiers / omit standard optimization
techniques that would be normal practice for TCN? This re-audit checks source code and existing
reports directly rather than relying on recollection. No retraining, no NEW5, no new HLS runs.

## Finding 1 (correction): at matched verification tier, TCN is not slower than PHC's headline number

Prior discussion compared PHC's fully-verified headline (`v8`, 8,316 cycles / 83.16us, routed
P&R + cosim) against TCN's **unverified, csynth-only** round-3 speed-optimization result (5,036
cycles / 50.36us, cosim not completed) and concluded PHC was ahead. That comparison mixes
verification tiers and is not the fair one.

The fair, matched-tier comparison already existed before any of the round 1-4 speed-optimization
work: `07c2d_tcn_streaming_iface_slim/AUDIT_tcn_streaming_iface_slim_pnr.md`, "Selected Closure
Points" table — TCN `distilled_shared_u4`, **7,438 cycles / 74.38us at 100MHz, csim + csynth +
1008-event cosim + routed P&R met**. Same target part (`xc7z020clg400-1`), same fixed-point
precision (`ap_fixed<24,12>` data / `ap_fixed<40,16>` accumulator, identical to PHC's
`phc_fixed_t`/`phc_acc_t` typedefs — confirmed by direct diff, so precision is not an unfair
variable), same 100MHz target clock.

At this matched tier: **TCN 74.38us < PHC v8 83.16us.** TCN's fully-verified number is
approximately 10.5% faster than PHC's fully-verified headline number, not slower.

## Finding 2: the speed advantage is not free — resource cost

| metric (routed P&R) | PHC v8 | TCN distilled_shared_u4 (07C2D, 100MHz) | ratio |
|---|---:|---:|---:|
| latency | 83.16us | 74.38us | TCN 1.12x faster |
| DSP | 20.9% (46/220) | 49.09% | TCN 2.3x more |
| LUT | 14.2% | 32.00% | TCN 2.3x more |
| FF | 13.6% | 14.44% | roughly matched |
| BRAM_18K | 2.1% (6/280) | 23.57% | TCN 11.2x more |

The defensible claim is not "PHC is faster than TCN." It is: **PHC reaches comparable (in fact
very slightly slower) matched-tier latency using a fraction of the silicon** — roughly 2x less
DSP/LUT and 11x less BRAM. This is a resource-efficiency finding, not a raw-speed finding, and it
is a stronger, more defensible claim for a design-quality/difficulty rubric than "PHC is faster."

## Finding 3: PHC's real speed advantage comes from variants TCN has no matched-tier answer to

PHC has additional fully-P&R+cosim-verified variants that go faster than v8: `unroll16`
(2,316 cycles / 25.733us, actual burned-board variant) and a diagonal-SSM `V3` variant (530
cycles / 5.3us, full P&R+cosim, but disclosed weaker signal quality, not adopted as the main
narrative). TCN has no equivalent matched-tier (routed P&R + cosim) variant faster than 74.38us —
its only faster number (50.36us) is csynth-only, cosim incomplete. **This asymmetry is real and
should be stated as "PHC has more verified fast variants," not as "PHC is faster than TCN."**

## Finding 4: one standard TCN-acceleration technique was never attempted — but it targets a different metric

Checked directly (`grep -rl DATAFLOW` across all `07c2*/hls_src/*.hpp` and `*.cpp`): **zero
matches.** Task-level pipelining across the 5 TCN blocks via `#pragma HLS DATAFLOW` (streaming
blocks via FIFO/ping-pong channels so block N processes event t while block N-1 processes event
t+1) was never attempted in any round. This is standard practice for exactly this kind of
block-sequential streaming architecture and is uniquely available to TCN's feedforward structure
(PHC's recurrent single state cannot decouple in the same way). It was not attempted because (a)
it is a materially larger, riskier restructuring than anything tried in rounds 1-4, given each
block already carries persistent history-buffer state, and (b) as established in the prior
turn's discussion, this lever improves **interval/throughput**, not **latency** — the metric
actually used in every PHC-vs-CPU and PHC-vs-TCN comparison in this project. So this is a real,
disclosable gap in what was attempted, but correcting it would not change the 74.38us vs 83.16us
latency comparison above.

## Finding 5: the two numbers are explicitly flagged by prior work as not strictly the same "latency object"

Both `07c2d_tcn_streaming_iface_slim/AUDIT_tcn_streaming_iface_slim_pnr.md:58` and
`07c2_tcn_streaming_closure/AUDIT_tcn_streaming_closure.md:7` independently state: "[TCN]
latency/interval ... should not be ranked directly against PHC recurrent single-step latency."
Investigated why: both `tcn_streaming_step` and `phc_step` are one-call-per-incoming-event
functions that produce one decision per call in steady state (after a short, one-time ring-buffer
warm-up on TCN's side that does not recur per-decision), so structurally the cycle counts do
measure the same kind of thing — "cycles to turn one new market event into one decision." No
deeper mechanism explaining a hidden non-equivalence was found beyond general caution appropriate
whenever comparing cycle counts across two structurally different architectures. This should be
carried into the paper as an explicit disclosed caveat (the comparison is informative but not a
fully controlled ablation, e.g. FLOPs/decision are not matched), rather than silently dropped or
silently treated as fully equivalent.

## Corrected Claim For The Paper/Report

Replace "PHC is faster than TCN" with: **"At matched full verification tier (routed P&R +
1008-event RTL cosim, same board, same clock, same fixed-point precision), PHC's conservative
headline variant (83.16us) and TCN's best closed variant (74.38us) are approximately tied on
latency, with TCN marginally faster; PHC achieves this at roughly 2-11x lower resource
utilization across DSP/LUT/BRAM. PHC's larger apparent speed advantage comes from additional
fully-verified fast variants (25.7us burned-board, 5.3us P&R+cosim-verified diagonal-SSM) that
TCN has no matched-tier equivalent for, not from TCN being architecturally slower at equal
effort. TCN's own further speed-optimization line (rounds 1-4) reached 50.36us at csynth level
but has not yet passed cosim, so it cannot yet be cited at the same confidence tier."**

## What This Changes

- Task 1 of this audit was launched skeptically ("is the PHC-beats-TCN framing missing a fair
  comparison") and the answer is yes, partially: the framing overstated PHC's speed advantage by
  comparing across verification tiers. The corrected, resource-efficiency framing is honest,
  still favorable to PHC, and more defensible under scrutiny than the original.
- Nothing here changes the underlying verified numbers themselves — only which of them should be
  compared against which, and how the resulting claim should be worded.
