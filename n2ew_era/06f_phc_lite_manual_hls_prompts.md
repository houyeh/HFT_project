# 06f — PHC-lite Manual HLS C++ : Coding Agent Prompts

Run these five prompts **in order**. Each one assumes the previous step's output files exist.
Create a new directory `n2ew_era/06f_phc_lite_manual_hls/` for all outputs (mirrors the 06a/06e layout).

Hard constraints that apply to ALL five prompts (paste into every prompt or keep visible):
- NEW5 is locked: do not use it, do not retrain, do not retune thresholds.
- Do not claim "FPGA improves PnL", "causal-fill proxy = true exchange replay",
  "PHC-lite = full PHCSSM", "STP-like gain = full biological STP", or
  "PYNQ-Z2 = production HFT deployment".
- The adopted model is frozen: `05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt`
  (PHCLiteSTP, seed=2). Do not alter its weights.

---

## PROMPT 1 — Export trained weights to a fixed-point C++ header

```
You are working in C:\Users\holbo\00VS_projjects\00_HFT_project. Create a new
directory n2ew_era\06f_phc_lite_manual_hls\ for all outputs of this phase
(subfolders: models, results, hls_src).

GOAL: Export the trained weights of the adopted PHC-lite step model into a
synthesizable C++ header of fixed-point constants, ready for Vitis HLS.

CONTEXT / WHAT ALREADY EXISTS (read these first, do not re-derive from scratch):
- Source checkpoint: n2ew_era\05c4_phc_stp_cosine_lr\models\phc_stp_v2_seed2_best_val.pt
  (PHCLiteSTP, seed=2 — this IS the adopted/frozen model, do not retrain it)
- Step-form model class: src\models\phc_stp_step_hls.py (PHCSTPStepHLS) — already
  proven numerically equivalent to the source model (max_abs_diff = 3.87e-7,
  see n2ew_era\06e_phc_stp_step_hls4ml\PHC_STP_STEP_HLS4ML_FINAL_REPORT.md)
- Migration helper: n2ew_era\06e_phc_stp_step_hls4ml\common_06e.py has
  load_step_model() / migrate_weights() that already produces
  n2ew_era\06e_phc_stp_step_hls4ml\models\phc_stp_step_hls_seed2.pt — reuse
  that checkpoint (do not re-migrate weights; load this .pt directly).
- IMPORTANT alpha note: the model uses RAW alpha (NOT sigmoid(alpha)) as a
  fixed scalar buffer. Confirm the loaded buffer "alpha_scalar" is the raw
  value (~0.5917 per the 06e migration audit), not a sigmoid-transformed one.
- Fixed-point precision: use Q24.12 == ap_fixed<24,12> (24 total bits, 12
  fractional bits). This exact scheme was already validated for PHC-lite in
  n2ew_era\06b_phc_lite_streaming_fixed_point_equivalence\PHC_STREAMING_FIXED_POINT_EQUIVALENCE_REPORT.md
  (action_agreement_rate ~0.998, max PnL delta ~2455 against multi-million
  baseline PnL — i.e. negligible). Reusing this exact scheme means you can
  cite that report's evidence directly; do not invent a new bit-width.

WHAT TO BUILD — a script n2ew_era\06f_phc_lite_manual_hls\00_export_weights.py that:
1. Loads n2ew_era\06e_phc_stp_step_hls4ml\models\phc_stp_step_hls_seed2.pt via
   PHCSTPStepHLS (feature_dim=16, hidden_dim=16).
2. For every learnable parameter AND buffer in the model
   (input_x, input_state, enc_linear, enc_norm [weight=gamma, bias=beta],
   gate, candidate, gain_from_encoded, gain_from_prev_state, synapse,
   loop_norm [gamma, beta], readout, output_pack, decay buffer [16],
   alpha_scalar buffer [1]):
   - Quantize each float32 value to Q24.12 fixed-point: round(value * 2^12),
     clip to the representable int24 range [-2^23, 2^23 - 1], and record how
     many values (if any) saturated/clipped — this is an important thing a
     reviewer will ask about, so report it explicitly per tensor.
   - Emit the values as a C++ header n2ew_era\06f_phc_lite_manual_hls\hls_src\phc_weights.hpp
     using `static const ap_fixed<24,12>` 1D/2D arrays with clear names
     matching the PyTorch attribute names (e.g. ENC_LINEAR_W[16][16],
     ENC_LINEAR_B[16], ENC_NORM_GAMMA[16], ENC_NORM_BETA[16], DECAY[16],
     ALPHA_SCALAR, ...). Include a generated-by comment header with source
     checkpoint path, seed, and quantization scheme.
   - NOTE: input_x, input_state, output_pack are fixed routing/identity
     matrices (not learned in the usual sense — they were frozen at
     construction in PHCSTPStepHLS._init_fixed_plumbing). You may either (a)
     export them as constants too for completeness, or (b) implement the
     slicing/concatenation they perform directly in C++ without a matmul —
     the latter is more HLS-resource-efficient and is the recommended
     approach; if you choose (b), document in the report why these three
     layers are omitted from the exported weight set.
3. Writes a migration/quantization audit CSV to
   n2ew_era\06f_phc_lite_manual_hls\results\weight_export_audit.csv with one
   row per tensor: name, shape, float_min, float_max, quantized_min,
   quantized_max, num_saturated, frac_bits, total_bits.
4. Writes a short markdown report
   n2ew_era\06f_phc_lite_manual_hls\PHC_WEIGHT_EXPORT_REPORT.md summarizing:
   the source checkpoint, the Q24.12 scheme and why it was chosen (cite the
   06b report numbers), the full audit table, and a clear PASS/FAIL line on
   whether ANY values saturated during quantization (if any did, flag this
   prominently — it would materially weaken the equivalence claim and must
   be investigated before proceeding).

Do not run any training, do not touch NEW5, do not modify phc_stp_step_hls.py.
Print the final audit table to stdout when done.
```

---

## PROMPT 2 — Generate Python-reference test vectors for the HLS testbench

```
You are working in C:\Users\holbo\00VS_projjects\00_HFT_project, directory
n2ew_era\06f_phc_lite_manual_hls\ already exists from a prior step (created
models/, results/, hls_src/ subfolders).

GOAL: Produce a CSV of (input -> expected_output) pairs from the frozen
PHC-lite step model, in float32, to serve as ground truth for an HLS C++
testbench — mirroring how n2ew_era\06a_hls_queue_rule_baseline\01_prepare_and_validate_queue_rule.py
produced queue_rule_test_vectors.csv for the queue-rule baseline.

CONTEXT (read first):
- n2ew_era\06e_phc_stp_step_hls4ml\common_06e.py contains a function
  verify_equivalence() that already does almost exactly this: it loads the
  step model via load_step_model(), loads validation sequences via
  p05a.load_sequences("val") (from n2ew_era\05a_phc_lite_microstructure_sequence_feasibility\common_05a.py),
  and replays the recurrence step-by-step:
      state = zeros(n, 16)
      for t in range(seq_len):
          x_cat = cat([x[:, t, :], state], dim=-1)   # shape (n, 32)
          out = step(x_cat)                           # shape (n, 18)
          readout, state = step.split_output(out)
  Reuse this exact loop structure and these exact loaders — do not write a
  new data pipeline. Import common_06e and common_05a directly (add their
  directories to sys.path the same way common_06e.py does).
- HARD RULE: verify the loaded payload does NOT have new5_used=True before
  proceeding (raise if it does) — common_06e.verify_equivalence already shows
  the correct guard pattern, copy it.

WHAT TO BUILD — a script n2ew_era\06f_phc_lite_manual_hls\01_generate_test_vectors.py that:
1. Loads the step model (load_step_model from common_06e) and the "val" split
   sequences (p05a.load_sequences("val")).
2. Replays the step recurrence over AT LEAST the first 200 sequences for the
   FULL sequence length available (check x.shape[1] — likely 128 timesteps;
   use all of them, do not truncate). This should yield on the order of tens
   of thousands of (input, output) rows — large enough to be a meaningful
   HLS cosim corpus (06a used ~405k rows; aim for at least 10k-50k here,
   adjust sample_count upward if the resulting CSV is too small, downward if
   it would make C/RTL cosimulation impractically slow — note that cosim runs
   at roughly 1 row per several seconds of wall-clock based on the 06a run,
   which took ~7.5 minutes for 405k rows of TRIVIAL combinational logic; a
   recurrent model with LayerNorm/sigmoid/tanh will be far slower per row, so
   pick a number you can actually finish — recommend starting with 2,000-5,000
   rows and noting in the report that this is a subsample, not the full split).
3. For every (timestep, sample) pair, write one CSV row with columns:
   row_id, sample_idx, timestep,
   x_t_0..x_t_15 (16 input features, float32, full precision),
   state_prev_0..state_prev_15 (16 values, float32),
   ref_readout_0, ref_readout_1 (2 values, float32 — the Python step-model output),
   ref_state_next_0..ref_state_next_15 (16 values, float32)
   to n2ew_era\06f_phc_lite_manual_hls\results\phc_step_test_vectors.csv
   Use enough decimal precision in the float formatting (e.g. %.9g) that
   round-tripping through the CSV does not itself introduce error larger than
   float32 epsilon — this matters because the HLS testbench will treat these
   as ground truth.
4. Cross-check: also run the FULL sequence model (the 05c4 PHCLiteSTP source,
   not the step model) on the same input sequences and confirm its final-step
   readout matches the step-replay's final readout within the existing
   equivalence tolerance (atol=1e-4, same as 06e) — this re-confirms the step
   model is still equivalent before you build a whole HLS pipeline on top of
   it. Abort with a clear error if this check fails.
5. Write a short report n2ew_era\06f_phc_lite_manual_hls\PHC_TEST_VECTOR_REPORT.md
   stating: how many sequences/timesteps/total rows were used and why, the
   equivalence cross-check result, the CSV schema, and the file path/size.

Do not touch NEW5. Print row counts and the equivalence cross-check result to stdout.
```

---

## PROMPT 3 — Hand-write the PHC-lite step forward pass in HLS-synthesizable C++

```
You are working in C:\Users\holbo\00VS_projjects\00_HFT_project. The directory
n2ew_era\06f_phc_lite_manual_hls\hls_src\ already contains phc_weights.hpp
(generated weight constants in ap_fixed<24,12>) from a prior step.

GOAL: Hand-write a synthesizable Vitis HLS C++ implementation of one PHC-lite
recurrent step — the same computation as src\models\phc_stp_step_hls.py's
PHCSTPStepHLS.forward(), but in fixed-point C++ instead of PyTorch. This
bypasses hls4ml entirely (hls4ml v1.3.0 cannot convert this model: its
LayerNorm parser requires 3D (batch,seq,features) input and this model uses
2D (batch,features) — see n2ew_era\06e_phc_stp_step_hls4ml\PHC_STP_STEP_HLS4ML_FINAL_REPORT.md
section 4 for the exact failure). Manual C++ -> Vitis HLS is the proven
fallback path (see n2ew_era\06a_hls_queue_rule_baseline for a working example
of the exact same flow on a simpler design — read queue_rule_hls.cpp/.hpp for
the coding-style/pragma conventions used in this project, e.g. #pragma HLS
PIPELINE II=1, #pragma HLS INLINE).

THE COMPUTATION TO IMPLEMENT (copy this faithfully from
src\models\phc_stp_step_hls.py forward(); feature_dim=16, hidden_dim=16):

  Inputs:  x_t[16]        (current causal features)
           state_prev[16] (previous recurrent state)
  Outputs: readout[2]     (bid_score, ask_score)
           state_next[16] (next recurrent state, to be fed back externally)

  encoded_pre  = LayerNorm_enc( Linear_enc(x_t) )                 // gamma,beta = enc_norm
  encoded      = encoded_pre * sigmoid(encoded_pre)                // SiLU/Swish
  gate_t       = sigmoid( Linear_gate(encoded) )
  candidate_t  = tanh( Linear_candidate(encoded) )
  state_temporal = decay[i] * state_prev[i] + gate_t[i] * candidate_t[i]   // elementwise, decay is a per-channel buffer
  gain_t       = sigmoid( Linear_gain_from_encoded(encoded)
                          + Linear_gain_from_prev_state(state_prev) )      // gain_from_prev_state has NO bias
  message_pre  = Linear_synapse(state_temporal)
  message      = message_pre * sigmoid(message_pre)                // SiLU
  msg_gated[i] = gain_t[i] * message[i]
  z            = LayerNorm_loop( state_temporal[i] + alpha_scalar * msg_gated[i] )   // alpha_scalar is RAW (not sigmoid)
  readout      = Linear_readout(z)        -> 2 outputs
  state_next   = z                          (PHCSTPStepHLS packs z directly into the
                                             second half of its output via output_pack,
                                             which is a frozen identity mapping — so
                                             state_next IS z; you do not need to apply
                                             a learned transform here, just emit z)

  Linear(v) means: out[j] = bias[j] + sum_i( weight[j][i] * v[i] )   (standard nn.Linear)
  LayerNorm(v) means: mu = mean(v); var = mean((v-mu)^2);
                      out[i] = gamma[i] * (v[i]-mu) / sqrt(var + eps) + beta[i]
                      (PyTorch default eps = 1e-5 — confirm this matches enc_norm/loop_norm
                      in the source model; if the checkpoint stored a different eps, use that)

WHAT TO BUILD, in n2ew_era\06f_phc_lite_manual_hls\hls_src\:

1. phc_step_hls.hpp — type definitions and function prototype:
   - typedef ap_fixed<24,12> phc_fixed_t;  (matches the weight quantization)
   - Consider a wider accumulator type for sums/dot-products to avoid
     intermediate overflow, e.g. ap_fixed<40,16> or similar — document your
     choice and why (this is exactly the kind of detail a reviewer will probe:
     "how did you prevent intermediate overflow in the dot products?").
   - void phc_step(const phc_fixed_t x_t[16], const phc_fixed_t state_prev[16],
                   phc_fixed_t readout[2], phc_fixed_t state_next[16]);

2. phc_step_hls.cpp — the implementation:
   - #include phc_weights.hpp for the constant arrays
   - Write small reusable helper functions for: linear_16_16, linear_16_2,
     layernorm_16, silu_16, sigmoid_16, tanh_16 (apply elementwise) — keep
     them INLINE-able (#pragma HLS INLINE) so the top function pipelines cleanly
   - For sigmoid/tanh: use the Vitis HLS math library (#include <hls_math.h>,
     hls::exp, or check whether hls4ml-style hls::sigmoid/hls::tanh helpers
     are available in this Vitis version) OR implement a piecewise-linear /
     lookup-table approximation. Pick ONE approach, implement it, and clearly
     document the approximation error you expect/measure — this WILL be
     scrutinized ("how accurate is your sigmoid approximation, and how does
     that compound across two activations and two LayerNorms per step?").
   - For LayerNorm's 1/sqrt(variance+eps): use hls::sqrt / reciprocal, or a
     lookup table over the expected variance range — again, document the
     choice and its error characteristics.
   - Add #pragma HLS PIPELINE II=1 and #pragma HLS ARRAY_PARTITION on the
     weight arrays / intermediate vectors as appropriate to get a sensible
     pipelined design (look at how queue_rule_hls.cpp uses pragmas as your
     style reference, but expect this design to need MORE pragma tuning since
     it has real arithmetic, unlike the trivial comparator in 06a).

3. A short design-notes file n2ew_era\06f_phc_lite_manual_hls\PHC_HLS_DESIGN_NOTES.md
   documenting: the fixed-point types chosen and why, the
   sigmoid/tanh/layernorm approximation strategy and its expected/measured
   error, the accumulator widths chosen to avoid overflow, and a list of every
   #pragma applied and its purpose. This file is what you'll lean on when
   someone asks "why does your design look the way it does."

Do not write a testbench yet (that's the next step). Do not touch NEW5.
Make sure the code compiles with a plain g++ -c first (catch syntax errors
cheaply before invoking Vitis HLS) — note in your notes file that you did this
sanity check.
```

---

## PROMPT 4 — Write the HLS testbench and run the full Vitis HLS flow

```
You are working in C:\Users\holbo\00VS_projjects\00_HFT_project. The directory
n2ew_era\06f_phc_lite_manual_hls\ contains:
  - hls_src\phc_weights.hpp, phc_step_hls.hpp, phc_step_hls.cpp  (from prior steps)
  - results\phc_step_test_vectors.csv  (Python reference vectors, from prior step)

GOAL: Write a C++ testbench that feeds the reference vectors through phc_step(),
compares fixed-point output against the float32 Python reference within a
documented tolerance, and run the complete Vitis HLS flow (C-sim, C-synthesis,
C/RTL co-simulation) targeting the PYNQ-Z2 (xc7z020clg400-1).

CONTEXT — copy the proven working pattern from n2ew_era\06a_hls_queue_rule_baseline:
- queue_rule_testbench.cpp shows the CSV-driven compare-and-count pattern
  this project uses (reads CSV, calls the HLS function, writes a result CSV
  with a match column, prints "total=N mismatches=M" at the end, returns
  nonzero on any mismatch). Follow the same structure and the same final
  stdout line format ("total=... mismatches=...") because that exact string
  is what the Vitis cosim log searches for / what prior reports parsed.
- run_hls.tcl in that same directory shows a working Tcl synthesis script
  (though note: in practice the project ended up using the Vitis Unified IDE's
  GUI-driven HLS Component flow with an hls_config.cfg file rather than the
  classic vitis_hls -f run_hls.tcl flow — check whether
  n2ew_era\06a_hls_queue_rule_baseline\hls_component\hls_config.cfg exists and
  use IT as your template for the new component's config, since that's the
  flow that was actually exercised successfully on this machine).
- IMPORTANT — read n2ew_era\06a_hls_queue_rule_baseline\hls_component\hls_config.cfg
  directly; it is the VERIFIED-WORKING template (cosim PASSED 405,528/405,528
  vectors using exactly this config). Its content is, in full:
      part=xc7z020clg400-1
      [hls]
      flow_target=vivado
      package.output.format=ip_catalog
      package.output.syn=1
      syn.top=queue_rule_action
      syn.file=../queue_rule_hls.cpp
      syn.file=../queue_rule_hls.hpp
      tb.file=../queue_rule_testbench.cpp
      clock=10ns
      csim.clean=0
      sim.O=0
      csim.argv=C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06a_hls_queue_rule_baseline/vectors/queue_rule_test_vectors.csv C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06a_hls_queue_rule_baseline/results/csim_out.csv
      cosim.argv=C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06a_hls_queue_rule_baseline/vectors/queue_rule_test_vectors.csv C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06a_hls_queue_rule_baseline/results/csim_out.csv
  THE CORRECT MECHANISM FOR PASSING TESTBENCH ARGUMENTS IS: plain
  `csim.argv=<abs_input_csv> <abs_output_csv>` and
  `cosim.argv=<abs_input_csv> <abs_output_csv>` keys directly in the `[hls]`
  section — ABSOLUTE Windows paths, forward slashes, space-separated
  argv[1]/argv[2]. Build the new component's cfg by copying this file's
  structure verbatim and substituting: syn.top=phc_step, the new source/
  testbench filenames, and paths under n2ew_era\06f_phc_lite_manual_hls\
  (e.g. .../06f_phc_lite_manual_hls/results/phc_step_test_vectors.csv and
  .../06f_phc_lite_manual_hls/results/phc_step_csim_out.csv).
  Two WRONG forms were tried earlier on 06a and must NOT be repeated:
    (a) `tb.sim_argv=...` — not a recognized key; silently skipped with a
        "Skipping unknown config ini entry" warning, leaving the testbench
        with no argv and producing a usage error.
    (b) `csim.setup=1` — makes Vitis SKIP execution entirely ("Skipping
        execution of C Simulation due to 'setup' option"); only compiles,
        never runs. Omit this key altogether (the working cfg has no
        csim.setup line at all).
  Note also `csim.clean=0` (not 1) in the proven cfg — match it unless you
  have a specific documented reason to diverge.
  After running, VERIFY the log contains an actual "total=N mismatches=M"
  line from the testbench — "CSim file generation done with 0 errors" alone
  only proves it compiled, not that it executed against real data.
- syn.top in hls_config.cfg must be the top function name: `syn.top=phc_step`
  — omitting it caused a "Top function not specified" error in the 06a run.

WHAT TO BUILD:

1. n2ew_era\06f_phc_lite_manual_hls\hls_src\phc_step_testbench.cpp:
   - argv[1] = input vectors CSV path, argv[2] = output results CSV path
     (same convention as queue_rule_testbench.cpp)
   - Parses each row of phc_step_test_vectors.csv (16 x_t + 16 state_prev
     floats as input; 2 readout + 16 state_next floats as Python reference)
   - Quantizes the float inputs to phc_fixed_t (ap_fixed<24,12>) — this models
     what a real streaming pipeline would feed the core
   - Calls phc_step(...)
   - Compares each of the 18 output values against the Python float32
     reference. CHOOSE AND JUSTIFY a tolerance (this is fixed-point vs fp32,
     so exact equality is impossible — recommend something like
     abs_diff < 2^-12 * K for small K, i.e. a few ULPs of the Q24.12 format,
     OR reuse a tolerance comparable to the one 06b validated functional
     impact at — document your reasoning either way)
   - Writes a per-row results CSV (row_id, all values, per-output abs_diff,
     a pass/fail flag) to the output path
   - Prints "total=<N> mismatches=<M>" to stdout at the end (M = rows where
     ANY of the 18 outputs exceeded tolerance), matching the 06a convention,
     and returns 0 if M==0 else 1

2. Set up the HLS component (mirror n2ew_era\06a_hls_queue_rule_baseline\hls_component\
   structure): create hls_config.cfg with
     part = xc7z020clg400-1
     flow_target = vivado
     clock = 10ns
     syn.top = phc_step
     syn.file = ../hls_src/phc_weights.hpp
     syn.file = ../hls_src/phc_step_hls.hpp
     syn.file = ../hls_src/phc_step_hls.cpp
     tb.file  = ../hls_src/phc_step_testbench.cpp
   (adjust relative paths to match wherever the component directory actually
   sits — check the 06a hls_config.cfg for the exact relative-path convention
   used there).

3. Run, in order, and capture/save each report:
   a. C SIMULATION — verify it actually executes (see warning above) and
      capture the "total=N mismatches=M" line. If M > 0, do NOT proceed to
      synthesis — instead investigate: is it a tolerance problem, a fixed-point
      type/overflow problem, a sigmoid/tanh/layernorm approximation problem,
      or a weight-quantization problem (check the saturation count from
      Prompt 1's audit)? Iterate on phc_step_hls.cpp / tolerances until
      mismatches=0 or you have a well-understood, documented residual error
      rate — do not silently raise the tolerance to make failures disappear.
   b. C SYNTHESIS — save n2ew_era\06f_phc_lite_manual_hls\results\phc_step_csynth.rpt
      (latency, II, LUT/FF/DSP/BRAM utilization, timing slack)
   c. C/RTL CO-SIMULATION — save n2ew_era\06f_phc_lite_manual_hls\results\phc_step_cosim.rpt
      and the full cosim log (which contains the authoritative
      "total=N mismatches=M" / "*** C/RTL co-simulation finished: PASS/FAIL ***"
      line — this is the gold-standard verification, not the C-sim one)

4. Write n2ew_era\06f_phc_lite_manual_hls\PHC_HLS_SYNTHESIS_REPORT.md with:
   - Whether C-sim, C-synthesis, and C/RTL cosim each passed
   - The exact latency/II/LUT/FF/DSP/BRAM numbers and timing slack from synthesis
   - The exact total/mismatches numbers from cosim (the authoritative ones)
   - A clear statement of what tolerance was used and why, and the measured
     max/mean absolute error between fixed-point-RTL and float32-Python output
   - An explicit "post-synthesis estimate vs post-implementation (place &
     route)" caveat — note that these numbers have NOT gone through Vivado
     Implementation, so timing slack in particular could change
   - All required protocol-boundary disclaimers (NEW5 locked, no FPGA-improves-
     PnL claim, not a PYNQ-Z2 deployment claim, fixed-point != fp32 claim, etc.)

This step may take a long time to run (cosim of a recurrent model with
LayerNorm/sigmoid/tanh over thousands of rows will be much slower than the
06a queue-rule's ~7.5 minutes for 405k trivial-logic rows) — budget accordingly
and consider running cosim on a smaller subsample first to sanity-check before
committing to the full corpus.
```

---

## PROMPT 5 — Build the final queue_rule vs PHC-lite comparison

```
You are working in C:\Users\holbo\00VS_projjects\00_HFT_project. Two HLS
designs have now been fully synthesized and verified:
  - n2ew_era\06a_hls_queue_rule_baseline  (rule-based baseline — DONE)
  - n2ew_era\06f_phc_lite_manual_hls      (ML model — just completed)

GOAL: Produce a single comparison report that a competition reviewer could
read end-to-end, contrasting hardware cost/latency of the rule-based baseline
vs. the adopted ML model on the same target FPGA (xc7z020clg400-1 / PYNQ-Z2).

WHAT TO BUILD — n2ew_era\06f_phc_lite_manual_hls\PHC_VS_QUEUE_RULE_FPGA_COMPARISON.md
containing:

1. A table with these rows for BOTH designs (pull exact numbers from
   n2ew_era\06a_hls_queue_rule_baseline\hls_component\hls_component\hls\syn\report\queue_rule_action_csynth.rpt
   and n2ew_era\06a_hls_queue_rule_baseline\hls_component\hls_component\hls\sim\report\queue_rule_action_cosim.rpt
   for the queue-rule side, and the new reports from 06f for the PHC-lite side):
   - Latency (cycles, and ns at the 10ns/100MHz clock)
   - Initiation Interval (II)
   - LUT / FF / DSP / BRAM (absolute counts AND % of xc7z020 totals: LUT
     53,200 / FF 106,400 / DSP 220 / BRAM_18K 280)
   - Timing slack (post-synthesis estimate)
   - C/RTL cosim status and total/mismatch counts

2. A short interpretation section covering, in plain prose (no bullet-point
   reflex — write it as paragraphs):
   - The order-of-magnitude difference in resource cost (expect PHC-lite to
     use meaningfully more LUT and likely nonzero DSP, given it has ~9 matrix
     multiplies, 2 LayerNorms, and several sigmoid/tanh activations per step,
     vs. the queue rule's pure-comparator logic)
   - The latency difference and what it means for achievable decision
     throughput on this FPGA
   - What this comparison DOES and DOES NOT prove — be explicit that it shows
     "the hardware cost of running the adopted ML model's per-step inference
     vs. a trivial rule, on this specific target part, under these specific
     fixed-point/approximation choices" — and that it does NOT show or imply
     anything about end-to-end strategy profitability, real exchange deployment,
     or that the FPGA implementation improves PnL versus a CPU implementation

3. A "what a reviewer will ask" section listing, honestly, the weakest points
   of this comparison as it stands — e.g.: post-synthesis vs post-implementation
   numbers, the chosen sigmoid/tanh/layernorm approximation's measured error
   and whether it could compound problematically in a long deployment, whether
   reuse_factor / pipelining choices were tuned or left at defaults, and
   whether a larger/representative cosim corpus would change confidence in the
   "mismatches=0" (or whatever residual rate was found) result.

4. Restate all protocol-boundary disclaimers one final time at the bottom.

This is meant to be a citation-ready section that can be dropped into the
構想書 (competition proposal) FPGA chapter with minimal further editing —
write it in a register suitable for that (precise, hedged where appropriate,
no overclaiming), and make sure every numeric claim traces to a specific
report file path so it can be checked.
```
