# PHC-lite Manual HLS Design Notes

## Scope

This phase hand-writes one PHC-lite recurrent step in C++ for Vitis HLS. It implements the frozen Phase 05C4 / 06E step computation using the exported Q24.12 constants in `hls_src/phc_weights.hpp`.

No model training, threshold selection, NEW5 access, or PnL evaluation is performed here. This is a hardware mapping artifact only.

## Fixed-Point Types

- External input/output and stored weights use `phc_fixed_t = ap_fixed<24,12>`, matching Phase 06F `phc_weights.hpp`.
- Dot-product and nonlinear intermediate values use `phc_acc_t = ap_fixed<40,16>`.
- LayerNorm variance uses `phc_var_t = ap_fixed<48,24>`.

The wider accumulator exists because each 16-wide dot product sums products plus bias. With Q24.12 inputs and weights, the raw products can need more headroom than Q24.12 before final output quantization. `ap_fixed<40,16>` gives roughly +/-32768 integer range with 24 fractional bits, which is intentionally conservative for the observed exported weights (all roughly within [-1.6, 1.6]) and normalized state/features. LayerNorm variance uses still wider `ap_fixed<48,24>` to protect the square-and-sum path.

## Routing Layers

`input_x`, `input_state`, and `output_pack` from `PHCSTPStepHLS` are fixed frontend plumbing, not learned microstructure parameters. They are intentionally not implemented as matrix multiplies.

- `x_t` and `state_prev` are separate function arguments instead of slicing a 32-vector.
- `readout` is written from the final readout linear layer.
- `state_next` is written directly as `z`.
- `ALPHA_SCALAR` is exported as a one-element array, so the datapath uses `ALPHA_SCALAR[0]`; this is the raw alpha value confirmed in Phase 06E, not `sigmoid(alpha)`.

This avoids wasting HLS resources on identity matrices.

## Nonlinear Functions

The implementation uses Vitis HLS math functions:

- `sigmoid(x) = 1 / (1 + hls::exp(-x))`
- `tanh(x) = hls::tanh(x)`
- `SiLU(x) = x * sigmoid(x)`
- LayerNorm reciprocal standard deviation uses `hls::sqrt(var + 1e-5)`

This is the most faithful first manual-HLS implementation because it mirrors the PyTorch floating formula directly. It is not yet a resource-optimized LUT or piecewise-linear approximation. Approximation error versus the Python float32 step has not been measured in this substep; the next testbench phase should compare against `results/phc_step_test_vectors.csv` and quantify max/mean error per output.

## LayerNorm

LayerNorm follows PyTorch default `eps = 1e-5`:

```text
mu  = mean(v)
var = mean((v - mu)^2)
out[i] = gamma[i] * (v[i] - mu) / sqrt(var + 1e-5) + beta[i]
```

Both encoder LayerNorm and loop LayerNorm use the same helper with different gamma/beta constants.

## Pragmas

- `#pragma HLS PIPELINE II=1` on `phc_step`: requests one new step per cycle after pipeline fill, subject to math operator/resource constraints.
- `#pragma HLS INLINE` on helper functions: allows HLS to flatten helper calls into the top datapath.
- `#pragma HLS ARRAY_PARTITION complete` on inputs, outputs, weights, and all 16-wide intermediate arrays: exposes parallel element access and avoids bottlenecking the unrolled 16-wide loops on array ports.
- `#pragma HLS UNROLL` on fixed-size loops: expresses the 16-wide vector operations and 16x16 dot products as parallel datapath candidates.

These pragmas are intentionally aggressive for the first synthesis attempt. If resource use is too high, the next phase can introduce reuse factors or partial unrolling.

## Plain g++ Sanity Check

The source includes guarded fallbacks for plain `g++ -c` syntax checking when Vitis HLS headers are not available. Those fallbacks are only for syntax checking; Vitis HLS must use the real `ap_fixed.h` and `hls_math.h`.

The plain g++ sanity check was run after generating this file:

```text
g++ -std=c++17 -I n2ew_era/06f_phc_lite_manual_hls/hls_src -c n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp
gpp_exit=0
```

## Current Claim Boundary

This code is intended to be synthesizable C++ source for Vitis HLS, but this phase does not run Vitis synthesis or C/RTL cosimulation. It does not prove FPGA deployment, FPGA PnL improvement, true exchange replay, or real-world profitability.
