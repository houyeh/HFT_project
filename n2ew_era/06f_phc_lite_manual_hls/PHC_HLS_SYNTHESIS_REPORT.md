# PHC HLS Synthesis Report

Status: `setup_ready_simulation_not_run`.

This report intentionally does **not** claim C-simulation, C-synthesis, or C/RTL co-simulation success. Per PM instruction for this turn, simulation was not run. The work completed here is limited to writing the C++ testbench, creating the Vitis HLS component config, and running a plain `g++ -c` syntax sanity check.

## Completed Artifacts

- HLS implementation: `hls_src/phc_step_hls.cpp`
- HLS interface: `hls_src/phc_step_hls.hpp`
- Weight constants: `hls_src/phc_weights.hpp`
- CSV-driven testbench: `hls_src/phc_step_testbench.cpp`
- Vitis HLS component config: `hls_component/hls_config.cfg`
- Python reference vectors: `results/phc_step_test_vectors.csv`
- Plain g++ compile audit: `results/phc_step_plain_gpp_compile_check.csv`

## Testbench Behavior

The testbench follows the Phase 06A CSV-driven convention:

```text
phc_step_testbench <input_vectors.csv> <output_results.csv>
```

It reads each row from `phc_step_test_vectors.csv`, quantizes the 16 current features and 16 previous-state values to `phc_fixed_t = ap_fixed<24,12>`, calls:

```cpp
void phc_step(
    const phc_fixed_t x_t[16],
    const phc_fixed_t state_prev[16],
    phc_fixed_t readout[2],
    phc_fixed_t state_next[16]
);
```

and compares 18 outputs against the Python float32 reference:

- `ref_readout_0..1`
- `ref_state_next_0..15`

The testbench prints the required Phase 06A-style final line:

```text
total=<N> mismatches=<M>
```

It returns `0` if `M == 0`, otherwise `1`.

## Tolerance

Initial absolute tolerance:

```text
PHC_ABS_TOL = 1.0
```

This is intentionally documented as an initial fixed-point-vs-float32 tolerance, not a hidden pass lever. Q24.12 has ULP `2^-12 ~= 0.000244`, but this recurrent step includes two LayerNorms plus sigmoid/tanh/exp/sqrt nonlinear paths. Phase 06B showed full streaming fixed-point action agreement around `0.998` and max score deltas below about `0.956` without material PnL degradation. Therefore `1.0` is a conservative first C-sim tolerance for the first manual HLS testbench run.

The next simulation phase should report max/mean absolute errors and tighten this tolerance if the measured distribution supports it. Do not silently increase this tolerance to hide mismatches.

## HLS Component Config

Config path:

```text
n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_config.cfg
```

Key settings:

```text
part=xc7z020clg400-1
flow_target=vivado
syn.top=phc_step
clock=10ns
csim.clean=0
sim.O=0
csim.argv=C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/results/phc_step_test_vectors.csv C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/results/phc_step_csim_out.csv
cosim.argv=C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/results/phc_step_test_vectors.csv C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/results/phc_step_cosim_out.csv
```

The config deliberately uses the verified Phase 06A mechanism: direct `csim.argv` and `cosim.argv` keys in the `[hls]` section with absolute Windows paths using forward slashes. It does **not** use the previously failing `tb.sim_argv` or `csim.setup=1` forms.

## Plain g++ Sanity Check

Plain syntax checks were run only with `g++ -c`; no executable testbench was run.

| check | exit_code | ran_simulation |
|---|---:|---|
| `phc_step_hls.cpp` syntax | 0 | false |
| `phc_step_testbench.cpp` syntax | 0 | false |

## Not Run Yet

| stage | status | reason |
|---|---|---|
| C simulation | `not_run` | PM requested not to run simulation in this step |
| C synthesis | `not_run` | held until simulation plan is approved |
| C/RTL co-simulation | `not_run` | PM requested not to run simulation in this step |

Because C-sim/cosim were not run, the following are not available yet:

- authoritative `total=N mismatches=M` from C-sim/cosim
- max/mean HLS-vs-Python output error
- latency / II / LUT / FF / DSP / BRAM
- timing slack
- C/RTL PASS/FAIL

## Protocol Boundaries

- NEW5 remains locked and untouched.
- This is not a PYNQ-Z2 deployment claim.
- This does not prove FPGA improves PnL.
- This does not prove real-world profitability.
- Fixed-point C++ is not identical to fp32 PyTorch until C-sim/cosim error is measured.
- Post-synthesis estimates, when later produced, will still not be post-implementation place-and-route results.
