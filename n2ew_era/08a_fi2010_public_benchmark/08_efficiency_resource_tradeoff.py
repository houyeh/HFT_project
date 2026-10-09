from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "EFFICIENCY_RESOURCE_TRADEOFF_REPORT.md"

PHC_PARAMS = 2115
PHC_MAC_NOMINAL = 1968
PHC_MAC_ACTIVE_HLS = 1200

DEEPL_OB_PAPER_PARAMS = 60_000
DEEPL_OB_REF_IMPL_PARAMS = 142_435
DEEPL_OB_MACS = 39_040_192
DEEPL_OB_F1 = {10: 83.40, 20: 72.82, 50: 80.35}

ZYNQ = {"DSP": 220, "LUT": 53_200, "FF": 106_400, "BRAM_18K": 280}


def pct(x: float) -> str:
    return f"{x:.2f}"


def md_table(rows: list[dict[str, object]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    primary = pd.read_csv(RESULTS / "stage1_primary_metrics_v2.csv")
    phc = primary[(primary["split"] == "test") & (primary["horizon"].isin([10, 20, 50]))].copy()
    phc_f1 = {int(row.horizon): 100.0 * float(row.macro_f1_mean) for row in phc.itertuples()}
    trainable_params = int(phc["trainable_parameters"].iloc[0])
    if trainable_params != PHC_PARAMS:
        raise RuntimeError(f"Expected PHC params {PHC_PARAMS}, got {trainable_params}")

    csynth = pd.read_csv(RESULTS / "stage2_csynth_summary.csv").iloc[0]
    measured_dsp = int(csynth["DSP"])
    measured_lut = int(csynth["LUT"])
    measured_ff = int(csynth["FF"])
    measured_bram = int(csynth["BRAM_18K"])
    latency = int(csynth["latency_cycles_max"])
    latency_us = float(csynth["latency_us_max_at_100mhz"])

    efficiency_rows = []
    for k in [10, 20, 50]:
        efficiency_rows.append(
            {
                "model": "PHC-lite FI-2010 v2",
                "k": k,
                "macro_F1_pct": pct(phc_f1[k]),
                "params_used": f"{PHC_PARAMS} measured",
                "F1_pct_per_1k_params": pct(phc_f1[k] / (PHC_PARAMS / 1000.0)),
                "MACs_used": f"{PHC_MAC_NOMINAL:,} nominal",
                "F1_pct_per_1M_MACs": pct(phc_f1[k] / (PHC_MAC_NOMINAL / 1_000_000.0)),
                "verification": "software_measured_F1 + local_MAC_formula",
            }
        )
        efficiency_rows.append(
            {
                "model": "DeepLOB",
                "k": k,
                "macro_F1_pct": pct(DEEPL_OB_F1[k]),
                "params_used": "60,000 paper-rounded",
                "F1_pct_per_1k_params": pct(DEEPL_OB_F1[k] / (DEEPL_OB_PAPER_PARAMS / 1000.0)),
                "MACs_used": f"{DEEPL_OB_MACS:,} estimated",
                "F1_pct_per_1M_MACs": pct(DEEPL_OB_F1[k] / (DEEPL_OB_MACS / 1_000_000.0)),
                "verification": "published_F1 + estimated_MACs",
            }
        )

    param_alt_rows = []
    for k in [10, 20, 50]:
        param_alt_rows.append(
            {
                "k": k,
                "DeepLOB_F1_pct": pct(DEEPL_OB_F1[k]),
                "paper_60k_F1_per_1k_params": pct(DEEPL_OB_F1[k] / 60.0),
                "ref_impl_142435_F1_per_1k_params": pct(DEEPL_OB_F1[k] / (DEEPL_OB_REF_IMPL_PARAMS / 1000.0)),
            }
        )

    phc_mac_rows = [
        {"block": "enc_linear", "formula": "40 * 16", "MACs": 40 * 16, "active_when_decay_alpha_zero": "yes"},
        {"block": "gate", "formula": "16 * 16", "MACs": 16 * 16, "active_when_decay_alpha_zero": "yes"},
        {"block": "candidate", "formula": "16 * 16", "MACs": 16 * 16, "active_when_decay_alpha_zero": "yes"},
        {"block": "gain_from_encoded", "formula": "16 * 16", "MACs": 16 * 16, "active_when_decay_alpha_zero": "no"},
        {"block": "gain_from_prev_state", "formula": "16 * 16", "MACs": 16 * 16, "active_when_decay_alpha_zero": "no"},
        {"block": "synapse", "formula": "16 * 16", "MACs": 16 * 16, "active_when_decay_alpha_zero": "no"},
        {"block": "readout", "formula": "16 * 3", "MACs": 16 * 3, "active_when_decay_alpha_zero": "yes"},
        {"block": "total_nominal_source", "formula": "640 + 5*256 + 48", "MACs": PHC_MAC_NOMINAL, "active_when_decay_alpha_zero": "mixed"},
        {"block": "total_active_HLS_pruned", "formula": "640 + 2*256 + 48", "MACs": PHC_MAC_ACTIVE_HLS, "active_when_decay_alpha_zero": "yes"},
    ]

    deeplob_mac_rows = [
        {"block": "conv block 1", "formula": "128k + 8.192M + 8.192M", "MACs": 16_512_000},
        {"block": "conv block 2", "formula": "2.048M + 4.096M + 4.096M", "MACs": 10_240_000},
        {"block": "conv block 3", "formula": "1.024M + 409.6k + 409.6k", "MACs": 1_843_200},
        {"block": "inception", "formula": "204.8k + 1.2288M + 204.8k + 2.048M + 204.8k", "MACs": 3_891_200},
        {"block": "LSTM", "formula": "100 * 4 * 64 * (192 + 64)", "MACs": 6_553_600},
        {"block": "dense", "formula": "64 * 3", "MACs": 192},
        {"block": "total", "formula": "sum above", "MACs": DEEPL_OB_MACS},
    ]

    dsp_per_1k_mac = measured_dsp / (PHC_MAC_NOMINAL / 1000.0)
    lut_per_1k_mac = measured_lut / (PHC_MAC_NOMINAL / 1000.0)
    mac_ratio = DEEPL_OB_MACS / PHC_MAC_NOMINAL
    estimated_dsp = measured_dsp * mac_ratio
    estimated_lut = measured_lut * mac_ratio

    resource_rows = [
        {
            "basis": "PHC-lite Stage2 measured baseline",
            "verification": "measured_csynth",
            "MACs": f"{PHC_MAC_NOMINAL:,}",
            "DSP": measured_dsp,
            "LUT": measured_lut,
            "DSP_vs_Zynq7020": f"{measured_dsp / ZYNQ['DSP']:.2f}x",
            "LUT_vs_Zynq7020": f"{measured_lut / ZYNQ['LUT']:.2f}x",
        },
        {
            "basis": "DeepLOB MAC-linear extrapolation from PHC Stage2",
            "verification": "estimate_only",
            "MACs": f"{DEEPL_OB_MACS:,}",
            "DSP": f"{estimated_dsp:,.0f}",
            "LUT": f"{estimated_lut:,.0f}",
            "DSP_vs_Zynq7020": f"{estimated_dsp / ZYNQ['DSP']:.1f}x",
            "LUT_vs_Zynq7020": f"{estimated_lut / ZYNQ['LUT']:.1f}x",
        },
        {
            "basis": "DeepLOB parameter-linear lower-bound using paper 60k",
            "verification": "estimate_only_sanity_check",
            "MACs": "not used",
            "DSP": f"{measured_dsp * (DEEPL_OB_PAPER_PARAMS / PHC_PARAMS):,.0f}",
            "LUT": f"{measured_lut * (DEEPL_OB_PAPER_PARAMS / PHC_PARAMS):,.0f}",
            "DSP_vs_Zynq7020": f"{measured_dsp * (DEEPL_OB_PAPER_PARAMS / PHC_PARAMS) / ZYNQ['DSP']:.1f}x",
            "LUT_vs_Zynq7020": f"{measured_lut * (DEEPL_OB_PAPER_PARAMS / PHC_PARAMS) / ZYNQ['LUT']:.1f}x",
        },
        {
            "basis": "DeepLOB parameter-linear using reference implementation 142,435",
            "verification": "estimate_only_sanity_check",
            "MACs": "not used",
            "DSP": f"{measured_dsp * (DEEPL_OB_REF_IMPL_PARAMS / PHC_PARAMS):,.0f}",
            "LUT": f"{measured_lut * (DEEPL_OB_REF_IMPL_PARAMS / PHC_PARAMS):,.0f}",
            "DSP_vs_Zynq7020": f"{measured_dsp * (DEEPL_OB_REF_IMPL_PARAMS / PHC_PARAMS) / ZYNQ['DSP']:.1f}x",
            "LUT_vs_Zynq7020": f"{measured_lut * (DEEPL_OB_REF_IMPL_PARAMS / PHC_PARAMS) / ZYNQ['LUT']:.1f}x",
        },
    ]

    report = f"""# PHC-lite vs DeepLOB Efficiency / Resource Trade-off

status: analysis_complete_no_deeplob_hls_claim

## Scope

This is an analysis-only comparison. It does not implement DeepLOB in HLS, does not retrain either model, does not run new csim/csynth, and does not use project-internal NEW1-NEW5 data.

The goal is to put measured PHC-lite FI-2010 numbers and published DeepLOB numbers into one efficiency/resource framework while clearly labeling `measured` versus `estimated`.

## Sources And Verification Labels

| item | value |
| --- | --- |
| PHC-lite FI-2010 F1 source | `results/stage1_primary_metrics_v2.csv`, `REPORT_STAGE1.md` Stage 1 v2 |
| PHC-lite Stage2 HLS source | `results/stage2_csynth_summary.csv`, `REPORT_STAGE2.md` |
| PHC-lite trainable parameters | {PHC_PARAMS:,}, measured locally |
| PHC-lite Stage2 csynth resource | DSP {measured_dsp}, LUT {measured_lut}, FF {measured_ff}, BRAM_18K {measured_bram}, latency {latency} cycles / {latency_us:.3f} us |
| DeepLOB F1 source | Zhang, Zohren, Roberts, arXiv:1808.03668, Table II, as already quoted in `REPORT_STAGE1.md` |
| DeepLOB paper parameter source | Zhang, Zohren, Roberts, Table III; printed as `60k`, not an exact integer |
| DeepLOB reference implementation parameter source | public `zcakhaa/DeepLOB...` TensorFlow notebook summary; exact count 142,435 for the inspected Keras implementation |

Important: because the DeepLOB paper table prints `60k` rather than an exact parameter integer, this report uses 60,000 as the paper-table value and explicitly labels it `paper-rounded`. That is favorable to DeepLOB in the F1-per-parameter calculation. The reference implementation exact count is shown separately, not silently substituted.

## Efficiency Table

F1 values are macro-F1 percentage points. `F1_pct_per_1M_MACs` uses nominal linear/conv/recurrent MACs and excludes nonlinearities, normalization, memory access, control, and softmax.

{md_table(efficiency_rows, ["model", "k", "macro_F1_pct", "params_used", "F1_pct_per_1k_params", "MACs_used", "F1_pct_per_1M_MACs", "verification"])}

## DeepLOB Parameter Count Sensitivity

The paper-table parameter count and the public reference implementation parameter count differ materially. This table makes that visible instead of blending the two.

{md_table(param_alt_rows, ["k", "DeepLOB_F1_pct", "paper_60k_F1_per_1k_params", "ref_impl_142435_F1_per_1k_params"])}

## PHC-lite MAC Accounting

The FI-2010 v2 classifier has zero `decay` and zero `alpha_scalar` buffers across all surveyed v2 checkpoints (`results/decay_parameter_survey.csv`). Therefore gain/synapse branches exist in the source architecture but are inactive in the mathematical Stage2 path and may be pruned by HLS.

For the main efficiency/resource estimate, this report uses the conservative nominal source count of {PHC_MAC_NOMINAL:,} MACs. The active pruned count is {PHC_MAC_ACTIVE_HLS:,} MACs and would make PHC-lite look even more resource-efficient.

{md_table(phc_mac_rows, ["block", "formula", "MACs", "active_when_decay_alpha_zero"])}

## DeepLOB MAC Estimate

DeepLOB MACs are estimated from the public CNN + inception + LSTM architecture commonly used for FI-2010: input 100x40x1, convolutional blocks reducing the feature axis, inception branches, LSTM with input width 192 and hidden size 64, then a 3-way dense output.

{md_table(deeplob_mac_rows, ["block", "formula", "MACs"])}

DeepLOB total estimated MACs per inference: **{DEEPL_OB_MACS:,}**.

## Resource Extrapolation

PHC-lite Stage2 measured csynth gives {measured_dsp} DSP and {measured_lut} LUT for {PHC_MAC_NOMINAL:,} nominal MACs. That corresponds to:

- DSP per 1,000 nominal PHC MACs: {dsp_per_1k_mac:.3f}
- LUT per 1,000 nominal PHC MACs: {lut_per_1k_mac:.3f}

**The DeepLOB resource rows below are estimate_only. They are a linear order-of-magnitude extrapolation, not a synthesis result. CNN convolution and LSTM gates may scale better or worse than this due to weight reuse, reuse factor, pipeline scheduling, memory banking, quantization, and hand optimization. These estimates must not be cited as real DeepLOB HLS utilization.**

{md_table(resource_rows, ["basis", "verification", "MACs", "DSP", "LUT", "DSP_vs_Zynq7020", "LUT_vs_Zynq7020"])}

## Literature Check For DeepLOB FPGA/HLS Implementations

Searches performed on 2026-07-30:

- `DeepLOB FPGA`
- `DeepLOB HLS`
- `DeepLOB hls4ml`
- `DeepLOB Vitis FPGA`
- `FINN-GL ConvLSTM DeepLOB`

I found the DeepLOB paper and the public architecture notebooks. I also found FINN-GL, a 2025 FPGA/LSTM deployment paper that targets quantized ConvLSTM-style short-term stock prediction and mentions a DeepLOB-like setting. It is not a direct public HLS/Vitis implementation of the original DeepLOB CNN+inception+LSTM architecture on FI-2010 with comparable Table II F1 numbers. Therefore this report keeps Section `Resource Extrapolation` as `estimate_only` instead of replacing it with real DeepLOB FPGA numbers.

## Honest Conclusion

PHC-lite does **not** beat DeepLOB on FI-2010 accuracy. On the Stage 1 v2 public benchmark, PHC-lite reaches roughly 40-41 macro-F1 for k=10/20/50, while DeepLOB reports roughly 73-83 macro-F1. The gap is large and should not be softened.

PHC-lite's advantage is efficiency and low-resource feasibility. With 2,115 trainable parameters and a csynth-measured Stage2 design using 33 DSP and 20,582 LUT on Zynq-7020, it is a small model that can be synthesized under this project's manual HLS method. DeepLOB is much stronger statistically, but the published CNN+LSTM architecture is much larger. Under the requested MAC-linear extrapolation, DeepLOB would exceed Zynq-7020 by about **{estimated_dsp / ZYNQ['DSP']:.1f}x DSP** and **{estimated_lut / ZYNQ['LUT']:.1f}x LUT**. Even the much softer parameter-linear sanity check using the paper-rounded 60k parameter count suggests about **{measured_dsp * (DEEPL_OB_PAPER_PARAMS / PHC_PARAMS) / ZYNQ['DSP']:.1f}x DSP** and **{measured_lut * (DEEPL_OB_PAPER_PARAMS / PHC_PARAMS) / ZYNQ['LUT']:.1f}x LUT**.

The correct claim is therefore:

> PHC-lite is weaker than DeepLOB in FI-2010 predictive accuracy, but it has a large parameter/MAC/resource-efficiency advantage. A DeepLOB-class model may need compression, quantization, reuse-factor tuning, or a different FPGA target before fitting the same low-cost Zynq-7020 envelope. This is an efficiency/resource trade-off, not an overall performance win.

## Source Links

- DeepLOB arXiv page: https://arxiv.org/abs/1808.03668
- DeepLOB PDF: https://arxiv.org/pdf/1808.03668
- DeepLOB public GitHub notebook repository: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
- FINN-GL related FPGA/LSTM paper: https://arxiv.org/abs/2506.20810
"""

    OUT.write_text(report, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
