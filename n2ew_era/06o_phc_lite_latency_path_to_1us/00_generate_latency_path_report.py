from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROJECT_ROOT = ROOT.parents[1]

BASELINE = {
    "latency_cycles": 8316,
    "latency_us_100mhz": 83.16,
    "ii": 8317,
    "dsp": 46,
    "lut": 24836,
    "ff": 14518,
    "bram": 6,
    "wns_ns": 0.020,
    "period_ns": 10.000,
    "inferred_critical_period_ns": 9.980,
}

STAGE_ROWS = [
    ("ENC_LINEAR", "16x16 linear", 1328, "VITIS_LOOP_101_1", "csynth loop table"),
    ("ENC_LAYER_NORM_VISIBLE", "LayerNorm visible subtotal", 300, "mean+var+sqrt+normalize", "derived from csynth loops"),
    ("ENC_SILU", "SiLU activation", 176, "VITIS_LOOP_88_1", "csynth loop table"),
    ("GATE_LINEAR", "16x16 linear", 1328, "VITIS_LOOP_119_1", "csynth loop table"),
    ("CANDIDATE_LINEAR", "16x16 linear", 1329, "Outline_VITIS_LOOP_119_1", "csynth instance table"),
    ("GAIN_FROM_ENCODED_LINEAR", "16x16 linear", 1329, "Outline_VITIS_LOOP_119_11", "csynth instance table"),
    ("GAIN_FROM_PREV_STATE_LINEAR", "16x16 linear no-bias", 1313, "Outline_VITIS_LOOP_136_1", "csynth instance table"),
    ("SYNAPSE_LINEAR", "16x16 linear", 1329, "Outline_VITIS_LOOP_119_13", "csynth instance table"),
    ("DECAY_STATE_UPDATE", "elementwise recurrence", 81, "Outline_VITIS_LOOP_233_1", "csynth instance table"),
    ("GAIN_ADD", "elementwise add", 49, "Outline_VITIS_LOOP_241_2", "csynth instance table"),
    ("GAIN_SIGMOID", "sigmoid activation", 145, "Outline_VITIS_LOOP_72_12", "csynth instance table"),
    ("MESSAGE_SILU", "SiLU activation", 177, "Outline_VITIS_LOOP_88_1", "csynth instance table"),
    ("LOOP_INPUT_COMBINE", "elementwise combine", 160, "VITIS_LOOP_250_3", "csynth loop table"),
    ("LOOP_LAYER_NORM_VISIBLE", "LayerNorm visible subtotal", 300, "mean+var+sqrt+normalize", "derived from csynth loops"),
    ("READOUT_LINEAR", "2x16 linear", 164, "VITIS_LOOP_154_1", "csynth loop table"),
    ("OUTPUT_COPY", "readout/state copy wait group", 33, "max(3,33)", "derived"),
    ("RESIDUAL_CONTROL", "control/divider/handshake residual", 137, "8316 - attributed total", "derived"),
]


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = RESULTS / name
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_sweep_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "n2ew_era" / "06l_phc_lite_pipeline_budget_sweep" / "results" / "pipeline_budget_sweep_summary.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stage_rows = []
    for stage, kind, cycles, entry, source in STAGE_ROWS:
        stage_rows.append({
            "stage": stage,
            "kind": kind,
            "cycles": cycles,
            "pct_of_8316": cycles / BASELINE["latency_cycles"] * 100.0,
            "csynth_entry": entry,
            "source": source,
        })
    write_csv("latency_stage_breakdown.csv", stage_rows)

    linear_total = sum(r["cycles"] for r in stage_rows if "16x16 linear" in str(r["kind"]))
    category_rows = [
        {"category": "six 16x16 linear blocks", "cycles": linear_total, "pct_of_8316": linear_total / 8316 * 100.0},
        {"category": "two LayerNorm visible subtotals", "cycles": 600, "pct_of_8316": 600 / 8316 * 100.0},
        {"category": "activation and elementwise loops", "cycles": 821, "pct_of_8316": 821 / 8316 * 100.0},
        {"category": "readout/output/residual", "cycles": 334, "pct_of_8316": 334 / 8316 * 100.0},
    ]
    write_csv("latency_category_summary.csv", category_rows)

    freq_rows = []
    for freq, period in [(100, 10.0), (150, 1000.0 / 150.0), (200, 5.0)]:
        latency_us = BASELINE["latency_cycles"] / freq
        slack_proxy = period - BASELINE["inferred_critical_period_ns"]
        freq_rows.append({
            "frequency_mhz": freq,
            "period_ns": period,
            "latency_us_if_cycles_unchanged": latency_us,
            "distance_to_1us_x": latency_us / 1.0,
            "existing_routed_timing_status": "verified_pass" if freq == 100 else "not_verified_expected_fail_without_retiming",
            "slack_proxy_vs_existing_routed_path_ns": slack_proxy,
            "evidence": "06F routed timing" if freq == 100 else "linear estimate only; requires fresh P&R timing report",
        })
    write_csv("frequency_scaling_projection.csv", freq_rows)

    sweep_rows_raw = read_sweep_rows()
    unroll_rows = []
    for row in sweep_rows_raw:
        latency_cycles = float(row["latency_cycles"])
        latency_us = float(row["latency_us"])
        unroll_rows.append({
            "lever": f"linear inner-loop unroll factor {row['unroll_factor']}",
            "tool_stage": "csim+csynth+cosim_existing_06L_no_P&R",
            "latency_cycles": int(latency_cycles),
            "latency_us_100mhz": latency_us,
            "improvement_vs_v8_x": float(row["speedup_vs_v8_latency"]),
            "distance_to_1us_x_at_100mhz": latency_us,
            "DSP": int(row["DSP"]),
            "DSP_pct": float(row["DSP_pct"]),
            "LUT": int(row["LUT"]),
            "LUT_pct": float(row["LUT_pct"]),
            "FF": int(row["FF"]),
            "FF_pct": float(row["FF_pct"]),
            "BRAM_18K": int(row["BRAM_18K"]),
            "BRAM_pct": float(row["BRAM_pct"]),
            "fits_budget": row["fits_budget"],
            "timing_closure_evidence": "not_routed_in_06L_summary",
        })
    write_csv("lever_unroll_existing_results.csv", unroll_rows)

    lever_rows = [
        {
            "lever": "clock 100->150MHz on unchanged 06F netlist",
            "best_supported_latency_us": 55.44,
            "improvement_vs_v8_x": 1.5,
            "distance_to_1us_x": 55.44,
            "resource_cost": "none if same netlist",
            "tool_evidence": "no new P&R; existing 100MHz WNS implies likely fail",
            "timing_status": "unverified_expected_fail_without_retiming",
            "recommendation": "do not claim until fresh 150MHz P&R shows nonnegative WNS",
        },
        {
            "lever": "clock 100->200MHz on unchanged 06F netlist",
            "best_supported_latency_us": 41.58,
            "improvement_vs_v8_x": 2.0,
            "distance_to_1us_x": 41.58,
            "resource_cost": "none if same netlist",
            "tool_evidence": "no new P&R; existing 100MHz WNS implies likely fail",
            "timing_status": "unverified_expected_fail_without_retiming",
            "recommendation": "not credible without major retiming/re-P&R",
        },
        {
            "lever": "linear helper unroll factor 16 (06L)",
            "best_supported_latency_us": 23.16,
            "improvement_vs_v8_x": 3.5906735751295336,
            "distance_to_1us_x": 23.16,
            "resource_cost": "DSP 121/220, LUT 36792/53200, FF 35774/106400, BRAM 0/280",
            "tool_evidence": "06L csim+csynth+cosim, no P&R timing yet",
            "timing_status": "synthesis_and_cosim_pass_route_unverified",
            "recommendation": "best measured latency lever so far; next candidate for P&R timing",
        },
        {
            "lever": "Q16.8 quantization only",
            "best_supported_latency_us": 83.16,
            "improvement_vs_v8_x": 1.0,
            "distance_to_1us_x": 83.16,
            "resource_cost": "DSP unchanged 46, LUT slightly lower 24746 vs 24836; numerical error much worse",
            "tool_evidence": "06l_OLD phase2 csim+csynth, no P&R",
            "timing_status": "not_recommended",
            "recommendation": "not useful for latency; do not use as path to 1us",
        },
        {
            "lever": "diagonal/structured recurrence replacement (06G)",
            "best_supported_latency_us": 20.87,
            "improvement_vs_v8_x": 3.984667944204121,
            "distance_to_1us_x": 20.87,
            "resource_cost": "DSP 34/220, LUT 16058/53200, FF 9066/106400, BRAM 1/280",
            "tool_evidence": "06G csynth only; different model, not PHC-lite production",
            "timing_status": "route_unverified_and_model_regresses_vs_best_sequence_model",
            "recommendation": "architectural clue, not a drop-in replacement",
        },
    ]
    write_csv("lever_tradeoff_summary.csv", lever_rows)

    lines = []
    lines.extend([
        "# PHC-lite Latency Bottleneck and Path to 1us Report",
        "",
        "Status: `existing_tool_output_analysis_no_new_PnR_run`.",
        "",
        "This report analyzes the frozen PHC-lite manual HLS implementation on Zynq-7020 and the already-run 06L/06G experiments. It does not rerun P&R, retrain a model, change HLS source, or touch NEW5.",
        "",
        "## Executive Conclusion",
        "",
        "The current 83.16us latency is not a timing-closure problem. The routed 100MHz implementation meets timing with WNS +0.020ns, but the single-step schedule has 8316 cycles and interval 8317. Because the recurrent state update makes the next event depend on the previous event's `state_next`, throughput II effectively equals single-step latency for one live state stream.",
        "",
        "The dominant bottleneck is the repeated unpipelined 16x16 linear/matrix-vector loops, not LayerNorm sqrt/division. Six 16x16 linear blocks account for about 7956 visible cycles, or 95.67% of the 8316-cycle budget if counted as raw block latency. In the scheduler-attributed view, they still dominate the critical path.",
        "",
        "The best existing measured lever is 06L inner-loop unroll factor 16: 2316 cycles / 23.16us at 100MHz, with csim/cosim passing and csynth resources still inside PYNQ-Z2 budget. This is a real 3.59x improvement over v8, but it is still 23.16x away from 1us at 100MHz and has not yet received routed P&R timing closure.",
        "",
        "## Baseline Evidence",
        "",
        "| item | value | source |",
        "|---|---:|---|",
        "| latency | 8316 cycles / 83.16us | 06F `phc_step_csynth.rpt` and RTL cosim |",
        "| interval / II | 8317 cycles | 06F `phc_step_csynth.rpt` |",
        "| P&R WNS | +0.020ns | 06F routed timing report |",
        "| clock | 10.000ns / 100MHz | 06F routed timing report |",
        "| inferred routed critical period | 9.980ns | 10.000ns - 0.020ns WNS |",
        "| critical path shape | FSM register to DSP48E1 PCIN, data path delay 8.496ns | 06F routed timing report |",
        "| routed utilization fingerprint | LUT 14.20%, FD/FF 7.72%, DSP 21.36%, RAMB/FIFO 2.14% | 06F `hls_impl_pnr.rpt` |",
        "",
        "## Stage / Loop Latency Breakdown",
        "",
        "| stage | kind | cycles | pct of 8316 | csynth entry |",
        "|---|---|---:|---:|---|",
    ])
    for row in stage_rows:
        lines.append(f"| {row['stage']} | {row['kind']} | {row['cycles']} | {row['pct_of_8316']:.2f}% | {row['csynth_entry']} |")
    lines.extend([
        "",
        "Category summary:",
        "",
        "| category | cycles | pct of 8316 |",
        "|---|---:|---:|",
    ])
    for row in category_rows:
        lines.append(f"| {row['category']} | {row['cycles']} | {row['pct_of_8316']:.2f}% |")
    lines.extend([
        "",
        "Interpretation: The 16x16 matrix-vector loops are the latency wall. LayerNorm is visible and should not be ignored, but the current report data does not support treating it as the primary bottleneck.",
        "",
        "## Clock Frequency Lever",
        "",
        "This is a linear cycle-time estimate only. The current routed critical path is already 9.980ns. Therefore the unchanged 06F netlist would be expected to fail 150MHz and 200MHz timing unless the design is retimed/reimplemented and a fresh P&R timing report shows nonnegative WNS.",
        "",
        "| frequency | period ns | latency if 8316 cycles unchanged | distance to 1us | timing evidence | slack proxy |",
        "|---:|---:|---:|---:|---|---:|",
    ])
    for row in freq_rows:
        lines.append(f"| {row['frequency_mhz']}MHz | {row['period_ns']:.3f} | {row['latency_us_if_cycles_unchanged']:.2f}us | {row['distance_to_1us_x']:.2f}x | {row['existing_routed_timing_status']} | {row['slack_proxy_vs_existing_routed_path_ns']:.3f}ns |")
    lines.extend([
        "",
        "Conclusion: clocking alone cannot reach 1us. Even a hypothetical timing-clean 200MHz implementation of the current 8316-cycle schedule would be 41.58us.",
        "",
        "## Existing Unroll Sweep Evidence",
        "",
        "| factor | latency cycles | latency us | DSP | LUT | FF | BRAM | fits budget | speedup vs v8 | cosim |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|---:|---|",
    ])
    for row in unroll_rows:
        lines.append(f"| {row['lever'].split()[-1]} | {row['latency_cycles']} | {row['latency_us_100mhz']:.2f} | {row['DSP']} | {row['LUT']} | {row['FF']} | {row['BRAM_18K']} | {row['fits_budget']} | {row['improvement_vs_v8_x']:.2f}x | PASS |")
    lines.extend([
        "",
        "The unroll factor 16 point is the best existing data-backed latency/resource tradeoff: it uses 121 DSP (55%), 36,792 LUT (69%), and 35,774 FF (33%), all under Zynq-7020 budget in csynth. It still needs routed timing closure before it can replace the 06F frozen baseline.",
        "",
        "## Lever Tradeoff Table",
        "",
        "| lever | best supported latency | improvement vs v8 | distance to 1us | resource cost | evidence / status | recommendation |",
        "|---|---:|---:|---:|---|---|---|",
    ])
    for row in lever_rows:
        lines.append(f"| {row['lever']} | {row['best_supported_latency_us']:.2f}us | {row['improvement_vs_v8_x']:.2f}x | {row['distance_to_1us_x']:.2f}x | {row['resource_cost']} | {row['tool_evidence']}; {row['timing_status']} | {row['recommendation']} |")
    lines.extend([
        "",
        "## Partial Time-Step Unroll / Recurrent Pipeline Assessment",
        "",
        "For a single live market state stream, partial unroll across time is not causally valid: event t+1 depends on `state_next` from event t. This is why the design has interval approximately equal to latency. Multi-stream interleaving could improve aggregate throughput across independent symbols/sessions, but it would not reduce the single-stream decision latency needed for an HFT quote gate.",
        "",
        "To reach 1us at 100MHz, the step must be at most 100 cycles. The best existing PHC-lite-style datapoint is 2316 cycles, so the gap is still about 23x. At 200MHz, that same factor-16 design would still be 11.58us if timing closed, which remains about 11.6x from 1us.",
        "",
        "## Concrete Technical Path Toward 1us",
        "",
        "1. First route the 06L factor-16 design. This is the lowest-risk next experiment because csim/cosim already pass and csynth fits resources. It can potentially reduce 83.16us to 23.16us, but timing closure must be verified.",
        "2. If factor-16 routes, test modest clock targets only after routing: 125MHz or 150MHz before 200MHz. The current 06F routed path implies unchanged 150MHz/200MHz would fail, so fresh P&R WNS is mandatory.",
        "3. To move below ~10us, the model architecture must reduce the number of dense 16x16 linear blocks or replace them with structured/low-rank/diagonal transforms. The 06G diagonal experiment gives a 2087-cycle csynth datapoint, but it is a different model and not a drop-in replacement.",
        "4. To approach 1us, dense PHC-lite needs a deeper architecture/HLS redesign: e.g. fewer dense blocks, parallel MAC trees with more DSP than factor16, carefully placed pipeline registers inside state-dependent computations, and potentially a simplified activation/normalization path. This is not achievable by clock frequency or Q16.8 quantization alone on the current Zynq-7020 evidence.",
        "",
        "## Claim Boundary",
        "",
        "- This report uses existing csynth/cosim/P&R artifacts only; no new P&R was run here.",
        "- 150MHz/200MHz numbers are cycle-time projections, not verified timing-closed implementations.",
        "- No FPGA PnL, exchange replay, or real-world profitability claim is made.",
        "- The 06L factor-16 result is not a frozen replacement until it receives routed P&R timing closure and a separate freeze decision.",
    ])
    (ROOT / "PHC_LATENCY_PATH_TO_1US_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(ROOT / "PHC_LATENCY_PATH_TO_1US_REPORT.md")
    print(RESULTS / "latency_stage_breakdown.csv")
    print(RESULTS / "lever_tradeoff_summary.csv")


if __name__ == "__main__":
    main()
