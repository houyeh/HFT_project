from __future__ import annotations

from common_phase02d import (
    ROOT,
    SELECTED_HORIZON,
    SELECTED_POLICY_ID,
    SELECTED_RULE_DEFINITION,
    dataframe_to_html_table,
    read_csv_if_exists,
)


EXACT_RANDOM_ZERO_PVALUE_WORDING = (
    "In 1000 exact random trials, no random trial exceeded the selected policy. "
    "Therefore empirical p-value &lt; 0.001."
)
ELIGIBILITY_NOTE = (
    "`eligible_for_final_selection_at_1x` is meaningful only on validation. "
    "Test and NEW4 are frozen evaluation splits and are not used for selection."
)


def main() -> None:
    controls = read_csv_if_exists(ROOT / "results" / "selected_policy_vs_controls.csv")
    exact = read_csv_if_exists(ROOT / "results" / "exact_random_oos_summary.csv")
    fee = read_csv_if_exists(ROOT / "results" / "fee_break_even_points.csv")
    stress = read_csv_if_exists(ROOT / "results" / "queue_stress_selected_policy_summary.csv")
    claims = read_csv_if_exists(ROOT / "results" / "phase02d_claim_safety_table.csv")

    controls_small = controls[
        ["split", "total_pnl", "old_bid_only_pnl", "selected_minus_old_bid_only", "matched_random_95p", "beats_random_95p", "beats_old_bid_only"]
    ] if not controls.empty else controls
    if not exact.empty:
        exact_display = exact.copy()
        exact_display["empirical_p_value_display"] = exact_display[
            "empirical_p_value_exact_random_ge_model"
        ].map(lambda value: "<0.001" if float(value) == 0.0 else f"{float(value):.6f}")
        exact_small = exact_display[
            ["split", "model_pnl", "exact_random_95p", "beats_exact_random_95p", "empirical_p_value_display"]
        ]
        exact_note = EXACT_RANDOM_ZERO_PVALUE_WORDING if (exact["empirical_p_value_exact_random_ge_model"].astype(float) == 0.0).any() else ""
    else:
        exact_small = exact
        exact_note = ""
    stress_small = stress[
        ["split", "stress_total_pnl", "selected_action_count", "win_rate", "max_drawdown", "positive_under_stress"]
    ] if not stress.empty else stress

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phase 02D Final Strict Baseline Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #20242b; background: #ffffff; }}
    h1 {{ font-size: 26px; margin-bottom: 4px; }}
    h2 {{ font-size: 16px; margin: 20px 0 8px; border-bottom: 1px solid #d9dee7; padding-bottom: 4px; }}
    .subtitle {{ color: #5b6472; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }}
    .card {{ border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; background: #fbfcfe; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
    th, td {{ border: 1px solid #d9dee7; padding: 5px 6px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    .claim-pass {{ color: #0b7a38; font-weight: bold; }}
    .claim-fail {{ color: #b42318; font-weight: bold; }}
    .muted {{ color: #667085; }}
    img {{ max-width: 100%; border: 1px solid #d9dee7; border-radius: 6px; background: white; }}
    .safe {{ background: #eef8f1; border-left: 4px solid #2f9e44; padding: 10px; }}
    .warn {{ background: #fff5f5; border-left: 4px solid #c43c35; padding: 10px; }}
  </style>
</head>
<body>
  <h1>Phase 02D Final Strict Baseline Audit</h1>
  <div class="subtitle">Fixed policy: <b>{SELECTED_POLICY_ID}</b>, horizon <b>H{SELECTED_HORIZON}</b></div>

  <div class="grid">
    <div class="card">
      <h2>Rule Definition</h2>
      <p>{SELECTED_RULE_DEFINITION}</p>
      <p class="safe">Safe claim: non-trivial L2 quote-ranking signal under short-horizon proxy markout.</p>
      <p class="warn">Forbidden claim: real-world profitability, true queue replay, or FPGA-proven PnL.</p>
    </div>
    <div class="card">
      <h2>Claim Safety</h2>
      {dataframe_to_html_table(claims)}
    </div>
  </div>

  <h2>Selected vs Controls</h2>
  {dataframe_to_html_table(controls_small)}
  <img src="figures/selected_vs_controls_bar.png" alt="Selected vs controls">

  <div class="grid">
    <div class="card">
      <h2>Random Hurdle</h2>
      <p class="muted">{exact_note}</p>
      {dataframe_to_html_table(exact_small)}
      <img src="figures/random_hurdle_bar.png" alt="Random hurdle">
    </div>
    <div class="card">
      <h2>Fee Break-Even</h2>
      <p class="muted">{ELIGIBILITY_NOTE}</p>
      {dataframe_to_html_table(fee)}
      <img src="figures/fee_sensitivity_selected_policy.png" alt="Fee sensitivity">
    </div>
  </div>

  <h2>Moderate Queue Stress</h2>
  <p class="muted">Moderate stress = fill_keep_rate 0.5, adverse_multiplier 1.5, delay 0.</p>
  <p class="muted">{ELIGIBILITY_NOTE}</p>
  {dataframe_to_html_table(stress_small)}
  <div class="grid">
    <img src="figures/queue_stress_heatmap_selected_policy.png" alt="Queue stress heatmap">
    <img src="figures/delay_sensitivity_selected_policy.png" alt="Delay sensitivity">
  </div>
</body>
</html>
"""
    (ROOT / "phase02d_final_audit_onepage.html").write_text(html, encoding="utf-8")
    print(f"Wrote {ROOT / 'phase02d_final_audit_onepage.html'}")


if __name__ == "__main__":
    main()
