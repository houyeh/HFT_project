# Next Tasks

## Priority 1 — 018 Queue-aware Validation
Goal: Quantify strategy survivability under strict queue-position proxies and fill-probability haircuts.
Inputs: baseline predictions, market data without aggressor_side.
Outputs: Queue sensitivity matrices, adverse selection stressed PnL.
Success criteria: Strategy maintains positive expected value under reasonable queue proxies.
Failure criteria: Strategy PnL completely collapses when toxic fill probability is maximized.
Why it matters: Cannot claim profitability without acknowledging LOB queue mechanics.

## Priority 2 — Update Slides
Goal: Sync latest exact figures to presentation materials.
Inputs: `new_era/project_control/` documentation.
Outputs: Updated PPTX/HTML slides.

## Priority 3 — 016 Fix / All-event Label
Only after queue-aware or if time permits. Fix the binary model performance gap so it can outperform random baselines.

## Priority 4 — More Data / Regime Calibration
Need more sessions before reliable auto-gating. One validation session is insufficient for robust causal regime selector calibration.
