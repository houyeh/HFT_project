# Start Here

Welcome.

- **What this project is**: HFT FPGA Trading Research Project focusing on an ML-based toxic-flow cancel filter. We are NOT building a directional alpha trader.
- **Where to start reading**: `new_era/project_control/00_PROJECT_STATE.md` and `new_era/project_control/05_CURRENT_SLIDE_STORY.md`.
- **Current baseline**: TCN 0.55 threshold on NEW dataset (Net PnL 84,980.0, beats matched random).
- **Completed experiments**: 015 (baseline), 016 (binary label PoC), 017 (regime auto-gate).
- **Current verdict**: Strategy exhibits local edge but is heavily regime-dependent. Auto-gating and binary labels are not yet production-ready.
- **Forbidden claims**: DO NOT claim production readiness, do not claim latency > 3 completely collapses PnL, do not claim we have true exchange queue replay.
- **Next priority**: 018 Queue-aware Validation.
- **How to add a new experiment**: Create a distinct folder (e.g. `019_xxx/`), write non-destructive scripts, save outputs to `outputs/`.
- **Where to save outputs**: Respective `outputs/` directories or `eval_outputs/`.
- **Who should review results**: PM Agent and Professor Agent.
