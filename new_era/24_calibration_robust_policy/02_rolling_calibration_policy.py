from __future__ import annotations

from calibration_utils import RESULTS_DIR, build_grid, ensure_dirs, load_context


def run() -> None:
    ensure_dirs()
    _, actions = load_context("val")
    grid, _ = build_grid(actions, family="rolling_global", group_cols=["session_id"])
    grid.to_csv(RESULTS_DIR / "rolling_calibration_validation_grid.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote {RESULTS_DIR / 'rolling_calibration_validation_grid.csv'}")

