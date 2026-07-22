from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase02e import EVAL_SPLITS, HORIZONS, MAKER_FEE, POINT_VALUE, RESULTS, ROOT, ensure_dirs, iter_event_frames, write_markdown


def summarize(values: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if values.empty:
        return pd.DataFrame()
    return (
        values.groupby(keys, as_index=False)
        .agg(
            selected_action_count=("total_pnl", "size"),
            total_pnl=("total_pnl", "sum"),
            half_spread_advantage_pnl=("half_spread_advantage_pnl", "sum"),
            mid_price_drift_pnl=("mid_price_drift_pnl", "sum"),
            maker_fee_total=("maker_fee", "sum"),
            avg_total_pnl=("total_pnl", "mean"),
            avg_half_spread_advantage_pnl=("half_spread_advantage_pnl", "mean"),
            avg_mid_price_drift_pnl=("mid_price_drift_pnl", "mean"),
            positive_rate=("total_pnl", lambda s: float((s > 0.0).mean())),
        )
        .assign(
            spread_component_share_of_positive_components=lambda df: df["half_spread_advantage_pnl"]
            / (df["half_spread_advantage_pnl"].abs() + df["mid_price_drift_pnl"].abs()).replace(0.0, np.nan),
            drift_component_share_of_positive_components=lambda df: df["mid_price_drift_pnl"]
            / (df["half_spread_advantage_pnl"].abs() + df["mid_price_drift_pnl"].abs()).replace(0.0, np.nan),
        )
    )


def main() -> None:
    ensure_dirs()
    rows: list[pd.DataFrame] = []
    for split in EVAL_SPLITS:
        for frame in iter_event_frames(split, horizons=HORIZONS):
            session_id = str(frame["session_id"].iloc[0]) if len(frame) else ""
            selected = pd.to_numeric(frame["spread_ticks"], errors="coerce").to_numpy(dtype=float) >= 1.0
            if not selected.any():
                continue
            base = frame.loc[selected, ["split", "session_id", "best_bid", "mid_price"]].copy()
            base["session_id"] = session_id
            for horizon in HORIZONS:
                future_mid = frame.loc[selected, f"future_mid_H{horizon}"].astype(float).to_numpy()
                finite = np.isfinite(future_mid)
                if not finite.any():
                    continue
                work = base.loc[finite].copy()
                mid = work["mid_price"].astype(float).to_numpy()
                bid = work["best_bid"].astype(float).to_numpy()
                fmid = future_mid[finite]
                work["horizon"] = int(horizon)
                work["half_spread_advantage_pnl"] = (mid - bid) * POINT_VALUE
                work["mid_price_drift_pnl"] = (fmid - mid) * POINT_VALUE
                work["maker_fee"] = MAKER_FEE
                work["total_pnl"] = work["half_spread_advantage_pnl"] + work["mid_price_drift_pnl"] - work["maker_fee"]
                rows.append(work)

    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    by_session = summarize(detail, ["split", "session_id", "horizon"])
    by_split = summarize(detail, ["split", "horizon"])
    by_split.insert(1, "session_id", "__ALL__")
    out = pd.concat([by_session, by_split], ignore_index=True, sort=False)
    out.to_csv(RESULTS / "old_bid_only_pnl_decomposition.csv", index=False)

    h20 = by_split[by_split["horizon"].astype(int) == 20].copy()
    if not h20.empty:
        h20["spread_dominated"] = h20["half_spread_advantage_pnl"].abs() > h20["mid_price_drift_pnl"].abs()
        spread_dom_count = int(h20["spread_dominated"].sum())
        split_count = int(len(h20))
    else:
        spread_dom_count = 0
        split_count = 0
    lines = [
        "# Old Bid-Only Interpretation",
        "",
        "For `old_bid_only_spread_ge_1`, the Phase 01 bid markout formula can be decomposed as:",
        "",
        "```text",
        "bid_pnl_H = (future_mid_H - best_bid) * POINT_VALUE - MAKER_FEE",
        "          = (mid_t - best_bid) * POINT_VALUE + (future_mid_H - mid_t) * POINT_VALUE - MAKER_FEE",
        "```",
        "",
        "`(mid_t - best_bid)` is the half-spread advantage. `(future_mid_H - mid_t)` is directional mid-price drift. This audit checks whether the apparent PnL is mostly mechanical spread capture rather than a directional forecasting edge.",
        "",
        "## H20 Split-Level Decomposition",
        "",
        h20.to_markdown(index=False) if not h20.empty else "No H20 decomposition rows were available.",
        "",
        "## Interpretation",
        "",
        f"At H20, spread component absolute magnitude exceeds drift component in {spread_dom_count}/{split_count} evaluated split-level rows.",
        "",
        "Advisor-safe conclusion: a large half-spread component indicates proxy artifact risk because Phase 01 assumes candidate quotes receive markout labels without true queue priority or fill probability. Strong old-bid-only PnL should therefore not be promoted to a final trading strategy without stricter execution simulation.",
    ]
    write_markdown(ROOT / "OLD_BID_ONLY_INTERPRETATION.md", lines)
    print("Old bid-only PnL decomposition complete.")
    if not h20.empty:
        print(h20[["split", "total_pnl", "half_spread_advantage_pnl", "mid_price_drift_pnl", "maker_fee_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
