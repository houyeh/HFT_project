from __future__ import annotations

from ev_utils import DATA_DIR, ensure_dirs, make_quote_ev_labels


def run() -> None:
    ensure_dirs()
    labels = make_quote_ev_labels()
    (DATA_DIR / "quote_ev_label_definition.md").write_text(
        "# Quote-EV Label Definition\n\n"
        "These labels are simulator-derived proxy labels. They must not be described as true exchange-level queue-aware PnL.\n\n"
        "For each event `t` and side `s in {bid, ask}`, we construct a side-specific quote row.\n\n"
        "- `fill_proxy_s(t)` is the existing proxy fill condition from 016 labels.\n"
        "- `neutral_u_moderate` is a deterministic hash-based pseudo-random number keyed by `(event_id, side)`.\n"
        "- `fill_kept_moderate = fill_proxy_s(t) AND neutral_u_moderate <= 0.5`.\n"
        "- `side_markout_ticks = future_mid_return_H_ticks` for bid and `-future_mid_return_H_ticks` for ask.\n"
        "- `realized_quote_pnl_moderate = fill_kept_moderate * (0.5 * spread_ticks * POINT_VALUE + side_markout_ticks * POINT_VALUE - MAKER_FEE - 0.5 * max(0, -side_markout_ticks) * POINT_VALUE)`.\n"
        "- `quote_positive_ev_moderate = 1[realized_quote_pnl_moderate > 0]`.\n\n"
        "Moderate stress is locked to `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, and `delay=0`. The label is a research proxy for stress-adjusted quote value, not an exchange-confirmed queue fill or realized trading PnL.\n",
        encoding="utf-8",
    )
    print(f"rows={len(labels)}")


if __name__ == "__main__":
    run()
    print(f"Wrote {DATA_DIR / 'quote_ev_labels.parquet'}")
