from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common_034 import FIGURES, ROOT, ensure_dirs, set_style, write_markdown


def draw_baseline_flowchart() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(11, 3.7))
    ax.axis("off")
    boxes = [
        ("Quote-eligible event", "valid L1 book\nbid_px > 0, ask_px > 0\nask >= bid"),
        ("Candidate actions", "make passive bid action\nmake passive ask action"),
        ("Core rule", "keep only\nside == bid\nand spread_ticks >= 1"),
        ("Output action", "quote bid\nblock ask and invalid rows"),
        ("Interpretation", "regime filter\nnot fine-grained alpha"),
    ]
    xs = [0.04, 0.25, 0.46, 0.67, 0.84]
    colors = ["#e7eef8", "#eaf7f0", "#fff3d6", "#e9f1ff", "#f5e9ef"]
    for i, ((title, body), x, color) in enumerate(zip(boxes, xs, colors)):
        patch = FancyBboxPatch(
            (x, 0.31),
            0.15,
            0.40,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color,
            edgecolor="#344054",
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + 0.075, 0.62, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(x + 0.075, 0.45, body, ha="center", va="center", fontsize=8.8)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.015, 0.51),
                xytext=(x + 0.165, 0.51),
                arrowprops=dict(arrowstyle="->", color="#344054", lw=1.4),
            )
    ax.set_title("bid_only_spread_ge_1 Baseline Mechanics", fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_flowchart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, width, height, title, body="", face="#eef2ff", edge="#344054", fontsize=9):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.68, title, ha="center", va="center", fontsize=fontsize + 0.5, fontweight="bold")
    if body:
        ax.text(xy[0] + width / 2, xy[1] + height * 0.38, body, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, lw=1.3, color="#344054"))


def draw_quote_bid_ask_diagram() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axvline(0.5, color="#d9dee7", lw=1)
    add_box(ax, (0.08, 0.62), 0.28, 0.20, "Passive Bid", "buy limit at best bid\nfilled if sell flow hits bid", face="#e8f3ff")
    add_box(ax, (0.08, 0.22), 0.28, 0.20, "Bid Adverse Selection", "buy, then mid price falls\ninventory loses value", face="#ffe8e5")
    add_box(ax, (0.64, 0.62), 0.28, 0.20, "Passive Ask", "sell limit at best ask\nfilled if buy flow lifts ask", face="#e8f7ed")
    add_box(ax, (0.64, 0.22), 0.28, 0.20, "Ask Adverse Selection", "sell, then mid price rises\nsold too low", face="#ffe8e5")
    arrow(ax, (0.22, 0.62), (0.22, 0.43))
    arrow(ax, (0.78, 0.62), (0.78, 0.43))
    ax.text(0.5, 0.93, "Passive Quote Mechanics and Side-Specific Adverse Selection", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.06, "Spread capture compensates liquidity provision; adverse selection is the cost of fills when price moves against the quote.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "quote_bid_ask_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_l1_l2_book_diagram() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.94, "L1 Baseline vs L2 ML Feature Usage", ha="center", fontsize=14, fontweight="bold")
    add_box(ax, (0.07, 0.55), 0.34, 0.25, "L1 Baseline Inputs", "best bid\nbest ask\nspread ticks\nquote side", face="#e8f3ff")
    add_box(ax, (0.59, 0.55), 0.34, 0.25, "L2 / History ML Inputs", "depth levels\nqueue imbalance\nvolatility\nevent rate", face="#f3e8ff")
    add_box(ax, (0.07, 0.18), 0.34, 0.20, "Hardware Mapping", "integer comparators\nsimple control gate", face="#e8f7ed")
    add_box(ax, (0.59, 0.18), 0.34, 0.20, "Extra Audit Burden", "feature extraction\nscore calibration\nOOS transfer checks", face="#fff3d6")
    arrow(ax, (0.24, 0.55), (0.24, 0.39))
    arrow(ax, (0.76, 0.55), (0.76, 0.39))
    fig.tight_layout()
    fig.savefig(FIGURES / "l1_l2_book_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_simulation_environment_table() -> None:
    set_style()
    rows = [
        ["Market data", "L1/L2 book snapshots", "full queue replay"],
        ["Execution", "passive quote proxy", "real exchange acks/fills"],
        ["Queue", "haircut/delay stress", "true queue priority"],
        ["Labels", "proxy quote EV", "ground-truth aggressor fills"],
        ["Hardware", "HLS comparator plan", "PnL necessity proof"],
    ]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Area", "Considered", "Omitted"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8f3ff")
            cell.set_text_props(weight="bold")
        elif c == 2:
            cell.set_facecolor("#fff0ee")
        else:
            cell.set_facecolor("#ffffff")
    ax.set_title("Simulation Environment: Considered vs Omitted", fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "simulation_environment_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_latency_path_diagram() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nodes = [
        ("Market Data", "feed receive"),
        ("Parser", "normalize book"),
        ("Controller", "side/spread gate"),
        ("Order I/O", "send/cancel"),
        ("Exchange Path", "network + venue"),
    ]
    xs = [0.05, 0.25, 0.45, 0.65, 0.83]
    for (title, body), x in zip(nodes, xs):
        add_box(ax, (x, 0.43), 0.13, 0.23, title, body, face="#e8f3ff")
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + 0.13, 0.54), (xs[i + 1], 0.54))
    ax.add_patch(Rectangle((0.43, 0.34), 0.17, 0.40, fill=False, edgecolor="#c43c35", lw=1.5, linestyle="--"))
    ax.text(0.515, 0.28, "FPGA targets deterministic compute jitter here", ha="center", fontsize=9, color="#7a211b")
    ax.text(0.5, 0.90, "CPU vs FPGA Latency Path: compute is only one component", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.12, "Do not claim FPGA improves PnL; use only latency risk-control and jitter framing.", ha="center", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "latency_path_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_ml_future_work_decision_tree() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.94, "ML Comeback Decision Tree", ha="center", fontsize=14, fontweight="bold")
    add_box(ax, (0.38, 0.74), 0.24, 0.14, "Residual veto failed?", "test/NEW4 did not pass", face="#ffe8e5")
    add_box(ax, (0.08, 0.48), 0.22, 0.15, "Too narrow?", "baseline universe only", face="#fff3d6")
    add_box(ax, (0.39, 0.48), 0.22, 0.15, "Calibration shift?", "static thresholds fragile", face="#fff3d6")
    add_box(ax, (0.70, 0.48), 0.22, 0.15, "Missing sequence?", "tabular signal weak", face="#fff3d6")
    add_box(ax, (0.06, 0.18), 0.24, 0.16, "Full-action EV model", "bid + ask selector", face="#e8f7ed")
    add_box(ax, (0.38, 0.18), 0.24, 0.16, "Rolling calibration", "past-only quantiles", face="#e8f7ed")
    add_box(ax, (0.70, 0.18), 0.24, 0.16, "Causal TCN research", "external GPU pack", face="#e8f7ed")
    arrow(ax, (0.50, 0.74), (0.19, 0.63))
    arrow(ax, (0.50, 0.74), (0.50, 0.63))
    arrow(ax, (0.50, 0.74), (0.81, 0.63))
    arrow(ax, (0.19, 0.48), (0.18, 0.34))
    arrow(ax, (0.50, 0.48), (0.50, 0.34))
    arrow(ax, (0.81, 0.48), (0.82, 0.34))
    ax.text(0.5, 0.06, "Any ML comeback must beat non-ML baseline, matched random controls, and fresh OOS/walk-forward tests.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_future_work_decision_tree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_modules() -> None:
    write_markdown(
        ROOT / "baseline_method_explanation.md",
        [
            "# Baseline Method Explanation",
            "",
            "The final non-ML controller is `bid_only_spread_ge_1`.",
            "",
            "Decision logic:",
            "",
            "- Start from quote-eligible rows with a valid L1 book: positive best bid, positive best ask, and non-negative spread.",
            "- Construct two passive candidate actions per event: one bid quote and one ask quote.",
            "- Keep a candidate only when `side == bid` and `spread_ticks >= 1`.",
            "- The rule does not read future returns, quote EV, toxicity labels, oracle labels, or model scores.",
            "",
            "Important interpretation:",
            "",
            "- In the current quote-eligible universe, all candidate spreads are at least one tick, so `spread_ticks >= 1` is saturated.",
            "- The controller is therefore best described as a simple bid-side regime filter, not a fine-grained alpha model.",
            "- The evidence is simulator-derived proxy robustness, not proof of real-world profitability.",
            "",
            "Figure: `figures/baseline_flowchart.png`.",
        ],
    )
    write_markdown(
        ROOT / "quote_bid_ask_explanation.md",
        [
            "# Quote Bid/Ask Mechanics",
            "",
            "A passive market maker posts limit orders instead of crossing the spread.",
            "",
            "- Passive bid: place a buy limit order at or near the best bid. If filled, the strategy buys inventory.",
            "- Passive ask: place a sell limit order at or near the best ask. If filled, the strategy sells inventory.",
            "- Spread capture is the intended compensation for providing liquidity.",
            "- Adverse selection is the risk that the quote is filled because informed flow is moving against it.",
            "",
            "Side-specific adverse examples:",
            "",
            "- Bid adverse selection: the quote buys, then the mid price falls. The position is immediately underwater.",
            "- Ask adverse selection: the quote sells, then the mid price rises. The strategy has sold too low.",
            "",
            "The project uses proxy quote EV and stress-adjusted fill assumptions because true exchange fills and queue position are not available.",
            "",
            "Figure: `figures/quote_bid_ask_diagram.png`.",
        ],
    )
    write_markdown(
        ROOT / "l1_vs_l2_data_usage.md",
        [
            "# L1 vs L2 Data Usage",
            "",
            "The deployed baseline uses only L1-level information:",
            "",
            "- best bid price",
            "- best ask price",
            "- spread in ticks",
            "- quote side",
            "",
            "The rejected ML extensions used richer causal features derived from L2 or event history, such as queue imbalance, depth ratios, volatility, and event rate. Those features may still be useful for future research, but they did not pass the incremental value criteria in the residual-veto setting.",
            "",
            "This distinction matters for hardware design: the baseline maps to simple comparators, while ML/L2 policies require feature extraction, score calibration, and additional audit burden.",
            "",
            "Figure: `figures/l1_l2_book_diagram.png`.",
        ],
    )
    write_markdown(
        ROOT / "exact_matched_random_explanation.md",
        [
            "# Exact Matched Random Explanation",
            "",
            "The Phase 032 exact matched random quote baseline produced p-value `1.0` and zero variance. That is not evidence against or for the controller; it is a saturation diagnostic.",
            "",
            "Why it saturates:",
            "",
            "- The controller selects `side == bid` and `spread_ticks >= 1`.",
            "- In the quote-eligible action universe, every bid action has `spread_ticks >= 1`.",
            "- Therefore the policy selects the entire exact candidate universe.",
            "- A random selector constrained to the same exact universe and the same session/spread-bucket counts has no alternative quotes left to sample.",
            "",
            "Correct wording: the exact matched random test is non-informative for quote-selection skill.",
            "",
            "Forbidden wording: the baseline beat exact matched random quote selection.",
        ],
    )
    write_markdown(
        ROOT / "ml_model_results_explanation.md",
        [
            "# ML Model Results Explanation",
            "",
            "The ML workstreams tested whether residual ML could improve the conservative baseline by vetoing bad quotes inside the `bid_only_spread_ge_1` universe.",
            "",
            "Inputs and safeguards:",
            "",
            "- Training used causal tabular features such as spread, queue imbalance, depth, volatility, event rate, and imbalance changes.",
            "- Thresholds were selected on validation only.",
            "- Frozen test and NEW4 were not used for model selection.",
            "- ML had to beat matched random veto 95p and improve baseline PnL at low veto rates.",
            "",
            "Observed outcome:",
            "",
            "- Validation sometimes showed small improvements.",
            "- Frozen test produced zero useful vetoes or no improvement.",
            "- NEW4 did not survive the matched-random veto hurdle.",
            "",
            "Conservative interpretation: residual ML did not pass the incremental-value standard. This does not prove all ML is useless; it says this residual veto framing was too narrow or unstable.",
            "",
            "Phase 034 recovered this gap by running `small_mlp_64_32_epoch_rerun` on the same residual universe with validation-only early stopping. The real epoch curves are now included as `figures/ml_epoch_loss_curves.png`, `figures/ml_epoch_metric_curves.png`, and `figures/ml_overfit_diagnostic.png`.",
            "",
            "The recovered curves improve transparency but do not revive ML: the frozen rerun still failed to improve both test and NEW4 over the non-ML baseline.",
        ],
    )
    write_markdown(
        ROOT / "ml_comeback_future_work.md",
        [
            "# ML Comeback Future Work",
            "",
            "The conservative baseline has two pain points:",
            "",
            "- It is blunt: in this universe it is effectively `bid_all`, not fine-grained quote selection.",
            "- It may miss positive-EV opportunities outside the bid-only region.",
            "",
            "Future ML routes should avoid repeating the residual-veto trap:",
            "",
            "- Full-action EV model: predict stress-adjusted quote EV across bid and ask actions, not only inside the baseline universe.",
            "- Dynamic regime classifier: learn when the bid-side regime is active rather than hard-coding it globally.",
            "- Rolling calibration: use past-only quantile thresholds instead of static absolute scores.",
            "- Sequence model research: use the exported RTX training pack for causal TCNs, while keeping validation-only policy selection.",
            "",
            "Any comeback must still beat strong non-ML baselines, matched random controls, and fresh OOS/walk-forward tests.",
            "",
            "Figure: `figures/ml_future_work_decision_tree.png`.",
        ],
    )
    write_markdown(
        ROOT / "oos_stress_methodology.md",
        [
            "# OOS and Stress Methodology",
            "",
            "The report separates three types of evidence:",
            "",
            "- Frozen historical test: the original held-out split used before later diagnostic phases burned it as pristine.",
            "- External proxy OOS: NEW/NEW3/NEW4-style rebuilt datasets, with NEW/NEW1 dedup caveats.",
            "- Stress tests: simulator-derived perturbations of fills, adverse losses, fees, and timing.",
            "",
            "The stress figures should be read as robustness diagnostics. They do not replay real exchange queue priority or acknowledge true fill events.",
        ],
    )
    write_markdown(
        ROOT / "queue_aware_stress_explanation.md",
        [
            "# Queue-Aware Stress Explanation",
            "",
            "Queue-aware stress is implemented as a proxy stress framework.",
            "",
            "- Fill haircut reduces realized fills to simulate worse queue priority.",
            "- Good-fill haircut preferentially removes favorable fills for pessimistic analysis.",
            "- Toxic multiplier increases the penalty of adverse fills.",
            "- Cancel delay shifts the action timing to mimic slower control.",
            "",
            "Exact proxy formula from `new_era/18_queue_aware/queue_stress_core.py`:",
            "",
            "- Effective cancel mask: if `delay <= 0`, `pull_eff[t] = pull[t]`; otherwise `pull_eff[t] = pull[t - delay]` and the first `delay` rows cannot be cancelled by that signal.",
            "- A candidate fill is kept only if `neutral_u <= neutral_fill_haircut`.",
            "- For favorable fills, an additional pessimistic filter applies: keep only if `good_u <= good_fill_haircut`.",
            "- Favorable definition: bid fill is favorable when `future_mid_return_H_ticks > 0`; ask fill is favorable when `future_mid_return_H_ticks < 0`.",
            "- `extra_mult = max(toxic_multiplier - 1, 0)`.",
            "- Bid adverse ticks: `max(0, -future_mid_return_H_ticks)`.",
            "- Ask adverse ticks: `max(0, future_mid_return_H_ticks)`.",
            "- `toxic_penalty_ticks = extra_mult * adverse_ticks`.",
            "- Bid fill cash update: `cash -= bid_price * POINT_VALUE + toxic_penalty_ticks * POINT_VALUE`; `fees += MAKER_FEE`; `inventory += 1`.",
            "- Ask fill cash update: `cash += ask_price * POINT_VALUE - toxic_penalty_ticks * POINT_VALUE`; `fees += MAKER_FEE`; `inventory -= 1`.",
            "- Session final equity: `cash - fees + inventory * final_mid * POINT_VALUE`.",
            "",
            "Equivalent single-fill final mark-to-market contribution:",
            "",
            "- Bid: `(final_mid - bid_price) * POINT_VALUE - toxic_penalty_ticks * POINT_VALUE - MAKER_FEE`.",
            "- Ask: `(ask_price - final_mid) * POINT_VALUE - toxic_penalty_ticks * POINT_VALUE - MAKER_FEE`.",
            "",
            "These stressors are useful for conservative sensitivity analysis, but they are not true queue replay. The data lacks real order IDs, acknowledgements, queue position, and exchange-level fill priority.",
        ],
    )
    write_markdown(
        ROOT / "oracle_upper_bound_explanation.md",
        [
            "# Oracle Upper Bound Explanation",
            "",
            "Oracle policies use future information and are not tradable. They answer a feasibility question: is positive moderate-stress PnL mathematically available in the proxy environment?",
            "",
            "Two relevant oracle concepts:",
            "",
            "- Perfect toxic cancel oracle: blocks fills known to be toxic using future information.",
            "- Perfect quote PnL filter oracle: keeps only quotes whose realized stress-adjusted proxy EV is positive.",
            "",
            "A large gap between the oracle and causal policies means opportunity exists in the proxy data, but current causal rules or ML features do not capture it reliably.",
        ],
    )
    write_markdown(
        ROOT / "simulation_environment_and_limitations.md",
        [
            "# Simulation Environment and Limitations",
            "",
            "| Area | Considered | Omitted |",
            "| --- | --- | --- |",
            "| Market data | Best bid/ask, spread, L2-derived features for ML diagnostics | Full order-by-order queue reconstruction |",
            "| Execution | Passive quote proxy and stress-adjusted quote EV | Real exchange acknowledgements and cancellations |",
            "| Queue | Haircuts and delay stress proxies | True queue priority and order ID matching |",
            "| Labels | Simulator-derived quote EV and toxicity proxies | Ground-truth aggressor-side exchange fills |",
            "| Hardware | HLS-friendly comparator logic and latency framing | Proof that FPGA deployment improves PnL |",
            "",
            "All profitability statements must remain proxy statements.",
            "",
            "Figure: `figures/simulation_environment_table.png`.",
        ],
    )
    write_markdown(
        ROOT / "cpu_fpga_latency_plan.md",
        [
            "# CPU vs FPGA Latency Plan",
            "",
            "Latency should be decomposed into separate components:",
            "",
            "- Compute latency: time to evaluate the controller once market data is available.",
            "- I/O latency: time to receive data and transmit orders.",
            "- Exchange/network latency: venue path, gateway, acknowledgement, and matching-engine timing.",
            "- Jitter: variability in latency, often more important than average latency for risk-control systems.",
            "",
            "Conservative FPGA claim:",
            "",
            "- FPGA implementation can reduce deterministic compute jitter for simple comparator logic.",
            "- This supports a latency risk-control framing.",
            "- It does not prove FPGA improves PnL or is necessary for production profitability.",
            "",
            "Figure: `figures/latency_path_diagram.png`.",
        ],
    )


def main() -> None:
    ensure_dirs()
    draw_baseline_flowchart()
    draw_quote_bid_ask_diagram()
    draw_l1_l2_book_diagram()
    draw_simulation_environment_table()
    draw_latency_path_diagram()
    draw_ml_future_work_decision_tree()
    write_modules()


if __name__ == "__main__":
    main()
    print(f"Wrote method modules and baseline flowchart to {ROOT}")
