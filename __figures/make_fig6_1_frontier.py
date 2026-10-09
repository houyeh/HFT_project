#!/usr/bin/env python3
"""
Figure 6-1: Latency-Value Pareto frontier + energy per inference.

All values are post-route (Vivado P&R, timing met) unless noted.
Power values are Vivado vectorless estimates at "Medium" confidence,
scope = HLS core (phc_step / phc_diag_step), NOT the full board wrapper.

Sources:
  latency/resource : n2ew_era/06u_phc_latency_breakthrough/AUDIT_06u_phc_latency_breakthrough.md
                     n2ew_era/06o_phc_lite_latency_path_to_1us/LATENCY_ROADMAP_SLIDE_TABLE.md
  task value       : n2ew_era/05c4,05e,05f frozen OOS action-level
                     H20 markout sums (NEW3)
  power            : */impl/*_power_routed.rpt
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# name, latency_us, NEW3 action-level H20 markout sum (TWD proxy), DSP %,
# WNS ns, total on-chip power W (None = not exported)
PTS = [
    ("06F v8 baseline",      83.160, 5_237_410, 21.36, "+0.020", 0.357),
    ("06L unroll16 (board)", 25.733, 5_237_410, 56.80, "met",    0.564),
    ("06U V1 @90MHz",        24.778, 5_237_410, 68.64, "+0.352", None),
    ("06U V1 @100MHz",       22.360, 5_237_410, 68.64, "+0.004", None),
    ("06U V2 hidden8",       12.020, 5_032_255, 59.55, "+0.445", None),
    ("06U V3 diagonal",       5.300, 4_727_425, 89.09, "+0.130", 0.449),
]
INFEASIBLE = ("V3 unroll16\nDSP 452 > 220", 3.830)

fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(9.5, 9.0), sharex=True,
    gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.10})

# ---------- Panel (a): latency vs task value ----------
xs = [p[1] for p in PTS]
ys = [p[2] for p in PTS]
sizes = [p[3] * 7 for p in PTS]

ax.plot(xs, ys, "-", color="#9aa7b4", lw=1.6, zorder=1)
for (name, x, y, dsp, wns, pw) in PTS:
    board = "board" in name
    ax.scatter(x, y, s=dsp * 7,
               facecolor="#c23b22" if board else "#2b6cb0",
               edgecolor="black", linewidth=1.1, zorder=3, alpha=0.9)

# infeasible point, placed on its own baseline below the data
Y_INF = 4.555e6
ax.scatter([INFEASIBLE[1]], [Y_INF], marker="x", s=190,
           color="#b00020", linewidth=3, zorder=4)
ax.annotate("V3 unroll16 (csynth only): 3.830 us, DSP 452 > 220 - does not fit",
            (INFEASIBLE[1], Y_INF), textcoords="offset points",
            xytext=(16, -4), fontsize=8.6, color="#b00020", va="center")

box = dict(boxstyle="round,pad=0.42", fc="white", ec="#2b6cb0", lw=0.9, alpha=0.95)
arrow = dict(arrowstyle="-", color="#2b6cb0", lw=0.9)

ax.annotate("06F v8 baseline\n83.160 us | DSP 21.4% | WNS +0.020 | 0.357 W",
            (83.160, 5_237_410), textcoords="offset points", xytext=(-10, -92),
            fontsize=8.4, ha="right", linespacing=1.3, bbox=box, arrowprops=arrow)

ax.annotate("three points share the same model and weights\n"
            "06U V1 @100MHz : 22.360 us | DSP 68.6% | WNS +0.004\n"
            "06U V1 @90MHz  : 24.778 us | DSP 68.6% | WNS +0.352\n"
            "06L unroll16   : 25.733 us | DSP 56.8% | WNS met | 0.564 W   <- ON BOARD",
            (24.0, 5_237_410), textcoords="offset points", xytext=(14, 58),
            fontsize=8.2, ha="center", linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="#c23b22", lw=1.0, alpha=0.96),
            arrowprops=dict(arrowstyle="-", color="#c23b22", lw=0.9))

ax.annotate("06U V2 hidden8\n12.020 us | DSP 59.5% | WNS +0.445",
            (12.020, 5_032_255), textcoords="offset points", xytext=(26, 20),
            fontsize=8.4, linespacing=1.3, bbox=box, arrowprops=arrow)

ax.annotate("06U V3 diagonal\n5.300 us | DSP 89.1% | WNS +0.130 | 0.449 W",
            (5.300, 4_727_425), textcoords="offset points", xytext=(30, 26),
            fontsize=8.4, linespacing=1.3, bbox=box, arrowprops=arrow)

ax.set_xscale("log")
ax.set_xlim(2.9, 190)
ax.set_ylabel("NEW3 action-level H20 markout sum  [TWD proxy]", fontsize=10.5)
ax.set_title("Figure 6-1 (a)  Latency-value frontier on Zynq-7020\n"
             "every point is post-route with timing met; marker area proportional to DSP utilisation",
             fontsize=11, loc="left")
ax.grid(True, which="both", alpha=0.25, ls=":")
ax.set_ylim(4.50e6, 5.56e6)
ax.set_yticks([4.6e6, 4.8e6, 5.0e6, 5.2e6, 5.4e6])
ax.set_yticklabels(["4.6M", "4.8M", "5.0M", "5.2M", "5.4M"])

ax.annotate("", xy=(22.36, 5.345e6), xytext=(83.16, 5.345e6),
            arrowprops=dict(arrowstyle="<->", color="#2f855a", lw=1.9))
ax.text(43, 5.375e6, "3.72x speed-up at ZERO task cost",
        color="#2f855a", fontsize=9.5, ha="center")
ax.annotate("", xy=(5.30, 4.655e6), xytext=(22.36, 4.655e6),
            arrowprops=dict(arrowstyle="<->", color="#b7791f", lw=1.9))
ax.text(10.9, 4.685e6, "further 4.22x costs -9.74% markout",
        color="#b7791f", fontsize=9.5, ha="center")

# ---------- Panel (b): energy per inference ----------
ex = [p[1] for p in PTS if p[5]]
ey = [p[5] * p[1] for p in PTS if p[5]]      # W * us = uJ
en = [p[0] for p in PTS if p[5]]
ax2.plot(ex, ey, "o-", color="#6b46c1", ms=9, lw=1.8,
         markeredgecolor="black", markeredgewidth=1.0)
for x, y, n in zip(ex, ey, en):
    ax2.annotate(f"{n}\n{y:.2f} uJ", (x, y), textcoords="offset points",
                 xytext=(10, -22) if "06F" in n else (10, 6), fontsize=8.2, linespacing=1.25)
ax2.set_xscale("log")
ax2.set_xlabel("Single-inference latency  [us, log scale]", fontsize=10.5)
ax2.set_ylabel("Energy per inference  [uJ]", fontsize=10.5)
ax2.set_title("Figure 6-1 (b)  Energy per inference (12.5x improvement, 29.69 -> 2.38 uJ)",
              fontsize=11, loc="left")
ax2.grid(True, which="both", alpha=0.25, ls=":")
ax2.set_ylim(0, 34)
ax2.set_xticks([5, 10, 20, 50, 100])
ax2.set_xticklabels(["5", "10", "20", "50", "100"])
ax2.set_xlim(2.9, 190)

fig.text(0.012, 0.012,
         "Power: Vivado vectorless estimate, Medium confidence, scope = HLS core only "
         "(not the full board wrapper). Power not exported for 06U V1/V2.\n"
         "Task-value axis: action-level total_excess_pnl_H20 (selected-action H20 markout sum), "
         "not causal-fill sequential PnL.\n"
         "Units are a TWD proxy under POINT_VALUE = 200 and MAKER_FEE = 15; "
         "NEW3 was never used for model or threshold selection.",
         fontsize=7.2, color="#444444", linespacing=1.4)

fig.subplots_adjust(left=0.105, right=0.985, top=0.935, bottom=0.115)
for ext in ("png", "svg"):
    fig.savefig(f"fig6_1_latency_value_frontier.{ext}", dpi=200)
print("written: fig6_1_latency_value_frontier.png / .svg")
for n, x, y, d, w, p in PTS:
    if p: print(f"{n:24s} {x:7.3f} us  {p:.3f} W  -> {p*x:6.2f} uJ")
