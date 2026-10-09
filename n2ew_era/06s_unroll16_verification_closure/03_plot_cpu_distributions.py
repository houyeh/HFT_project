from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s_unroll16_verification_closure"
CPU_RESULTS = ROOT / "results" / "cpu"
FIGURES = ROOT / "figures"


def load_raw(path: Path) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                round_id = int(float(row["round"]))
                latency = float(row["latency_us"])
            except (KeyError, ValueError):
                continue
            out.setdefault(round_id, []).append(latency)
    return out


def plot_ecdf() -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in [1, 4, 16]:
        raw = load_raw(CPU_RESULTS / f"cpu_concurrent_N{n}_06s_raw.csv")
        for round_id, values in sorted(raw.items()):
            if not values:
                continue
            xs = sorted(values)
            ys = [(i + 1) / len(xs) for i in range(len(xs))]
            ax.plot(xs, ys, linewidth=1.4, label=f"N={n} round={round_id}")
    ax.axvline(25.733, color="tab:red", linestyle="--", linewidth=1.1, label="25.733 us")
    ax.axvline(83.160, color="tab:green", linestyle="--", linewidth=1.1, label="83.160 us")
    ax.set_xscale("log")
    ax.set_xlabel("Latency per inference (us, log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("06S LibTorch CPU Latency Distributions")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    out = FIGURES / "cpu_libtorch_latency_ecdf_06s.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_ecdf()
    print(out)


if __name__ == "__main__":
    main()
