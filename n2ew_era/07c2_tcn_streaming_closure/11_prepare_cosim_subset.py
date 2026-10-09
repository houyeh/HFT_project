from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

VARIANTS = ("scratch", "distilled")
SAMPLES_FOR_COSIM = 4
EVENTS_PER_SAMPLE = 252


def component_dir(variant: str) -> Path:
    return ROOT / f"hls_component_stage2_resource_{variant}_shared_u1"


def subset_path(variant: str) -> Path:
    return RESULTS / f"stage2_streaming_{variant}_test_vectors_first4_windows.csv"


def summary_path(variant: str) -> Path:
    return RESULTS / f"stage2_closure_{variant}_shared_u1_event1008_tb_summary.csv"


def make_subset(variant: str) -> dict[str, object]:
    src = RESULTS / f"stage2_streaming_{variant}_test_vectors.csv"
    dst = subset_path(variant)
    if not src.exists():
        raise FileNotFoundError(src)

    rows_written = 0
    with src.open("r", newline="", encoding="utf-8") as inp, dst.open("w", newline="", encoding="utf-8") as out:
        reader = csv.reader(inp)
        writer = csv.writer(out, lineterminator="\n")
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if rows_written >= SAMPLES_FOR_COSIM:
                break
            writer.writerow(row)
            rows_written += 1

    if rows_written != SAMPLES_FOR_COSIM:
        raise RuntimeError(f"{variant}: wrote {rows_written} rows, expected {SAMPLES_FOR_COSIM}")

    cfg = component_dir(variant) / "hls_config.cfg"
    text = cfg.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.startswith("csim.argv="):
            line = f"csim.argv={dst.resolve().as_posix()} {summary_path(variant).resolve().as_posix()}"
        lines.append(line)
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "variant": variant,
        "source_full_vectors": str(src),
        "subset_vectors": str(dst),
        "window_rows": rows_written,
        "streaming_top_event_vectors": rows_written * EVENTS_PER_SAMPLE,
        "readout_decision_rows": rows_written,
        "tb_summary": str(summary_path(variant)),
        "hls_config": str(cfg),
        "new5_used": False,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = [make_subset(variant) for variant in VARIANTS]
    out = RESULTS / "stage2_closure_cosim_subset_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for row in manifest:
        print(
            f"{row['variant']}: {row['window_rows']} windows, "
            f"{row['streaming_top_event_vectors']} streaming event vectors -> {row['subset_vectors']}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
