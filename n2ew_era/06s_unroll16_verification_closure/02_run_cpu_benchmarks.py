from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06s_unroll16_verification_closure"
RESULTS = ROOT / "results"
CPU_RESULTS = RESULTS / "cpu"

P06F = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls"
BENCH_EXE = P06F / "results" / "libtorch_build" / "cpp_libtorch_concurrent_benchmark.exe"
TORCHSCRIPT_MODEL = P06F / "models" / "phc_stp_step_hls_seed2_torchscript.pt"
TORCH_LIB_DIR_CANDIDATES = [
    PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
]

CONCURRENCY_LEVELS = [1, 4, 16]
ROUNDS = 2
TIMING_ITERS_PER_STREAM = 10000
WARMUP_ITERS = 100
TORCH_THREADS = 8
DEADLINES_US = {
    "83p16": 83.160,
    "25p733": 25.733,
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def run_powershell(command: str, timeout: int = 30) -> dict[str, object]:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def capture_cpu_snapshot(label: str) -> Path:
    CPU_RESULTS.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "label": label,
        "captured_at_unix": time.time(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "os_cpu_count": None,
        "wmic_cpu": None,
        "load_counter": None,
        "top_processes": None,
    }
    try:
        snapshot["os_cpu_count"] = __import__("os").cpu_count()
    except Exception as exc:  # pragma: no cover - diagnostic only
        snapshot["os_cpu_count_error"] = repr(exc)

    snapshot["wmic_cpu"] = run_powershell(
        "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress"
    )
    snapshot["load_counter"] = run_powershell(
        "Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 1 | "
        "Select-Object -ExpandProperty CounterSamples | Select-Object CookedValue | ConvertTo-Json -Compress"
    )
    snapshot["top_processes"] = run_powershell(
        "Get-Process | Sort-Object CPU -Descending | Select-Object -First 15 ProcessName,Id,CPU,WorkingSet64 | ConvertTo-Json -Compress",
        timeout=45,
    )
    out = CPU_RESULTS / f"cpu_snapshot_{label}.json"
    out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_raw(raw_csv: Path, source_json: Path) -> list[dict[str, object]]:
    rows_by_round: dict[int, list[float]] = {}
    if not raw_csv.exists():
        return []
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                round_id = int(float(row["round"]))
                latency = float(row["latency_us"])
            except (KeyError, ValueError):
                continue
            rows_by_round.setdefault(round_id, []).append(latency)

    out: list[dict[str, object]] = []
    for round_id, values in sorted(rows_by_round.items()):
        n = len(values)
        if n == 0:
            continue
        mean = sum(values) / n
        median = percentile(values, 0.5)
        p99 = percentile(values, 0.99)
        row = {
            "source": rel(source_json),
            "raw_latency_csv": rel(raw_csv),
            "round": round_id,
            "total_measurements": n,
            "concurrency": None,
            "warmup_iters": WARMUP_ITERS,
            "timing_iters_per_stream": TIMING_ITERS_PER_STREAM,
            "torch_threads": TORCH_THREADS,
            "mean_us": mean,
            "median_us": median,
            "p99_us": p99,
            "min_us": min(values),
            "max_us": max(values),
            "jitter_p99_minus_median_us": (p99 - median) if p99 is not None and median is not None else None,
        }
        for key, deadline in DEADLINES_US.items():
            misses = sum(1 for v in values if v > deadline)
            row[f"deadline_miss_count_{key}"] = misses
            row[f"deadline_miss_rate_{key}_pct"] = 100.0 * misses / n
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one(concurrency: int) -> dict[str, object]:
    if not BENCH_EXE.exists():
        raise FileNotFoundError(BENCH_EXE)
    if not TORCHSCRIPT_MODEL.exists():
        raise FileNotFoundError(TORCHSCRIPT_MODEL)

    out_json = CPU_RESULTS / f"cpu_concurrent_N{concurrency}_06s.json"
    raw_csv = CPU_RESULTS / f"cpu_concurrent_N{concurrency}_06s_raw.csv"
    log_path = CPU_RESULTS / f"cpu_concurrent_N{concurrency}_06s.log"
    command = [
        str(BENCH_EXE),
        str(TORCHSCRIPT_MODEL),
        str(out_json),
        str(raw_csv),
        str(concurrency),
        str(ROUNDS),
        str(TIMING_ITERS_PER_STREAM),
        str(WARMUP_ITERS),
        str(TORCH_THREADS),
    ]
    env = os.environ.copy()
    torch_lib_dirs = [p for p in TORCH_LIB_DIR_CANDIDATES if (p / "torch_cpu.dll").exists()]
    if torch_lib_dirs:
        env["PATH"] = str(torch_lib_dirs[0]) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=60 * 30,
        env=env,
    )
    log = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "prepended_torch_lib_dir": rel(torch_lib_dirs[0]) if torch_lib_dirs else None,
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return {
        "concurrency": concurrency,
        "returncode": proc.returncode,
        "out_json": out_json,
        "raw_csv": raw_csv,
        "log_path": log_path,
    }


def main() -> None:
    CPU_RESULTS.mkdir(parents=True, exist_ok=True)
    before = capture_cpu_snapshot("before")
    run_records = []
    summary_rows: list[dict[str, object]] = []
    for concurrency in CONCURRENCY_LEVELS:
        print(f"running LibTorch benchmark N={concurrency}")
        record = run_one(concurrency)
        run_records.append({k: rel(v) if isinstance(v, Path) else v for k, v in record.items()})
        rows = summarize_raw(record["raw_csv"], record["out_json"])
        for row in rows:
            row["concurrency"] = concurrency
        summary_rows.extend(rows)
        if record["returncode"] != 0:
            print(f"WARNING: benchmark N={concurrency} exited {record['returncode']}; see {rel(record['log_path'])}")
    after = capture_cpu_snapshot("after")

    write_csv(CPU_RESULTS / "cpu_concurrent_benchmark_06s_summary.csv", summary_rows)
    manifest = {
        "phase": "06s",
        "benchmark_exe": rel(BENCH_EXE),
        "torchscript_model": rel(TORCHSCRIPT_MODEL),
        "concurrency_levels": CONCURRENCY_LEVELS,
        "rounds": ROUNDS,
        "timing_iters_per_stream": TIMING_ITERS_PER_STREAM,
        "warmup_iters": WARMUP_ITERS,
        "torch_threads": TORCH_THREADS,
        "deadline_us": DEADLINES_US,
        "cpu_snapshot_before": rel(before),
        "cpu_snapshot_after": rel(after),
        "runs": run_records,
        "new5_used": False,
    }
    (CPU_RESULTS / "cpu_benchmark_manifest_06s.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {rel(CPU_RESULTS / 'cpu_concurrent_benchmark_06s_summary.csv')}")
    for row in summary_rows:
        print(
            "N={concurrency} round={round} mean_us={mean_us:.3f} median_us={median_us:.3f} "
            "p99_us={p99_us:.3f} max_us={max_us:.3f} miss83={deadline_miss_rate_83p16_pct:.3f}% "
            "miss25={deadline_miss_rate_25p733_pct:.3f}%".format(**row)
        )


if __name__ == "__main__":
    main()
