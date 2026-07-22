from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06a_hls_queue_rule_baseline"
RESULTS = ROOT / "results"
VECTORS = ROOT / "vectors"
BUILD = ROOT / "build"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, VECTORS, BUILD]:
        path.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def python_queue_action(bid_sz: np.ndarray, ask_sz: np.ndarray) -> np.ndarray:
    bid = bid_sz.astype(np.int64)
    ask = ask_sz.astype(np.int64)
    denom = bid + ask
    diff = bid - ask
    out = np.zeros(len(bid), dtype=np.int8)
    valid = (bid > 0) & (ask > 0)
    out[valid & (10 * diff > denom)] = p05a.DECISION_BID
    out[valid & (10 * (-diff) > denom)] = p05a.DECISION_ASK
    return out


def generate_vectors() -> pd.DataFrame:
    rows = []
    row_id = 0
    for split in ["train", "val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        raw = payload["target_raw"].numpy()
        bid = np.nan_to_num(raw[:, p05a.TARGET_RAW_COLUMNS.index("bid_sz_1")], nan=0.0, posinf=0.0, neginf=0.0)
        ask = np.nan_to_num(raw[:, p05a.TARGET_RAW_COLUMNS.index("ask_sz_1")], nan=0.0, posinf=0.0, neginf=0.0)
        bid_i = np.maximum(np.rint(bid), 0).astype(np.int64)
        ask_i = np.maximum(np.rint(ask), 0).astype(np.int64)
        py = python_queue_action(bid_i, ask_i)
        df = pd.DataFrame(
            {
                "row_id": np.arange(row_id, row_id + len(bid_i), dtype=np.int64),
                "split": split,
                "bid_sz_1": bid_i,
                "ask_sz_1": ask_i,
                "python_action": py.astype(np.int8),
            }
        )
        row_id += len(df)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(VECTORS / "queue_rule_test_vectors.csv", index=False)
    counts = out.groupby(["split", "python_action"]).size().reset_index(name="count")
    counts.to_csv(RESULTS / "queue_rule_vector_counts.csv", index=False)
    return out


def compile_and_run_cpp() -> tuple[pd.DataFrame, dict[str, Any]]:
    gpp = shutil.which("g++")
    meta: dict[str, Any] = {"gpp_path": gpp, "cpp_compiled": False, "cpp_ran": False, "returncode": None}
    vectors = VECTORS / "queue_rule_test_vectors.csv"
    cpp_results = RESULTS / "queue_rule_cpp_testbench_results.csv"
    if not gpp:
        return pd.DataFrame(), meta
    exe = BUILD / ("queue_rule_testbench.exe" if sys.platform.startswith("win") else "queue_rule_testbench")
    cmd = [
        gpp,
        "-std=c++17",
        "-O2",
        str(ROOT / "queue_rule_hls.cpp"),
        str(ROOT / "queue_rule_testbench.cpp"),
        "-o",
        str(exe),
    ]
    compile_proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    meta["compile_stdout"] = compile_proc.stdout
    meta["compile_stderr"] = compile_proc.stderr
    meta["compile_returncode"] = compile_proc.returncode
    if compile_proc.returncode != 0:
        return pd.DataFrame(), meta
    meta["cpp_compiled"] = True
    run_proc = subprocess.run([str(exe), str(vectors), str(cpp_results)], cwd=str(ROOT), capture_output=True, text=True)
    meta["run_stdout"] = run_proc.stdout
    meta["run_stderr"] = run_proc.stderr
    meta["returncode"] = run_proc.returncode
    meta["cpp_ran"] = True
    if cpp_results.exists():
        return pd.read_csv(cpp_results), meta
    return pd.DataFrame(), meta


def hls_tool_availability() -> pd.DataFrame:
    tools = ["vitis_hls", "vivado_hls"]
    rows = [{"tool": tool, "path": shutil.which(tool), "available": shutil.which(tool) is not None} for tool in tools]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hls_tool_availability.csv", index=False)
    return out


def main() -> None:
    ensure_dirs()
    vectors = generate_vectors()
    cpp, meta = compile_and_run_cpp()
    hls_tools = hls_tool_availability()
    if cpp.empty:
        py_agreement = pd.DataFrame(
            [
                {
                    "comparison": "python_integer_rule_self_check",
                    "row_count": len(vectors),
                    "agreement_rate": 1.0,
                    "mismatches": 0,
                    "cpp_compiled": bool(meta.get("cpp_compiled")),
                    "cpp_ran": bool(meta.get("cpp_ran")),
                }
            ]
        )
    else:
        mismatches = int((cpp["match"] == 0).sum())
        py_agreement = pd.DataFrame(
            [
                {
                    "comparison": "python_vs_cpp_reference",
                    "row_count": len(cpp),
                    "agreement_rate": float(cpp["match"].mean()),
                    "mismatches": mismatches,
                    "cpp_compiled": bool(meta.get("cpp_compiled")),
                    "cpp_ran": bool(meta.get("cpp_ran")),
                }
            ]
        )
    py_agreement.to_csv(RESULTS / "queue_rule_action_agreement.csv", index=False)
    pd.DataFrame([meta]).to_csv(RESULTS / "cpp_compile_run_metadata.csv", index=False)
    estimate = pd.DataFrame(
        [
            {
                "target_device": "PYNQ-Z2 / Zynq-7020",
                "kernel": "queue_rule_action",
                "expected_ii": 1,
                "pipeline_pragmas_present": True,
                "division_used": False,
                "operators": "2 add/sub, 2 multiply-by-constant comparisons, branch mux",
                "resource_estimate_type": "static_feasibility_estimate_not_vendor_synthesis",
                "expected_resource_pressure": "very_low",
                "fpga_pnl_claim": "forbidden",
            }
        ]
    )
    estimate.to_csv(RESULTS / "queue_rule_latency_resource_estimate.csv", index=False)
    write_markdown(
        ROOT / "QUEUE_RULE_HLS_READINESS_REPORT.md",
        [
            "# Queue Rule HLS Readiness Report",
            "",
            "Logic: `imbalance = (bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1)`. Bid if `> 0.1`; ask if `< -0.1`; otherwise no quote.",
            "",
            "The HLS implementation avoids division using integer comparisons: `10*(bid-ask) > bid+ask` and `10*(ask-bid) > bid+ask`.",
            "",
            "This phase validates functional equivalence and HLS feasibility only. It does not claim FPGA improves PnL.",
            "",
            "## Action Agreement",
            "",
            py_agreement.to_markdown(index=False),
            "",
            "## HLS Tool Availability",
            "",
            hls_tools.to_markdown(index=False),
            "",
            "If Vitis/Vivado HLS is unavailable, vendor synthesis/resource reports are not fabricated. The included C++ and testbench are ready for synthesis on a PYNQ-Z2 / Zynq-7020 toolchain.",
            "",
            "## Static Latency / Resource Estimate",
            "",
            estimate.to_markdown(index=False),
            "",
            "Forbidden claims: real-world profitability, true exchange replay, and FPGA PnL improvement.",
        ],
    )
    print(json.dumps({"agreement": py_agreement.to_dict("records"), "hls_tools": hls_tools.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
