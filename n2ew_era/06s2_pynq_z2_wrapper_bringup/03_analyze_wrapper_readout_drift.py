from __future__ import annotations

import argparse
import csv
from pathlib import Path


WORD_BITS = 24
SIGN_BIT = 1 << (WORD_BITS - 1)
WORD_MASK = (1 << WORD_BITS) - 1
Q_SCALE = 4096.0

DECISION_NO_QUOTE = 0
DECISION_BID = 1
DECISION_ASK = 2


def signed24(value: int) -> int:
    value &= WORD_MASK
    if value & SIGN_BIT:
        return value - (1 << WORD_BITS)
    return value


def parse_hex24(text: str) -> int:
    return int(text.strip().replace("0x", ""), 16) & WORD_MASK


def policy_from_lsb(bid_lsb: int, ask_lsb: int, threshold_lsb: int = 0) -> int:
    if bid_lsb >= ask_lsb and bid_lsb > threshold_lsb:
        return DECISION_BID
    if ask_lsb > bid_lsb and ask_lsb > threshold_lsb:
        return DECISION_ASK
    return DECISION_NO_QUOTE


def decision_name(decision: int) -> str:
    return {
        DECISION_NO_QUOTE: "no_quote",
        DECISION_BID: "bid",
        DECISION_ASK: "ask",
    }[decision]


def policy_margin_lsb(bid_lsb: int, ask_lsb: int, threshold_lsb: int = 0) -> int:
    decision = policy_from_lsb(bid_lsb, ask_lsb, threshold_lsb)
    if decision == DECISION_BID:
        return min(bid_lsb - threshold_lsb, bid_lsb - ask_lsb)
    if decision == DECISION_ASK:
        return min(ask_lsb - threshold_lsb, ask_lsb - bid_lsb)
    return max(threshold_lsb - bid_lsb, threshold_lsb - ask_lsb)


def analyze_run(run_name: str, dump_path: Path, out_dir: Path) -> dict[str, str | int | float]:
    rows: list[dict[str, str]] = []
    with dump_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    drift_rows: list[dict[str, str | int | float]] = []
    nonzero_deltas: list[int] = []
    decision_flip_count = 0
    max_abs_diff = 0
    min_abs_expected_best = None
    min_policy_margin = None

    for row in rows:
        actual0 = signed24(parse_hex24(row["actual0_hex"]))
        expected0 = signed24(parse_hex24(row["expected0_hex"]))
        actual1 = signed24(parse_hex24(row["actual1_hex"]))
        expected1 = signed24(parse_hex24(row["expected1_hex"]))
        delta0 = actual0 - expected0
        delta1 = actual1 - expected1
        diff0 = abs(delta0)
        diff1 = abs(delta1)
        max_abs_diff = max(max_abs_diff, diff0, diff1)

        expected_decision = policy_from_lsb(expected0, expected1)
        actual_decision = policy_from_lsb(actual0, actual1)
        decision_changed = expected_decision != actual_decision
        if decision_changed:
            decision_flip_count += 1

        expected_best_abs = abs(max(expected0, expected1))
        expected_margin = policy_margin_lsb(expected0, expected1)
        if min_abs_expected_best is None or expected_best_abs < min_abs_expected_best:
            min_abs_expected_best = expected_best_abs
        if min_policy_margin is None or expected_margin < min_policy_margin:
            min_policy_margin = expected_margin

        if diff0 != 0 or diff1 != 0:
            if delta0 != 0:
                nonzero_deltas.append(delta0)
            if delta1 != 0:
                nonzero_deltas.append(delta1)
            drift_rows.append(
                {
                    "run": run_name,
                    "index": int(row["index"]),
                    "actual0_hex": row["actual0_hex"],
                    "expected0_hex": row["expected0_hex"],
                    "delta0_lsb": delta0,
                    "diff0_lsb": diff0,
                    "actual1_hex": row["actual1_hex"],
                    "expected1_hex": row["expected1_hex"],
                    "delta1_lsb": delta1,
                    "diff1_lsb": diff1,
                    "expected_bid_lsb": expected0,
                    "expected_ask_lsb": expected1,
                    "actual_bid_lsb": actual0,
                    "actual_ask_lsb": actual1,
                    "expected_bid_q24_12": expected0 / Q_SCALE,
                    "expected_ask_q24_12": expected1 / Q_SCALE,
                    "actual_bid_q24_12": actual0 / Q_SCALE,
                    "actual_ask_q24_12": actual1 / Q_SCALE,
                    "expected_decision": decision_name(expected_decision),
                    "actual_decision": decision_name(actual_decision),
                    "decision_changed": int(decision_changed),
                    "expected_policy_margin_lsb": expected_margin,
                }
            )

    drift_out = out_dir / f"wrapper_1lsb_drift_rows_{run_name}.csv"
    if drift_rows:
        fieldnames = list(drift_rows[0].keys())
    else:
        fieldnames = [
            "run",
            "index",
            "actual0_hex",
            "expected0_hex",
            "delta0_lsb",
            "diff0_lsb",
            "actual1_hex",
            "expected1_hex",
            "delta1_lsb",
            "diff1_lsb",
            "expected_bid_lsb",
            "expected_ask_lsb",
            "actual_bid_lsb",
            "actual_ask_lsb",
            "expected_bid_q24_12",
            "expected_ask_q24_12",
            "actual_bid_q24_12",
            "actual_ask_q24_12",
            "expected_decision",
            "actual_decision",
            "decision_changed",
            "expected_policy_margin_lsb",
        ]
    with drift_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(drift_rows)

    delta_minus1 = sum(1 for d in nonzero_deltas if d == -1)
    delta_plus1 = sum(1 for d in nonzero_deltas if d == 1)
    delta_other = sum(1 for d in nonzero_deltas if d not in (-1, 1))
    direction = "none"
    if nonzero_deltas:
        signs = {1 if d > 0 else -1 for d in nonzero_deltas}
        if len(signs) == 1:
            direction = "positive" if 1 in signs else "negative"
        else:
            direction = "mixed"

    return {
        "run": run_name,
        "dump_path": str(dump_path),
        "row_count": len(rows),
        "drift_row_count": len(drift_rows),
        "max_abs_diff_lsb": max_abs_diff,
        "nonzero_delta_count": len(nonzero_deltas),
        "delta_minus1_count": delta_minus1,
        "delta_plus1_count": delta_plus1,
        "delta_other_count": delta_other,
        "nonzero_delta_direction": direction,
        "decision_flip_count": decision_flip_count,
        "min_abs_expected_best_lsb_all_rows": min_abs_expected_best if min_abs_expected_best is not None else "",
        "min_policy_margin_lsb_all_rows": min_policy_margin if min_policy_margin is not None else "",
        "drift_rows_csv": str(drift_out),
    }


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Run spec must be NAME=PATH")
    name, path = spec.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Run name is empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run_spec, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [analyze_run(name, path, args.out_dir) for name, path in args.run]
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    for summary in summaries:
        print(
            "READOUT_DRIFT_SUMMARY "
            f"run={summary['run']} rows={summary['row_count']} "
            f"drift_rows={summary['drift_row_count']} "
            f"max_abs_diff_lsb={summary['max_abs_diff_lsb']} "
            f"direction={summary['nonzero_delta_direction']} "
            f"decision_flips={summary['decision_flip_count']}"
        )


if __name__ == "__main__":
    main()
