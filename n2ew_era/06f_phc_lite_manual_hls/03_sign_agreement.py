from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
INPUT_CSV = RESULTS / "phc_step_cosim_out.csv"
OUTPUT_CSV = RESULTS / "phc_step_sign_agreement.csv"


def sign_with_zero_positive(values: pd.Series) -> np.ndarray:
    """Return -1 for negative values and +1 for zero/positive values."""
    arr = values.to_numpy(dtype=np.float64, copy=False)
    if np.isnan(arr).any():
        raise ValueError(f"NaN values found in column {values.name}")
    return np.where(arr < 0.0, -1, 1)


def agreement_row(channel: str, ref_sign: np.ndarray, dut_sign: np.ndarray) -> dict[str, float | int | str]:
    total = int(ref_sign.size)
    agree = int(np.sum(ref_sign == dut_sign))
    disagree = total - agree
    pct = (agree / total * 100.0) if total else float("nan")
    return {
        "channel": channel,
        "total_rows": total,
        "agree_rows": agree,
        "disagree_rows": disagree,
        "sign_agreement_pct": pct,
    }


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing cosim output CSV: {INPUT_CSV}")

    required = ["ref_readout_0", "dut_readout_0", "ref_readout_1", "dut_readout_1"]
    header = pd.read_csv(INPUT_CSV, nrows=0)
    usecols = required + (["pass"] if "pass" in header.columns else [])
    df = pd.read_csv(INPUT_CSV, usecols=usecols)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if "pass" in df.columns:
        failed = int((df["pass"] == 0).sum())
        if failed:
            print(f"WARNING: filtering out {failed} failed cosim rows (pass=0)")
        df = df[df["pass"] == 1].reset_index(drop=True)

    ref0 = sign_with_zero_positive(df["ref_readout_0"])
    dut0 = sign_with_zero_positive(df["dut_readout_0"])
    ref1 = sign_with_zero_positive(df["ref_readout_1"])
    dut1 = sign_with_zero_positive(df["dut_readout_1"])

    agree0 = ref0 == dut0
    agree1 = ref1 == dut1
    joint_ref = np.column_stack([ref0, ref1])
    joint_dut = np.column_stack([dut0, dut1])
    joint_agree = np.all(joint_ref == joint_dut, axis=1)

    rows = [
        agreement_row("readout_0", ref0, dut0),
        agreement_row("readout_1", ref1, dut1),
        {
            "channel": "joint_readout_0_1",
            "total_rows": int(len(df)),
            "agree_rows": int(np.sum(joint_agree)),
            "disagree_rows": int(len(df) - np.sum(joint_agree)),
            "sign_agreement_pct": (float(np.mean(joint_agree)) * 100.0) if len(df) else float("nan"),
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print("Sign convention: negative -> -1; zero/positive -> +1 (tie-breaking: 0 -> +1)")
    print(f"Input: {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print("")
    print("channel,total_rows,agree_rows,disagree_rows,sign_agreement_pct")
    for row in rows:
        print(
            f"{row['channel']},{row['total_rows']},{row['agree_rows']},"
            f"{row['disagree_rows']},{row['sign_agreement_pct']:.6f}"
        )


if __name__ == "__main__":
    main()
