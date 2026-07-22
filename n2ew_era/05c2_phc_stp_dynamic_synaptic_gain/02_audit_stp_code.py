from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COMMON = ROOT / "common_05c2.py"
MODEL = ROOT / "models" / "phc_lite_stp.py"


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def literal_list_present(source: str, values: list[str]) -> bool:
    pattern = r"\[" + r"\s*,\s*".join([repr(v) for v in values]) + r"\]"
    return re.search(pattern, source) is not None


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = COMMON.read_text(encoding="utf-8")
    model_source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    selected_path = RESULTS / "phc_stp_selected_policy.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
    threshold_grid = pd.read_csv(RESULTS / "phc_stp_validation_threshold_search.csv") if (RESULTS / "phc_stp_validation_threshold_search.csv").exists() else pd.DataFrame()
    oos = pd.read_csv(RESULTS / "phc_stp_vs_v1_hgb_queue.csv") if (RESULTS / "phc_stp_vs_v1_hgb_queue.csv").exists() else pd.DataFrame()
    matched = pd.read_csv(RESULTS / "phc_stp_matched_random_summary.csv") if (RESULTS / "phc_stp_matched_random_summary.csv").exists() else pd.DataFrame()
    causal = pd.read_csv(RESULTS / "phc_stp_causal_fill_vs_v1.csv") if (RESULTS / "phc_stp_causal_fill_vs_v1.csv").exists() else pd.DataFrame()
    oracle = pd.read_csv(RESULTS / "phc_stp_window_oracle_capture.csv") if (RESULTS / "phc_stp_window_oracle_capture.csv").exists() else pd.DataFrame()

    checks: list[dict[str, object]] = []

    def add(item: str, status: str, evidence: str, file_path: Path = COMMON) -> None:
        checks.append(
            {
                "audit_item": item,
                "status": status,
                "evidence": evidence,
                "file_path": str(file_path),
            }
        )

    add("train_val_new3_new4_split_handling", "PASS" if 'for split in ["val", "new3", "new4"]' in source and "p05a.load_sequences(\"train\")" in source else "FAIL", "Training uses train payload; evaluation loops over val/new3/new4.")
    add("no_new5_usage", "PASS" if "load_sequences(\"new5\")" not in source.lower() and "new5_used_for_selection" in source and not bool(selected.get("new5_used_for_selection", True)) else "FAIL", "No NEW5 sequence loading found; selected JSON marks new5_used_for_selection=false.")
    add("validation_only_epoch_selection", "PASS" if "selection_metric\": \"validation_policy_excess_pnl_at_threshold_0\"" in source and "val_payload" in source else "FAIL", "Checkpoint selected by validation policy excess PnL.")
    add("validation_only_threshold_seed_selection", "PASS" if not threshold_grid.empty and set(threshold_grid.get("selection_split", [])) == {"val"} and bool((threshold_grid.get("new3_used_for_selection", pd.Series([False])).astype(str).str.lower() == "false").all()) else "FAIL", "Threshold grid CSV contains only selection_split=val and false OOS selection flags.")
    add("feature_list", "PASS" if "p05a.PHC_FEATURES" in source else "FAIL", "Model feature_dim is len(p05a.PHC_FEATURES); inherited forbidden-feature contract from Phase 05A.")
    add("target_excess_pnl_H20", "PASS" if selected.get("target") == "excess_pnl_H20" and "p05a.HORIZON" in source else "FAIL", "Selected policy target is excess_pnl_H20; metrics use Phase 05A horizon constant H20.")
    add("matched_random_implementation", "PASS" if "p05a.matched_random_for_split" in source and not matched.empty and bool(matched["beats_random_95p"].astype(bool).all()) else "FAIL", "Matched random delegates to Phase 05A count/session/side/spread-bucket matcher.")
    add("causal_fill_implementation", "PASS" if "p05a.fill_probabilities" in source and "stable_uniform" in source and not causal.empty and bool((causal.get("fill_uses_future_pnl", pd.Series([False])).astype(str).str.lower().isin(["false", "0"])).all()) else "FAIL", "Causal-fill uses train-derived norms, stable uniforms, and no future-PnL fill decision.")
    add("oracle_capture_computation", "PASS" if "oracle_uses_future_labels" in source and not oracle.empty and bool(oracle["oracle_uses_future_labels"].astype(bool).all()) else "FAIL", "Oracle capture explicitly uses future labels and is diagnostic only.")
    add("stp_gain_definition", "PASS" if "gain_t = torch.sigmoid" in model_source and "gain_from_encoded" in model_source and "gain_from_prev_state" in model_source else "FAIL", "gain_t = sigmoid(W_g encoded_t + U_g state_{t-1}) exists in model source.", MODEL)
    add("not_full_biological_stp", "PASS" if "not full biological STP" in model_source or "not full biological" in model_source else "FAIL", "Model docstring states it is not full biological STP.", MODEL)
    add("new5_not_in_outputs", "PASS" if (oos.empty or "new5" not in set(oos.get("split", []))) and (oracle.empty or "new5" not in set(oracle.get("split", []))) else "FAIL", "OOS/oracle output splits exclude NEW5.")

    out = pd.DataFrame(checks)
    out.to_csv(RESULTS / "phc_stp_code_audit_checklist.csv", index=False)
    status = "PASS" if (out["status"] == "PASS").all() else "FAIL"
    write_markdown(
        ROOT / "PHC_STP_CODE_AUDIT_REPORT.md",
        [
            "# PHC STP-like Code Audit Report",
            "",
            f"Overall status: `{status}`.",
            "",
            "Audited implementation: `common_05c2.py` and `models/phc_lite_stp.py`.",
            "",
            out.to_markdown(index=False),
            "",
            "This audit checks code structure and produced artifacts. It does not adopt STP-like PHC and does not use NEW5.",
        ],
    )
    print(status)


if __name__ == "__main__":
    main()
