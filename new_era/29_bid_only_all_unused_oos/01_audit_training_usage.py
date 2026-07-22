from __future__ import annotations

import pandas as pd

from common_032 import DATA_DIR_NAMES, DATA_ROOT, RESULTS, csv_session_inventory, ensure_dirs, excluded_session_map, training_selection_sessions, write_markdown


def main() -> None:
    ensure_dirs()
    provenance = training_selection_sessions()
    provenance.to_csv(RESULTS / "training_selection_sessions.csv", index=False)
    excluded = excluded_session_map()
    inventory_parts = []
    for name in DATA_DIR_NAMES:
        data_dir = DATA_ROOT / name
        inv = csv_session_inventory(data_dir)
        if inv.empty:
            inv = pd.DataFrame([{"data_dir": name, "session_id": "", "file": "", "bytes": 0}])
        inv["included_in_unused_oos"] = ~inv["session_id"].astype(str).isin(excluded)
        inv["exclusion_reason"] = inv["session_id"].astype(str).map(excluded).fillna("included_unused_oos")
        inventory_parts.append(inv)
    inventory = pd.concat(inventory_parts, ignore_index=True)
    inventory.to_csv(RESULTS / "data_new_to_new4_session_inventory.csv", index=False)

    lines = [
        "# Training/Selection Data Usage Audit",
        "",
        "`bid_only_spread_ge_1` is a non-ML rule, but the rule was selected in Phase 026 on validation. For strict OOS testing, both train sessions and validation-selection sessions are excluded.",
        "",
        "## Sessions Used Upstream",
        "",
    ]
    for _, row in provenance.iterrows():
        lines.append(f"- `{row['session_id']}` split `{row['split']}` dataset `{row['dataset']}` symbol `{row['symbol']}` rows `{int(row['label_rows'])}` role `{row['used_for_bid_only_policy']}`.")
    lines.extend(["", "## Exclusion Rule", ""])
    for sid, reason in sorted(excluded.items()):
        lines.append(f"- Exclude `{sid}`: `{reason}`.")
    lines.append("")
    lines.append("All other sessions under `data/NEW` through `data/NEW4` are treated as unused OOS for this audit.")
    write_markdown(RESULTS / "training_usage_audit.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote training usage audit to {RESULTS}")

