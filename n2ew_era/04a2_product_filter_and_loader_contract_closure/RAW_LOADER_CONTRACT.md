# Raw Loader Contract

Malformed `info_time` rows must be dropped before feature or label construction and explicitly logged. Silently coercing timestamps is strictly forbidden.

| rule_id    | rule                                                                                          | implementation_requirement                                                                | silent_coercion_allowed   | status   |
|:-----------|:----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:--------------------------|:---------|
| loader_001 | Malformed info_time rows MUST be dropped before feature or label construction.                | Log physical_path, session_id, row locator, raw info_time, and product_id when available. | False                     | FROZEN   |
| loader_002 | Rows must pass the frozen primary product filter before final target-universe evaluation.     | product_id must match ^TXF[A-Z0-9]*$                                                      | False                     | FROZEN   |
| loader_003 | Valid info_time parser interprets hour:minute:second:millisecond:flag and requires flag == 0. | Non-zero or missing fifth segment is malformed until separately audited.                  | False                     | FROZEN   |

## Malformed Rows Found During Closure Scan

| source_group   | session_id    | file_name                         |   malformed_info_time_rows | malformed_info_time_examples                                |
|:---------------|:--------------|:----------------------------------|---------------------------:|:------------------------------------------------------------|
| NEW1           | 20260127Night | my_orderbook_20260331_9004_3.csv  |                          1 | global_row_index=302023; product_id=; info_time=23:21:58:   |
| NEW3           | 20260330      | my_orderbook_20260515_22548_1.csv |                          1 | global_row_index=130845; product_id=; info_time=9:54:5:373: |
