# Data Compatibility Audit

- NEW4 source: `data\NEW4`.
- Detected NEW4 symbol: `TXFA6`.
- Inferred NEW4 tick size: `1.0`.
- Point value: `200.0`.
- Maker fee: `15.0`.

## Audit Results

- `point_value`: `match`. Historical=`200.0`, NEW4=`200.0`.
- `maker_fee`: `match`. Historical=`15.0`, NEW4=`15.0`.
- `tick_size`: `match`. Historical=`1.0`, NEW4=`1.0`.
- `symbol`: `review`. Historical=`TXFB6`, NEW4=`TXFA6`.
- `loader`: `match`. Historical=`load_new_tw_data / binary_labels_v2 derived from same raw format`, NEW4=`load_new_tw_data`.

## Use In Reports

NEW4 is treated as external out-of-sample evidence only. No policy threshold, side rule, stress setting, fee, multiplier, or tick-size assumption is tuned on NEW4.
