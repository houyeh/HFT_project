# Product Universe Contract

Phase 04A.2 freezes the primary final product universe as **TXF-family only** using regex `^TXF[A-Z0-9]*$`.

TXF, MXF, and TMF are not treated as directly comparable under the inherited proxy constants. The project has repeatedly used `POINT_VALUE = 200` and `MAKER_FEE = 15`; those assumptions are compatible with a TXF-style proxy contract, but they are not exchange-certified and should not be pooled with MXF/TMF without product-specific normalization.

MXF/TMF may be studied only as a separate normalized diagnostic, with separate point value, tick size, fee, and reporting rows.

| contract_item                  | decision                                      | regex                    | allowed_in_final_evaluation   | reason                                                                                                                                                    |
|:-------------------------------|:----------------------------------------------|:-------------------------|:------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|
| primary_final_product_universe | TXF family only                               | ^TXF[A-Z0-9]*$           | True                          | Inherited proxy point_value=200 and maker_fee=15 are TXF-compatible project assumptions; pooling MXF/TMF without normalization would mix contract scales. |
| legacy_broad_target_universe   | TXF/MXF/TMF diagnostic only unless normalized | ^(TXF|MXF|TMF)[A-Z0-9]*$ | False                         | MXF and TMF are not assumed comparable to TXF in point value, tick value, or fee. They require separate metrics or product-normalized PnL.                |
| raw_mixed_product_stream       | forbidden for pooled final evaluation         | all product_id values    | False                         | Raw files contain hundreds of product IDs; unfiltered evaluation would mix unrelated markets.                                                             |
