# 不能過度宣稱的避雷清單 (Claims to Avoid)

- **絕對不要說** 模型是 directional trader (主動方向性交易)。
- **絕對不要說** 0.55 已經是 production threshold (它只是 test-sweep candidate)。
- **絕對不要說** 策略目前已經 production-ready。
- **絕對不要說** Delay > 3 ticks 策略就會完全崩盤 (正確說法：Net PnL 非單調，在 delay=10 時明顯變差)。
- **絕對不要說** FPGA 已經被完全證明在絕對獲利上的必要性 (正確說法：FPGA 是 risk-control-oriented)。
- **絕對不要說** fill-conditional label 是真實的交易所 fill (正確說法：基於 proxy 且無 aggressor-side data)。
- **絕對不要說** 舊版的 toxic labels 還有重試的價值。
- **絕對不要說** NEW3 表現不好所以被排除不算 (正確說法：NEW3 是 stress regime，需要 causal gate)。
