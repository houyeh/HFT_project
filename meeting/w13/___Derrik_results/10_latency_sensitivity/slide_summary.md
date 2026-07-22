# 延遲敏感度分析 (Latency Sensitivity)

- 隨著撤單延遲 (Cancel delay) 增加，診斷出的「延遲毒性虧損 (Delayed toxic loss)」也會隨之增加。
- Net PnL 對延遲的反應不是單調的：
  - 在 Delay 3~5 時，PnL 有時反而會上升，這可能是因為過濾掉了假陽性撤單。
  - **但在 Delay 10 時，所有關鍵切片的表現都會明顯惡化。**
- 結論：推動 FPGA 硬體加速的主要學術理由，在於**極致的風險控管 (Risk-control)**，防止延遲過大造成防禦崩潰，而不是單純為了賺更多錢。
