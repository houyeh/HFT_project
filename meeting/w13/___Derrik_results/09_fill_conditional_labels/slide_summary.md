# 條件成交標籤審查 (Fill-Conditional Label Audit)

- 使用了 Proxy C (`price_level_removed_proxy`) 在 horizon=10 的條件下進行審查。
- 發現「有成交條件下」的毒性率（Toxic Rate）相比「無條件下」有顯著的倍數提升 (Lift)。
- ⚠️ 注意：目前沒有對手單 (Aggressor-side) 數據，所以這只是替代指標 (Proxy) 證據，不是交易所真實成交的 Ground Truth。
- 這個結果給了我們後續訓練 `bid_toxic` / `ask_toxic` binary model 極大的信心與方向。
