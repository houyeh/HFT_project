# Week13 會議資料包總覽 (Meeting Pack Summary)

## 輸出資訊
- 輸出目錄: `week13_meeting_pack`
- 簡報資料夾: 00_result 到 12_result
- 檔案清單: `week13_output_manifest.csv`

## 缺失檔案 (Missing Inputs)
- 無

## 警告與備註 (Warnings)
- 未執行新的回測。
- 未執行模型重新訓練。
- Threshold 0.55 並未被標示為可上線生產 (production) 的閾值。
- Fill-conditional (有成交條件) 的標籤審查是基於替代指標 (Proxy)，並非真實交易所成交數據。
- 延遲 (Latency) 的解釋避免了單調崩潰的過度宣稱。

## 須避免的宣稱 (Claims To Avoid)
- 絕對不要宣稱具備主動方向性交易 (Directional trading) 的能力。
- 絕對不要宣稱延遲大於 3 ticks 策略就會完全崩盤。
- 絕對不要宣稱替代指標 (Proxy) 估算的成交是真實成交。
- 絕對不要宣稱已經解決了統一的生產環境 Gate 選擇器問題。
