# 0428 新台指期資料測試報告 (NEW Data)

## 1. 資料格式與對齊結果
- 解析了 `data/NEW` 中的新格式 CSV。
- 提取的 Symbol：TXFB6, MXFB6
  - **TXFB6**: LOB Quotes: 2852399, Trades: 72014
  - **MXFB6**: LOB Quotes: 4593840, Trades: 169420

## 2. 預測結果 (Raw Metrics)
未經過濾的原始分類表現：
```json
{
  "accuracy": 0.20438273818397187,
  "macro_precision": 0.38354299556582777,
  "macro_recall": 0.3533501255933318,
  "macro_f1": 0.14408031608993735,
  "support": 744603,
  "per_class": {
    "Down": {
      "precision": 0.01716632645532574,
      "recall": 0.005850649187363268,
      "f1": 0.008726963566802531,
      "support": 119303
    },
    "Stationary": {
      "precision": 0.9573634608833882,
      "recall": 0.06664479820415133,
      "f1": 0.12461480479748155,
      "support": 506056
    },
    "Up": {
      "precision": 0.17609919935876922,
      "recall": 0.9875549293884808,
      "f1": 0.29889917990552795,
      "support": 119244
    }
  },
  "confusion_matrix": [
    [
      698,
      719,
      117886
    ],
    [
      39262,
      33726,
      433068
    ],
    [
      701,
      783,
      117760
    ]
  ]
}
```

## 3. Gating 後的可交易訊號表現 (Probability Gating)
應用機率門檻與邊際門檻過濾後的分類表現：
```json
{
  "accuracy": 0.21722850968905577,
  "macro_precision": 0.3842880550345911,
  "macro_recall": 0.35927248825660807,
  "macro_f1": 0.15474156146232765,
  "support": 744603,
  "per_class": {
    "Down": {
      "precision": 0.017425892603466237,
      "recall": 0.004626874428974963,
      "f1": 0.007312226785004636,
      "support": 119303
    },
    "Stationary": {
      "precision": 0.9590927143550594,
      "recall": 0.08589563210395687,
      "f1": 0.15767041847879315,
      "support": 506056
    },
    "Up": {
      "precision": 0.17634555814524777,
      "recall": 0.9872949582368924,
      "f1": 0.29924203912318514,
      "support": 119244
    }
  },
  "confusion_matrix": [
    [
      552,
      899,
      117852
    ],
    [
      30565,
      43468,
      432023
    ],
    [
      560,
      955,
      117729
    ]
  ]
}
```

## 4. 向量化回測結果 (Signal Backtest)
依據 Gating 後的訊號進行的快速回測：
```json
{
  "samples": 744603,
  "trades": 699281,
  "trade_rate": 0.9391326653263551,
  "gross_pnl_ticks": -1299.0,
  "net_pnl_ticks": -1433269.0,
  "avg_net_pnl_ticks": -2.0496324081449373,
  "hit_rate": 0.29933746233631403,
  "long_trades": 667604,
  "short_trades": 31677,
  "sharpe_like": -9.048620424575104
}
```

### 各商品回測拆解
```
  symbol  samples  trades  gross_pnl_ticks  net_pnl_ticks
0  TXFB6   285229  273717           5505.5      -627990.5
1  MXFB6   459374  425564          -6804.5      -805278.5
```
