"""Execution-aware features aligned from trade prints back onto quote events."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_queue_imbalance(df: pd.DataFrame, depth: int = 5) -> pd.Series:
    """
    計算委託簿不平衡 (Order Book Imbalance, OBI)，使用按檔位加權的掛單量。
    越靠近最佳買賣價 (Top of book) 的檔位，權重越高。
    
    公式: (加權買單量 - 加權賣單量) / (加權買單量 + 加權賣單量)
    其中 加權量 = Sum(各檔位數量_i / i)，i 從 1 到 depth
    """
    # 反向檔位加權：第 1 檔權重為 1.0，第 2 檔為 0.5，依此類推
    weighted_bid = sum(df[f"bid_vol_{level}"].astype(float) / level for level in range(1, depth + 1))
    weighted_ask = sum(df[f"ask_vol_{level}"].astype(float) / level for level in range(1, depth + 1))
    
    total = weighted_bid + weighted_ask
    # 避免除以零，將 0 替換為 1.0 (如果雙方皆為 0，imbalance 將為 0)
    imbalance = (weighted_bid - weighted_ask) / total.replace(0, 1.0)
    return imbalance.fillna(0.0)



def infer_trade_signs(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    tick_size: float,
) -> pd.DataFrame:
    """
    使用層級分類規則推斷成交方向（主動買或主動賣）。
    
    因為許多資料源並未明確標示成交是由買方還是賣方發起，
    我們使用標準的市場微觀結構經驗法則（heuristics）：
    1. 報價法則 (Quote Rule)：比較成交價與當時的最佳買賣價。
    2. 中間價法則 (Mid-Price Rule)：若成交在買賣價中間，則與前一個中間價比較。
    3. 跳動法則 (Tick Rule)：比較本次成交價與前一次成交價。
    """
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume", "trade_sign"])

    # 準備報價參考資料以進行 asof join
    quote_ref = quotes[["timestamp", "bid_px_1", "ask_px_1"]].sort_values("timestamp").copy()
    quote_ref["mid_price"] = (quote_ref["bid_px_1"] + quote_ref["ask_px_1"]) / 2.0

    # 將成交紀錄對齊到最近的一筆報價（向後匹配 backward match）
    aligned = pd.merge_asof(
        trades.sort_values("timestamp").copy(),
        quote_ref,
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    ).dropna(subset=["bid_px_1", "ask_px_1"])

    if aligned.empty:
        return pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume", "trade_sign"])

    # 跳動法則 (Tick Rule): 連續成交之間價格變動的符號
    price_diff = aligned["trade_price"].diff()
    tick_rule = pd.Series(np.sign(price_diff), index=aligned.index).replace(0.0, np.nan).ffill().fillna(0.0)
    
    # 中間價法則 (Mid Rule): 在中間價的哪一側 (+1 若大於 mid，-1 若小於 mid)
    mid_rule = np.sign(aligned["trade_price"] - aligned["mid_price"])

    # 報價法則 (Quote Rule): 定義成交在賣價（含）以上，或買價（含）以下的遮罩
    # 我們使用 0.5 * tick_size 的容差來處理可能的進位誤差或小幅滑價
    buy_mask = aligned["trade_price"] >= (aligned["ask_px_1"] - 0.5 * tick_size)
    sell_mask = aligned["trade_price"] <= (aligned["bid_px_1"] + 0.5 * tick_size)
    
    # 層級邏輯：
    # 第一優先: 報價法則 (Quote Rule，明確的主動方)
    # 第二優先: 中間價法則 (Mid Rule，適用於成交在價差內且不在中間價的情況)
    # 第三優先: 跳動法則 (Tick Rule，適用於成交正好在中間價或價差未知的情況)
    trade_sign = np.where(
        buy_mask,
        1.0,
        np.where(
            sell_mask,
            -1.0,
            np.where(mid_rule != 0, mid_rule, tick_rule.to_numpy(dtype=float)),
        ),
    )
    
    # 最後的備案：如果仍未分類，預設為主動買 (偏好將其視為消耗流動性)
    trade_sign = np.where(trade_sign == 0, 1.0, trade_sign)

    aligned["trade_sign"] = trade_sign.astype(np.int8)
    return aligned[["timestamp", "trade_price", "trade_volume", "trade_sign"]].reset_index(drop=True)



def align_trade_features_to_quotes(
    quotes: pd.DataFrame,
    trades: pd.DataFrame,
    tick_size: float,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """
    將成交事件對齊到委託簿快照，並計算 execution-aware（考慮執行情況的）特徵。
    
    此函式橋接了離散的成交紀錄 (trade prints) 與委託簿快照 (order book snapshots) 之間的落差。
    它計算描述以下特性的特徵：
    1. 交易流向 (Trade Flow)：主動單不平衡 (Aggressor imbalance) 與帶號成交量比例。
    2. 交易強度 (Trade Intensity)：相對於掛單量，報價佇列被消耗的速度。
    3. 撤單壓力 (Order Cancellations)：用來近似沒有發生交易的情況下流動性被抽走的程度。

    產生的特徵包含：
    - aggressor_imbalance_interval: 該報價區間內的 (買方主動量 - 賣方主動量) / 總量。
    - rolling_signed_flow_ratio: 滾動視窗內的 累計帶號成交量 / 累計總成交量。
    - trade_intensity_short: 滾動成交總量 / 平均委託簿掛單量 (L1+L2)。
    - net_cancel_pressure: (賣單撤銷 proxy - 買單撤銷 proxy) / 總撤銷 proxy。
    - cancel_pressure_short: 滾動視窗內的淨撤單壓力。
    """
    features = pd.DataFrame(index=quotes.index)
    
    # 初始化所有執行特徵為 0
    for column in [
        "trade_count_interval",
        "trade_volume_interval",
        "signed_trade_volume_interval",
        "buy_aggressor_volume_interval",
        "sell_aggressor_volume_interval",
        "aggressor_imbalance_interval",
        "rolling_signed_flow_ratio",
        "trade_intensity_short",
        "bid_cancel_proxy",
        "ask_cancel_proxy",
        "net_cancel_pressure",
        "cancel_pressure_short",
    ]:
        features[column] = 0.0

    if trades.empty or quotes.empty:
        return features

    # 1. 使用 price-plus-tick-rule 邏輯推斷成交方向
    signed_trades = infer_trade_signs(trades, quotes, tick_size)
    if signed_trades.empty:
        return features

    # 2. 將成交紀錄映射到"下一個"報價事件 (向後對齊 backward alignment)
    # 這確保了時間點 t 的特徵反映了自 t-1 以來的所有市場活動
    quote_timestamps = quotes["timestamp"].to_numpy(dtype=np.int64, copy=False)
    quote_idx = np.searchsorted(quote_timestamps, signed_trades["timestamp"].to_numpy(dtype=np.int64), side="left")
    
    valid_mask = quote_idx < len(quotes)
    if not np.any(valid_mask):
        return features

    signed_trades = signed_trades.loc[valid_mask].copy()
    signed_trades["quote_idx"] = quote_idx[valid_mask]
    signed_trades["signed_volume"] = signed_trades["trade_sign"] * signed_trades["trade_volume"]
    signed_trades["buy_volume"] = np.where(signed_trades["trade_sign"] > 0, signed_trades["trade_volume"], 0.0)
    signed_trades["sell_volume"] = np.where(signed_trades["trade_sign"] < 0, signed_trades["trade_volume"], 0.0)

    # 3. 聚合在兩次報價更新之間發生的所有成交紀錄
    grouped = (
        signed_trades.groupby("quote_idx", as_index=True)
        .agg(
            trade_count_interval=("trade_volume", "size"),
            trade_volume_interval=("trade_volume", "sum"),
            signed_trade_volume_interval=("signed_volume", "sum"),
            buy_aggressor_volume_interval=("buy_volume", "sum"),
            sell_aggressor_volume_interval=("sell_volume", "sum"),
        )
        .sort_index()
    )
    features.loc[grouped.index, grouped.columns] = grouped.values

    # 4. 計算區間與滾動的交易流特徵 (Flow Features)
    total_volume = features["trade_volume_interval"].replace(0.0, 1.0)
    # 將區間內的成交量不平衡標準化至 [-1, 1] 之間
    features["aggressor_imbalance_interval"] = features["signed_trade_volume_interval"] / total_volume

    rolling_volume = features["trade_volume_interval"].rolling(window=rolling_window, min_periods=1).sum()
    rolling_signed = features["signed_trade_volume_interval"].rolling(window=rolling_window, min_periods=1).sum()
    features["rolling_signed_flow_ratio"] = rolling_signed / rolling_volume.replace(0.0, 1.0)

    # 5. 計算交易強度 (Trade Intensity) (透過 LOB 深度 2 進行標準化)
    total_visible_queue = (
        quotes["bid_vol_1"].astype(float)
        + quotes["ask_vol_1"].astype(float)
        + quotes.get("bid_vol_2", pd.Series(np.zeros(len(quotes), dtype=float))).astype(float)
        + quotes.get("ask_vol_2", pd.Series(np.zeros(len(quotes), dtype=float))).astype(float)
    ).replace(0.0, 1.0)
    
    # trade_intensity_short = (滾動成交總量) / (移動平均掛單量)
    # 數值遠大於 1.0 時，暗示掛單被「吃掉」的速度遠快於補充的速度。
    features["trade_intensity_short"] = (
        rolling_volume / total_visible_queue.rolling(window=rolling_window, min_periods=1).mean().replace(0.0, 1.0)
    )

    # 6. 撤單近似值計算邏輯 (Cancellation Proxy Logic)
    # 我們將「撤單」預估為：無法被觀測到的成交量所解釋的 LOB 掛單減少量。
    prev_bid_px = quotes["bid_px_1"].astype(float).shift(1)
    prev_ask_px = quotes["ask_px_1"].astype(float).shift(1)
    prev_bid_vol = quotes["bid_vol_1"].astype(float).shift(1)
    prev_ask_vol = quotes["ask_vol_1"].astype(float).shift(1)

    # 只有當價格層級沒有改變時才計算。
    # 如果價格層級改變，數量的變化會被層級移動所混淆。
    same_bid = quotes["bid_px_1"].astype(float) == prev_bid_px
    same_ask = quotes["ask_px_1"].astype(float) == prev_ask_px
    
    # 最佳買賣價 (Top of the book) 的原始掛單減少量
    bid_reduction = (prev_bid_vol - quotes["bid_vol_1"].astype(float)).clip(lower=0.0).fillna(0.0)
    ask_reduction = (prev_ask_vol - quotes["ask_vol_1"].astype(float)).clip(lower=0.0).fillna(0.0)

    # 撤單 Proxy = Max(掛單減少量 - 主動成交量, 0)
    # 例如：如果最佳買價 (bid) 數量減少 100，但主動賣出 (sell trades) 只有 40，我們就預估有 60 是被撤單的。
    features["bid_cancel_proxy"] = np.where(
        same_bid,
        np.maximum(bid_reduction - features["sell_aggressor_volume_interval"], 0.0),
        0.0,
    )
    features["ask_cancel_proxy"] = np.where(
        same_ask,
        np.maximum(ask_reduction - features["buy_aggressor_volume_interval"], 0.0),
        0.0,
    )

    # 7. 撤單壓力標準化 (Cancel Pressure Normalization)
    cancel_total = (features["bid_cancel_proxy"] + features["ask_cancel_proxy"]).replace(0.0, 1.0)
    features["net_cancel_pressure"] = (features["ask_cancel_proxy"] - features["bid_cancel_proxy"]) / cancel_total

    # 視窗內的滾動淨撤單壓力
    rolling_cancel = (
        (features["ask_cancel_proxy"] - features["bid_cancel_proxy"])
        .rolling(window=rolling_window, min_periods=1)
        .sum()
    )
    rolling_cancel_scale = (
        (features["ask_cancel_proxy"] + features["bid_cancel_proxy"])
        .rolling(window=rolling_window, min_periods=1)
        .sum()
        .replace(0.0, 1.0)
    )
    features["cancel_pressure_short"] = rolling_cancel / rolling_cancel_scale

    return features.fillna(0.0)

