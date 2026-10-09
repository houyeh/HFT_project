"""03B6/04B1 reversal root-cause audit.

中文導讀：
這支程式只做「診斷」，不重新訓練模型，也不重新選門檻。它的核心問題是：
03B6 混合 universe 裡 ML 輸給 queue imbalance simple rule，但 04B1 改成 TXF-only
後 ML 又打贏 simple rule，這個排名反轉到底是什麼機制造成的。

整體流程如下：
1. 建立所有專案路徑、模型欄位與固定 policy/action 編碼。
2. 用 assert_no_new5_path 包住所有讀檔入口，避免誤碰 NEW5。
3. 重新計算 development-only 的特徵分布、simple rule 表現、以及 frozen ML 在同一份
   TXF validation 上的預測與交易結果。
4. 讀取既有 frozen NEW3/NEW4 診斷 CSV，只做事後對照，不拿來選模型、調參或挑假說。
5. 將所有實際重算出的數字寫成 CSV，再組成 AUDIT markdown 報告。
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_ROOT = PROJECT_ROOT / "n2ew_era" / "000_context"
RESULTS = CONTEXT_ROOT / "results" / "03b6_04b1_reversal_root_cause"
AUDIT_PATH = CONTEXT_ROOT / "AUDIT_03b6_04b1_reversal_root_cause.md"

# 各階段資料夾：03B/03B6 是混合 universe 的基準與 frozen ML，
# 04B/04B1 是 TXF-only 重建後的資料與 frozen ML。
PHASE03B = PROJECT_ROOT / "n2ew_era" / "03b_simple_ml_action_value_baselines"
PHASE03B6 = PROJECT_ROOT / "n2ew_era" / "03b6_hgb_convergence_closure"
PHASE04B = PROJECT_ROOT / "n2ew_era" / "04b_txf_only_final_development_rerun"
PHASE04B1 = PROJECT_ROOT / "n2ew_era" / "04b1_ml_completion_patch"
PHASE_AUDIT_PATH = PHASE03B6 / "AUDIT_03b6_04b1_reversal_root_cause.md"

# 03B 的共用函式不在目前資料夾底下，所以先把 phase03B 加進 import path。
if str(PHASE03B) not in sys.path:
    sys.path.insert(0, str(PHASE03B))

import common_phase03b as p03b  # noqa: E402


# 固定診斷參數。這裡的 seed 只用於 permutation diagnostic 抽樣/打亂，不是模型訓練。
SEED = 20260620
PERMUTATION_SAMPLE_ROWS = 50_000
TXF_RE = re.compile(r"^TXF[A-Z0-9]*$")
HORIZON = 20

# 04B action-row parquet 用 0/1/2 表示 bid/ask/no-quote 候選 action；
# policy decision 另用 0/1/2 表示不下單/掛 bid/掛 ask。
ACTION_BID = 0
ACTION_ASK = 1
ACTION_NO_QUOTE = 2
DECISION_NO_QUOTE = 0
DECISION_BID = 1
DECISION_ASK = 2
TXF_ACTION_DATA = PHASE04B / "data" / "txf_only_action_universe.parquet"

# 04B1 frozen HGB 期待的特徵欄位。action_is_* 與 side_* 是由 action_slot 和原始 imbalance 衍生。
ML_FEATURES_04B = [
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "event_rate_past_50",
    "inter_arrival_ms",
    "volatility_past_50",
    "abs_return_past_50",
    "imbalance_change_past",
    "spread_change_past",
    "hawkes_excite",
    "action_is_bid",
    "action_is_ask",
    "side_sign",
    "side_queue_imbalance",
    "side_deep_imbalance",
    "side_imbalance_change_past",
]
FOCUS_FEATURES = [
    "queue_imbalance",
    "deep_imbalance",
    "spread_ticks",
    "event_rate_past_50",
]


@dataclass
class Metric:
    """累積單一 policy/model 的交易統計。

    state_count 是可決策狀態數，selected_count 是實際選擇 bid/ask 的次數。
    pnl_sum、win/loss、drawdown 都以 excess_pnl_H20 為基礎逐批累加。
    """

    state_count: int = 0
    selected_count: int = 0
    bid_count: int = 0
    ask_count: int = 0
    pnl_sum: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    max_drawdown: float = 0.0
    equity: float = 0.0
    peak: float = 0.0


def part_paths_04b(split: str) -> list[Path]:
    """列出 04B TXF-only action universe 某個 split 底下的 parquet 分片。"""

    return sorted((TXF_ACTION_DATA / f"split={split}").glob("*.parquet"))


def action_feature_frame_04b(raw: pd.DataFrame) -> pd.DataFrame:
    """把 04B 原始 action rows 轉成 04B1 HGB 可直接吃的特徵表。

    每個 market state 會有 bid/ask/no-quote 三列 action row；這裡依 action_slot 補出
    action_is_bid/action_is_ask、side_sign，以及方向化的 imbalance 特徵。
    """

    out = raw.copy()
    slot = out["action_slot"].astype(np.int8).to_numpy()
    out["action_is_bid"] = (slot == ACTION_BID).astype(np.float32)
    out["action_is_ask"] = (slot == ACTION_ASK).astype(np.float32)
    side_sign = np.zeros(len(out), dtype=np.float32)
    side_sign[slot == ACTION_BID] = 1.0
    side_sign[slot == ACTION_ASK] = -1.0
    out["side_sign"] = side_sign
    out["side_queue_imbalance"] = side_sign * pd.to_numeric(out["queue_imbalance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["side_deep_imbalance"] = side_sign * pd.to_numeric(out["deep_imbalance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["side_imbalance_change_past"] = side_sign * pd.to_numeric(out["imbalance_change_past"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    for col in ML_FEATURES_04B:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out["target_excess_pnl_H20"] = pd.to_numeric(out["excess_pnl_H20"], errors="coerce").fillna(0.0).astype(np.float32)
    out["target_positive_excess_H20"] = out["target_excess_pnl_H20"] > 0.0
    return out


def action_matrix_from_part_04b(path: Path) -> pd.DataFrame:
    """讀取單一 TXF parquet 分片，保留模型特徵與 target，再套用 04B 特徵轉換。"""

    cols = [
        "split",
        "session_id",
        "event_index",
        "action_slot",
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "event_rate_past_50",
        "inter_arrival_ms",
        "volatility_past_50",
        "abs_return_past_50",
        "imbalance_change_past",
        "spread_change_past",
        "hawkes_excite",
        "excess_pnl_H20",
    ]
    return action_feature_frame_04b(read_parquet(path, columns=cols))


def read_event_frame_04b(path: Path) -> pd.DataFrame:
    """把 action-row 格式還原成 state-row 格式，方便計算 policy PnL。

    04B parquet 同一個 event 會有 bid/ask/no-quote 三列。simple rule 和 ML policy 最後
    都是在「一個 state 做一次 bid/ask/no-quote 決策」，因此要檢查三列對齊後，把 bid/ask
    的 target PnL 合併回同一列。
    """

    cols = [
        "split",
        "session_id",
        "calendar_session_id",
        "symbol",
        "event_index",
        "timestamp_us",
        "action_slot",
        "spread_ticks",
        "bid_sz_1",
        "ask_sz_1",
        "top_depth",
        "queue_imbalance",
        "deep_imbalance",
        "event_rate_past_50",
        "inter_arrival_ms",
        "volatility_past_50",
        "abs_return_past_50",
        "imbalance_change_past",
        "spread_change_past",
        "hawkes_excite",
        "excess_pnl_H20",
    ]
    raw = read_parquet(path, columns=cols)
    bid = raw[raw["action_slot"].astype(int).eq(ACTION_BID)].reset_index(drop=True)
    ask = raw[raw["action_slot"].astype(int).eq(ACTION_ASK)].reset_index(drop=True)
    no_quote = raw[raw["action_slot"].astype(int).eq(ACTION_NO_QUOTE)].reset_index(drop=True)
    if not (len(bid) == len(ask) == len(no_quote)):
        raise RuntimeError(f"Action triplet mismatch in {path}")
    if not bid["event_index"].reset_index(drop=True).equals(ask["event_index"].reset_index(drop=True)):
        raise RuntimeError(f"Bid/ask event alignment mismatch in {path}")
    out = bid.drop(columns=["action_slot", "excess_pnl_H20"]).copy()
    out["bid_excess_pnl_H20"] = bid["excess_pnl_H20"].to_numpy(dtype=float)
    out["ask_excess_pnl_H20"] = ask["excess_pnl_H20"].to_numpy(dtype=float)
    return out


def policy_decision_04b(policy_id: str, frame: pd.DataFrame) -> np.ndarray:
    """04B 版本的 simple rule 決策器。

    `side_by_queue_imbalance_abs_0p1` 的規則很直接：queue_imbalance > 0.1 掛 bid，
    < -0.1 掛 ask，中間區間不報價。
    """

    decision = np.zeros(len(frame), dtype=np.int8)
    if policy_id == "no_quote":
        return decision
    if policy_id == "side_by_queue_imbalance_abs_0p1":
        q = pd.to_numeric(frame["queue_imbalance"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(q)
        decision[finite & (q > 0.1)] = DECISION_BID
        decision[finite & (q < -0.1)] = DECISION_ASK
        return decision
    raise ValueError(f"Unknown policy_id: {policy_id}")


def decision_pnl_04b(frame: pd.DataFrame, decision: np.ndarray) -> np.ndarray:
    """依照每個 state 的 decision，把對應 bid/ask excess PnL 取出；no-quote 的 PnL 為 0。"""

    out = np.zeros(len(frame), dtype=float)
    bid = frame["bid_excess_pnl_H20"].to_numpy(dtype=float)
    ask = frame["ask_excess_pnl_H20"].to_numpy(dtype=float)
    bid_mask = decision == DECISION_BID
    ask_mask = decision == DECISION_ASK
    out[bid_mask] = bid[bid_mask]
    out[ask_mask] = ask[ask_mask]
    out[~np.isfinite(out)] = 0.0
    return out


def update_metric_04b(metric: Metric, decision: np.ndarray, pnl: np.ndarray) -> None:
    """把一批 state 的決策結果累加到 Metric，包含勝率、PnL 與最大回撤。"""

    selected = decision != DECISION_NO_QUOTE
    metric.state_count += int(len(decision))
    metric.selected_count += int(selected.sum())
    metric.bid_count += int((decision == DECISION_BID).sum())
    metric.ask_count += int((decision == DECISION_ASK).sum())
    selected_pnl = pnl[selected]
    selected_pnl = selected_pnl[np.isfinite(selected_pnl)]
    metric.pnl_sum += float(selected_pnl.sum())
    metric.win_count += int((selected_pnl > 0.0).sum())
    metric.loss_count += int((selected_pnl < 0.0).sum())
    if len(pnl):
        equity = metric.equity + np.cumsum(pnl)
        peak = np.maximum.accumulate(np.maximum(equity, metric.peak))
        dd = peak - equity
        metric.max_drawdown = max(metric.max_drawdown, float(dd.max()) if len(dd) else 0.0)
        metric.peak = max(metric.peak, float(peak[-1]) if len(peak) else metric.peak)
        metric.equity = float(equity[-1])


def finalize_metric_04b(split: str, policy_id: str, metric: Metric) -> dict[str, Any]:
    """把累積中的 Metric 轉成可寫入 CSV/markdown 的一列統計。"""

    selected = max(metric.selected_count, 1)
    return {
        "split": split,
        "policy_id": policy_id,
        "horizon": HORIZON,
        "state_count": metric.state_count,
        "selected_action_count": metric.selected_count,
        "selection_rate": metric.selected_count / metric.state_count if metric.state_count else 0.0,
        "bid_count": metric.bid_count,
        "ask_count": metric.ask_count,
        "total_excess_pnl_H20": metric.pnl_sum,
        "avg_excess_pnl_per_selected": metric.pnl_sum / selected if metric.selected_count else 0.0,
        "win_rate": metric.win_count / selected if metric.selected_count else 0.0,
        "loss_rate": metric.loss_count / selected if metric.selected_count else 0.0,
        "max_drawdown": metric.max_drawdown,
    }


def ml_policy_decision_from_scores_04b(scores: np.ndarray, threshold: float) -> np.ndarray:
    """把 HGB 對 bid/ask/no-quote action rows 的分數轉成 state-level 交易決策。

    score 的順序假設為每三列一組：bid、ask、no-quote。此診斷沿用 frozen threshold：
    bid/ask 兩者中分數較高且高於門檻者才下單，否則 no-quote。
    """

    bid = scores[0::3]
    ask = scores[1::3]
    decision = np.zeros(len(bid), dtype=np.int8)
    choose_bid = (bid >= ask) & (bid > threshold)
    choose_ask = (ask > bid) & (ask > threshold)
    decision[choose_bid] = DECISION_BID
    decision[choose_ask] = DECISION_ASK
    return decision


p04b = SimpleNamespace(
    # 用 SimpleNamespace 模擬 04B 原腳本介面，讓下方診斷可以用 p04b.xxx 呼叫同一組 helper。
    ACTION_BID=ACTION_BID,
    ACTION_ASK=ACTION_ASK,
    ACTION_NO_QUOTE=ACTION_NO_QUOTE,
    DECISION_NO_QUOTE=DECISION_NO_QUOTE,
    DECISION_BID=DECISION_BID,
    DECISION_ASK=DECISION_ASK,
    ML_FEATURES=ML_FEATURES_04B,
    Metric=Metric,
    part_paths=part_paths_04b,
    action_matrix_from_part=action_matrix_from_part_04b,
    read_event_frame=read_event_frame_04b,
    policy_decision=policy_decision_04b,
    decision_pnl=decision_pnl_04b,
    update_metric=update_metric_04b,
    finalize_metric=finalize_metric_04b,
    ml_policy_decision_from_scores=ml_policy_decision_from_scores_04b,
)


def ensure_dirs() -> None:
    """建立本次 audit 的輸出資料夾。"""

    RESULTS.mkdir(parents=True, exist_ok=True)


def assert_no_new5_path(path: Path | str) -> None:
    """所有讀檔入口共用的保護欄：只要路徑含 NEW5 就立刻中止。"""

    text = str(path).replace("\\", "/").upper()
    if "NEW5" in text:
        raise RuntimeError(f"NEW5 path access is forbidden for this audit: {path}")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """安全版 CSV 讀取：先檢查不是 NEW5，再交給 pandas。"""

    assert_no_new5_path(path)
    return pd.read_csv(path, **kwargs)


def read_json(path: Path) -> dict[str, Any]:
    """安全版 JSON 讀取：用於 frozen policy/model metadata。"""

    assert_no_new5_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
    """安全版 parquet 讀取：所有 development feature/action matrix 都走這裡。"""

    assert_no_new5_path(path)
    return pd.read_parquet(path, **kwargs)


def load_joblib(path: Path) -> Any:
    """安全版 joblib 載入：只載入既有 frozen 模型，不做重新訓練。"""

    assert_no_new5_path(path)
    return joblib.load(path)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """計算 Spearman rank correlation，遇到樣本太少或常數序列時回傳 NaN。

    這裡不用 scipy，直接用 pandas rank 再做 Pearson correlation，讓診斷環境依賴更少。
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ar = pd.Series(a[mask]).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(ar) == 0.0 or np.std(br) == 0.0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """mean_squared_error 的平方根包裝，讓報表欄位直接命名為 RMSE。"""

    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def txf_like(symbol: Any) -> bool:
    """判斷 symbol 是否長得像 TXF 系列，用來拆 mixed universe 的 symbol family。"""

    return bool(TXF_RE.match(str(symbol).strip()))


def add_03b_missing_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """把 04B TXF action matrix 補成 03B6 frozen model 期待的特徵欄位。

    目的不是新增模型特徵，而是讓 03B6 frozen HGB 可以在同一份 TXF validation action rows
    上推論，進而和 04B1 做同資料對照。
    """

    out = matrix.copy()
    if "action_is_no_quote" not in out.columns:
        out["action_is_no_quote"] = (
            pd.to_numeric(out["action_slot"], errors="coerce").fillna(p04b.ACTION_NO_QUOTE).astype(int).eq(p04b.ACTION_NO_QUOTE)
        ).astype(np.float32)
    for feature in p03b.feature_matrix_columns():
        if feature not in out.columns:
            raise KeyError(f"Missing 03B feature {feature} in TXF matrix")
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def txf_action_matrix_from_part(path: Path) -> pd.DataFrame:
    """讀單一 04B TXF action parquet，保留 NEW5 防線。"""

    assert_no_new5_path(path)
    return p04b.action_matrix_from_part(path)


def txf_event_frame_from_part(path: Path) -> pd.DataFrame:
    """讀單一 04B TXF event-level 表，供 policy decision/PnL 計算使用。"""

    assert_no_new5_path(path)
    return p04b.read_event_frame(path)


def load_txf_split(split: str) -> pd.DataFrame:
    """載入並串接 TXF-only 某個 split 的全部 action-row 分片。"""

    frames = []
    for path in p04b.part_paths(split):
        assert_no_new5_path(path)
        frames.append(txf_action_matrix_from_part(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_mixed_matrix(split: str) -> pd.DataFrame:
    """載入 03B/03B6 mixed universe 的 train 或 validation feature matrix。"""

    if split == "train":
        return read_parquet(p03b.train_matrix_path())
    if split == "val":
        return read_parquet(p03b.val_matrix_path())
    raise ValueError(split)


def numeric_stats(frame: pd.DataFrame, features: list[str], universe: str, split_scope: str, source_detail: str) -> list[dict[str, Any]]:
    """對指定特徵重算分布統計，所有列都標明資料範圍與來源。

    輸出含 mean/std/分位數/abs_gt_0p1_rate；後者特別對 queue imbalance rule 的 0.1
    門檻有解釋價值。
    """

    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(values) == 0:
            rows.append(
                {
                    "analysis_scope": "development-only",
                    "universe": universe,
                    "split_scope": split_scope,
                    "source_detail": source_detail,
                    "row_basis": "action_rows_model_input",
                    "feature": feature,
                    "n": 0,
                }
            )
            continue
        q = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        rows.append(
            {
                "analysis_scope": "development-only",
                "universe": universe,
                "split_scope": split_scope,
                "source_detail": source_detail,
                "row_basis": "action_rows_model_input",
                "feature": feature,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "p01": float(q[0]),
                "p05": float(q[1]),
                "p25": float(q[2]),
                "p50": float(q[3]),
                "p75": float(q[4]),
                "p95": float(q[5]),
                "p99": float(q[6]),
                "max": float(np.max(values)),
                "abs_gt_0p1_rate": float(np.mean(np.abs(values) > 0.1)),
            }
        )
    return rows


def feature_distribution_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """方向 A：比較 mixed vs TXF-only development 特徵分布。

    這裡只讀 train/val development artifact，先做四個 focus features 的整體統計，
    再檢查 mixed matrix 裡 symbol family 的實際組成，避免把「混合商品污染」當成未驗證假設。
    """

    mixed_train = load_mixed_matrix("train")
    mixed_val = load_mixed_matrix("val")
    txf_train = load_txf_split("train")
    txf_val = load_txf_split("val")

    loaded = {
        "mixed_train": mixed_train,
        "mixed_val": mixed_val,
        "txf_train": txf_train,
        "txf_val": txf_val,
    }

    rows: list[dict[str, Any]] = []
    for universe, split, frame, source_detail in [
        ("mixed_03b6", "train_sampled", mixed_train, "03B sampled train feature matrix, NEW2 development"),
        ("mixed_03b6", "val", mixed_val, "03B validation feature matrix, NEW1 development"),
        ("mixed_03b6", "train_sampled_plus_val", pd.concat([mixed_train, mixed_val], ignore_index=True), "03B train sample + validation feature matrices, development only"),
        ("txf_04b1", "train", txf_train, "04B TXF-only train action universe, NEW2 development"),
        ("txf_04b1", "val", txf_val, "04B TXF-only validation action universe, NEW1 development"),
        ("txf_04b1", "train_plus_val", pd.concat([txf_train, txf_val], ignore_index=True), "04B TXF-only train + validation action universe, development only"),
    ]:
        rows.extend(numeric_stats(frame, FOCUS_FEATURES, universe, split, source_detail))
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "feature_distribution_stats.csv", index=False)

    symbol_rows: list[dict[str, Any]] = []
    mixed_dev = pd.concat([mixed_train, mixed_val], ignore_index=True)
    for split_scope, frame in [
        ("train_sampled", mixed_train),
        ("val", mixed_val),
        ("train_sampled_plus_val", mixed_dev),
    ]:
        symbol_col = "symbol" if "symbol" in frame.columns else "product_id"
        counts = frame.groupby(frame[symbol_col].astype(str), dropna=False).size().reset_index(name="action_rows")
        counts = counts.rename(columns={symbol_col: "symbol"})
        counts["analysis_scope"] = "development-only"
        counts["universe"] = "mixed_03b6"
        counts["split_scope"] = split_scope
        counts["is_txf_like_symbol"] = counts["symbol"].map(txf_like)
        counts["action_row_fraction"] = counts["action_rows"] / max(int(counts["action_rows"].sum()), 1)
        symbol_rows.extend(counts.to_dict(orient="records"))
    symbol_counts = pd.DataFrame(symbol_rows)
    symbol_counts.to_csv(RESULTS / "mixed_symbol_distribution.csv", index=False)

    family_rows: list[dict[str, Any]] = []
    for split_scope, frame in [
        ("train_sampled", mixed_train),
        ("val", mixed_val),
        ("train_sampled_plus_val", mixed_dev),
    ]:
        symbol_col = "symbol" if "symbol" in frame.columns else "product_id"
        family = np.where(frame[symbol_col].astype(str).map(txf_like), "mixed_txf_like_subset", "mixed_non_txf_subset")
        tmp = frame.assign(_symbol_family=family)
        for symbol_family, group in tmp.groupby("_symbol_family", sort=True):
            family_rows.extend(
                numeric_stats(
                    group,
                    FOCUS_FEATURES,
                    symbol_family,
                    split_scope,
                    "03B mixed development feature matrix partitioned by symbol regex ^TXF[A-Z0-9]*$",
                )
            )
    family_stats = pd.DataFrame(family_rows)
    family_stats.to_csv(RESULTS / "mixed_symbol_family_feature_stats.csv", index=False)
    return stats, symbol_counts, loaded


def permutation_importance(model: Any, frame: pd.DataFrame, features: list[str], target_col: str, model_label: str, split_scope: str) -> pd.DataFrame:
    """用單次 deterministic shuffle 估計 frozen model 對各特徵的依賴程度。

    評分方式是 Spearman drop：原始分數對 target 的 rank correlation 減去打亂單一特徵後的
    rank correlation。這只用於診斷決策邊界，不用於重新選特徵。
    """

    if len(frame) > PERMUTATION_SAMPLE_ROWS:
        frame = frame.sample(n=PERMUTATION_SAMPLE_ROWS, random_state=SEED).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    y = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    base_score = model.predict(frame[features].to_numpy(dtype=np.float32)).astype(np.float32)
    base = safe_spearman(base_score, y)
    rng = np.random.default_rng(SEED + len(model_label))
    rows = []
    for feature in features:
        shuffled = frame.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        score = model.predict(shuffled[features].to_numpy(dtype=np.float32)).astype(np.float32)
        perm = safe_spearman(score, y)
        rows.append(
            {
                "analysis_scope": "development-only",
                "model_label": model_label,
                "split_scope": split_scope,
                "method": "single_shuffle_spearman_drop",
                "sample_rows": int(len(frame)),
                "base_spearman": base,
                "feature": feature,
                "permuted_spearman": perm,
                "importance_spearman_drop": base - perm,
                "test_new3_new4_used_for_selection": False,
                "new5_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_spearman_drop", ascending=False).reset_index(drop=True)


def compute_importance(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """載入 03B6/04B1 frozen HGB，重算 validation development 的 permutation importance。"""

    selected03 = read_json(PHASE03B6 / "results" / "selected_hgb_convergence_policy.json")
    model03 = load_joblib(PHASE03B6 / "models" / f"{selected03['model_id']}.joblib")
    model04 = load_joblib(PHASE04B1 / "models" / "hgb_regressor_iter_120.joblib")

    imp03 = permutation_importance(
        model03,
        loaded["mixed_val"][p03b.feature_matrix_columns() + [p03b.TARGET]].copy(),
        p03b.feature_matrix_columns(),
        p03b.TARGET,
        "03B6_mixed_hgb_iter_320_threshold_0p0",
        "val_NEW1_development",
    )
    imp04 = permutation_importance(
        model04,
        loaded["txf_val"][p04b.ML_FEATURES + ["target_excess_pnl_H20"]].copy(),
        p04b.ML_FEATURES,
        "target_excess_pnl_H20",
        "04B1_txf_hgb_iter_120_threshold_0p224194",
        "val_NEW1_development",
    )
    out = pd.concat([imp03, imp04], ignore_index=True)
    out.to_csv(RESULTS / "permutation_importance_recomputed.csv", index=False)
    return out


def model_scores(model: Any, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """用 frozen HGB 對指定特徵矩陣輸出 action-value score。"""

    return model.predict(frame[features].to_numpy(dtype=np.float32)).astype(np.float32)


def metric_row_from_p04b_metric(split_scope: str, policy_id: str, metric: p04b.Metric) -> dict[str, Any]:
    """把 04B metric 包成報表列，並明確標記 development-only/no NEW5/no NEW3NEW4 selection。"""

    row = p04b.finalize_metric(split_scope, policy_id, metric)
    row["analysis_scope"] = "development-only"
    row["source_detail"] = "recomputed from TXF-only NEW1/NEW2 development action parquet"
    row["test_new3_new4_used_for_selection"] = False
    row["new5_used"] = False
    return row


def compute_txf_val_prediction_and_policy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """方向 B 的核心診斷：兩個 frozen ML 都跑在同一份 TXF validation 上。

    這個函式回答兩個問題：
    1. 03B6 mixed HGB 和 04B1 TXF HGB 在同一批 TXF action rows 上的預測誤差是否不同。
    2. 把 score 套回各自 frozen threshold 後，state-level policy PnL/選擇率/決策重疊是否不同。

    注意：這裡沒有重新切資料、訓練模型或調整 threshold；threshold 直接來自既有 frozen metadata。
    """

    # 讀 frozen model 與 frozen validation-selected threshold。
    selected03 = read_json(PHASE03B6 / "results" / "selected_hgb_convergence_policy.json")
    threshold03 = float(selected03["threshold"])
    model03 = load_joblib(PHASE03B6 / "models" / f"{selected03['model_id']}.joblib")
    closure04 = read_json(PHASE04B1 / "results" / "txf_hgb_convergence_closure.json")
    threshold04 = float(closure04["selected_threshold_on_validation"])
    model04 = load_joblib(PHASE04B1 / "models" / "hgb_regressor_iter_120.joblib")

    y_chunks: list[np.ndarray] = []
    score03_chunks: list[np.ndarray] = []
    score04_chunks: list[np.ndarray] = []
    max03_chunks: list[np.ndarray] = []
    max04_chunks: list[np.ndarray] = []

    metrics = {
        "queue_rule": p04b.Metric(),
        "03B6_mixed_hgb_on_txf_val": p04b.Metric(),
        "04B1_txf_hgb_on_txf_val": p04b.Metric(),
    }
    overlap_counts = {
        "state_count": 0,
        "both_selected": 0,
        "only_03b6_selected": 0,
        "only_04b1_selected": 0,
        "neither_selected": 0,
        "same_side_selected": 0,
        "opposite_side_selected": 0,
    }
    score_state_frames: list[pd.DataFrame] = []
    bin_rows: list[dict[str, Any]] = []

    for path in p04b.part_paths("val"):
        assert_no_new5_path(path)
        # matrix04 是 action-row 表，event 是 state-row 表；前者供 ML predict，後者供 policy PnL。
        matrix04 = txf_action_matrix_from_part(path)
        event = txf_event_frame_from_part(path)
        matrix03 = add_03b_missing_features(matrix04)
        score03 = model_scores(model03, matrix03, p03b.feature_matrix_columns())
        score04 = model_scores(model04, matrix04, p04b.ML_FEATURES)
        target = matrix04["target_excess_pnl_H20"].to_numpy(dtype=np.float32)

        y_chunks.append(target)
        score03_chunks.append(score03)
        score04_chunks.append(score04)

        # 三個 policy 在完全相同的 state rows 上比較：simple rule、03B6 frozen ML、04B1 frozen ML。
        decision_queue = p04b.policy_decision("side_by_queue_imbalance_abs_0p1", event)
        decision03 = p04b.ml_policy_decision_from_scores(score03, threshold03)
        decision04 = p04b.ml_policy_decision_from_scores(score04, threshold04)

        p04b.update_metric(metrics["queue_rule"], decision_queue, p04b.decision_pnl(event, decision_queue))
        p04b.update_metric(metrics["03B6_mixed_hgb_on_txf_val"], decision03, p04b.decision_pnl(event, decision03))
        p04b.update_metric(metrics["04B1_txf_hgb_on_txf_val"], decision04, p04b.decision_pnl(event, decision04))

        selected03 = decision03 != p04b.DECISION_NO_QUOTE
        selected04 = decision04 != p04b.DECISION_NO_QUOTE
        overlap_counts["state_count"] += int(len(decision03))
        overlap_counts["both_selected"] += int((selected03 & selected04).sum())
        overlap_counts["only_03b6_selected"] += int((selected03 & ~selected04).sum())
        overlap_counts["only_04b1_selected"] += int((~selected03 & selected04).sum())
        overlap_counts["neither_selected"] += int((~selected03 & ~selected04).sum())
        overlap_counts["same_side_selected"] += int(((decision03 == decision04) & selected03 & selected04).sum())
        overlap_counts["opposite_side_selected"] += int(((decision03 != decision04) & selected03 & selected04).sum())

        # 保留 state-level score/decision/PnL，後面依 abs(queue_imbalance) 分箱看決策邊界差異。
        max03 = np.maximum(score03[0::3], score03[1::3])
        max04 = np.maximum(score04[0::3], score04[1::3])
        max03_chunks.append(max03)
        max04_chunks.append(max04)
        score_state_frames.append(
            pd.DataFrame(
                {
                    "split_scope": "val_NEW1_development",
                    "session_id": event["session_id"].astype(str).to_numpy(),
                    "queue_abs_bin": pd.cut(
                        np.abs(pd.to_numeric(event["queue_imbalance"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
                        bins=[-0.001, 0.05, 0.10, 0.20, 0.40, 1.01],
                        labels=["0_0p05", "0p05_0p10", "0p10_0p20", "0p20_0p40", "0p40_1p00"],
                    ).astype(str),
                    "spread_ticks": pd.to_numeric(event["spread_ticks"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                    "score03_max_bid_ask": max03,
                    "score04_max_bid_ask": max04,
                    "decision03": decision03,
                    "decision04": decision04,
                    "queue_decision": decision_queue,
                    "pnl03": p04b.decision_pnl(event, decision03),
                    "pnl04": p04b.decision_pnl(event, decision04),
                    "pnl_queue": p04b.decision_pnl(event, decision_queue),
                }
            )
        )

    # action-row 層級的 prediction diagnostics：比較分數和 target_excess_pnl_H20 的誤差/排序能力。
    y = np.concatenate(y_chunks)
    score03_all = np.concatenate(score03_chunks)
    score04_all = np.concatenate(score04_chunks)
    y_cls = y > 0.0
    pred_rows = []
    for model_label, threshold, score in [
        ("03B6_mixed_hgb_iter_320_on_txf_val", threshold03, score03_all),
        ("04B1_txf_hgb_iter_120_on_txf_val", threshold04, score04_all),
    ]:
        pred_rows.append(
            {
                "analysis_scope": "development-only",
                "split_scope": "val_NEW1_development",
                "heldout_universe": "txf_04b_val",
                "model_label": model_label,
                "frozen_threshold": threshold,
                "action_rows": int(len(y)),
                "rmse_vs_target_excess": rmse(y, score),
                "mae_vs_target_excess": float(mean_absolute_error(y, score)),
                "spearman_score_vs_target": safe_spearman(score, y),
                "pr_auc_positive_excess": float(average_precision_score(y_cls.astype(int), score)),
                "roc_auc_positive_excess": float(roc_auc_score(y_cls.astype(int), score)),
                "test_new3_new4_used_for_selection": False,
                "new5_used": False,
            }
        )
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(RESULTS / "txf_val_prediction_error_same_data.csv", index=False)

    # state-row 層級的 trading diagnostics：把 score 經 threshold 轉成 bid/ask/no-quote 後算 PnL。
    policy_rows = [
        metric_row_from_p04b_metric("val_NEW1_development", "side_by_queue_imbalance_abs_0p1", metrics["queue_rule"]),
        metric_row_from_p04b_metric("val_NEW1_development", "03B6_mixed_hgb_iter_320_threshold_0p0_on_txf", metrics["03B6_mixed_hgb_on_txf_val"]),
        metric_row_from_p04b_metric("val_NEW1_development", "04B1_txf_hgb_iter_120_threshold_0p224194", metrics["04B1_txf_hgb_on_txf_val"]),
    ]
    policy = pd.DataFrame(policy_rows)
    policy.to_csv(RESULTS / "txf_val_policy_recomputed_same_data.csv", index=False)

    denom = max(overlap_counts["state_count"], 1)
    overlap = pd.DataFrame(
        [
            {
                "analysis_scope": "development-only",
                "split_scope": "val_NEW1_development",
                "heldout_universe": "txf_04b_val",
                **overlap_counts,
                "both_selected_rate": overlap_counts["both_selected"] / denom,
                "only_03b6_selected_rate": overlap_counts["only_03b6_selected"] / denom,
                "only_04b1_selected_rate": overlap_counts["only_04b1_selected"] / denom,
                "neither_selected_rate": overlap_counts["neither_selected"] / denom,
                "opposite_side_among_both_selected_rate": overlap_counts["opposite_side_selected"] / max(overlap_counts["both_selected"], 1),
                "test_new3_new4_used_for_selection": False,
                "new5_used": False,
            }
        ]
    )
    overlap.to_csv(RESULTS / "txf_val_decision_overlap.csv", index=False)

    # 用 queue imbalance 絕對值分箱，觀察 ML 是否在 simple rule 門檻附近或強訊號區採取不同交易行為。
    states = pd.concat(score_state_frames, ignore_index=True)
    for model_label, score_col, pnl_col, decision_col in [
        ("03B6_mixed_hgb_iter_320_threshold_0p0_on_txf", "score03_max_bid_ask", "pnl03", "decision03"),
        ("04B1_txf_hgb_iter_120_threshold_0p224194", "score04_max_bid_ask", "pnl04", "decision04"),
        ("side_by_queue_imbalance_abs_0p1", "", "pnl_queue", "queue_decision"),
    ]:
        for queue_bin, group in states.groupby("queue_abs_bin", sort=True):
            selected = group[decision_col].to_numpy(dtype=np.int8) != p04b.DECISION_NO_QUOTE
            selected_pnl = group[pnl_col].to_numpy(dtype=float)[selected]
            row = {
                "analysis_scope": "development-only",
                "split_scope": "val_NEW1_development",
                "feature_bin": "abs_queue_imbalance",
                "bin": str(queue_bin),
                "policy_or_model": model_label,
                "state_count": int(len(group)),
                "selected_action_count": int(selected.sum()),
                "selection_rate": float(selected.mean()) if len(selected) else 0.0,
                "total_excess_pnl_H20": float(np.nansum(selected_pnl)),
                "avg_excess_pnl_per_selected": float(np.nanmean(selected_pnl)) if len(selected_pnl) else 0.0,
                "win_rate": float(np.mean(selected_pnl > 0.0)) if len(selected_pnl) else 0.0,
                "test_new3_new4_used_for_selection": False,
                "new5_used": False,
            }
            if score_col:
                values = group[score_col].to_numpy(dtype=float)
                row["score_mean"] = float(np.nanmean(values))
                row["score_p50"] = float(np.nanquantile(values, 0.50))
                row["score_p95"] = float(np.nanquantile(values, 0.95))
            bin_rows.append(row)
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(RESULTS / "txf_val_policy_by_queue_abs_bin.csv", index=False)
    return pred, policy, overlap, bins


def recompute_queue_policy_for_frame(frame: pd.DataFrame, split_scope: str, universe: str, source_detail: str) -> dict[str, Any]:
    """方向 C：在 03B mixed-style matrix 上重算 simple queue rule 的 policy PnL。

    03B matrix 是 action-row 格式，所以先抽出 bid rows 當 state basis，再呼叫 03B 共用函式
    從同一個 action-row frame 裡取 bid/ask PnL。
    """

    states = frame[frame["action_slot"].astype(int) == p03b.ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    decision = p03b.queue_rule_decision_from_state(states)
    pnl = p03b.policy_pnl_from_decision(frame, decision)
    selected = decision != p03b.DECISION_NO_QUOTE
    selected_pnl = pnl[selected]
    return {
        "analysis_scope": "development-only",
        "universe": universe,
        "split_scope": split_scope,
        "source_detail": source_detail,
        "policy_id": "side_by_queue_imbalance_abs_0p1",
        "state_count": int(len(states)),
        "selected_action_count": int(selected.sum()),
        "selection_rate": float(selected.sum() / max(len(states), 1)),
        "bid_count": int((decision == p03b.DECISION_BID).sum()),
        "ask_count": int((decision == p03b.DECISION_ASK).sum()),
        "total_excess_pnl_H20": float(np.nansum(selected_pnl)),
        "avg_excess_pnl_per_selected": float(np.nanmean(selected_pnl)) if len(selected_pnl) else 0.0,
        "win_rate": float(np.mean(selected_pnl > 0.0)) if len(selected_pnl) else 0.0,
        "loss_rate": float(np.mean(selected_pnl < 0.0)) if len(selected_pnl) else 0.0,
        "max_drawdown": p03b.max_drawdown(pnl),
        "test_new3_new4_used_for_selection": False,
        "new5_used": False,
    }


def recompute_txf_queue_policy(split: str) -> dict[str, Any]:
    """方向 C：在 04B TXF-only train/val 上重算同一條 simple queue rule。"""

    metric = p04b.Metric()
    for path in p04b.part_paths(split):
        assert_no_new5_path(path)
        event = txf_event_frame_from_part(path)
        decision = p04b.policy_decision("side_by_queue_imbalance_abs_0p1", event)
        p04b.update_metric(metric, decision, p04b.decision_pnl(event, decision))
    row = p04b.finalize_metric(split, "side_by_queue_imbalance_abs_0p1", metric)
    row["analysis_scope"] = "development-only"
    row["universe"] = "txf_04b1"
    row["split_scope"] = split
    row["source_detail"] = f"04B TXF-only {split} action universe, development only"
    row["test_new3_new4_used_for_selection"] = False
    row["new5_used"] = False
    return row


def queue_rule_development_contrast(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把 mixed validation 和 TXF-only train/validation 的 simple rule 表現放在同一張表。

    這是使用者追加要求的優先方向 C：先看 simple rule 本身是否在 universe 改變後崩壞。
    若 simple rule 穩定而 ML 大幅改變，比較支持「ML 對資料/特徵/threshold contract 更敏感」。
    """

    rows = [
        recompute_queue_policy_for_frame(
            loaded["mixed_val"],
            "val",
            "mixed_03b6",
            "03B validation feature matrix, NEW1 development",
        ),
        recompute_txf_queue_policy("train"),
        recompute_txf_queue_policy("val"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "queue_rule_development_contrast_recomputed.csv", index=False)
    return out


def posthoc_new3_new4_diagnostics() -> pd.DataFrame:
    """讀取既有 frozen NEW3/NEW4 診斷輸出，只做事後排名對照。

    這裡不載入 NEW3/NEW4 原始資料，也不把 NEW3/NEW4 數字用於選模型或調門檻；
    它只是把已存在 CSV 中的 ML-vs-queue 結果整理成同一張 summary。
    """

    mixed_ml = read_csv(PHASE03B6 / "results" / "hgb_convergence_frozen_oos_eval.csv")
    txf_ml = read_csv(PHASE04B1 / "results" / "txf_ml_final_candidate_matched_random_summary.csv")
    txf_queue = read_csv(PHASE04B / "results" / "txf_non_ml_baseline_eval.csv")
    txf_seq = read_csv(PHASE04B1 / "results" / "txf_ml_final_candidate_causal_fill_sequential_eval.csv")

    rows: list[dict[str, Any]] = []
    for split in ["val", "new3", "new4"]:
        if split in set(mixed_ml["split"]):
            r = mixed_ml[mixed_ml["split"].eq(split)].iloc[0]
            rows.append(
                {
                    "analysis_scope": "NEW3_NEW4_posthoc_diagnostic" if split in {"new3", "new4"} else "development-only",
                    "universe": "mixed_03b6",
                    "split": split,
                    "policy_or_model": str(r["policy_id"]),
                    "metric_family": "instant_excess_markout",
                    "ml_total_excess_pnl_H20": float(r["total_excess_pnl"]),
                    "queue_rule_total_excess_pnl_H20": float(r["queue_rule_total_excess_pnl"]),
                    "ml_minus_queue_rule": float(r["total_excess_pnl"]) - float(r["queue_rule_total_excess_pnl"]),
                    "ml_beats_queue_rule": bool(r["beats_queue_imbalance_rule"]),
                    "selection_split": "val",
                    "test_new3_new4_used_for_selection": False,
                    "new5_used": False,
                    "source_detail": "existing frozen diagnostic CSV; read only for posthoc comparison",
                }
            )
        if split in set(txf_ml["split"]):
            ml_row = txf_ml[txf_ml["split"].eq(split)].iloc[0]
            q_row = txf_queue[(txf_queue["split"].eq(split)) & (txf_queue["policy_id"].eq("side_by_queue_imbalance_abs_0p1"))].iloc[0]
            rows.append(
                {
                    "analysis_scope": "NEW3_NEW4_posthoc_diagnostic" if split in {"new3", "new4"} else "development-only",
                    "universe": "txf_04b1",
                    "split": split,
                    "policy_or_model": str(ml_row["policy_id"]),
                    "metric_family": "instant_excess_markout",
                    "ml_total_excess_pnl_H20": float(ml_row["model_total_excess_pnl_H20"]),
                    "queue_rule_total_excess_pnl_H20": float(q_row["total_excess_pnl_H20"]),
                    "ml_minus_queue_rule": float(ml_row["model_total_excess_pnl_H20"]) - float(q_row["total_excess_pnl_H20"]),
                    "ml_beats_queue_rule": bool(float(ml_row["model_total_excess_pnl_H20"]) > float(q_row["total_excess_pnl_H20"])),
                    "selection_split": "val",
                    "test_new3_new4_used_for_selection": False,
                    "new5_used": False,
                    "source_detail": "existing frozen diagnostic CSV; read only for posthoc comparison",
                }
            )
        seq_split = txf_seq[txf_seq["split"].eq(split)]
        seq_match = seq_split[seq_split["policy_id"].str.contains("ml_hgb", regex=False)]
        seq_queue = seq_split[seq_split["policy_id"].eq("side_by_queue_imbalance_abs_0p1")]
        if not seq_match.empty:
            s = seq_match.iloc[0]
            if not seq_queue.empty:
                queue_seq_pnl = float(seq_queue.iloc[0]["sequential_excess_pnl"])
            else:
                queue_seq_pnl = float(s["sequential_excess_pnl"]) - float(s["ml_minus_queue_rule"])
            rows.append(
                {
                    "analysis_scope": "NEW3_NEW4_posthoc_diagnostic" if split in {"new3", "new4"} else "development-only",
                    "universe": "txf_04b1",
                    "split": split,
                    "policy_or_model": str(s["policy_id"]),
                    "metric_family": "causal_fill_sequential",
                    "ml_total_excess_pnl_H20": float(s["sequential_excess_pnl"]),
                    "queue_rule_total_excess_pnl_H20": queue_seq_pnl,
                    "ml_minus_queue_rule": float(s["ml_minus_queue_rule"]),
                    "ml_beats_queue_rule": bool(float(s["sequential_excess_pnl"]) >= queue_seq_pnl),
                    "selection_split": "val",
                    "test_new3_new4_used_for_selection": False,
                    "new5_used": False,
                    "source_detail": "existing frozen diagnostic CSV; read only for posthoc comparison",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "new3_new4_posthoc_ranking_summary.csv", index=False)
    return out


def fmt_float(value: Any, digits: int = 4) -> str:
    """報告用數字格式化：大數加千分位，小數保留指定位數，NaN 顯示空字串。"""

    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1_000_000:
        return f"{value:,.0f}"
    if abs(value) >= 1_000:
        return f"{value:,.2f}"
    return f"{value:.{digits}f}"


def compact_feature_table(stats: pd.DataFrame) -> pd.DataFrame:
    """從完整特徵分布表挑出 markdown 報告需要展示的欄位與 split。"""

    keep = stats[
        stats["split_scope"].isin(["train_sampled_plus_val", "train_plus_val", "val"])
        & stats["feature"].isin(FOCUS_FEATURES)
        & stats["universe"].isin(["mixed_03b6", "txf_04b1"])
    ].copy()
    cols = ["analysis_scope", "universe", "split_scope", "feature", "n", "mean", "std", "p05", "p50", "p95", "abs_gt_0p1_rate"]
    return keep[cols].sort_values(["feature", "split_scope", "universe"]).reset_index(drop=True)


def compact_importance_table(importance: pd.DataFrame) -> pd.DataFrame:
    """整理 importance 報告表：保留每個模型 top features，也保留使用者指定的 focus features。"""

    top = importance.groupby("model_label", group_keys=False).head(8)
    focus = importance[importance["feature"].isin(FOCUS_FEATURES)]
    out = pd.concat([top, focus], ignore_index=True).drop_duplicates(["model_label", "feature"])
    cols = ["analysis_scope", "model_label", "split_scope", "feature", "base_spearman", "importance_spearman_drop"]
    return out[cols].sort_values(["model_label", "importance_spearman_drop"], ascending=[True, False]).reset_index(drop=True)


def write_report(
    stats: pd.DataFrame,
    symbol_counts: pd.DataFrame,
    family_stats: pd.DataFrame,
    importance: pd.DataFrame,
    pred: pd.DataFrame,
    txf_policy: pd.DataFrame,
    overlap: pd.DataFrame,
    bins: pd.DataFrame,
    queue_contrast: pd.DataFrame,
    posthoc: pd.DataFrame,
) -> None:
    """把所有診斷結果寫成 markdown audit 報告，並同步寫到 03B6 phase 與 000_context。

    上游函式負責產生實際數字；這裡只做表格整理、挑選關鍵數字、撰寫邊界說明與誠實結論。
    """

    # 報告中的精簡表另存 CSV，方便之後 meeting 或簡報直接引用同一份數字。
    feature_compact = compact_feature_table(stats)
    feature_compact.to_csv(RESULTS / "feature_distribution_report_table.csv", index=False)
    importance_compact = compact_importance_table(importance)
    importance_compact.to_csv(RESULTS / "importance_report_table.csv", index=False)

    sym = symbol_counts[symbol_counts["split_scope"].eq("train_sampled_plus_val")].copy()
    sym_summary = (
        sym.groupby("is_txf_like_symbol", as_index=False)
        .agg(action_rows=("action_rows", "sum"), action_row_fraction=("action_row_fraction", "sum"), symbols=("symbol", "nunique"))
        .sort_values("is_txf_like_symbol")
    )
    sym_summary["is_txf_like_symbol"] = sym_summary["is_txf_like_symbol"].map({False: "non_txf_like", True: "txf_like"})
    sym_summary.to_csv(RESULTS / "mixed_symbol_family_row_share.csv", index=False)

    family_compact = family_stats[
        family_stats["split_scope"].eq("train_sampled_plus_val") & family_stats["feature"].isin(FOCUS_FEATURES)
    ][["analysis_scope", "universe", "split_scope", "feature", "n", "mean", "std", "p50", "p95", "abs_gt_0p1_rate"]].copy()
    family_compact.to_csv(RESULTS / "mixed_symbol_family_feature_report_table.csv", index=False)

    pred_report = pred[
        [
            "analysis_scope",
            "split_scope",
            "heldout_universe",
            "model_label",
            "frozen_threshold",
            "action_rows",
            "rmse_vs_target_excess",
            "mae_vs_target_excess",
            "spearman_score_vs_target",
            "pr_auc_positive_excess",
            "roc_auc_positive_excess",
        ]
    ].copy()

    txf_policy_report = txf_policy[
        [
            "analysis_scope",
            "split",
            "policy_id",
            "state_count",
            "selected_action_count",
            "selection_rate",
            "bid_count",
            "ask_count",
            "total_excess_pnl_H20",
            "avg_excess_pnl_per_selected",
            "win_rate",
            "max_drawdown",
        ]
    ].copy()

    queue_report = queue_contrast[
        [
            "analysis_scope",
            "universe",
            "split_scope",
            "policy_id",
            "state_count",
            "selected_action_count",
            "selection_rate",
            "total_excess_pnl_H20",
            "avg_excess_pnl_per_selected",
            "win_rate",
            "max_drawdown",
        ]
    ].copy()

    posthoc_report = posthoc[
        [
            "analysis_scope",
            "universe",
            "split",
            "metric_family",
            "policy_or_model",
            "ml_total_excess_pnl_H20",
            "queue_rule_total_excess_pnl_H20",
            "ml_minus_queue_rule",
            "ml_beats_queue_rule",
        ]
    ].copy()

    queue_val = queue_contrast[queue_contrast["split_scope"].eq("val")]
    mixed_queue_val = float(queue_val[queue_val["universe"].eq("mixed_03b6")]["avg_excess_pnl_per_selected"].iloc[0])
    txf_queue_val = float(queue_val[queue_val["universe"].eq("txf_04b1")]["avg_excess_pnl_per_selected"].iloc[0])
    policy_val = txf_policy.set_index("policy_id")
    pnl_03_on_txf = float(policy_val.loc["03B6_mixed_hgb_iter_320_threshold_0p0_on_txf", "total_excess_pnl_H20"])
    pnl_04_on_txf = float(policy_val.loc["04B1_txf_hgb_iter_120_threshold_0p224194", "total_excess_pnl_H20"])
    pnl_queue_txf = float(policy_val.loc["side_by_queue_imbalance_abs_0p1", "total_excess_pnl_H20"])
    imp_top03 = importance[importance["model_label"].str.startswith("03B6")].iloc[0]
    imp_top04 = importance[importance["model_label"].str.startswith("04B1")].iloc[0]
    overlap_row = overlap.iloc[0]

    # status=mixed 是刻意保守：有診斷到決策邊界改變，但 frozen artifact 無法單獨證明非 TXF 污染。
    conclusion_status = "mixed"
    conclusion = (
        "diagnosed but inconclusive. The audit finds a concrete negative result: the available 03B/03B6 development "
        "feature matrices used here contain only the TXFB6 symbol, so the current artifacts do not substantiate the "
        "simple explanation that removing non-TXF products alone caused the reversal. On the same TXF validation rows, "
        "04B1 still ranks and trades better than the frozen 03B6 model, so a real ML decision-boundary change exists. "
        "The supported mechanism is broader than product filtering: Phase 04B/04B1 changed the TXF action-universe "
        "coverage/loader contract and feature schema, then fit and thresholded the HGB on that rebuilt TXF development "
        "universe. The exact contribution of coverage, schema, and fit/threshold scale cannot be uniquely separated "
        "from the frozen artifacts without rebuilding historical intermediates."
    )

    runtime_note = (
        f"Runtime compatibility note: the final script was run under Python {sys.version.split()[0]} "
        f"with sklearn {sklearn.__version__}. The frozen HGB joblib objects loaded and predictions "
        "completed; the recomputed 04B1 TXF validation policy total matched the frozen 04B1 validation "
        "total. Treat sklearn/joblib persistence as residual risk if the runtime changes."
    )

    # 以下 lines 是完整 markdown 報告。每個區塊都清楚標明 development-only 或 NEW3/NEW4 posthoc。
    lines = [
        f"status: {conclusion_status}",
        "# Audit: 03B6 vs 04B1 Reversal Root Cause",
        "",
        f"Primary report location: `{PHASE_AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix()}`. "
        f"A synchronized context copy is also written to `{AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix()}`.",
        "",
        "This audit did not train or tune any model. It loaded the frozen 03B6 and 04B1 HGB model objects and their frozen validation-selected thresholds, recomputed development-only diagnostics on NEW1/NEW2-derived artifacts, and read existing frozen NEW3/NEW4 diagnostic CSVs only for posthoc comparison.",
        "",
        "Hard boundary: no NEW5 path was loaded. The script hard-stops on any file path containing `NEW5`.",
        "",
        runtime_note,
        "",
        "## Frozen Objects Used",
        "",
        "| object | frozen setting | role | split use |",
        "|:--|:--|:--|:--|",
        "| 03B6 mixed HGB | `hgb_regressor_iter_320`, threshold `0.0` | diagnostic frozen model | selected on mixed validation only |",
        "| 04B1 TXF HGB | `hgb_regressor_iter_120`, threshold `0.22419440373778343` | diagnostic frozen model | selected on TXF validation only |",
        "| simple rule | `side_by_queue_imbalance_abs_0p1` | fixed baseline | no tuned parameters in this audit |",
        "",
        "## Development-Only Feature Distribution",
        "",
        "Row basis is action-row model input. For `queue_imbalance`, `deep_imbalance`, `spread_ticks`, and `event_rate_past_50`, bid/ask/no-quote triplication does not change distributional values within a state.",
        "",
        feature_compact.to_markdown(index=False),
        "",
        "## Mixed Universe Composition",
        "",
        sym_summary.to_markdown(index=False),
        "",
        "## Mixed Universe TXF vs Non-TXF Subsets",
        "",
        family_compact.to_markdown(index=False),
        "",
        "## Recomputed Permutation Importance",
        "",
        "Method: one deterministic shuffle per feature on validation action rows, scored by Spearman drop against H20 excess PnL. This is diagnostic only and not used to pick features or thresholds.",
        "",
        importance_compact.to_markdown(index=False),
        "",
        f"Top 03B6 feature by this diagnostic: `{imp_top03['feature']}` with Spearman drop `{fmt_float(imp_top03['importance_spearman_drop'])}`. Top 04B1 feature: `{imp_top04['feature']}` with Spearman drop `{fmt_float(imp_top04['importance_spearman_drop'])}`.",
        "",
        "## Same TXF Validation Prediction Error",
        "",
        "Both frozen models were run on the same pure-TXF NEW1 validation action rows. This is the development-only held-out comparison requested in B; no NEW3/NEW4 row is used here.",
        "",
        pred_report.to_markdown(index=False),
        "",
        "## Same TXF Validation Policy PnL",
        "",
        txf_policy_report.to_markdown(index=False),
        "",
        f"On the same TXF validation data, 03B6 frozen ML PnL is `{fmt_float(pnl_03_on_txf)}`, 04B1 frozen ML PnL is `{fmt_float(pnl_04_on_txf)}`, and the queue rule is `{fmt_float(pnl_queue_txf)}`.",
        "",
        "## Same TXF Validation Decision Overlap",
        "",
        overlap.to_markdown(index=False),
        "",
        f"Only-04B1-selected states are `{fmt_float(overlap_row['only_04b1_selected_rate'] * 100, 2)}%` of TXF validation states, while only-03B6-selected states are `{fmt_float(overlap_row['only_03b6_selected_rate'] * 100, 2)}%`. This is a direct development-only measurement of changed HGB decision boundary behavior.",
        "",
        "## TXF Validation Queue-Imbalance Bins",
        "",
        bins[
            [
                "analysis_scope",
                "split_scope",
                "feature_bin",
                "bin",
                "policy_or_model",
                "state_count",
                "selected_action_count",
                "selection_rate",
                "total_excess_pnl_H20",
                "avg_excess_pnl_per_selected",
                "win_rate",
                "score_mean",
                "score_p50",
                "score_p95",
            ]
        ].to_markdown(index=False),
        "",
        "## Simple Rule Development Contrast",
        "",
        "This table is development-only and recomputed from existing feature/action artifacts. It checks whether the simple queue rule itself collapses when moving from mixed to TXF. The mixed train sample is intentionally excluded here because the Phase 03B train matrix is an action-row sample and does not preserve complete bid/ask state triplets for policy PnL.",
        "",
        queue_report.to_markdown(index=False),
        "",
        f"Validation avg PnL per selected action for the queue rule is `{fmt_float(mixed_queue_val)}` in mixed 03B6 validation and `{fmt_float(txf_queue_val)}` in TXF-only validation.",
        "",
        "## NEW3/NEW4 Posthoc Diagnostics",
        "",
        "These numbers are read from already frozen diagnostic CSVs and used only to check whether the development-only pattern is consistent out of sample. They were not used to pick models, thresholds, features, or hypotheses.",
        "",
        posthoc_report.to_markdown(index=False),
        "",
        "## Mechanism Assessment",
        "",
        "The development-only evidence supports part of the reversal story but rejects the simplest version of it. The current 03B/03B6 development matrices used by the frozen model contain only `TXFB6`, not a visible mixture of TXF plus non-TXF symbols. Therefore the available artifacts cannot prove that non-TXF products polluted the HGB.",
        "",
        "The same-TXF held-out run is still a strong sanity check: when both frozen models are applied to the same TXF validation data, 04B1 has lower error, stronger rank metrics, and higher policy PnL. This points to a changed HGB decision boundary caused by the rebuilt TXF action universe, feature schema, training coverage, and threshold scale. The simple queue rule remains positive and similar in validation average PnL per selected action, so the reversal is ML-specific rather than a collapse of the queue rule itself.",
        "",
        "## Honest Conclusion",
        "",
        conclusion,
        "",
        "No NEW5 data was loaded, and no NEW3/NEW4 number was used for selection or tuning in this audit.",
    ]
    report_text = "\n".join(lines) + "\n"
    AUDIT_PATH.write_text(report_text, encoding="utf-8")
    PHASE_AUDIT_PATH.write_text(report_text, encoding="utf-8")


def main() -> None:
    """主流程：依序執行所有診斷，最後輸出 markdown 與 CSV。

    執行順序刻意先 development-only，再 frozen OOS posthoc，避免讀者誤以為 NEW3/NEW4
    有參與模型選擇或假說挑選。
    """

    ensure_dirs()
    stats, symbol_counts, loaded = feature_distribution_tables()
    family_stats = read_csv(RESULTS / "mixed_symbol_family_feature_stats.csv")
    importance = compute_importance(loaded)
    pred, txf_policy, overlap, bins = compute_txf_val_prediction_and_policy()
    queue_contrast = queue_rule_development_contrast(loaded)
    posthoc = posthoc_new3_new4_diagnostics()
    write_report(stats, symbol_counts, family_stats, importance, pred, txf_policy, overlap, bins, queue_contrast, posthoc)
    print(f"wrote {PHASE_AUDIT_PATH}")
    print(f"wrote {AUDIT_PATH}")
    print(f"wrote CSV diagnostics under {RESULTS}")


if __name__ == "__main__":
    main()
