"""Create simple mock order-book CSV files for early pipeline smoke tests."""

import os
import time
import numpy as np
import pandas as pd

# This file predates the Taiwan replay adapter and is kept as a lightweight
# way to test the preprocessing stack without large raw datasets.

def generate_mock_lob(symbol, tick_size, num_ticks=10000, base_price=100.0):
    """
    產生模擬的 Level-2 LOB 逐筆資料 (Tick-level LOB snapshot)
    包含 5 檔深度，時間戳記精確至奈秒。
    """
    # 模擬起始時間: 2023-01-01 09:30:00 (奈秒)
    start_ts = int(pd.Timestamp("2023-01-01 09:30:00").timestamp() * 1e9)
    current_ts = start_ts
    
    records = []
    current_mid = base_price
    
    for _ in range(num_ticks):
        # 模擬時間推進: 隨機加 1,000 ~ 50,000 奈秒
        current_ts += np.random.randint(1000, 50000)
        
        # 模擬隨機漫步更新 Mid Price (20%漲, 60%平, 20%跌)
        step = np.random.choice([-tick_size, 0, tick_size], p=[0.2, 0.6, 0.2])
        current_mid += step
        
        # 確保價格恰好對齊在 tick_size 的網格上
        current_mid = round(current_mid / tick_size) * tick_size
        
        # 隨機產生 Spread (1 到 3 個 Tick)
        spread = np.random.randint(1, 4) * tick_size
        
        ask_1 = current_mid + spread / 2
        bid_1 = current_mid - spread / 2
        
        # 確保 Ask_1 與 Bid_1 符合 Tick Size 邊界限制
        ask_1 = np.ceil(ask_1 / tick_size) * tick_size
        bid_1 = np.floor(bid_1 / tick_size) * tick_size
        
        # 若 Ask_1 與 Bid_1 重合，則自動撐開一個 Tick Size
        if ask_1 <= bid_1:
            ask_1 = bid_1 + tick_size
            
        record = {'timestamp': current_ts}
        
        # 產生 5 檔深度的委託價格與掛單量
        for i in range(1, 6):
            record[f'ask_px_{i}'] = round(ask_1 + (i - 1) * tick_size, 4)
            record[f'ask_vol_{i}'] = np.random.randint(1, 100)
            
            record[f'bid_px_{i}'] = round(bid_1 - (i - 1) * tick_size, 4)
            record[f'bid_vol_{i}'] = np.random.randint(1, 100)
            
        records.append(record)
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_dir = os.path.join(project_root, "data", "raw")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 依專題設定的三種標的(ES, NQ, CL)
    assets = [
        {"symbol": "ES", "tick_size": 0.25, "base_price": 4000.0},
        {"symbol": "NQ", "tick_size": 0.25, "base_price": 12000.0},
        {"symbol": "CL", "tick_size": 0.01, "base_price": 80.0}
    ]
    
    for asset in assets:
        print(f"Generating mock data for {asset['symbol']}...")
        df = generate_mock_lob(
            symbol=asset['symbol'], 
            tick_size=asset['tick_size'], 
            num_ticks=10000, 
            base_price=asset['base_price']
        )
        save_path = os.path.join(save_dir, f"mock_{asset['symbol']}.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} ticks of mock data to {save_path}")
