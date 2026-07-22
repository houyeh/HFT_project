import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

def build_hls4ml_friendly_mlp(input_shape):
    """
    建立一個專為硬體 (hls4ml 與 PYNQ-Z2 晶片) 設計的輕量化多層感知機 (MLP)。
    
    重點設計考量：
    - 為了節省 FPGA 的硬體資源 (DSPs 和 LUTs)，我們避免使用過深的網路或過於複雜的運算。
    - 模型規模需極小化，使得之後硬體轉換時可以達到 1 個 Clock 週期的超低延遲。
    """
    model = Sequential([
        # 將輸入的 (100 時間步, 8 特徵) 序列攤平為 800 維度的單一向量
        Flatten(input_shape=input_shape, name="flatten_input"),
        
        # 第一層全連接層：將 800 維壓縮為 32 維
        Dense(32, name="fc1"),
        BatchNormalization(name="bn1"), # 加入批次正規化穩定訓練
        Activation("relu", name="relu1"),
        
        # 第二層全連接層：再壓縮至 16 維
        Dense(16, name="fc2"),
        BatchNormalization(name="bn2"),
        Activation("relu", name="relu2"),
        
        # 輸出層：輸出 3 個類別的機率 (下跌 Down, 盤整 Stationary, 上漲 Up)
        Dense(3, activation="softmax", name="output_probabilities")
    ], name="Baseline_MLP")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy", # 因為標籤 y 是整數 (0, 1, 2)
        metrics=["accuracy"]
    )
    return model

def main():
    """
    主程式：
    1. 讀取 Step 2 儲存的 .npz 壓縮檔。
    2. 將不同商品 (大台/小台) 的資料合併成統一的訓練池。
    3. 進行 Train/Validation 資料集分割。
    4. 計算類別權重 (Class Weights) 以解決金融資料中「盤整多、漲跌少」的不平衡問題。
    5. 建立並訓練輕量化 MLP 模型，同時設定 Early Stopping 防止過度擬合。
    6. 將訓練好的模型存為 .h5 檔案供後續 HLS 轉換使用。
    """
    npz_path = project_root / "new_era" / "002_zscore_dataset.npz"
    if not npz_path.exists():
        print(f"Error: {npz_path} not found. Did you run 02_feature_engineering.py?")
        sys.exit(1)
        
    print(f"Loading high-purity Z-score dataset from {npz_path}...")
    data = np.load(npz_path)
    
    # 將 TXFB6 與 MXFB6 的特徵 (X) 與標籤 (y) 分別合併進一個全域訓練池 (Training Pool)
    X_list, y_list = [], []
    for key in data.files:
        if key.endswith("_X"):
            X_list.append(data[key])
        elif key.endswith("_y"):
            y_list.append(data[key])
            
    if not X_list:
        print("No data found in npz file.")
        sys.exit(1)
        
    # 合併所有 numpy 陣列
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"Total Combined Data: X shape {X.shape}, y shape {y.shape}")
    
    # 隨機切分 80% 作為訓練集，20% 作為驗證集 (作為概念驗證 POC)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 動態計算類別權重 (Class weights) 來處理資料不平衡問題 (Up/Down 數量通常遠少於 Stationary)
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {int(c): float(total / (len(classes) * count)) for c, count in zip(classes, counts)}
    print(f"Class Weights applied to fight imbalance: {class_weights}")
    
    # 輸入維度基於前處理階段的設定 (時間步=100, 特徵數=8)
    model = build_hls4ml_friendly_mlp(input_shape=(X.shape[1], X.shape[2]))
    model.summary()
    
    model_save_path = project_root / "new_era" / "003_baseline_mlp.h5"
    
    # 設定回呼函數 (Callbacks)
    callbacks = [
        # 當 val_loss 連續 5 個 Epoch 沒有改善時，提早停止訓練並恢復最佳權重
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        # 只儲存 val_loss 最低的那個模型
        ModelCheckpoint(filepath=str(model_save_path), save_best_only=True, monitor="val_loss")
    ]
    
    print("\nStarting Keras Model Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=1024,  # 使用較大 Batch size 來加速訓練，並讓梯度更新較穩定
        class_weight=class_weights, # 套用類別權重
        callbacks=callbacks,
        verbose=1
    )
    
    val_acc = max(history.history['val_accuracy'])
    print(f"\nTraining completed. Best Validation Accuracy: {val_acc:.4f}")
    print(f"Keras h5 model saved to: {model_save_path}")

if __name__ == "__main__":
    # 程式進入點
    main()
