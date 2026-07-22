# GPU Training Pack for Phase 030

This pack trains residual Quote-EV veto models on the baseline universe selected by `bid_only_spread_ge_1`.

The target machine is an RTX 4050 Laptop GPU with 6GB VRAM and 16GB RAM. All PyTorch scripts expose `--batch-size` and call `torch.cuda.empty_cache()` so you can scale down if CUDA runs out of memory.

## Setup

```bash
conda env create -f environment.yml
conda activate hft-ev-veto-gpu
python 01_training_pipeline_audit.py
```

## Files

- `train.parquet`, `val.parquet`, `test.parquet`, `new4.parquet`: baseline-universe quote samples.
- `sequence_index.parquet`: past-only sequence pointers; it does not expand full windows into RAM.
- `01_training_pipeline_audit.py`: data/label audit.
- `train_mlp_smoke_test.py`: fast CUDA/dataloader/loss smoke test.
- `04_model_family_training.py`: tabular logistic, histogram gradient boosting, and PyTorch MLP training.
- `05_tcn_sequence_training.py`: causal TCN training with dynamic sequence slicing.
- `06_evaluate_incremental_value.py`: frozen test and NEW4 incremental value evaluation.

## Recommended Order

```bash
python 01_training_pipeline_audit.py
python train_mlp_smoke_test.py --epochs 5 --batch-size 64
python 04_model_family_training.py --model all --epochs 80 --batch-size 64
python 05_tcn_sequence_training.py --window 50 --epochs 40 --batch-size 32 --channels 16
python 06_evaluate_incremental_value.py --model-path models/mlp_model.pt --model-type mlp
python 06_evaluate_incremental_value.py --model-path models/tcn_model.pt --model-type tcn --window 50
```

If CUDA OOM occurs, retry with smaller batches:

```bash
python 05_tcn_sequence_training.py --window 50 --epochs 40 --batch-size 16 --channels 8
```

## Research Rules

- Train on `train.parquet`.
- Select thresholds only on `val.parquet`.
- Evaluate `test.parquet` and `new4.parquet` only after the model and threshold are frozen.
- NEW4 is external OOS evidence only and must not be used for tuning.
- The target is a simulator-derived proxy: `negative_ev_moderate = 1 if quote_ev_moderate < 0`.

## Feature Columns

```text
spread_ticks, queue_imbalance_calc, deep_imbalance_calc, top_depth, depth_ratio,
volatility_past_50, abs_return_past_50, event_rate_past_50,
queue_imbalance_change_past, normalized_spread_calc
```

## Sequence Note

`sequence_index.parquet` stores `[sample_id, start_row_id, end_row_id, target_row_id]` using per-session `local_row_id`. The TCN script slices windows dynamically inside the PyTorch Dataset and pads missing history on the left. It never materializes all sequences at once.
