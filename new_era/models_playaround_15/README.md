# 015 Models Playaround (core library)

This folder contains the **core reusable code** for the 015 playaround experiment system.

你日常要跑的「有 01/02/03/04 執行順序」腳本在：

`new_era/15_models_playaround/`

## Entry points

- Notebook: `notebooks/015_models_playaround.ipynb`
- Dataset: `new_era/models_playaround_15/dataset_generation.py`
- Train: `new_era/models_playaround_15/train_tcn.py`
- Eval (full plots/CSVs): `new_era/models_playaround_15/evaluate_tcn_full.py`
- Backtest (val sweep -> fixed test): `new_era/models_playaround_15/backtest_val_sweep.py`

## Experiment outputs

All results go under:

`new_era/15_models_playaround/015_models_playaround_result/<experiment_id>/`

where `experiment_id` includes at least:

- `label_type`
- `feature_count`
- `window_size`
- `horizon`
- `threshold_ticks`

## Matrix + summary

- Run a batch matrix: `new_era/models_playaround_15/run_matrix.py`
- Aggregate results into `experiment_summary.csv` + `experiment_report.md`:
  `new_era/models_playaround_15/generate_experiment_report.py`
- Collect all plots into one place: `new_era/models_playaround_15/charts_gallery.py`
