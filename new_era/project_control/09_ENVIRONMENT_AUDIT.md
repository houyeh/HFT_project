# Environment Audit

- Python version: (Assumed standard via conda/venv)
- conda / venv: `.venv/` and `venv/` directories detected.
- requirements files: `requirements.txt`, `new_era/15_models_playaround/requirements_015_models_playaround.txt`
- pyproject / setup files: Not immediately visible at root.
- large directories: `data/`, `.venv/`
- generated output folders: `015_models_playaround_result/`, `meeting/w13/__results/`, `016_new_binary/outputs/`, `017_regime_auto_gate/outputs/`
- duplicate result folders: `015_models_playaround_result_expC_only/`, `015_models_playaround_result_newall_h5_only/`
- old caches: `.pytest_cache/`, `__pycache__`
- .ipynb_checkpoints: Likely scattered in `notebooks/`
- large model checkpoints: `new_era/*.h5`, `new_era/*.pth`, checkpoints inside `015_models_playaround_result/checkpoints/`
- large CSV / PNG output folders: Inside `eval_outputs/` and `diagnosis/`

**Conclusion**: The repository is functional but has accumulating caches and duplicated snapshot folders that require hygiene. No active deletions are performed in this audit.
