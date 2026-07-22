# Cleanup Plan

## Safe to keep
- `data/`
- `new_era/15_models_playaround/015_models_playaround_result/` (core baseline)
- `new_era/016_new_binary/outputs/`
- `new_era/017_regime_auto_gate/outputs/`
- `new_era/018_queue_aware_validation/outputs/`
- `new_era/project_control/`
- `meeting/w13/__results/`

## Safe to archive after confirmation
- Duplicate run folders like `015_models_playaround_result_expC_only/`, `015_models_playaround_result_newall_h5_only/`
- Old `meeting/` subfolders (if any preceding w13)
- Experimental scripts that are superseded.

## Safe to delete after confirmation
- `.pytest_cache/`
- `__pycache__/` in all subdirectories.
- `.ipynb_checkpoints/`
- Temporary unreferenced `.pth` files left in root if duplicated in `checkpoints/`.

## Never delete
- raw data
- 015 baseline results
- 016 outputs
- 017 outputs
- 018 outputs
- project_control
- final meeting deck assets
- scripts that generated important results
