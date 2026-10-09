# `new_era/` Archive Note

- Archive path: `C:\Users\holbo\00VS_projjects\new_era_archive_20260808.tar.gz`
- Archive date: `2026-08-08`
- Archive contents: full `new_era/` directory preserved without restructuring
- Restore only if needed for one scenario: rerunning `01_full_l2_action_universe_builder` from scratch after discarding the already-frozen outputs in `data/full_l2_action_universe*.parquet`

This archive exists because the `n2ew_era/` scripts only depend on `new_era/` for the frozen upstream build phase, and the produced parquet outputs are already present in the main project data area.
