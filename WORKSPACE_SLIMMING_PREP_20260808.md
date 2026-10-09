# 00_HFT_project Workspace Slimming Prep (2026-08-08)

This log records every destructive or potentially destructive file operation performed for the workspace slimming task.

## Safety constraints respected

- `src/` was not modified, moved, or deleted.
- `data/NEW5` was not read for content analysis, moved, or deleted.
- Before deleting any file, a project-wide Python-reference search was run and recorded below.
- No Git history rewrite commands were used.

## Step 1: Safe deduplication checks

### Target files

- `n2ew_era/06c_tcn_action_value_hls4ml/results/feat18_sequences_train.pt`
- `n2ew_era/06c_tcn_action_value_hls4ml/results/feat18_sequences_val.pt`
- `n2ew_era/06c_tcn_action_value_hls4ml/results/feat18_sequences_new3.pt`
- `n2ew_era/06c_tcn_action_value_hls4ml/results/feat18_sequences_new4.pt`
- `n2ew_era/06d_tcn_5block_prepad_hls4ml/results/feat18_sequences_train.pt`
- `n2ew_era/06d_tcn_5block_prepad_hls4ml/results/feat18_sequences_val.pt`
- `n2ew_era/06d_tcn_5block_prepad_hls4ml/results/feat18_sequences_new3.pt`
- `n2ew_era/06d_tcn_5block_prepad_hls4ml/results/feat18_sequences_new4.pt`

### SHA256 results

- `train.pt`: `35EFBBD7E8307E5A06F758623808954A0089258291F98E3AF10047C83A28DD57` in both `06c` and `06d`
- `val.pt`: `10B97CCAEF731129E1492057C06CB56DE47FB86709F7F8B8D12C22BF7B70FD54` in both `06c` and `06d`
- `new3.pt`: `5226C2D06C03A956F8A6DDC638CE29261CCE100B6123719FB6B201249F1EBC72` in both `06c` and `06d`
- `new4.pt`: `41143AE127417FC8B97AD5F73184E1F13B4A439E96BAB6DB1FAC4190156186A2` in both `06c` and `06d`

### Project-wide Python reference search before deletion

Command intent: search for either exact filenames or directory-qualified paths before replacing any duplicate file.

Results:

```text
.\n2ew_era\06d_tcn_5block_prepad_hls4ml\common_06d.py:672:    if not (RESULTS / "feat18_sequences_train.pt").exists():
.\n2ew_era\06c_tcn_action_value_hls4ml\common_06c.py:564:    if not (RESULTS / "feat18_sequences_train.pt").exists():
```

Interpretation:

- Only same-directory relative filename checks were found.
- No Python file referenced a hard-coded cross-directory path to these `.pt` files.

### Symlink replacement attempt and recovery

- Attempted action: keep the `06c` copies and replace the `06d` copies with symbolic links.
- Result: Windows refused symbolic-link creation with `NewItemSymbolicLinkElevationRequired`.
- Recovery action: every deleted `06d` file was immediately restored by copying back from the identical `06c` source.
- Net result after recovery: no `.pt` file remains missing; no deduplication was completed yet.

### `NEW_ALL` vs `NEW1`/`NEW2` duplicate report

- Report file created: `data/NEW_ALL_duplicate_report_20260808.csv`
- Scope: `data/NEW_ALL`, `data/NEW1`, `data/NEW2`, `data/NEW3`, `data/NEW4`
- `NEW5` content was not read. Only later metadata inventory touched `NEW5` file sizes.
- Duplicate findings:
  - `NEW_ALL` and `NEW1`: 18 duplicate CSV entries by filename, size, and SHA256
  - `NEW_ALL` and `NEW2`: 16 duplicate CSV entries by filename, size, and SHA256
  - `NEW3` and `NEW4`: no duplicate CSV entries against `NEW_ALL` by filename
- No deletion was performed for these CSVs, pending user review.

## Step 2: `new_era/` archive

### Archive statistics before compression

- File count: `10,955`
- Total size: `20,810,551,989` bytes (`19.381 GiB`)

### Archive output

- Archive path: `C:\Users\holbo\00VS_projjects\new_era_archive_20260808.tar.gz`
- Archive size: `9,283,708,473` bytes (`8.646 GiB`)
- Archive date: `2026-08-08`

### Archive verification

- Archive entry count: `10,955`
- Extracted verification file count: `10,955`
- Counts match source: `True`

## Step 3: nested Git cleanup

### `n2ew_era/.git` status check

Command: `git -c safe.directory=C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era -C n2ew_era log --oneline -n 5`

Result:

```text
fatal: your current branch 'main' does not have any commits yet
```

Interpretation:

- `n2ew_era/.git` exists, but the nested repo has no commits.

### Project-wide Python reference search before deletion

Command intent: search Python files for any explicit `n2ew_era/.git` path dependency before removing the nested repo metadata.

Results:

```text
[no matches]
```

### Deletion result

- Deleted: `n2ew_era/.git`
- Post-delete check: parent repo `git status --short` no longer reports a nested-repository warning for `n2ew_era/`

## Step 4: large binary inventory

### Inventory output

- Report file created: `LARGE_BINARY_INVENTORY_20260808.csv`
- Filter: files larger than `100 MB`
- Scope:
  - `data/**/*.pt|parquet|csv`
  - `n2ew_era/**/models/**/*.pt`
  - `n2ew_era/**/results/**/*.pt|parquet|csv`
- Note: `NEW5` content was not opened, but its file metadata appears in this inventory because the task explicitly requested a size/path report under `data/`.
