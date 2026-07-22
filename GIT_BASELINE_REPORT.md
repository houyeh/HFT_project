# Git Baseline Report

## Baseline Summary

- Repository root: `C:\Users\holbo\00VS_projjects\00_HFT_project`
- Baseline commit message: `baseline: first snapshot of 00_HFT_project (pre-departure freeze, 2026-07)`
- Baseline tag: `v0-baseline-20260714`
- Remote origin: `https://github.com/houyeh/HFT_project.git`
- Push status: local only; no push executed.
- Commit hash: resolve after commit with `git rev-parse v0-baseline-20260714`.

This report is included in the baseline snapshot. A Git commit cannot contain its own final hash inside a tracked file without rewriting history, so the immutable tag above is the in-repo hash anchor.

## Files Included

- Staged file count before commit: 7,906
- Staged working-tree payload size before commit: 950.69 MB
- Staged payload size before commit: 0.928 GB
- Single staged files over 50 MB: 0
- `data/raw/` staged files: 0
- `data/` staged files: 0
- Python virtual environment files staged: 0

Included classes:

- Markdown reports and project documentation.
- Python scripts and tests.
- `results/*.csv` provenance outputs below the 50 MB single-file threshold.
- HLS source code under `hls_src/`.
- Text synthesis, timing, and diagnostic reports that are not ignored build intermediates.
- Figures and presentation assets below the size threshold.

## Files Excluded

The detailed manifest is in `EXCLUDED_FILES_MANIFEST.md`.

- Excluded large-file count: 535
- Excluded large-file total size: 71.38 GB
- `data/raw/` is excluded as a whole directory for copyright/licensing hygiene, regardless of file size.
- Large raw/generated data payloads, checkpoints, model binaries, and HLS/Vivado build products are excluded and should be regenerated from the listed scripts or phase pipelines.

Primary excluded classes:

- `.venv/`, Python caches, editor state.
- `data/raw/`, `data/`, and nested phase `data/` payloads.
- `*.csv`, `*.parquet`, `*.pt`, and similar generated payloads over 50 MB.
- HLS/Vivado implementation/simulation intermediates such as `.Xil/`, `solution*/impl/`, `solution*/sim/`, `hls4ml_prj/`, and generated HDL folders.
- Temporary/discarded work directories.

## Pre-Commit Checks

Commands run before committing:

```powershell
git status --short --branch
git count-objects -vH
git ls-files | Measure-Object
git ls-files | Where-Object { $_ -like 'data/raw/*' -or $_ -like 'data/*' -or $_ -like '*/data/*' }
```

Observed before commit:

```text
git count-objects -vH
count: 7114
size: 211.36 MiB
in-pack: 0
packs: 0
size-pack: 0 bytes
prune-packable: 0
garbage: 0
size-garbage: 0 bytes
```

## Nested Repository Note

`n2ew_era/` contains its own `.git` directory with no commits. The nested `.git` metadata was not deleted, reset, rewritten, or moved.

To include `n2ew_era` contents in the parent baseline without creating a broken gitlink, its non-ignored leaf files were staged directly into the parent repository index using Git object hashing and `git update-index`. This preserves the nested metadata on disk while making the parent baseline self-contained for the tracked source/report files.

## Verification After Commit

Run:

```powershell
git log --oneline --stat -1
git rev-parse HEAD
git rev-parse v0-baseline-20260714
git status --short --branch
git tag --list v0-baseline-20260714
```

Expected:

- `git rev-parse HEAD` and `git rev-parse v0-baseline-20260714` resolve to the same commit immediately after tagging.
- `git status` may still show ignored/generated data and nested `n2ew_era/.git` state on disk, but the baseline source/report snapshot is committed locally.
