from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

from new_era.models_playaround_15.config import ExperimentConfig


def _float_tag(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace(".", "p").replace("-", "m")


def _legacy_float_tag(value: float) -> str:
    # Match legacy folder naming used previously, e.g. "1" instead of "1.0".
    return f"{float(value):.6g}"


def experiment_id(cfg: ExperimentConfig) -> str:
    # Keep experiment directory names short to avoid Windows MAX_PATH issues
    # when saving plots (matplotlib savefig can fail with FileNotFoundError on long paths).
    parts: list[str] = [
        f"lt={cfg.labels.label_type}",
        f"fc={cfg.features.feature_count}",
        f"w={cfg.window.window_size}",
        f"H={cfg.window.horizon}",
        f"thr={_float_tag(cfg.window.threshold_ticks)}",
    ]
    if cfg.labels.label_type in (
        "toxic_bid_binary",
        "toxic_ask_binary",
        "adverse_bid_fill_binary",
        "adverse_ask_fill_binary",
    ):
        parts.append(f"adv={_float_tag(cfg.labels.adverse_threshold_ticks)}")
    if cfg.labels.label_type in ("active_buy_profitable_binary", "active_sell_profitable_binary"):
        parts.extend(
            [
                f"tfee={_float_tag(cfg.labels.taker_fee_twd)}",
                f"exit={_float_tag(cfg.labels.exit_cost_ticks)}",
                f"mar={_float_tag(cfg.labels.margin_ticks)}",
            ]
        )
    return "__".join(parts)


def legacy_experiment_id(cfg: ExperimentConfig) -> str:
    # Older (verbose) naming used in earlier runs.
    parts: list[str] = [
        f"label_type={cfg.labels.label_type}",
        f"feature_count={cfg.features.feature_count}",
        f"window_size={cfg.window.window_size}",
        f"horizon={cfg.window.horizon}",
        f"threshold_ticks={_legacy_float_tag(cfg.window.threshold_ticks)}",
    ]
    if cfg.labels.label_type in (
        "toxic_bid_binary",
        "toxic_ask_binary",
        "adverse_bid_fill_binary",
        "adverse_ask_fill_binary",
    ):
        parts.append(f"adverse_threshold_ticks={_legacy_float_tag(cfg.labels.adverse_threshold_ticks)}")
    if cfg.labels.label_type in ("active_buy_profitable_binary", "active_sell_profitable_binary"):
        parts.extend(
            [
                f"taker_fee_twd={_legacy_float_tag(cfg.labels.taker_fee_twd)}",
                f"exit_cost_ticks={_legacy_float_tag(cfg.labels.exit_cost_ticks)}",
                f"margin_ticks={_legacy_float_tag(cfg.labels.margin_ticks)}",
            ]
        )
    return "__".join(parts)


def _npz_has_sequences(npz_path: Path) -> bool:
    if not npz_path.exists():
        return False
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            return any(str(name).endswith("_X.npy") for name in zf.namelist())
    except Exception:
        return False


def _move_path(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        if not overwrite:
            return
        if dst.is_dir():
            import shutil

            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.rename(dst)
    except OSError:
        # Fallback to copy+delete (best-effort).
        import shutil

        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src)
        else:
            shutil.copy2(src, dst)
            try:
                src.unlink()
            except OSError:
                pass


def _maybe_migrate_legacy_experiment_dir(results_root: Path, cfg: ExperimentConfig) -> None:
    # If a legacy directory exists but the new short-id directory doesn't, rename it in-place.
    # This avoids re-generating huge datasets / re-training, while fixing Windows path-length issues.
    new_dir = results_root / experiment_id(cfg)
    legacy_dir = results_root / legacy_experiment_id(cfg)
    if not legacy_dir.exists():
        return

    if not new_dir.exists():
        try:
            legacy_dir.rename(new_dir)
        except OSError:
            # Best-effort only: if rename fails, we keep the legacy folder as-is.
            # The caller can still re-run the experiment into the short-id folder.
            return
        return

    # Both exist: try to merge critical artifacts from legacy into new, then remove legacy if empty.
    legacy_npz = legacy_dir / "dataset_sequences.npz"
    new_npz = new_dir / "dataset_sequences.npz"
    if _npz_has_sequences(legacy_npz) and (not _npz_has_sequences(new_npz)):
        _move_path(legacy_npz, new_npz, overwrite=True)
        _move_path(legacy_dir / "dataset_manifest.json", new_dir / "dataset_manifest.json", overwrite=True)
        _move_path(legacy_dir / "dataset_debug_stats.json", new_dir / "dataset_debug_stats.json", overwrite=True)

    legacy_weights = legacy_dir / "checkpoints" / "tcn_weights.pth"
    new_weights = new_dir / "checkpoints" / "tcn_weights.pth"
    if legacy_weights.exists() and (not new_weights.exists()):
        _move_path(legacy_dir / "checkpoints", new_dir / "checkpoints", overwrite=False)

    # If legacy still has train/eval/backtest outputs but new doesn't, move them.
    for folder in ("train_outputs", "eval_outputs", "backtest_outputs"):
        src = legacy_dir / folder
        dst = new_dir / folder
        if src.exists() and (not dst.exists()):
            _move_path(src, dst, overwrite=False)

    # Attempt to remove legacy dir if it's now empty.
    try:
        legacy_dir.rmdir()
    except OSError:
        pass


@dataclass(frozen=True)
class ExperimentPaths:
    root_dir: Path
    config_json: Path
    dataset_npz: Path
    checkpoints_dir: Path
    model_weights_pth: Path
    train_dir: Path
    eval_dir: Path
    backtest_dir: Path

    def ensure_dirs(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.backtest_dir.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_results_root() -> Path:
    project_root = get_project_root()
    return project_root / "new_era" / "15_models_playaround" / "015_models_playaround_result"


def build_experiment_paths(cfg: ExperimentConfig) -> ExperimentPaths:
    results_root = get_results_root()
    _maybe_migrate_legacy_experiment_dir(results_root, cfg)
    root_dir = results_root / experiment_id(cfg)
    return ExperimentPaths(
        root_dir=root_dir,
        config_json=root_dir / "experiment_config.json",
        dataset_npz=root_dir / "dataset_sequences.npz",
        checkpoints_dir=root_dir / "checkpoints",
        model_weights_pth=root_dir / "checkpoints" / "tcn_weights.pth",
        train_dir=root_dir / "train_outputs",
        eval_dir=root_dir / "eval_outputs",
        backtest_dir=root_dir / "backtest_outputs",
    )
