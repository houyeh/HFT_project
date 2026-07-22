from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from new_era.models_playaround_15.config import ExperimentConfig
from new_era.models_playaround_15.paths import ExperimentPaths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def save_experiment_config(paths: ExperimentPaths, cfg: ExperimentConfig) -> None:
    write_json(paths.config_json, cfg.to_dict())

