"""Minimal smoke test that the CME training entrypoint exists."""

from __future__ import annotations

from pathlib import Path


def test_train_cme_baseline_script_exists():
    script_path = Path("scripts/02_train_eval/train_cme_baseline.py")
    assert script_path.exists()
