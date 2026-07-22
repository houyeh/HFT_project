from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _try_import_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        return str(version) if version is not None else None
    except Exception:
        return None


def _torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        devices: list[dict[str, Any]] = []
        if cuda_available:
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": int(i),
                        "name": str(props.name),
                        "total_memory_bytes": int(props.total_memory),
                    }
                )

        return {
            "version": getattr(torch, "__version__", None),
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": int(torch.backends.cudnn.version() or 0) if cuda_available else 0,
            "device_count": device_count,
            "devices": devices,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def write_environment_snapshot(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": _try_import_version("numpy"),
            "pandas": _try_import_version("pandas"),
            "sklearn": _try_import_version("sklearn"),
            "matplotlib": _try_import_version("matplotlib"),
        },
        "torch": _torch_info(),
    }

    out_json = out_dir / "environment_snapshot.json"
    out_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        (out_dir / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as exc:
        (out_dir / "pip_freeze_error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")

    return out_json

