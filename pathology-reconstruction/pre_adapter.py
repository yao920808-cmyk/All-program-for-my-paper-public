from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

from config import DEFAULT_PATHS


def _candidate_search_paths(candidate: Path) -> list[Path]:
    """Return robust search locations for the base experiment module.

    The earlier version only checked the current working directory. That fails in
    Jupyter when the notebook is opened from a different directory. This version
    checks:
    1) the path exactly as provided,
    2) the modular framework directory,
    3) the current working directory.
    """
    module_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidate = Path(candidate)

    paths = []
    if candidate.is_absolute():
        paths.append(candidate)
    else:
        paths.append(module_dir / candidate)
        paths.append(cwd / candidate)
        paths.append(candidate)

    # Preserve order while removing duplicates.
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def load_base_module(candidates: Iterable[Path] | None = None) -> ModuleType:
    candidate_paths = list(candidates or DEFAULT_PATHS.base_module_candidates)
    tried: list[str] = []

    for candidate in candidate_paths:
        for path in _candidate_search_paths(Path(candidate)):
            tried.append(str(path))
            if not path.exists():
                continue

            safe_stem = path.stem.replace("(", "_").replace(")", "_").replace("-", "_").replace(" ", "_")
            module_name = f"pathology_reconstruction_experiment_dynamic_{safe_stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not find a pathology_reconstruction_experiment*.py base module. "
        f"Tried: {tried}"
    )


pre = load_base_module()
