from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import hashlib
import json
import os
import random

import numpy as np

DEFAULT_CACHE_DIR = os.getenv("ALNS_CACHE_DIR", "/tmp/.alns_cache")
DEFAULT_MATRIX_UNIT = "sec"  # "sec" or "ms"
PROBLEM_SALT = "alns-later-supernode-v1"


def set_deterministic(seed: int) -> None:
    """Best-effort determinism.

    - Fix random / numpy seeds
    - Limit BLAS threads to reduce tiny nondeterminism
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)


def quantize_matrix(m: np.ndarray, *, unit: str = DEFAULT_MATRIX_UNIT) -> np.ndarray:
    """Quantize to int64 to stabilize tie-breaks caused by float noise."""
    m = np.asarray(m, dtype=float)
    if unit == "sec":
        return np.rint(m).astype(np.int64)
    if unit == "ms":
        return np.rint(m * 1000.0).astype(np.int64)
    raise ValueError("unit must be 'sec' or 'ms'")


def problem_key(
    m_int: np.ndarray,
    *,
    node_ids: List[int],
    start_id: Optional[int],
    end_id: Optional[int],
    opts: Dict[str, Any],
    salt: str = PROBLEM_SALT,
) -> str:
    """Hash key that identifies a 'same problem + same options'."""
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(np.ascontiguousarray(m_int).tobytes())
    h.update(("|node_ids:" + ",".join(map(str, node_ids))).encode("utf-8"))
    h.update(("|start_id:" + str(start_id)).encode("utf-8"))
    h.update(("|end_id:" + str(end_id)).encode("utf-8"))
    h.update(("|opts:" + json.dumps(opts, sort_keys=True, ensure_ascii=False)).encode("utf-8"))
    return h.hexdigest()


def seed_from_key(key_hex: str) -> int:
    """Derive 32-bit seed from sha256 hex string."""
    return int(key_hex[:8], 16)


def cache_path(cache_dir: str, key_hex: str) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"alns_{key_hex[:16]}.json"


def load_cache(cache_dir: str, key_hex: str) -> Optional[Dict[str, Any]]:
    p = cache_path(cache_dir, key_hex)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_dir: str, key_hex: str, data: Dict[str, Any]) -> None:
    p = cache_path(cache_dir, key_hex)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
