from __future__ import annotations

"""
Non-breaking runtime patches for SadTalker (used only when eye_contact == "locked_plus").

Design goals:
- Never crash if SadTalker internal APIs differ.
- Apply small clamps that make the head look "locked on camera".
- Keep behavior unchanged for other modes.
"""

from contextlib import suppress
from typing import Any

import os


def _clamp_array(x: Any, lo: float, hi: float) -> Any:
    with suppress(Exception):
        import numpy as np  # type: ignore
        arr = np.asarray(x, dtype=float)
        arr = np.clip(arr, lo, hi)
        # preserve original type when possible
        with suppress(Exception):
            return arr.astype(getattr(x, "dtype", arr.dtype))
        return arr
    return x


def _clamp_pose_dict(d: Any, deg: float = 0.5) -> None:
    """Clamp yaw/pitch/roll sequences inside SadTalker facerender data dict."""
    if not isinstance(d, dict):
        return
    lo, hi = -float(deg), float(deg)
    for k in ("yaw_c_seq", "pitch_c_seq", "roll_c_seq"):
        if k in d:
            d[k] = _clamp_array(d[k], lo, hi)


def apply_patches(mode: str) -> None:
    """Apply patches for a given mode. Safe to call multiple times."""
    if (mode or "").strip().lower() != "locked_plus":
        return

    # Mark active (useful for downstream hooks if present)
    os.environ["WR_SADTALKER_PATCH_MODE"] = "locked_plus"

    # Patch 1: clamp facerender pose sequences (most common in SadTalker)
    with suppress(Exception):
        import importlib

        mod = importlib.import_module("src.facerender.animate")
        fn = getattr(mod, "get_facerender_data", None)
        if callable(fn):
            # Avoid double-wrapping
            if getattr(fn, "_wr_locked_plus_patched", False):
                return

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                out = fn(*args, **kwargs)
                _clamp_pose_dict(out, deg=0.5)
                return out

            setattr(wrapped, "_wr_locked_plus_patched", True)
            setattr(mod, "get_facerender_data", wrapped)

    # Patch 2: placeholder for fork-specific modules (kept additive)
    with suppress(Exception):
        import importlib

        _ = importlib.import_module("src.facerender.sync_batchnorm")
