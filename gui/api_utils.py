# hand_bone_tracker/gui/api_utils.py
"""
Pure helper functions shared by the FastAPI backend.

Kept free of heavy imports (torch, models) so they can be unit-tested quickly.
"""
import re
from pathlib import Path

_FILENAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")
MAX_STEM_LENGTH = 64


def sanitize_filename_stem(value: str, fallback: str = "sample") -> str:
    """Turn arbitrary user text into a safe file-name stem.

    - Lowercases and replaces every character outside [a-z0-9_-] with '_'
      (this also neutralizes path traversal like '../..' or '..\\..').
    - Strips leading/trailing separators and dots (no hidden files).
    - Caps length and falls back when the result is empty.
    """
    cleaned = _FILENAME_UNSAFE_RE.sub("_", str(value)).strip("._-")
    cleaned = cleaned[:MAX_STEM_LENGTH].strip("._-").lower()
    return cleaned or fallback


def resolve_safe_checkpoint_path(raw_path: str, base_dir: Path):
    """Resolve `raw_path` and enforce it stays inside `base_dir`.

    Returns an existing .ckpt Path, or None when the path is missing,
    has the wrong extension, or escapes the checkpoints directory
    (path-traversal / arbitrary-file-load guard).
    """
    if not raw_path or not isinstance(raw_path, str):
        return None
    try:
        candidate = Path(raw_path).resolve(strict=True)
        base = Path(base_dir).resolve(strict=True)
    except OSError:
        return None
    if candidate.suffix.lower() != ".ckpt":
        return None
    if base not in candidate.parents:
        return None
    return candidate
