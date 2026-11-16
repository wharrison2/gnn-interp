"""JSON-lines files that survive interruption (e.g. spot preemption mid-write).

``read`` drops a truncated last line (no trailing newline) from the file, so the next ``append`` starts on a
clean line. ``save_atomic`` writes via a temporary file, so an interrupted save keeps the previous version.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def read(path: Path) -> list[dict]:
    """Rows of ``path`` ([] if missing). A partial last line is removed from the file (with a warning)."""
    if not path.exists():
        return []
    text = path.read_text()
    if text and not text.endswith("\n"):
        keep = text[:text.rfind("\n") + 1]
        print(f"warning: dropping truncated last line of {path}: {text[len(keep):][:80]!r}", flush=True)
        path.write_text(keep)
        text = keep
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append(path: Path, *rows: dict):
    """Append ``rows`` in one write (a problem's rows land together or, if interrupted, as one partial line)."""
    with path.open("a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def save_atomic(obj, path: Path):
    """``torch.save`` to a temporary file, then rename over ``path``."""
    import torch

    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
