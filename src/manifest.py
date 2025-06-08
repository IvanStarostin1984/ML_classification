from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import sys
from typing import Sequence

__all__ = ["write_manifest"]


def write_manifest(
    files: Sequence[str], out: Path = Path("artefacts/SHA256_manifest.txt")
) -> Path:
    """Write SHA-256 checksums and environment info to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for fp in files:
        h = hashlib.sha256(Path(fp).read_bytes()).hexdigest()
        lines.append(f"{h}  {fp}")
    pyver = (
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    env = {"python": pyver, "platform": platform.platform()}
    lines.append(json.dumps(env, sort_keys=True))
    out.write_text("\n".join(lines))
    return out
