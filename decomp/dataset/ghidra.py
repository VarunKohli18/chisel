"""Run headless Ghidra on a binary and select the target function's pseudo-C by entry address."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_IMAGE = "ghidra_extractor"
# candidate Ghidra image bases
_BASES = (0x100000, 0x0, 0x400000, 0x10000)
_FUNC_RE = re.compile(r"/\* Function: (?P<name>.*?) \*/\s*\n/\* Entry: (?P<entry>0x[0-9a-fA-F]+)")


def run_ghidra(binary: Path) -> dict | None:
    """Run the container and return its JSON ({pseudo_c, cfg}), or None on failure."""
    d = binary.parent.resolve()
    try:
        r = subprocess.run(["docker", "run", "--rm", "-v", f"{d}:/binaries", _IMAGE,
                            f"/binaries/{binary.name}"],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def select_function(pseudo_c: str, vaddr: int) -> str | None:
    """The target function's C body (header comments stripped), found by entry address."""
    blocks = []
    for part in re.split(r"(?=/\* Function: )", pseudo_c):
        m = _FUNC_RE.search(part)
        if m:
            blocks.append((int(m.group("entry"), 16), part.strip()))
    by_entry = dict(blocks)
    for base in _BASES:
        body = by_entry.get(base + vaddr)
        if body is not None:
            return "\n".join(l for l in body.splitlines() if not l.startswith("/*")).strip()
    return None
