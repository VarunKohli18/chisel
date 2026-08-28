"""Record helpers. A record is one ExeBench function with func_def, deps, signature, io_pairs, and iospec."""

from __future__ import annotations

import re


def record_key(record: dict) -> str:
    return f"{record['kind']}__{record['fname']}"


def rename_to_func0(func_def: str, fname: str) -> str:
    """Rename the target function to func0, whole-word only, leaving string literals untouched."""
    pat = re.compile(r'("(?:[^"\\]|\\.)*")' + r"|\b" + re.escape(fname) + r"\b")
    return pat.sub(lambda m: m.group(1) if m.group(1) else "func0", func_def)


def func0_iospec(iospec: dict) -> dict:
    """iospec copy with funname set to func0."""
    spec = dict(iospec)
    spec["funname"] = "func0"
    return spec
