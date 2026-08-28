"""Build the x86_64 executable pair (unstripped and strip-all) for one function at one opt level."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_MISSING_HEADER = re.compile(r"fatal error:\s*([^\s:]+):\s*No such file")


@dataclass
class Built:
    fname: str
    vaddr: int
    unstripped: Path
    stripall: Path


def _delocalize(func_def: str) -> str:
    """Drop leading storage/inline qualifiers from the function definition."""
    return re.sub(r"^\s*(?:static\s+|extern\s+|inline\s+|__inline__\s+|__inline\s+|"
                  r"__forceinline\s+)+", "", func_def.lstrip())


def _drop_include(src: str, header: str) -> str:
    base = re.escape(header.split("/")[-1])
    keep = [ln for ln in src.splitlines()
            if not re.search(r'#\s*include\s*[<"][^>"]*' + base, ln)]
    return "\n".join(keep)


def _vaddr(binary: Path, fname: str):
    r = subprocess.run(["nm", str(binary)], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == fname and parts[1] in ("T", "t"):
            return int(parts[0], 16)
    return None


def build(func_def: str, deps: str, fname: str, opt: str, outdir: Path):
    """Compile and link the unstripped and strip-all executable pair, or None on failure. Unresolvable headers are dropped iteratively and undefined externals are ignored at link."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    func_c, func_o = outdir / "func.c", outdir / "func.o"
    src = (deps or "") + "\n\n" + _delocalize(func_def) + "\n"
    r = None
    for _ in range(40):
        func_c.write_text(src, encoding="utf-8")
        r = subprocess.run(["gcc", "-c", "-g", f"-{opt}", "-w", str(func_c), "-o", str(func_o)],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            break
        m = _MISSING_HEADER.search(r.stderr)
        if not m:
            return None
        pruned = _drop_include(src, m.group(1))
        if pruned == src:
            return None
        src = pruned
    if r is None or r.returncode != 0:
        return None

    main_c = outdir / "main.c"
    main_c.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    exe = outdir / "ghidra_input"
    r = subprocess.run(["gcc", "-g", f"-{opt}", "-w", f"-Wl,-u,{fname}",
                        "-Wl,--unresolved-symbols=ignore-all",
                        str(main_c), str(func_o), "-o", str(exe), "-lm"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    vaddr = _vaddr(exe, fname)
    if vaddr is None:
        return None
    stripall = outdir / "ghidra_input.stripall"
    shutil.copy(exe, stripall)
    if subprocess.run(["strip", "--strip-all", str(stripall)],
                      capture_output=True, timeout=30).returncode != 0:
        return None
    return Built(fname=fname, vaddr=vaddr, unstripped=exe, stripall=stripall)
