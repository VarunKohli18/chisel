"""Offline re-executability scoring against the held-out test suite. A candidate re-executes if
it compiles and matches the original on every suite pair. This is the only module that reads the
dataset's io pairs. From a function's iospec we generate a C driver that initializes the inputs
per pair, calls func0, and serializes the live-out args and return value."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from decomp.dataset.records import func0_iospec, record_key, rename_to_func0
from decomp.signature import externalize_func0

_ASLR_OFF = ["setarch", "-R"] if shutil.which("setarch") else []

_SCALAR_CTYPE = {"int8": "int8_t", "int16": "int16_t", "int32": "int32_t", "int64": "int64_t",
                 "uint8": "uint8_t", "uint16": "uint16_t", "uint32": "uint32_t",
                 "uint64": "uint64_t", "float32": "float", "float64": "double", "bool": "int"}
_FLOAT = {"float32", "float64"}
_SIGNED = {"int8", "int16", "int32", "int64", "bool"}
_ARRAY_RE = re.compile(r"array\((.+)#(\d+)\)")
_POINTER_RE = re.compile(r"pointer\((.+)\)")
# float comparison tolerance, relative with a small absolute floor
_FLOAT_RTOL = 1e-3
_FLOAT_ATOL = 1e-6


class Unsupported(Exception):
    """The function uses a type the driver cannot synthesize."""


@dataclass
class Divergence:
    pair_index: int
    var: str
    expected: object
    actual: object


@dataclass
class ScoreResult:
    key: str
    compiled: bool
    reexec: bool
    n_pairs: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    error: str = ""
    divergences: list = field(default_factory=list)


def _kind(t: str, classmap: dict):
    classmap = classmap or {}
    if t == "string":
        return ("string",)
    m = _ARRAY_RE.fullmatch(t)
    if m:
        elem, n = m.group(1).strip(), int(m.group(2))
        if elem in _SCALAR_CTYPE:
            return ("array", elem, n)
        if elem in classmap:
            return ("array_struct", elem, n)
        raise Unsupported(f"array of unsupported elem: {t}")
    mp = _POINTER_RE.fullmatch(t)
    if mp:
        return ("pointer", mp.group(1).strip())
    if t in _SCALAR_CTYPE:
        return ("scalar", _SCALAR_CTYPE[t], t)
    if t in classmap:
        return ("struct", t)
    raise Unsupported(f"unsupported type: {t}")


def _lit(token: str, value) -> str:
    if token in _FLOAT:
        return repr(float(value))
    if token == "bool":
        return "1" if value else "0"
    return str(int(value))


def _struct_init(struct_t: str, value, classmap: dict) -> str:
    entry = classmap.get(struct_t) or {}
    ftypes = entry.get("typemap", {})
    vd = value if isinstance(value, dict) else {}
    parts = []
    for f in entry.get("symbols", []):
        ft = ftypes.get(f)
        if ft is None:
            raise Unsupported(f"{struct_t}.{f} has no type")
        parts.append(f".{f} = {_value_init(ft, vd.get(f), classmap)}")
    return "{" + ", ".join(parts) + "}" if parts else "{0}"


def _value_init(t: str, value, classmap: dict) -> str:
    k = _kind(t, classmap)
    if k[0] == "scalar":
        return _lit(k[2], value) if value is not None else "0"
    if k[0] == "array":
        _, elem, n = k
        vals = (value if isinstance(value, list) else [])[:n]
        return "{" + ",".join(_lit(elem, v) for v in vals) + "}"
    if k[0] == "struct":
        return _struct_init(k[1], value, classmap)
    if k[0] == "array_struct":
        _, st, n = k
        vals = (value if isinstance(value, list) else [])[:n]
        return "{" + ",".join(_struct_init(st, v, classmap) for v in vals) + "}"
    if k[0] == "string":
        enc = (value if isinstance(value, str) else "").encode("utf-8", "surrogatepass")
        return "{" + ",".join(str(b) for b in enc) + (",0}" if enc else "0}")
    if not value:
        return "0"
    raise Unsupported(f"non-null pointer field of type {t}")


def _decl_init(var: str, t: str, value, lines: list, classmap: dict) -> str:
    k = _kind(t, classmap)
    cvar = f"a_{var}"
    if k[0] == "scalar":
        _, ctype, token = k
        lines.append(f"  {ctype} {cvar} = {_lit(token, value) if value is not None else '0'};")
        return cvar
    if k[0] == "string":
        enc = (value if isinstance(value, str) else "").encode("utf-8", "surrogatepass")
        size = max(256, len(enc) * 4 + 64)
        body = ",".join(str(b) for b in enc) + ("," if enc else "")
        lines.append(f"  char {cvar}[{size}] = {{{body}0}};")
        return cvar
    if k[0] == "array":
        _, elem, n = k
        vals = value if isinstance(value, list) else []
        size = max(n, len(vals), 1)
        lines.append(f"  {_SCALAR_CTYPE[elem]} {cvar}[{size}] = {{{','.join(_lit(elem, v) for v in vals)}}};")
        return cvar
    if k[0] == "struct":
        lines.append(f"  {k[1]} {cvar} = {_struct_init(k[1], value, classmap)};")
        return cvar
    if k[0] == "array_struct":
        _, st, n = k
        vals = (value if isinstance(value, list) else [])[:n]
        lines.append(f"  {st} {cvar}[{n}] = {{{','.join(_struct_init(st, v, classmap) for v in vals)}}};")
        return cvar
    inner = k[1]
    ik = _kind(inner, classmap)
    obj = f"{cvar}_obj"
    if ik[0] == "struct":
        lines.append(f"  {inner} {obj} = {_struct_init(inner, value, classmap)};")
        lines.append(f"  {inner} *{cvar} = &{obj};")
        return cvar
    if ik[0] == "scalar":
        lines.append(f"  {ik[1]} {obj} = {_lit(ik[2], value) if value is not None else '0'};")
        lines.append(f"  {ik[1]} *{cvar} = &{obj};")
        return cvar
    raise Unsupported(f"pointer to {inner}")


def _serialize(pidx: int, var: str, t: str, expr: str, lines: list, classmap: dict) -> None:
    k = _kind(t, classmap)
    if k[0] == "struct":
        entry = classmap.get(k[1]) or {}
        for f in entry.get("symbols", []):
            _serialize(pidx, f"{var}.{f}", entry["typemap"][f], f"({expr}).{f}", lines, classmap)
        return
    if k[0] == "array_struct":
        _, st, n = k
        for i in range(n):
            _serialize(pidx, f"{var}#{i}", st, f"({expr})[{i}]", lines, classmap)
        return
    if k[0] == "pointer":
        _serialize(pidx, var, k[1], f"(*({expr}))", lines, classmap)
        return
    if k[0] == "scalar":
        _, _ctype, token = k
        if token in _FLOAT:
            lines.append(f'  printf("P {pidx} {var} f %.17g\\n", (double)({expr}));')
        elif token in _SIGNED:
            lines.append(f'  printf("P {pidx} {var} i %lld\\n", (long long)({expr}));')
        else:
            lines.append(f'  printf("P {pidx} {var} u %llu\\n", (unsigned long long)({expr}));')
    elif k[0] == "string":
        lines.append(f'  printf("P {pidx} {var} s ");')
        lines.append(f'  for (size_t _k=0; {expr}[_k] && _k<65536; _k++) printf("%02x", (unsigned char){expr}[_k]);')
        lines.append('  printf("\\n");')
    else:
        _, elem, n = k
        code = "af" if elem in _FLOAT else ("ai" if elem in _SIGNED else "au")
        fmt = "%.17g" if elem in _FLOAT else ("%lld" if elem in _SIGNED else "%llu")
        cast = "(double)" if elem in _FLOAT else ("(long long)" if elem in _SIGNED else "(unsigned long long)")
        lines.append(f'  printf("P {pidx} {var} {code} {n}");')
        lines.append(f'  for (int _k=0; _k<{n}; _k++) printf(" {fmt}", {cast}{expr}[_k]);')
        lines.append('  printf("\\n");')


def build_driver(func_source: str, deps: str, iospec: dict, io_pairs: list) -> str:
    """Generate the full C driver. Raises Unsupported on a type the generator cannot handle."""
    func_source = externalize_func0(func_source)
    typemap = iospec.get("typemap", {})
    classmap = iospec.get("classmap", {})
    funargs = iospec.get("funargs", [])
    liveout = list(iospec.get("liveout", []))
    retnames = iospec.get("returnvarname", []) or []
    fname = iospec.get("funname") or iospec.get("fname")
    if not fname or not funargs:
        raise Unsupported("missing function name or args")
    for v in funargs + liveout + retnames:
        if v in typemap:
            _kind(typemap[v], classmap)
    for a in funargs:
        t = typemap.get(a)
        if a in liveout and t and _kind(t, classmap)[0] == "struct":
            raise Unsupported(f"by-value struct '{a}' in live-out, mutation not observable")

    out = ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>",
           "#include <stdint.h>", "#include <math.h>", "", deps or "", "",
           func_source, "", "int main(void) {"]
    for pidx, pair in enumerate(io_pairs):
        inp = pair.get("input") or {}
        out.append(f"  {{ /* pair {pidx} */")
        argexprs = {}
        for a in funargs:
            t = typemap.get(a)
            if t is None:
                raise Unsupported(f"arg {a} has no type")
            argexprs[a] = _decl_init(a, t, inp.get(a), out, classmap)
        ret_t = typemap.get(retnames[0]) if retnames else None
        call = f"{fname}(" + ", ".join(argexprs[a] for a in funargs) + ")"
        if ret_t is not None:
            rk = _kind(ret_t, classmap)
            if rk[0] == "scalar":
                rctype = rk[1]
            elif rk[0] == "string":
                rctype = "char*"
            elif rk[0] == "array":
                rctype = f"{_SCALAR_CTYPE[rk[1]]}*"
            else:
                raise Unsupported(f"unsupported return type {ret_t}")
            out.append(f"    {rctype} _ret = {call};")
        else:
            out.append(f"    {call};")
        for v in liveout:
            if v in argexprs:
                _serialize(pidx, v, typemap[v], argexprs[v], out, classmap)
        if ret_t is not None:
            _serialize(pidx, retnames[0], ret_t, "_ret", out, classmap)
        out.append("  }")
    out += ["  return 0;", "}"]
    return "\n".join(out)


def _parse(text: str) -> dict:
    res = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "P":
            continue
        pidx, var, code, rest = int(parts[1]), parts[2], parts[3], parts[4:]
        try:
            if code in ("i", "u"):
                val = int(rest[0])
            elif code == "f":
                val = float(rest[0])
            elif code in ("ai", "au"):
                val = [int(x) for x in rest[1:]]
            elif code == "af":
                val = [float(x) for x in rest[1:]]
            elif code == "s":
                val = rest[0] if rest else ""
            else:
                continue
        except (ValueError, IndexError):
            continue
        res[(pidx, var)] = (code, val)
    return res


def _float_eq(x, y) -> bool:
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    if math.isinf(x) or math.isinf(y):
        return x == y
    return x == y or abs(x - y) <= _FLOAT_ATOL + _FLOAT_RTOL * max(abs(x), abs(y))


def _eq(a, b) -> bool:
    (ca, va), (cb, vb) = a, b
    if ca != cb:
        return False
    if ca == "f":
        return _float_eq(va, vb)
    if ca == "af":
        return len(va) == len(vb) and all(_float_eq(x, y) for x, y in zip(va, vb))
    return va == vb


def _run_driver(driver_c: str, timeout: float = 10.0):
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / "drv.c", Path(d) / "drv"
        src.write_text(driver_c, encoding="utf-8")
        c = subprocess.run(["gcc", "-O0", "-w", str(src), "-o", str(exe), "-lm"],
                           capture_output=True, text=True, errors="replace", timeout=60)
        if c.returncode != 0:
            return False, "", c.stderr[-2000:]
        try:
            r = subprocess.run(_ASLR_OFF + [str(exe)], capture_output=True, text=True,
                               errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "", "driver timeout"
        return True, r.stdout, r.stderr[-500:]


def reference_outputs(func_def: str, deps: str, iospec: dict, io_pairs: list):
    """Run the original to get ground truth outputs, or None if it cannot build or run."""
    try:
        driver = build_driver(func_def, deps, iospec, io_pairs)
    except Unsupported:
        return None
    ok, out, _ = _run_driver(driver)
    return _parse(out) if ok else None


def score(record: dict, candidate_src: str, io_pairs: list) -> ScoreResult:
    """Score one candidate func0 against the suite. reexec is True only when it compiled and
    matched the original on every pair."""
    key = record_key(record)
    spec = func0_iospec(record["iospec"])
    deps = record.get("deps", "") or ""
    ref = reference_outputs(rename_to_func0(record["func_def"], record["fname"]),
                            deps, spec, io_pairs)
    if ref is None:
        return ScoreResult(key, False, False, error="reference build/run failed")
    n_pairs = len(set(k[0] for k in ref))
    if n_pairs == 0:
        return ScoreResult(key, False, False, error="empty reference, no observable output")
    try:
        driver = build_driver(candidate_src, deps, spec, io_pairs)
    except Unsupported as e:
        return ScoreResult(key, False, False, n_pairs=n_pairs, error=str(e))
    ok, out, err = _run_driver(driver)
    if not ok:
        return ScoreResult(key, False, False, n_pairs=n_pairs, error=err)
    got = _parse(out)
    divs = [Divergence(p, v, exp[1], (got.get((p, v)) or (None, None))[1])
            for (p, v), exp in ref.items()
            if (p, v) not in got or not _eq(exp, got[(p, v)])]
    div_pairs = len(set(d.pair_index for d in divs))
    passed = max(0, n_pairs - div_pairs)
    return ScoreResult(key, True, not divs, n_pairs=n_pairs, passed=passed,
                       pass_rate=passed / n_pairs, divergences=divs)
