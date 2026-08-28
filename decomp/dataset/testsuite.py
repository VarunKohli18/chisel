"""Generate the re-executability suite. Synthesize N random typed inputs from the iospec, run the original, and keep the inputs that execute cleanly. Falls back to the native io_pairs when generation fails."""

from __future__ import annotations

import random
from collections import Counter

from decomp.dataset.records import func0_iospec, rename_to_func0
from decomp.score import (_ARRAY_RE, _FLOAT, _SIGNED, Unsupported, _kind, _parse, _run_driver,
                          build_driver)


def _scalar(token: str, rng, span):
    if token in _FLOAT:
        lo, hi = span if span else (-1000.0, 1000.0)
        if rng.random() < 0.15:
            return rng.choice([0.0, 1.0, -1.0, 0.5])
        return rng.uniform(float(lo), float(hi))
    if token == "bool":
        return rng.randint(0, 1)
    bits = int("".join(c for c in token if c.isdigit()) or "32")
    if token in _SIGNED:
        cap = 2 ** (bits - 1)
        dlo, dhi = -min(cap, 10000), min(cap - 1, 10000)
    else:
        dlo, dhi = 0, min(2 ** bits - 1, 10000)
    lo, hi = span if span else (dlo, dhi)
    lo, hi = max(int(lo), dlo), min(int(hi), dhi)
    if lo > hi:
        lo, hi = dlo, dhi
    return rng.randint(lo, hi)


def _struct(struct_t, classmap, rng):
    entry = classmap.get(struct_t) or {}
    ftypes = entry.get("typemap", {})
    out = {}
    for f in entry.get("symbols", []):
        ft = ftypes.get(f)
        if ft is None:
            raise Unsupported(f"{struct_t}.{f} has no type")
        out[f] = _value(ft, classmap, rng)
    return out


def _value(token, classmap, rng, span=None):
    k = _kind(token, classmap)
    if k[0] == "scalar":
        return _scalar(k[2], rng, span)
    if k[0] == "string":
        n = rng.randint(0, 24)
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789 ") for _ in range(n))
    if k[0] == "array":
        _, elem, n = k
        return [_scalar(elem, rng, span) for _ in range(n)]
    if k[0] == "struct":
        return _struct(k[1], classmap, rng)
    if k[0] == "array_struct":
        _, st, n = k
        return [_struct(st, classmap, rng) for _ in range(n)]
    if k[0] == "pointer":
        ik = _kind(k[1], classmap)
        if ik[0] == "struct":
            return _struct(k[1], classmap, rng)
        if ik[0] == "scalar":
            return _scalar(ik[2], rng, span)
        raise Unsupported(f"pointer to {k[1]}")
    raise Unsupported(f"cannot generate {token}")


def _gen_inputs(iospec, n, rng, bound_scalars):
    typemap = iospec.get("typemap", {})
    classmap = iospec.get("classmap", {})
    funargs = iospec.get("funargs", [])
    ranges = iospec.get("range", {}) or {}
    if not funargs:
        raise Unsupported("no funargs")
    for a in funargs:
        if a not in typemap:
            raise Unsupported(f"arg {a} has no type")
        _kind(typemap[a], classmap)
    arr_lens = [int(m.group(2)) for a in funargs
                if (m := _ARRAY_RE.fullmatch(typemap[a]))]
    cap = min(arr_lens) - 1 if arr_lens else None
    pairs = []
    for _ in range(n):
        inp = {}
        for a in funargs:
            t, span = typemap[a], ranges.get(a)
            if (bound_scalars and span is None and cap is not None and cap >= 0
                    and _kind(t, classmap)[0] == "scalar" and t not in _FLOAT and t != "bool"):
                span = (0, cap)
            inp[a] = _value(t, classmap, rng, span)
        pairs.append({"input": inp})
    return pairs


def _complete(parsed: dict) -> set:
    """Pair indices whose serialized var-set matches the modal one."""
    by_pidx = {}
    for (pidx, var) in parsed:
        by_pidx.setdefault(pidx, set()).add(var)
    if not by_pidx:
        return set()
    target = Counter(frozenset(v) for v in by_pidx.values()).most_common(1)[0][0]
    return {p for p, v in by_pidx.items() if frozenset(v) == target}


def _attempt(func0_src, deps, iospec, n, rng, bound_scalars):
    try:
        pairs = _gen_inputs(iospec, n, rng, bound_scalars)
        driver = build_driver(func0_src, deps, iospec, pairs)
    except Unsupported as e:
        return None, f"unsupported: {e}"
    driver = driver.replace("int main(void) {",
                            "int main(void) {\n  setvbuf(stdout,0,_IONBF,0);", 1)
    ok, out, err = _run_driver(driver, timeout=30.0)
    if not ok:
        return None, f"driver build/run failed: {err[:120]}"
    keep = _complete(_parse(out))
    if not keep:
        return None, "no complete pairs"
    return [pairs[p] for p in sorted(keep)], {"requested": n, "kept": len(keep)}


def build_suite(record: dict, n: int, rng):
    """Return (kept_pairs, stats) for one function, or (None, reason). Natural ranges first, then a second pass bounding count and index scalars to the array length if every input fails."""
    iospec = func0_iospec(record["iospec"])
    func0_src = rename_to_func0(record["func_def"], record["fname"])
    deps = record.get("deps", "")
    kept, stats = _attempt(func0_src, deps, iospec, n, rng, bound_scalars=False)
    if kept is None and (stats == "no complete pairs" or "timeout" in str(stats)):
        kept, stats = _attempt(func0_src, deps, iospec, n, rng, bound_scalars=True)
    return kept, stats
