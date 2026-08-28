#!/usr/bin/env python3
"""Run the Agent4Decompile harness on our dataset with the gemma model, test-suite-free.

Feeds the Ghidra pseudo-C as initial code, runs gemma through A4D's refiner at
constraint_level=2, and scores the returned candidate on the held-out suite.
"""
import argparse, json, glob, os, re, sys, time
from pathlib import Path

A4D = "../agent4decompile/src"
sys.path.insert(0, A4D)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refinement.refiner import MCGDRefiner, strip_main_from_code  # noqa: E402
from decomp import score                            # noqa: E402
from decomp.dataset import records_path             # noqa: E402
from decomp.dataset.records import rename_to_func0  # noqa: E402

_C_KW = {"if", "for", "while", "switch", "do", "else", "return", "sizeof"}


def _to_func0(src: str, want_nargs=None) -> str:
    """Rename the target function to func0, arg-count matched when there are helpers."""
    defs = re.findall(r"\b([A-Za-z_]\w*)\s*\(([^;{)]*)\)\s*\{", src)
    cands = [(n, a) for n, a in defs if n not in _C_KW and n != "func0"]
    if not cands:
        return src
    target = cands[0][0]
    if want_nargs is not None and len(cands) > 1:
        for n, a in cands:
            na = 0 if a.strip() in ("", "void") else a.count(",") + 1
            if na == want_nargs:
                target = n
                break
    return rename_to_func0(src, target)

CACHE = Path(__file__).resolve().parent.parent / "data_cache" / "exebench_hard"


def load_records():
    return {f"{r['kind']}__{r['fname']}": r
            for r in (json.loads(l) for l in records_path("exebench_hard").read_text().splitlines() if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/runs_new/a4d_gemma")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gemma4:31b"))
    a = ap.parse_args()

    records = load_records()
    cells = sorted(p for p in CACHE.glob("*__*__*.json"))
    cells = [c for i, c in enumerate(cells) if i % a.nshards == a.shard]
    if a.limit:
        cells = cells[: a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    refiner = MCGDRefiner(llm_provider="ollama", model=a.model,
                          max_iterations=a.iters, constraint_level=2, architecture="x86_64")

    def norm_and_score(raw, want, io, rec):
        """Strip harness main, rename target to func0, then score."""
        if not raw or rec is None or not io:
            return False, False, 0.0
        cand = _to_func0(strip_main_from_code(raw), want)
        try:
            sc = score.score(rec, cand, io)
            return bool(sc.compiled), bool(sc.reexec), round(sc.pass_rate, 4)
        except Exception:
            return False, False, 0.0

    rows, t0 = [], time.time()
    for i, cp in enumerate(cells):
        d = json.loads(cp.read_text())
        key, opt, variant, pc = d["key"], d["opt"], d["variant"], d.get("pseudo_c", "")
        rec = records.get(key)
        io = (rec or {}).get("io_pairs") or []
        want = len((rec or {}).get("iospec", {}).get("funargs", []) or []) or None
        t = time.time()
        cand, iters, note, hist = "", 0, "", []
        try:
            res = refiner.refine(initial_code=pc, binary_name=key, decompiler="ghidra")
            cand = res.refined_code or ""
            iters = res.iterations
            hist = res.iteration_history or []
            if cand:
                cand = _to_func0(strip_main_from_code(cand), want)
        except Exception as e:
            note = f"a4d error: {str(e)[:120]}"
        compiled = reexec = False
        pass_rate = 0.0
        if cand and rec is not None and io:
            try:
                sc = score.score(rec, cand, io)
                compiled, reexec, pass_rate = bool(sc.compiled), bool(sc.reexec), round(sc.pass_rate, 4)
            except Exception as e:
                note = (note + f" | score error: {str(e)[:80]}").strip(" |")
        # first generation = first LLM-refined candidate, fall back to raw / final
        i1_raw = (hist[1]["code"] if len(hist) > 1 else (hist[0]["code"] if hist else cand))
        i1c, i1r, _ = norm_and_score(i1_raw, want, io, rec)
        rows.append({"key": key, "opt": opt, "variant": variant, "arm": "a4d_gemma",
                     "compiled": compiled, "reexec": reexec, "accepted": compiled,
                     "iter1_compiled": i1c, "iter1_reexec": i1r,
                     "iters": iters, "pass_rate": pass_rate, "candidate": cand, "note": note,
                     "seconds": round(time.time() - t, 1)})
        rate = (i + 1) / max(1e-9, (time.time() - t0)) * 60
        print(f"[{i+1}/{len(cells)}] {key} {opt}/{variant} compiled={compiled} reexec={reexec} "
              f"iters={iters} {rows[-1]['seconds']}s | {rate:.1f}/min", flush=True)

    fn = out / f"shard{a.shard}.json"
    tmp = str(fn) + ".tmp"; Path(tmp).write_text(json.dumps(rows, indent=1)); Path(tmp).replace(fn)
    print(f"[a4d] wrote {len(rows)} rows -> {fn}", flush=True)


if __name__ == "__main__":
    main()
