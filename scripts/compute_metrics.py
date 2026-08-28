#!/usr/bin/env python3
"""Compute every numerical result behind the paper from the run JSONs.

Usage: compute_metrics.py [runs_dir] [--rescore]   (default runs_dir: results/runs_latest)

Writes into <runs_dir>:
  _metrics.json        per-arm metrics for overall/stripall/unstripped + per-opt retain_best
  _llm_passrate.txt    llm one-shot Pass: suite pass-rate of the compiler arm's iter-1 candidates
  _iter_passrate.json  per-iteration mean pass-rate curves (observe, memory)

The last two execute candidates against the held-out suites and are slow, so they are reused
when already present and recomputed only when missing or with --rescore.
"""
import glob
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_args = [a for a in sys.argv[1:] if a != "--rescore"]
RESCORE = "--rescore" in sys.argv[1:]
RUNS = Path(_args[0] if _args else "results/runs_latest")
ARMS = ["llm", "compiler", "fuzzer", "observe", "memory", "retain_best"]
BASELINES = ["llm4decompile", "a4d_gemma"]
BENCH = "exebench_hard"

# ---- suite rescoring (process pool; executes candidates against the held-out suites) ----

_REC = None
_SUI = {}


def _init(bench):
    global _REC, _BENCH
    from decomp.dataset import records_path
    _REC = {f"{r['kind']}__{r['fname']}": r for r in
            (json.loads(l) for l in records_path(bench).read_text().splitlines() if l.strip())}
    _BENCH = bench


def _suite(k):
    if k not in _SUI:
        from decomp.dataset import suite_path
        p = suite_path(_BENCH, k)
        _SUI[k] = json.loads(p.read_text())["io_pairs"] if p.exists() else None
    return _SUI[k]


def _work(task):
    h, key, cand = task
    from decomp import score
    rec, su = _REC.get(key), _suite(key)
    if rec is None or not su or not cand:
        return h, None
    try:
        return h, round(score.score(rec, cand, su).pass_rate, 4)
    except Exception:
        return h, None


def _score_all(tasks):
    res = {}
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(BENCH,)) as ex:
        for h, pr in ex.map(_work, list(tasks.values())):
            res[h] = pr
    return res


def load(arm):
    rows = []
    src = "compiler" if arm == "llm" else arm     # llm derived from iter-1
    for f in sorted(glob.glob(f"{RUNS}/{src}/*.json")):
        rows += json.loads(Path(f).read_text())
    return rows


def llm_passrate():
    """The llm one-shot Pass: mean suite pass-rate of the compiler arm's iter-1 candidates
    (non-compiling = 0). Writes _llm_passrate.txt."""
    rows = load("compiler")
    tasks, refs = {}, []
    for r in rows:
        it = (r.get("per_iter") or [{}])[0]
        c = it.get("candidate")
        if it.get("compiled") and c:
            h = (r["key"], hashlib.sha1(c.encode()).hexdigest())
            tasks.setdefault(h, (h, r["key"], c))
            refs.append(h)
        else:
            refs.append(None)
    print(f"[llm Pass] scoring {len(tasks)} unique iter-1 candidates over {len(rows)} rows ...", flush=True)
    res = _score_all(tasks)
    prs = [(res.get(h) or 0.0) if h else 0.0 for h in refs]
    mean = sum(prs) / (len(prs) or 1)
    (RUNS / "_llm_passrate.txt").write_text(f"{mean:.4f}")
    print("[llm Pass] =", round(mean, 4))


def iter_passrate():
    """Per-iteration mean pass-rate curves for the observe and memory arms.
    Writes _iter_passrate.json."""
    import statistics as st
    arms = ["observe", "memory"]
    data = {a: load(a) for a in arms}
    tasks, refs = {}, []
    for a in arms:
        for ri, r in enumerate(data[a]):
            for ii, it in enumerate(r.get("per_iter", [])):
                c = it.get("candidate")
                if it.get("compiled") and c:
                    h = (r["key"], hashlib.sha1(c.encode()).hexdigest())
                    tasks.setdefault(h, (h, r["key"], c))
                    refs.append((a, ri, ii, h))
    print(f"[iter Pass] scoring {len(tasks)} unique candidates ...", flush=True)
    res = _score_all(tasks)
    for a, ri, ii, h in refs:
        data[a][ri]["per_iter"][ii]["pr"] = res.get(h)
    out = {}
    for a in arms:
        curve, ns = [], []
        for k in range(5):
            vals = [r["per_iter"][k].get("pr") for r in data[a]
                    if len(r.get("per_iter", [])) > k and r["per_iter"][k].get("pr") is not None]
            curve.append(round(100 * st.mean(vals), 1) if vals else None)
            ns.append(len(vals))
        out[a] = {"mean_passrate@k": curve, "n@k": ns}
        print("[iter Pass]", a, out[a], flush=True)
    (RUNS / "_iter_passrate.json").write_text(json.dumps(out, indent=1))


# ---- per-arm metrics ----

def metrics(rows, llm=False):
    n = len(rows) or 1
    if llm:
        comp = sum(r["iter1_compiled"] for r in rows)
        re_ = sum(r["iter1_reexec"] for r in rows)
        acc = n
        fa = sum(1 for r in rows if not r["iter1_reexec"])
        return dict(N=len(rows), RC=comp/n, RE=re_/n, Pass=None, R_comp=0.0, R_exec=0.0,
                    R_tot=0.0, FA=fa/acc, FR=0.0, regression=0.0, iters=1.0)
    comp = sum(r["compiled"] for r in rows)
    re_ = sum(r["reexec"] for r in rows)
    acc = sum(r["accepted"] for r in rows)
    notacc = n - acc
    fa = sum(1 for r in rows if r["accepted"] and not r["reexec"])
    fr = sum(1 for r in rows if not r["accepted"] and r["reexec"])
    nc1 = [r for r in rows if not r["iter1_compiled"]]
    Rcomp = sum(1 for r in nc1 if r["compiled"]) / (len(nc1) or 1)
    cw1 = [r for r in rows if r["iter1_compiled"] and not r["iter1_reexec"]]
    Rexec = sum(1 for r in cw1 if r["reexec"]) / (len(cw1) or 1)
    nf1 = [r for r in rows if not r["iter1_reexec"]]
    Rtot = sum(1 for r in nf1 if r["reexec"]) / (len(nf1) or 1)
    ok1 = [r for r in rows if r["iter1_reexec"]]
    reg = sum(1 for r in ok1 if not r["reexec"]) / (len(ok1) or 1)
    iters = sum(r["iters"] for r in rows) / n
    pas = sum(r.get("pass_rate") or 0.0 for r in rows) / n
    return dict(N=len(rows), RC=comp/n, RE=re_/n, Pass=pas, R_comp=Rcomp, R_exec=Rexec,
                R_tot=Rtot, FA=(fa/acc if acc else 0.0), FR=(fr/notacc if notacc else 0.0),
                regression=reg, iters=iters)


def main():
    if RESCORE or not (RUNS / "_llm_passrate.txt").exists():
        llm_passrate()
    if RESCORE or not (RUNS / "_iter_passrate.json").exists():
        iter_passrate()

    out = {}
    for split in ["overall", "stripall", "unstripped"]:
        out[split] = {}
        for arm in ARMS + BASELINES:
            rows = load(arm)
            if split != "overall":
                rows = [r for r in rows if r["variant"] == split]
            out[split][arm] = metrics(rows, llm=(arm == "llm"))

    # llm-arm Pass comes from the offline rescoring above
    llm_pr = RUNS / "_llm_passrate.txt"
    if llm_pr.exists():
        out["overall"]["llm"]["Pass"] = float(llm_pr.read_text().split()[0])

    # per-optimization-level breakdown of the full system (retain_best)
    rb = load("retain_best")
    out["per_opt"] = {v: {o: metrics([r for r in rb if r["variant"] == v and r["opt"] == o])
                          for o in ["O0", "O1", "O2", "O3"]}
                      for v in ["unstripped", "stripall"]}

    (RUNS / "_metrics.json").write_text(json.dumps(out, indent=1))

    cols = ["RC", "RE", "Pass", "R_comp", "R_exec", "R_tot", "FA", "FR", "regression", "iters"]
    for split in ["overall", "stripall", "unstripped"]:
        print(f"\n=== {split} ===")
        print(f"{'arm':13s} " + " ".join(f"{c:>7s}" for c in cols))
        for arm in ARMS + BASELINES:
            m = out[split][arm]
            vals = []
            for c in cols:
                v = m[c]
                if v is None:
                    vals.append(f"{'--':>7s}")
                else:
                    vals.append(f"{v:7.2f}" if c == "iters" else f"{100*v:6.1f}%")
            print(f"{arm:13s} " + " ".join(vals))


if __name__ == "__main__":
    main()
