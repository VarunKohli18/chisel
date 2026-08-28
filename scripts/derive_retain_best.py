#!/usr/bin/env python3
"""Derive the retain_best arm from the memory arm offline.

retain_best differs from memory only at the exhaustion path, so running it as its own arm would
re-pay the full LLM and mining cost to produce identical per-round records. Deriving it instead
keeps the comparison paired -- both arms see exactly the same candidates -- and costs one
differential replay per compiled candidate, on the ~24% of cells that exhaust their budget.

For accepted cells the result is unchanged. For budget-exhausted cells we re-judge every compiled
candidate on a common yardstick (decomp.select, the same code path loop.py uses in-loop) and
re-score the winner against the held-out suite.

The yardstick is M, the accumulated counterexamples, rebuilt from the per-round `div_seeds`
recorded by the loop. Runs produced before that field existed cannot be derived from -- rerun the
memory arm rather than silently ranking on a weaker corpus.
"""
import argparse
import copy
import glob
import json
import sys
import tempfile
from base64 import b64decode
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decomp import config, fuzz, score, select
from decomp.dataset import cell_path, records_path, suite_path


def load_records(benchmark):
    return {f"{r['kind']}__{r['fname']}": r
            for r in (json.loads(l) for l in records_path(benchmark).read_text().splitlines() if l.strip())}


def rebuild_memory(per_iter):
    """M as the loop accumulated it: counterexamples bucketed by the byte layout they were mined
    under, in order, deduplicated. Keyed rather than flat because the same seed decodes to a
    different call under a different signature."""
    mem = {}
    for it in per_iter:
        bucket = mem.setdefault(it.get("sig_key", "-"), [])
        have = set(bucket)
        for b64 in it.get("div_seeds") or []:
            s = b64decode(b64)
            if s not in have:
                have.add(s)
                bucket.append(s)
    return mem


def main():
    ap = argparse.ArgumentParser(description="derive the retain_best arm from memory-arm results")
    ap.add_argument("benchmark", nargs="?", default=None)
    ap.add_argument("out", nargs="?", default="results/runs_new")
    ap.add_argument("--config", default="config.yaml",
                    help="yaml the judging knobs come from; point at the config the memory run "
                         "used so the derived and live retain_best judge identically")
    ap.add_argument("--diff-seeds", dest="diff_seeds", type=int, default=None)
    ap.add_argument("--max-divergences", dest="max_divergences", type=int, default=None)
    ap.add_argument("--per-sample-timeout", dest="per_sample_timeout", type=float, default=None)
    ap.add_argument("--compile-mode", dest="compile_mode", default=None,
                    choices=["object", "link"])
    args = ap.parse_args()
    # The same loader the runs use, with the retain_best preset expanded on top -- judging must
    # happen under the knobs the memory run actually ran with, not a freshly built default Config.
    cfg = config.load(args.config, dict(arm="retain_best", benchmark=args.benchmark,
                                        diff_seeds=args.diff_seeds,
                                        max_divergences=args.max_divergences,
                                        per_sample_timeout=args.per_sample_timeout,
                                        compile_mode=args.compile_mode))
    benchmark, out = cfg.benchmark, Path(args.out)
    records = load_records(benchmark)
    suites = {}

    def suite(key):
        if key not in suites:
            p = suite_path(benchmark, key)
            suites[key] = json.loads(p.read_text())["io_pairs"] if p.exists() else None
        return suites[key]

    def pseudo_c(row):
        """The cell's Ghidra pseudo-C. Only used as the signature fallback when the incumbent
        candidate has no parseable one, so a missing cell is not fatal."""
        p = cell_path(benchmark, row["key"], row["opt"], row["variant"])
        if not p.exists():
            return ""
        return json.loads(p.read_text()).get("pseudo_c", "")

    (out / "retain_best").mkdir(parents=True, exist_ok=True)
    n_changed = n_exhausted = n_stale = 0
    for f in glob.glob(str(out / "memory" / "*.json")):
        rows = json.loads(Path(f).read_text())
        rb_rows = []
        for r in rows:
            rb = copy.deepcopy(r)
            rb["arm"] = "retain_best"
            per_iter = r.get("per_iter", [])
            if r.get("accepted") or not per_iter:
                rb_rows.append(rb)
                continue
            n_exhausted += 1
            rec, io = records.get(r["key"]), suite(r["key"])
            if rec is None or not io:
                rb_rows.append(rb)
                continue
            # A run without div_seeds predates the field; ranking it would silently use an empty
            # yardstick, which is exactly the "unmeasured scores as best" bug we removed.
            if not any("div_seeds" in it for it in per_iter):
                n_stale += 1
                rb["note"] = "retain_best skipped: run predates div_seeds, cannot rebuild M"
                rb_rows.append(rb)
                continue

            with tempfile.TemporaryDirectory(prefix="rb_") as d:
                work = Path(d)
                orig_obj = fuzz.compile_original(rec, work)
                if orig_obj is None:
                    rb["note"] = "retain_best skipped: original did not compile"
                    rb_rows.append(rb)
                    continue
                try:
                    chosen, ranked = select.choose(
                        per_iter, cfg=cfg, orig_obj=orig_obj, pseudo_c=pseudo_c(r),
                        memory=rebuild_memory(per_iter), work=work / "rank")
                except Exception as e:
                    rb["note"] = f"retain_best ranking failed, kept memory result: {str(e)[:80]}"
                    rb_rows.append(rb)
                    continue

            if not chosen or chosen == r.get("candidate"):
                rb_rows.append(rb)
                continue
            try:
                sc = score.score(rec, chosen, io)
            except Exception as e:                     # scoring crash keeps memory's row
                rb["note"] = f"retain_best score failed, kept memory result: {str(e)[:80]}"
                rb_rows.append(rb)
                continue
            rb["candidate"] = chosen
            rb["compiled"] = True                      # select only returns candidates that build
            rb["reexec"] = bool(sc.reexec)
            rb["pass_rate"] = round(getattr(sc, "pass_rate", 0.0), 4)
            rb["note"] = f"retained best of {sum(1 for x in ranked if x.judged)}/{len(ranked)} judged"
            n_changed += 1
            rb_rows.append(rb)

        name = Path(f).name
        tmp = str(out / "retain_best" / name) + ".tmp"
        Path(tmp).write_text(json.dumps(rb_rows, indent=1))
        Path(tmp).replace(out / "retain_best" / name)

    print(f"[retain_best] derived from memory; {n_exhausted} exhausted cells, "
          f"{n_changed} re-picked & re-scored, {n_stale} skipped (no div_seeds)", flush=True)


if __name__ == "__main__":
    main()
