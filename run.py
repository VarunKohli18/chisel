#!/usr/bin/env python3
"""Single entry point: data | run | suites.

  run.py data   [--benchmark X] [--opt ..] [--variant ..] [--limit N]
  run.py run    --arm retain_best --opt O0 --variant stripall [--shard i --nshards n] --out f.json
  run.py suites [--dir data_cache/exebench_hard]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path

from decomp import config
from decomp.dataset import cache_dir, cell_path, records_path, suite_path


def _load_records(benchmark: str) -> dict:
    return {f"{r['kind']}__{r['fname']}": r
            for r in (json.loads(l) for l in records_path(benchmark).read_text().splitlines() if l.strip())}


def _load_suite(benchmark: str, key: str):
    p = suite_path(benchmark, key)
    return json.loads(p.read_text())["io_pairs"] if p.exists() else None


def cmd_run(args) -> None:
    from decomp import loop

    cfg = config.from_args(args)
    benchmark, opt, variant = cfg.benchmark, cfg.opt, cfg.variant
    records = _load_records(benchmark)
    cells = sorted(glob.glob(str(cache_dir(benchmark) / f"*__{opt}__{variant}.json")))
    if args.key:
        cells = [c for c in cells if args.key in Path(c).stem]
        if not cells:
            raise SystemExit(f"no cell matching key {args.key!r} at {opt}/{variant}")
    if args.keys_file:
        wanted = {ln.strip() for ln in Path(args.keys_file).read_text().splitlines() if ln.strip()}
        want_stems = {f"{k}__{opt}__{variant}" for k in wanted}
        cells = [c for c in cells if Path(c).stem in want_stems]
        if not cells:
            raise SystemExit(f"no cells from {args.keys_file} at {opt}/{variant}")
    if args.exclude:
        drop = [s for s in args.exclude.split(",") if s]
        cells = [c for c in cells if not any(s in Path(c).stem for s in drop)]
    if args.limit:
        cells = cells[:args.limit]
    if args.nshards > 1:
        cells = [c for i, c in enumerate(cells) if i % args.nshards == args.shard]

    label = cfg.arm_label()
    results = []

    def dump():
        if args.out:
            tmp = args.out + ".tmp"
            Path(tmp).write_text(json.dumps([asdict(r) for r in results], indent=1))
            os.replace(tmp, args.out)

    for cf in cells:
        key = Path(cf).stem
        try:
            cell = json.loads(Path(cf).read_text())
            key = cell.get("key", key)
            record = records.get(key)
            if record is None:
                print(f"[skip] {key}: no record"); continue
            suite = _load_suite(benchmark, key)
            if suite is None:
                print(f"[skip] {key}: no suite"); continue
            if len(suite) < args.min_suite:
                print(f"[skip] {key}: suite {len(suite)} < min {args.min_suite}"); continue
            r = loop.run_one(record, cell, suite, cfg, verbose=args.verbose)
        except Exception as e:
            traceback.print_exc()
            r = loop.Result(key=key, opt=opt, variant=variant, arm=label, iters=0,
                            compiled=False, note=f"CRASHED {type(e).__name__}: {str(e)[:140]}")
        results.append(r)
        print(f"[{r.arm}] {r.key:36s} compiled={r.compiled} reexec={r.reexec} "
              f"pass_rate={r.pass_rate} iters={r.iters} {r.seconds}s {r.note}")
        dump()

    n = len(results)
    print(f"\n== {label} {opt}/{variant} n={n}: compiled "
          f"{sum(r.compiled for r in results)}/{n}, reexec {sum(r.reexec for r in results)}/{n} ==")
    dump()


def cmd_suites(args) -> None:
    """Report how many suite cases execute to completion per function."""
    from decomp import score
    from decomp.dataset.records import func0_iospec, record_key, rename_to_func0

    d = Path(args.dir)
    recs = {f"{r['kind']}__{r['fname']}": r
            for r in (json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if l.strip())}
    suite_dir = d / "testsuite" if (d / "testsuite").is_dir() else d / "test_suite"
    suites = sorted(suite_dir.glob("*.json"))
    print(f"{'function':40s} {'suite':>7} {'worked':>7} {'source':>10}")
    print("-" * 68)
    tot_n = tot_ok = kept = 0
    for sf in suites:
        s = json.loads(sf.read_text())
        key, pairs = s["key"], s["io_pairs"]
        rec = recs.get(key)
        worked = 0
        if rec is not None and pairs:
            ref = score.reference_outputs(rename_to_func0(rec["func_def"], rec["fname"]),
                                          rec.get("deps", ""), func0_iospec(rec["iospec"]), pairs)
            worked = len(set(k[0] for k in ref)) if ref else 0
        tot_n += len(pairs); tot_ok += worked
        kept += len(pairs) >= args.min_suite
        flag = "" if len(pairs) >= args.min_suite else "  DROP"
        print(f"{key:40s} {len(pairs):>7} {worked:>7} {s.get('source','?'):>10}{flag}")
    print("-" * 68)
    print(f"{'TOTAL ('+str(len(suites))+' functions)':40s} {tot_n:>7} {tot_ok:>7}")
    print(f"kept (suite >= {args.min_suite}): {kept}/{len(suites)}; dropped: {len(suites)-kept}")


def cmd_data(args) -> None:
    from decomp.dataset import build
    build.run(benchmark=args.benchmark, opts=args.opts, variants=args.variants,
              limit=args.limit, keys_file=args.keys, n_suite=args.n_suite,
              shard=args.shard, nshards=args.nshards)


def main() -> None:
    ap = argparse.ArgumentParser(description="iterative decompiler")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the refinement loop over a benchmark cell")
    config.add_run_args(p_run)
    p_run.add_argument("--key", default=None, help="run only functions whose cell name contains this")
    p_run.add_argument("--keys-file", dest="keys_file", default=None,
                       help="file of function keys (kind__fname, one per line) to restrict the run to")
    p_run.add_argument("--verbose", action="store_true",
                       help="print each iteration's prompt, llm output, and oracle feedback")
    p_run.add_argument("--min-suite", dest="min_suite", type=int, default=100,
                       help="skip functions whose held-out suite has fewer than this many io_pairs "
                            "(default 100, dropping the 17 small-suite functions; 0 keeps all 150)")
    p_run.add_argument("--exclude", default="bn_mul_comba8",
                       help="comma-separated substrings; cells whose name contains any are skipped "
                            "(default bn_mul_comba8, whose correct C exceeds num_predict and is "
                            "always truncated; pass '' to keep everything)")
    p_run.add_argument("--limit", type=int, default=0)
    p_run.add_argument("--shard", type=int, default=0)
    p_run.add_argument("--nshards", type=int, default=1)
    p_run.add_argument("--out", default="")
    p_run.set_defaults(func=cmd_run)

    p_dat = sub.add_parser("data", help="regenerate the dataset (records, binaries, ghidra, suites)")
    p_dat.add_argument("--benchmark", default="exebench_hard")
    p_dat.add_argument("--opts", type=lambda s: s.split(","), default=["O0", "O1", "O2", "O3"])
    p_dat.add_argument("--variants", type=lambda s: s.split(","), default=["stripall", "unstripped"])
    p_dat.add_argument("--keys", default=None, help="file of function keys to select")
    p_dat.add_argument("--n-suite", dest="n_suite", type=int, default=1000)
    p_dat.add_argument("--limit", type=int, default=0)
    p_dat.add_argument("--shard", type=int, default=0)
    p_dat.add_argument("--nshards", type=int, default=1)
    p_dat.set_defaults(func=cmd_data)

    p_sui = sub.add_parser("suites", help="report how many suite cases work per function")
    p_sui.add_argument("--dir", default="data_cache/exebench_hard",
                       help="dataset dir holding records.jsonl, testsuite/, bin/")
    p_sui.add_argument("--min-suite", dest="min_suite", type=int, default=100,
                       help="threshold for the kept/dropped tally (default 100)")
    p_sui.set_defaults(func=cmd_suites)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
