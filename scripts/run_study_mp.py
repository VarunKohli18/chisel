#!/usr/bin/env python3
"""Load-balanced study orchestrator, process-pool variant.

Resumable and non-destructive: existing results/<arm>/<opt>_<variant>.json rows are preloaded and
skipped, new rows are appended before each atomic rewrite. Each task claims the least-loaded
endpoint for its lifetime.

  export OLLAMA_ENDPOINTS="http://127.0.0.1:11435 ... 11438"
  python scripts/run_study_mp.py --out results/runs_new --only-arms compiler,fuzzer,observe,memory,retain_best \
      --workers 160 --iters 5
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decomp import config, loop
from decomp.dataset import cache_dir, records_path, suite_path

# module globals, set in main() before the pool is forked so workers inherit them
ENDPOINTS: list = []
EP_INFLIGHT = None          # shared Array across forked workers
EP_LOCK = None              # Lock guarding EP_INFLIGHT
RECORDS: dict = {}
BASE_CFG = None             # config.Config template
BENCHMARK = ""
SUITE_CACHE: dict = {}      # per-worker suite cache


def load_records(benchmark: str) -> dict:
    return {f"{r['kind']}__{r['fname']}": r
            for r in (json.loads(l) for l in records_path(benchmark).read_text().splitlines()
                      if l.strip())}


def _suite_for(key: str):
    if key in SUITE_CACHE:
        return SUITE_CACHE[key]
    p = suite_path(BENCHMARK, key)
    s = json.loads(p.read_text())["io_pairs"] if p.exists() else None
    SUITE_CACHE[key] = s
    return s


def _claim_ep() -> int:
    with EP_LOCK:
        i = min(range(len(ENDPOINTS)), key=lambda j: EP_INFLIGHT[j])
        EP_INFLIGHT[i] += 1
        return i


def _release_ep(i: int) -> None:
    with EP_LOCK:
        EP_INFLIGHT[i] -= 1


def _run_task(task):
    """Worker entry point. Returns (name, opt, variant, result_dict)."""
    name, oracle, features, opt, variant, cf, key = task
    i = _claim_ep()
    try:
        cell = json.loads(Path(cf).read_text())
        k = cell.get("key", key)
        record = RECORDS.get(k)
        suite = _suite_for(k)
        cfg = replace(BASE_CFG, oracle=list(oracle), features=list(features),
                      opt=opt, variant=variant, endpoint=ENDPOINTS[i])
        r = loop.run_one(record, cell, suite, cfg, verbose=False)
        rd = asdict(r)
    except Exception as e:
        traceback.print_exc()
        rd = asdict(loop.Result(key=key, opt=opt, variant=variant, arm=name, iters=0,
                                compiled=False, note=f"CRASHED {type(e).__name__}: {str(e)[:140]}"))
    finally:
        _release_ep(i)
    return name, opt, variant, rd


def main() -> None:
    global ENDPOINTS, EP_INFLIGHT, EP_LOCK, RECORDS, BASE_CFG, BENCHMARK

    ap = argparse.ArgumentParser(description="process-pool ablation study orchestrator")
    ap.add_argument("--config", default="config.yaml",
                    help="yaml the run knobs default from (same loader as run.py); "
                         "CLI flags override it")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--opts", type=lambda s: s.split(","), default=["O0", "O1", "O2", "O3"])
    ap.add_argument("--variants", type=lambda s: s.split(","), default=["stripall", "unstripped"])
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--workers", type=int, default=160,
                    help="worker PROCESSES; oversubscribe the GPU slots so the fuzz phase of one "
                         "task overlaps generation of another")
    ap.add_argument("--min-suite", dest="min_suite", type=int, default=100)
    ap.add_argument("--exclude", default="bn_mul_comba8",
                    help="comma-separated substrings; matching function keys are skipped")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--diff-seeds", dest="diff_seeds", type=int, default=None)
    ap.add_argument("--max-divergences", dest="max_divergences", type=int, default=None,
                    help="divergences collected per round before the scan stops")
    ap.add_argument("--no-seed-replay", dest="replay_seeds", action="store_const", const=False,
                    default=None,
                    help="typed seeds only steer libfuzzer mining, they are not replayed verbatim")
    ap.add_argument("--per-sample-timeout", dest="per_sample_timeout", type=float, default=None)
    ap.add_argument("--compile-mode", dest="compile_mode", default=None,
                    choices=["object", "link"], help="'link' = A4D-style standalone link gate")
    ap.add_argument("--endpoints", default=os.getenv("OLLAMA_ENDPOINTS", "http://127.0.0.1:11434"),
                    help="space-separated Ollama endpoints (defaults to $OLLAMA_ENDPOINTS)")
    ap.add_argument("--only-arms", default="",
                    help="comma-separated arm names to restrict the run, in priority order")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resume/task counts and exit without running or touching files")
    args = ap.parse_args()

    ENDPOINTS = args.endpoints.split()
    ne = len(ENDPOINTS)
    # One loader for every entry point: yaml defaults, non-None CLI overrides on top. The
    # per-task oracle/features/opt/variant/endpoint are replaced in _run_task, so whatever the
    # yaml says for those is inert here.
    BASE_CFG = config.load(args.config, dict(
        benchmark=args.benchmark, iters=args.iters, model=args.model,
        temperature=args.temperature, prompt=args.prompt, budget=args.budget,
        diff_seeds=args.diff_seeds, replay_seeds=args.replay_seeds,
        max_divergences=args.max_divergences, per_sample_timeout=args.per_sample_timeout,
        compile_mode=args.compile_mode))
    BENCHMARK = BASE_CFG.benchmark
    out_dir = Path(args.out)
    drop = [s for s in args.exclude.split(",") if s]

    ctx = mp.get_context("fork")
    EP_INFLIGHT = ctx.Array("i", ne, lock=False)
    EP_LOCK = ctx.Lock()

    ladder = [(name, p["oracle"], p["features"]) for name, p in config.ARMS.items()]
    if args.only_arms:
        by_name = {t[0]: t for t in ladder}
        want = [a for a in (s.strip() for s in args.only_arms.split(",")) if a]
        missing = [a for a in want if a not in by_name]
        if missing:
            raise SystemExit(f"unknown arm(s) {missing}, choose from {list(by_name)}")
        ladder = [by_name[a] for a in want]

    RECORDS = load_records(BENCHMARK)

    out_dir.mkdir(parents=True, exist_ok=True)

    # resume: preload existing group files, remember completed rows
    groups: dict = defaultdict(list)
    done: set = set()
    for f in glob.glob(str(out_dir / "*" / "*.json")):
        arm = Path(f).parent.name
        try:
            rows = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in rows:
            groups[(arm, r["opt"], r["variant"])].append(r)
            done.add((arm, r["opt"], r["variant"], r["key"]))

    def write_group(gk) -> None:
        arm, opt, variant = gk
        d = out_dir / arm
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{opt}_{variant}.json"
        tmp = str(f) + ".tmp"
        Path(tmp).write_text(json.dumps(groups[gk], indent=1))
        os.replace(tmp, f)

    # build the global task list, suite filter applied once here in the parent
    parent_suites: dict = {}

    def parent_suite(key):
        if key not in parent_suites:
            p = suite_path(BENCHMARK, key)
            parent_suites[key] = json.loads(p.read_text())["io_pairs"] if p.exists() else None
        return parent_suites[key]

    tasks = []
    skipped_suite = 0
    for name, oracle, features in ladder:
        for opt in args.opts:
            for variant in args.variants:
                for cf in sorted(glob.glob(str(cache_dir(BENCHMARK) / f"*__{opt}__{variant}.json"))):
                    stem = Path(cf).stem
                    if any(s in stem for s in drop):
                        continue
                    key = stem[: -len(f"__{opt}__{variant}")]
                    if (name, opt, variant, key) in done:
                        continue
                    suite = parent_suite(key)
                    if suite is None or len(suite) < args.min_suite:
                        skipped_suite += 1
                        continue
                    tasks.append((name, oracle, features, opt, variant, cf, key))

    total = len(tasks)
    print(f"[study-mp] {total} tasks ({len(done)} already done, {skipped_suite} skipped on suite), "
          f"{args.workers} worker processes over {ne} endpoints", flush=True)
    by_arm = defaultdict(int)
    for t in tasks:
        by_arm[t[0]] += 1
    print(f"[study-mp] remaining by arm: {dict(by_arm)}", flush=True)

    if args.dry_run:
        print("[study-mp] dry run, exiting without running or touching files", flush=True)
        return
    if not total:
        print("[study-mp] nothing to do", flush=True)
        return

    # Single-instance guard (stale lock from a dead pid is taken over).
    lock = out_dir / "study.lock"
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
            os.kill(other, 0)
            raise SystemExit(f"another orchestrator (pid {other}) holds {lock}; "
                             f"kill it or remove the lock to start a new run")
        except (ValueError, ProcessLookupError):
            pass
    lock.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: lock.exists() and lock.read_text().strip() == str(os.getpid())
                    and lock.unlink())

    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(_run_task, t) for t in tasks]
        for fut in as_completed(futs):
            name, opt, variant, rd = fut.result()
            gk = (name, opt, variant)
            groups[gk].append(rd)
            write_group(gk)
            n_done += 1
            el = time.time() - t0
            rate = n_done / el if el else 0
            eta = (total - n_done) / rate / 60 if rate else 0
            print(f"[{n_done}/{total}] {name:9s} {opt}/{variant} {rd.get('key',''):34s} "
                  f"compiled={rd.get('compiled')} reexec={rd.get('reexec')} "
                  f"iters={rd.get('iters')} {rd.get('seconds')}s "
                  f"| {rate*60:.1f}/min ETA {eta:.0f}m", flush=True)

    print(f"[study-mp] complete: {total} tasks in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
