#!/usr/bin/env python3
"""One-shot LLM4Decompile-v2 baseline. Feeds the Ghidra pseudo-C in LLM4Decompile's prompt
template, generates one candidate greedily, and scores it on the held-out suite.

Resumable and shardable across GPUs:

  CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline_llm4decompile.py --out results/runs_new/llm4decompile
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/run_baseline_llm4decompile.py \
      --out results/runs_new/llm4decompile --shard $i --nshards 4 & done
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decomp import score
from decomp.dataset import cache_dir, records_path, suite_path
from decomp.dataset.records import rename_to_func0

# Rename the recovered function to func0 before scoring; prefer the arg-count match when there
# are several definitions.
_C_KW = {"if", "for", "while", "switch", "do", "else", "return", "sizeof"}


def _to_func0(src: str, want_nargs=None) -> str:
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

# LLM4Decompile-v2 published prompt template.
_BEFORE = "# This is the assembly code:\n"
_AFTER = "\n# What is the source code?\n"


def load_records(benchmark: str) -> dict:
    return {f"{r['kind']}__{r['fname']}": r
            for r in (json.loads(l) for l in records_path(benchmark).read_text().splitlines()
                      if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM4Decompile-v2 one-shot baseline")
    ap.add_argument("--model", default="LLM4Binary/llm4decompile-9b-v2")
    ap.add_argument("--benchmark", default="exebench_hard")
    ap.add_argument("--out", default="results/runs_new/llm4decompile")
    ap.add_argument("--opts", type=lambda s: s.split(","), default=["O0", "O1", "O2", "O3"])
    ap.add_argument("--variants", type=lambda s: s.split(","), default=["stripall", "unstripped"])
    ap.add_argument("--min-suite", dest="min_suite", type=int, default=100)
    ap.add_argument("--exclude", default="bn_mul_comba8")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2048)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap tasks this shard runs (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="build the task list and exit, no GPU")
    args = ap.parse_args()

    benchmark = args.benchmark
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop = [s for s in args.exclude.split(",") if s]
    records = load_records(benchmark)

    # resume: preload existing rows, remember done keys
    from collections import defaultdict
    groups: dict = defaultdict(list)
    done: set = set()
    for f in glob.glob(str(out_dir / "*.json")):
        try:
            rows = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in rows:
            groups[(r["opt"], r["variant"])].append(r)
            done.add((r["opt"], r["variant"], r["key"]))

    def suite_for(key):
        p = suite_path(benchmark, key)
        return json.loads(p.read_text())["io_pairs"] if p.exists() else None

    def write_group(gk):
        opt, variant = gk
        f = out_dir / f"{opt}_{variant}.json"
        tmp = str(f) + f".tmp{args.shard}"
        Path(tmp).write_text(json.dumps(groups[gk], indent=1))
        Path(tmp).replace(f)

    # build this shard's task list
    tasks = []
    idx = 0
    for opt in args.opts:
        for variant in args.variants:
            for cf in sorted(glob.glob(str(cache_dir(benchmark) / f"*__{opt}__{variant}.json"))):
                stem = Path(cf).stem
                if any(s in stem for s in drop):
                    continue
                key = stem[: -len(f"__{opt}__{variant}")]
                if (opt, variant, key) in done:
                    continue
                suite = suite_for(key)
                if suite is None or len(suite) < args.min_suite:
                    continue
                if idx % args.nshards == args.shard:
                    tasks.append((opt, variant, cf, key))
                idx += 1

    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[l4d] shard {args.shard}/{args.nshards}: {len(tasks)} tasks ({len(done)} already done)",
          flush=True)
    if args.dry_run:
        for t in tasks[:5]:
            print("    sample:", t[0], t[1], t[3], flush=True)
        print("[l4d] dry run, exiting before GPU/model load", flush=True)
        return
    if not tasks:
        print("[l4d] nothing to do", flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[l4d] loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    print("[l4d] model ready", flush=True)

    t0 = time.time()
    for i, (opt, variant, cf, key) in enumerate(tasks, 1):
        cell = json.loads(Path(cf).read_text())
        k = cell.get("key", key)
        record = records.get(k)
        suite = suite_for(k)
        pseudo = cell["pseudo_c"]
        prompt = _BEFORE + pseudo.strip() + _AFTER
        ts = time.time()
        try:
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=16384).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, num_beams=1,
                                     pad_token_id=tok.eos_token_id)
            cand = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            cand, note = "", f"generation failed: {type(e).__name__}: {str(e)[:120]}"
            sc = None
        else:
            note = ""
        if cand and record is not None:
            cand = _to_func0(cand, len(record.get("iospec", {}).get("funargs", []) or []))
        if cand:
            try:
                sc = score.score(record, cand, suite)
            except Exception as e:
                sc = None
                note = f"score failed: {type(e).__name__}: {str(e)[:120]}"
        compiled = bool(sc and sc.compiled)
        reexec = bool(sc and sc.reexec)
        row = {"key": k, "opt": opt, "variant": variant, "arm": "llm4decompile", "iters": 1,
               "compiled": compiled, "accepted": True, "reexec": reexec,
               "pass_rate": round(getattr(sc, "pass_rate", 0.0), 4) if sc else 0.0,
               "n_pairs": getattr(sc, "n_pairs", 0) if sc else 0,
               "iter1_compiled": compiled, "iter1_reexec": reexec,
               "seconds": round(time.time() - ts, 1),
               "note": (note or (getattr(sc, "error", "") or ""))[:160], "candidate": cand,
               "per_iter": []}
        gk = (opt, variant)
        groups[gk].append(row)
        write_group(gk)
        el = time.time() - t0
        print(f"[l4d {args.shard}] [{i}/{len(tasks)}] {opt}/{variant} {k:34s} "
              f"compiled={compiled} reexec={reexec} {row['seconds']}s "
              f"| {i/el*60:.1f}/min", flush=True)

    print(f"[l4d] shard {args.shard} complete: {len(tasks)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
