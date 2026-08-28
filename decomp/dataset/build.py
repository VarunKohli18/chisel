"""Regenerate the dataset: records.jsonl, a Ghidra pseudo-C cell per (function, opt, variant), and a test suite per function."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path

from decomp.dataset import (bin_dir, cache_dir, cell_path, records_path, suite_path)
from decomp.dataset import compile as compile_bin
from decomp.dataset import ghidra, select, testsuite
from decomp.dataset.records import record_key

_ARCH = "x8664"


def _binary_path(benchmark: str, key: str, opt: str, variant: str) -> Path:
    return bin_dir(benchmark) / f"{key}__{opt}__{variant}"


def _ghidra_cell(record: dict, binary: Path, vaddr: int, opt: str, variant: str) -> dict | None:
    payload = ghidra.run_ghidra(binary)
    if not payload or not payload.get("pseudo_c"):
        return None
    pseudo_c = ghidra.select_function(payload["pseudo_c"], vaddr)
    if not pseudo_c:
        return None
    return {"key": record_key(record), "opt": opt, "variant": variant, "arch": _ARCH,
            "vaddr": vaddr, "pseudo_c": pseudo_c}


def _build_suite(record: dict, n: int) -> dict:
    key = record_key(record)
    rng = random.Random(f"1234:{key}")
    kept, stats = testsuite.build_suite(record, n, rng)
    if kept:
        return {"key": key, "n": len(kept), "io_pairs": kept, "source": "generated"}
    # fall back to native io_pairs
    native = record.get("io_pairs") or []
    return {"key": key, "n": len(native), "io_pairs": native, "source": "native",
            "note": str(stats)}


def run(*, benchmark: str, opts: list, variants: list, limit: int = 0,
        keys_file: str | None = None, n_suite: int = 1000,
        shard: int = 0, nshards: int = 1) -> None:
    cache_dir(benchmark).mkdir(parents=True, exist_ok=True)
    bin_dir(benchmark).mkdir(parents=True, exist_ok=True)

    if shard == 0 and not records_path(benchmark).is_file():
        kp = Path(keys_file) if keys_file else None
        n = select.write_records(benchmark, kp)
        print(f"wrote records.jsonl ({n} records)", flush=True)

    records = select.load_records(benchmark)
    if limit:
        records = records[:limit]
    records = [r for i, r in enumerate(records) if i % nshards == shard]

    built = skipped = failed = 0
    for record in records:
        key, fname, deps = record_key(record), record["fname"], record.get("deps", "")
        # suite first, one per function
        sp = suite_path(benchmark, key)
        if not sp.exists():
            sp.parent.mkdir(parents=True, exist_ok=True)
            suite = _build_suite(record, n_suite)
            sp.write_text(json.dumps(suite))
            print(f"suite {key}: {suite['n']} pairs ({suite['source']})", flush=True)

        for opt in opts:
            # one compile per (function, opt) yields both variants; rebuild when a cell or binary is missing
            missing_cells = [v for v in variants
                             if not cell_path(benchmark, key, opt, v).exists()]
            missing_bins = [v for v in ("unstripped", "stripall")
                            if not _binary_path(benchmark, key, opt, v).exists()]
            if not missing_cells and not missing_bins:
                skipped += len(variants)
                continue
            with tempfile.TemporaryDirectory(prefix="cell_") as td:
                b = compile_bin.build(record["func_def"], deps, fname, opt, Path(td))
                if b is None:
                    failed += len(missing_cells) or 1
                    print(f"FAIL build {key} {opt}", flush=True)
                    continue
                for v, path in (("unstripped", b.unstripped), ("stripall", b.stripall)):
                    shutil.copy(path, _binary_path(benchmark, key, opt, v))
                for v in missing_cells:
                    binary = b.unstripped if v == "unstripped" else b.stripall
                    cell = _ghidra_cell(record, binary, b.vaddr, opt, v)
                    if cell is None:
                        failed += 1
                        print(f"FAIL ghidra {key} {opt} {v}", flush=True)
                        continue
                    cell_path(benchmark, key, opt, v).write_text(json.dumps(cell))
                    built += 1
                    print(f"ok   {key} {opt} {v}", flush=True)

    print(f"\n=== shard {shard}/{nshards}: cells built={built} skipped={skipped} failed={failed} ===")
