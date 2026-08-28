"""Select the benchmark's functions from the ExeBench splits and write records.jsonl. The benchmark is defined by a keys file, one cache_key per line."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

from decomp.dataset import records_path

_EXEBENCH = Path(__file__).resolve().parents[3] / "data" / "exebench"
_SPLITS = [("real_test", "real"), ("valid_real", "valreal")]


def _cache_key(kind: str, fname: str) -> str:
    return f"{kind}__{re.sub(r'[^A-Za-z0-9_]', '_', fname)}"


def keys_file(benchmark: str) -> Path:
    return _EXEBENCH / f"{benchmark}_keys.txt"


def _iter_exebench():
    """Yield (cache_key, record_dict) from every present real split."""
    import zstandard
    dctx = zstandard.ZstdDecompressor()
    for subdir, kind in _SPLITS:
        zsts = sorted((_EXEBENCH / subdir).glob("*.jsonl.zst"))
        if not zsts:
            continue
        with open(zsts[0], "rb") as fh:
            for line in io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8"):
                d = json.loads(line)["text"]
                fname = d.get("fname") or ""
                if not fname:
                    continue
                rec = {"fname": fname, "kind": kind, "func_def": d.get("func_def") or "",
                       "deps": d.get("real_deps") or "", "signature": d.get("signature") or "",
                       "io_pairs": d.get("real_io_pairs") or [], "iospec": d.get("real_iospec"),
                       "path": d.get("path", "")}
                yield _cache_key(kind, fname), rec


def write_records(benchmark: str, keys_path: Path | None = None) -> int:
    """Write records.jsonl by matching the benchmark's keys against the ExeBench splits, returning the count written."""
    kf = keys_path or keys_file(benchmark)
    if not kf.is_file():
        raise SystemExit(f"no keys file at {kf}; provide one cache_key per line")
    wanted = {ln.strip() for ln in kf.read_text().splitlines() if ln.strip()}
    out, seen = [], set()
    for key, rec in _iter_exebench():
        if key in wanted and key not in seen:
            seen.add(key)
            out.append(rec)
    rf = records_path(benchmark)
    rf.parent.mkdir(parents=True, exist_ok=True)
    tmp = rf.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    tmp.replace(rf)
    missing = wanted - seen
    if missing:
        print(f"[warn] {len(missing)} keys not found in splits, e.g. {sorted(missing)[:3]}")
    return len(out)


def load_records(benchmark: str) -> list:
    rf = records_path(benchmark)
    return [json.loads(l) for l in rf.read_text().splitlines() if l.strip()]
