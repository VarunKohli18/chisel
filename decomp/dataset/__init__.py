"""Dataset store layout under data_cache/<benchmark>/."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE = ROOT / "data_cache"


def cache_dir(benchmark: str) -> Path:
    return DATA_CACHE / benchmark


def records_path(benchmark: str) -> Path:
    return cache_dir(benchmark) / "records.jsonl"


def cell_path(benchmark: str, key: str, opt: str, variant: str) -> Path:
    return cache_dir(benchmark) / f"{key}__{opt}__{variant}.json"


def suite_path(benchmark: str, key: str) -> Path:
    return cache_dir(benchmark) / "testsuite" / f"{key}.json"


def bin_dir(benchmark: str) -> Path:
    return cache_dir(benchmark) / "bin"
