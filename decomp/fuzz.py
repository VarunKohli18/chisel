"""Native differential execution on x86_64. The replay runner links one harness against the
original and the candidate object, runs both on every seed, and reports divergences in exit code,
signal, or stdout. The libFuzzer miner synthesizes new discriminating inputs in-process."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import signal
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:                       # non-Unix, cpu limit skipped
    resource = None

from decomp import harness
from decomp.signature import Signature, externalize_func0, parse_signature

# Disable ASLR so an out-of-bounds read gives the same bytes every run. No-op if setarch is absent.
_ASLR_OFF = ["setarch", "-R"] if shutil.which("setarch") else []
_DET_RUNS = 2000        # libFuzzer execution budget fallback
_DET_SEED = 1           # fixed PRNG seed


@dataclass
class Divergence:
    call: str           # decoded func0 call for feedback, or a hex preview
    orig: str           # what the original did, printed, crashed, or timed out
    cand: str
    seed: bytes
    detail: str = ""    # one description localized to where the outputs first differ


def _exec_root() -> str:
    """A directory on an exec-capable filesystem."""
    for base in [os.environ.get("DECOMPILE_TMPDIR"), tempfile.gettempdir(), os.getcwd()]:
        if not base:
            continue
        try:
            d = tempfile.mkdtemp(prefix=".execprobe_", dir=base)
            p = os.path.join(d, "p")
            Path(p).write_text("#!/bin/sh\nexit 0\n")
            os.chmod(p, 0o755)
            ok = subprocess.run([p], capture_output=True).returncode == 0
            shutil.rmtree(d, ignore_errors=True)
            if ok:
                return base
        except Exception:
            continue
    return tempfile.gettempdir()


@dataclass
class _Run:
    exit_code: int
    signal: int | None
    sha: str
    stdout: bytes


_TIMEOUT = _Run(-1, None, hashlib.sha256(b"<timeout>").hexdigest(), b"<timeout>")


def _cpu_limiter(cpu: int):
    """A preexec hook that caps the child's CPU seconds so a runaway loop dies in CPU time rather
    than wall time."""
    if resource is None:
        return None

    def _set():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    return _set


def _run(binary: Path, stdin: bytes, cpu: int, wall: float) -> _Run:
    """Run the binary on one input under a CPU-time limit, with a wall-clock backstop. A CPU-limit
    kill and a wall timeout both map to the same <timeout> observation."""
    try:
        p = subprocess.run(_ASLR_OFF + [str(binary)], input=stdin, capture_output=True,
                           timeout=wall, preexec_fn=_cpu_limiter(cpu))
    except subprocess.TimeoutExpired:
        return _TIMEOUT
    rc = p.returncode
    if rc < 0 and -rc == signal.SIGXCPU:      # CPU limit kill, treat as timeout
        return _TIMEOUT
    return _Run(rc if rc >= 0 else 128 - rc, (-rc if rc < 0 else None),
                hashlib.sha256(p.stdout).hexdigest(), p.stdout)


def _describe(r: _Run) -> str:
    if r.signal is not None:
        return f"crashed with signal {r.signal}"
    if r.stdout == b"<timeout>":
        return "timed out"
    text = r.stdout.decode("utf-8", "replace").rstrip("\n")
    if len(text) > 200:
        text = text[:200] + f"...<{len(text) - 200} more>"
    desc = f"printed `{text}`" if text else "printed nothing"
    return desc + (f" (exit={r.exit_code})" if r.exit_code else "")


_DIFF_WINDOW = 3        # tokens of context shown on each side of the first differing value
_DIFF_TAIL = 6          # extra tokens shown when one output is a prefix of the other


def _localized_diff(ro: _Run, rc: _Run) -> str:
    """One description pointing at where the two token streams first differ. A crash, timeout, or
    exit code mismatch falls back to describing both sides."""
    non_print = (ro.signal is not None or rc.signal is not None
                 or b"<timeout>" in (ro.stdout, rc.stdout) or ro.exit_code != rc.exit_code)
    ot = ro.stdout.decode("utf-8", "replace").split()
    ct = rc.stdout.decode("utf-8", "replace").split()
    if non_print or ot == ct:                       # nothing to localize, show both sides
        return f"original {_describe(ro)}, candidate {_describe(rc)}"
    n = min(len(ot), len(ct))
    i = 0
    while i < n and ot[i] == ct[i]:
        i += 1
    ndiff = sum(a != b for a, b in zip(ot, ct)) + abs(len(ot) - len(ct))
    counts = f"original has {len(ot)} values, candidate {len(ct)}"
    if i == n:                                      # one output is a prefix of the other
        who, extra = ("candidate", ct[n:]) if len(ct) > len(ot) else ("original", ot[n:])
        tail = " ".join(extra[:_DIFF_TAIL]) + (" ..." if len(extra) > _DIFF_TAIL else "")
        return (f"outputs agree on the first {n} value(s), then {who} prints "
                f"{abs(len(ot) - len(ct))} extra (`{tail}`); {counts}")
    lo = max(0, i - _DIFF_WINDOW)
    octx = " ".join(ot[lo:i] + [f"[{ot[i]}]"] + ot[i + 1:i + 1 + _DIFF_WINDOW])
    cctx = " ".join(ct[lo:i] + [f"[{ct[i]}]"] + ct[i + 1:i + 1 + _DIFF_WINDOW])
    return (f"outputs agree on the first {i} value(s), then differ at index {i}: "
            f"original `{octx}` vs candidate `{cctx}` "
            f"({ndiff} of {max(len(ot), len(ct))} values differ; {counts})")


def _link(harness_c: str, obj: Path, out: Path) -> str | None:
    """Compile the harness against obj. None on success, an error snippet otherwise."""
    src = out.parent / f"{out.name}.h.c"
    src.write_text(harness_c, encoding="utf-8")
    try:
        r = subprocess.run(["gcc", str(src), str(obj), "-O0", "-w", "-lm", "-o", str(out)],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "harness link timed out"
    return None if r.returncode == 0 else str(r.stderr.strip().splitlines()[-3:])


def differential_run(sig, orig_obj: Path, cand_obj: Path, cand_src: str, *, features: set,
                     seeds: list, timeout: float, max_divergences: int = 25):
    """Run the original and candidate on every seed and return the divergences and their seeds.
    A divergence is any mismatch in exit code, signal, or stdout. A one-sided timeout is rechecked
    with slack."""
    harness_c = harness.render_harness(
        sig, typed_decode="typed_decode" in features,
        rich_observables="rich_observables" in features,
        address_free="address_free" in features)

    with tempfile.TemporaryDirectory(prefix="fuzz_", dir=_exec_root()) as tmp:
        tp = Path(tmp)
        orig_run, cand_run = tp / "orig", tp / "cand"
        for label, obj, out in (("original", orig_obj, orig_run), ("candidate", cand_obj, cand_run)):
            err = _link(harness_c, obj, out)
            if err is not None:
                return None, f"could not link the harness against the {label} object: {err}"

        # cpu seconds of at least 1, with a wider wall backstop for non-CPU blocking
        cpu = max(1, math.ceil(timeout))
        wall = max(timeout * 4, cpu + 3)
        # contract gating counts a divergence only on inputs where the original neither crashes
        # nor times out
        gate = "contract_gate" in features
        divs, div_seeds = [], []
        for seed in seeds:
            ro, rc = _run(orig_run, seed, cpu, wall), _run(cand_run, seed, cpu, wall)
            if _diverged(ro, rc) and b"<timeout>" in (ro.stdout, rc.stdout):
                # one-sided timeout, confirm with slack
                ro, rc = _run(orig_run, seed, cpu * 2, wall * 2), _run(cand_run, seed, cpu * 2, wall * 2)
            if gate and (ro.signal is not None or ro.stdout == b"<timeout>"):
                continue                       # original out of contract here
            if _diverged(ro, rc):
                divs.append(Divergence(harness.decode_call(sig, seed), _describe(ro),
                                       _describe(rc), seed, detail=_localized_diff(ro, rc)))
                div_seeds.append(seed)
                if len(divs) >= max_divergences:
                    break
        return divs, div_seeds


def _diverged(a: _Run, b: _Run) -> bool:
    return a.exit_code != b.exit_code or a.signal != b.signal or a.sha != b.sha


# libFuzzer entropic miner

def libfuzzer_available() -> bool:
    return shutil.which("clang") is not None


def _prefix_original(orig_obj: Path, work: Path):
    """Copy orig_obj with every defined symbol prefixed orig_ so func0 becomes orig_func0 and
    helpers do not collide with the candidate's."""
    if shutil.which("nm") is None or shutil.which("objcopy") is None:
        return None
    try:
        r = subprocess.run(["nm", "-g", "--defined-only", str(orig_obj)],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    syms = {p[2] for p in (l.split() for l in r.stdout.splitlines())
            if len(p) >= 3 and p[1] in "TtDdBbRrW"}
    if "func0" not in syms:
        return None
    mapfile = work / "redefine.map"
    mapfile.write_text("".join(f"{s} orig_{s}\n" for s in sorted(syms)))
    out = work / "orig_prefixed.o"
    try:
        rc = subprocess.run(["objcopy", f"--redefine-syms={mapfile}", str(orig_obj), str(out)],
                            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out if rc.returncode == 0 and out.exists() else None


def _build_target(work: Path, sources: list, objs: list, out: Path) -> str | None:
    cmd = ["clang", "-O1", "-g", "-w", "-fsanitize=fuzzer", "-Dfunc0=cand_func0",
           *sources, *[str(o) for o in objs], "-lm", "-o", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "build timeout"
    return None if r.returncode == 0 else "build failed"


import re

_IMM_RE = re.compile(r"\$0x([0-9a-fA-F]+)")


# Immediates that carry no discriminating signal, such as alignment masks and the small frame
# constants a compiler emits regardless of the source.
_NOISE_CONSTS = {0xff, 0xffff, 0xffffffff, -1, 0xf, 0xfffffff0, 0x1f, 0x3f, 0x7f,
                 2, 3, 4, 8, 16, 32, -2, -4, -8, -16}


def extract_constants(orig_obj: Path, limit: int = 64) -> list[int]:
    """Immediate operands disassembled from func0 in the original object. The original links in
    uninstrumented, so these branch thresholds and magic numbers are invisible to libFuzzer unless
    fed to the miner. Restricted to func0's own body, folded to signed 64-bit, with the trivial
    and alignment constants dropped."""
    if shutil.which("objdump") is None or not Path(orig_obj).exists():
        return []
    try:
        r = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(orig_obj)],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    # keep only the lines inside the func0 block, not helpers or other functions
    body, in_func = [], False
    for line in r.stdout.splitlines():
        if line.endswith(">:"):
            in_func = line.endswith("<func0>:")
            continue
        if in_func:
            body.append(line)
    text = "\n".join(body) or r.stdout       # fall back to whole object if func0 unlabeled
    seen, out = set(), []
    for m in _IMM_RE.finditer(text):
        v = int(m.group(1), 16)
        if v >= (1 << 63):                      # fold to signed 64-bit
            v -= (1 << 64)
        if v in (0, 1, -1) or v in seen or v in _NOISE_CONSTS:
            continue
        seen.add(v); out.append(v)
        if len(out) >= limit:
            break
    return out


def _dict_entry(v: int) -> list[str]:
    """C-escaped little-endian byte strings for an integer constant at 4 and 8 byte widths.
    Narrower widths are the noisiest to splice, so they are dropped."""
    entries = []
    for width in (4, 8):
        lo, hi = -(1 << (width * 8 - 1)), (1 << (width * 8)) - 1
        if not (lo <= v <= hi):
            continue
        b = (v & ((1 << (width * 8)) - 1)).to_bytes(width, "little")
        entries.append('"' + "".join(f"\\x{x:02x}" for x in b) + '"')
    return entries


def write_dict(constants: list[int], path: Path) -> bool:
    """Write a libFuzzer -dict file of the original's constants. Returns False if empty."""
    lines, seen = [], set()
    for v in constants:
        for e in _dict_entry(v):
            if e not in seen:
                seen.add(e); lines.append(e)
    if not lines:
        return False
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def libfuzzer_mine(cand_src: str, sig: Signature, orig_obj: Path, *, budget: int,
                   seed_corpus: list | None = None, typed_decode: bool = True,
                   use_dict: bool = True) -> list:
    """Run differential libFuzzer, with a coverage-only fallback, and return discriminating and
    coverage-diverse inputs as seeds. Empty list when libFuzzer cannot run."""
    if not libfuzzer_available() or sig is None or not sig.args:
        return []
    consts = extract_constants(Path(orig_obj)) if use_dict else []
    with tempfile.TemporaryDirectory(prefix="libfuzz_") as d:
        work = Path(d)
        (work / "cand.c").write_text(cand_src, encoding="utf-8")
        target = work / "target"

        built = False
        prefixed = _prefix_original(Path(orig_obj), work) if Path(orig_obj).exists() else None
        if prefixed is not None:
            (work / "diff.c").write_text(
                harness.render_diff_target(sig, typed_decode=typed_decode, constants=consts),
                encoding="utf-8")
            if _build_target(work, [str(work / "diff.c"), str(work / "cand.c")],
                             [prefixed], target) is None:
                built = True
        if not built:
            (work / "cov.c").write_text(harness.render_cov_target(sig, typed_decode=typed_decode),
                                        encoding="utf-8")
            if _build_target(work, [str(work / "cov.c"), str(work / "cand.c")], [], target) is not None:
                return []   # the candidate will not build instrumented

        corpus = work / "corpus"
        corpus.mkdir()
        for i, s in enumerate(seed_corpus or []):
            (corpus / f"seed_{i:05d}").write_bytes(s)

        flags = [f"-runs={budget}", f"-seed={_DET_SEED}", f"-max_len={harness.max_input_len(sig, typed=typed_decode)}",
                 "-timeout=10", "-print_final_stats=0", "-entropic=1", "-use_value_profile=1"]
        # feed the original's own branch constants as a dictionary, since value profiling cannot
        # see into the uninstrumented original
        dict_path = work / "orig.dict"
        if use_dict and write_dict(consts, dict_path):
            flags.append(f"-dict={dict_path}")
        try:
            subprocess.run([str(target), str(corpus), *flags], capture_output=True,
                           text=True, timeout=600, cwd=str(work))
        except subprocess.TimeoutExpired:
            pass

        out, seen = [], set()
        files = list(corpus.iterdir()) + [p for p in work.iterdir()
                                          if p.is_file() and p.name.startswith(("crash-", "oom-", "timeout-"))]
        for p in files:
            try:
                b = p.read_bytes()
            except OSError:
                continue
            # keep the empty input, it decodes to the all-defaults call and libFuzzer writes a
            # zero-byte crash file when that call diverges
            if b not in seen:
                seen.add(b); out.append(b)
        return out


def compile_original(record: dict, work: Path):
    """Compile the original, renamed func0, to an object that serves as the differential
    reference."""
    from decomp.dataset.records import rename_to_func0
    src = (record.get("deps", "") or "") + "\n" + \
        externalize_func0(rename_to_func0(record["func_def"], record["fname"]))
    src_path = work / "orig.c"
    obj_path = work / "orig.o"
    src_path.write_text(src, encoding="utf-8")
    r = subprocess.run(["gcc", "-c", str(src_path), "-O0", "-w", "-o", str(obj_path)],
                       capture_output=True, text=True, timeout=60)
    return obj_path if r.returncode == 0 else None


def candidate_seed(cand_src: str) -> int:
    return zlib.crc32(cand_src.encode("utf-8"))
