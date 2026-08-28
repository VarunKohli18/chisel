"""The oracles. Each takes a compiled candidate and returns whether it passed plus the feedback
text the next prompt shows. The compiler oracle builds with gcc and reports the first diagnostics.
The fuzzer oracle mines inputs with libFuzzer, runs the differential harness, and reports
divergences. All inputs are signature-derived."""

from __future__ import annotations

import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from decomp import fuzz, harness
from decomp.signature import parse_any_signature, parse_signature

_DIAG_RE = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+):\s+"
                      r"(?P<sev>error|fatal error):\s+(?P<msg>.+)$")
_FEEDBACK_DIAGS = 5            # compiler diagnostics shown per round
_FEEDBACK_DIVERGENCES = 10     # counterexamples shown per round
# Backstop on the memory corpus. With the usual settings this never binds, it exists so a
# pathological run cannot grow M without bound.
MEMORY_CAP = 500


def compile_candidate(source: str, opt: str, work: Path, mode: str = "object",
                      max_diags: int = _FEEDBACK_DIAGS):
    """Compile a candidate to an object at the given opt level. Returns the object path or None,
    plus feedback built from the parsed diagnostics on failure. In link mode the candidate must
    also link as a standalone, and a link failure is reported as a compile failure."""
    work.mkdir(parents=True, exist_ok=True)
    src, obj = work / "cand.c", work / "cand.o"
    src.write_text(source, encoding="utf-8")
    lvl = opt if re.fullmatch(r"O[0-3s]", opt) else "O0"
    try:
        r = subprocess.run(["gcc", "-c", str(src), f"-{lvl}", "-o", str(obj)],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "Compilation timed out. Simplify the control flow and try again."
    if r.returncode != 0:
        diags = [f"{m.group('msg').strip()} (line {m.group('line')})"
                 for m in (_DIAG_RE.match(l.strip()) for l in r.stderr.splitlines()) if m]
        if not diags:
            diags = [" / ".join(r.stderr.strip().splitlines()[-3:]) or "compilation failed"]
        shown = diags[:max_diags]
        more = f" (and {len(diags) - len(shown)} more)" if len(diags) > len(shown) else ""
        return None, "Compilation failed: " + "; ".join(shown) + more + ". Fix it."
    if mode == "link":                    # additionally require a standalone link
        body = source if re.search(r"\bmain\s*\(", source) else source + "\nint main(void){return 0;}\n"
        lsrc, lexe = work / "cand_link.c", work / "cand_link"
        lsrc.write_text(body, encoding="utf-8")
        try:
            rl = subprocess.run(["gcc", "-w", "-o", str(lexe), str(lsrc), "-lm"],
                                capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None, "Compilation timed out. Simplify the control flow and try again."
        if rl.returncode != 0:
            und = list(dict.fromkeys(
                l.split("undefined reference to ")[1].strip().strip("`'\"")
                for l in rl.stderr.splitlines() if "undefined reference to " in l))[:5]
            if und:
                return None, ("Compilation failed: unresolved reference(s) " + ", ".join(und)
                              + " that the decompiler could not recover; remove them or supply the "
                              "definitions. Fix it.")
            return None, ("Compilation failed: " + (" / ".join(rl.stderr.strip().splitlines()[-3:])
                          or "link failed") + ". Fix it.")
    return obj, None


@dataclass
class FuzzOutcome:
    passed: bool
    feedback: str | None = None
    corpus: list = field(default_factory=list)     # divergence-inducing inputs, for memory
    # None means the differential check did not run and the candidate carries no divergence
    # evidence. That is distinct from 0, which means the check ran and found nothing. Ranking must
    # never treat an unmeasured candidate as a clean one.
    n_divergences: int | None = None
    n_seeds_run: int = 0                           # replay set size the count was measured over
    # This round's own counterexamples, not the whole of M. Persisted per round so a finished run
    # can reconstruct M offline for the retain_best derivation.
    new_seeds: list = field(default_factory=list)


def fuzzer_check(cand_src: str, cand_obj: Path, orig_obj: Path, sig, *, features: set,
                 budget: int, timeout: float, n_seeds: int = 200,
                 accumulated: list | None = None, replay_seeds: bool = True,
                 max_divergences: int = 25, orig_dict: bool = True,
                 max_feedback: int = _FEEDBACK_DIVERGENCES) -> FuzzOutcome:
    """Mine discriminating inputs with libFuzzer, run the differential harness, and report
    divergences. accumulated is M, the counterexamples from earlier rounds. It both seeds the
    miner and is replayed verbatim, so a candidate must keep satisfying every input that has
    already exposed a divergence. The returned corpus is M for the next round, counterexamples
    only, never the mined pool."""
    typed_decode = "typed_decode" in features
    typed_seeds = "typed_seeds" in features
    in_contract = "contract_gate" in features       # judge only inputs within the suite's domain
    acc = list(accumulated or [])
    sd = fuzz.candidate_seed(cand_src)
    # The fresh batch matches the arm's decode and seeds mining when there is no memory yet.
    # Under typed decode the curated corpus is the typed_seeds feature, so without it the batch
    # is empty and mining is the only input source.
    if typed_decode and not typed_seeds:
        batch = []
    else:
        batch = harness.fresh_batch(sig, typed_decode=typed_decode, n=n_seeds, seed=sd,
                                    in_contract=in_contract)
    mined = []
    try:
        # Seed mining with the typed corpus and the accumulated divergences. The typed batch is
        # the structure-rich starting population, memory adds past counterexamples.
        mined = fuzz.libfuzzer_mine(cand_src, sig, orig_obj, budget=budget,
                                    seed_corpus=harness.dedup_seeds(batch + acc), typed_decode=typed_decode,
                                    use_dict=orig_dict)
    except Exception:
        mined = []
    rng = random.Random(fuzz.candidate_seed(cand_src))
    # shuffle so collected divergences mix the fresh batch, memory corpus, and mined inputs
    if replay_seeds:
        seeds = harness.dedup_seeds(batch + acc + mined)
    else:
        # The typed batch only steers mining here. Drop the curated inputs from the replay set,
        # including any libfuzzer echoed back verbatim, so they are never tested.
        bset = set(batch)
        seeds = harness.dedup_seeds(acc + [m for m in mined if m not in bset])
    rng.shuffle(seeds)

    # An empty replay set would accept the candidate without ever running it. A no-arg function's
    # whole input space is the one empty invocation, and with no parseable signature at all the
    # harness declares a no-arg func0 and that single invocation is the only thing it can try.
    seeds = harness.whole_input_space(sig, seeds)
    if not seeds:
        if sig is None:
            seeds = [b""]
        else:
            return FuzzOutcome(False, corpus=acc, feedback=(
                "The differential check could not build any inputs for this signature. Give func0 "
                "explicit, ordinary C parameter types."))

    divs, div_seeds = fuzz.differential_run(sig, orig_obj, cand_obj, cand_src,
                                            features=features, seeds=seeds, timeout=timeout,
                                            max_divergences=max_divergences)
    if divs is None:                                # link failure, surface as feedback
        # n_divergences stays None since nothing was learned about this candidate, so it must not
        # be rankable against candidates that were actually measured
        return FuzzOutcome(False, feedback=div_seeds, corpus=acc)
    # Memory carries counterexamples only. The mined corpus is regenerated every round, so
    # persisting it would crowd the real counterexamples out of the cap and change what M means.
    corpus = harness.dedup_seeds(acc + div_seeds)[:MEMORY_CAP]
    if not divs:
        return FuzzOutcome(True, corpus=corpus, n_divergences=0, n_seeds_run=len(seeds))
    # show a random sample rather than the first few
    shown = divs if len(divs) <= max_feedback else rng.sample(divs, max_feedback)
    body = "\n".join(f"On {d.call}: {d.detail or f'original {d.orig}, candidate {d.cand}'}."
                     for d in shown)
    more = f"\n(and {len(divs) - len(shown)} more)" if len(divs) > len(shown) else ""
    # The scan stops at max_divergences, so the count is a floor, not a total. Say so rather than
    # report a capped count as if it were exhaustive.
    at_least = "at least " if len(divs) >= max_divergences else ""
    fb = (f"Your func0 diverges from the original on {at_least}{len(divs)} input(s). It must "
          f"reproduce the original on all of them:\n{body}{more}\nFix it.")
    return FuzzOutcome(False, feedback=fb, corpus=corpus, n_divergences=len(divs),
                       n_seeds_run=len(seeds), new_seeds=div_seeds)


def signature_for(cand_src: str, pseudo_c: str):
    """Prefer the candidate's signature, which has concrete C types, over the decompiler's
    recovered one. The fallback matters because Ghidra never names the function func0, so the
    pseudo C prototype has to be read by shape."""
    return parse_signature(cand_src) or parse_any_signature(pseudo_c)


