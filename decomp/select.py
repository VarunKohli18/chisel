"""Best-candidate retention. When the iteration budget runs out without an accept, keep_best
returns the best candidate the loop produced instead of whatever it ended on. The per-round
divergence counts are not comparable across rounds, so every compiled candidate is re-judged on
one shared seed set, the accumulated counterexamples plus a fresh typed batch, decoded under one
reference signature. This module is the single source of truth for that choice. The loop calls it
in-loop and derive_retain_best calls it offline, so the two cannot drift."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from decomp import fuzz, harness, oracles, signature


@dataclass
class Ranked:
    source: str
    index: int                    # per_iter position, used as a deterministic tie break
    n_divergences: int | None     # None when the candidate could not be judged
    n_seeds: int = 0

    @property
    def judged(self) -> bool:
        return self.n_divergences is not None


def compiled_candidates(per_iter: list) -> list:
    """The distinct compiled candidate sources from a finished loop, earliest occurrence first.
    Rounds whose candidate failed to compile are excluded. Rounds whose differential check never
    ran are kept, since re-judging them from scratch is the point."""
    out, seen = [], set()
    for i, it in enumerate(per_iter):
        src = it.get("candidate")
        if not it.get("compiled") or not src or src in seen:
            continue
        seen.add(src)
        out.append((i, src))
    return out


def _yardstick(sig, memory: list, *, n_seeds: int, in_contract: bool, seed: int,
               typed_decode: bool = True) -> list:
    """The shared evidence set, accumulated counterexamples first and then a fresh batch. M comes
    first so anything that truncates downstream keeps the inputs that have exposed divergences.
    The batch comes from harness.fresh_batch, the same regime mapping the oracle uses. With no
    signature there is nothing to decode a byte batch under, so none is built and the candidates
    go unjudged."""
    batch = [] if sig is None else harness.fresh_batch(sig, typed_decode=typed_decode, n=n_seeds,
                                                       seed=seed, in_contract=in_contract)
    return harness.whole_input_space(sig, harness.dedup_seeds(list(memory) + batch))


def rank(per_iter: list, *, cfg, orig_obj, pseudo_c: str, memory: dict,
         work: Path | None = None) -> list:
    """Re-judge every compiled candidate on one shared corpus and return Ranked, best first.
    Best means fewest divergences on the shared set, unjudgeable candidates sort last, and ties
    break toward the earlier round. Takes the run's Config whole so every caller judges under
    exactly the knobs the run used. memory is the loop's dict bucketed by sig_key."""
    features = set(cfg.features)
    cands = compiled_candidates(per_iter)
    if not cands:
        return []
    tmp = Path(work or tempfile.mkdtemp(prefix="rank_"))
    tmp.mkdir(parents=True, exist_ok=True)

    # The reference layout is the incumbent's signature, meaning the last candidate that
    # compiled. If the incumbent's layout holds no accumulated counterexamples but an earlier
    # compiled round's does, prefer the latest such layout, so a final-round signature switch
    # does not throw away every input that has already exposed a divergence.
    sig_ref = oracles.signature_for(cands[-1][1], pseudo_c)
    if sig_ref is None or not memory.get(signature.sig_key(sig_ref)):
        for _, src in reversed(cands[:-1]):
            s = oracles.signature_for(src, pseudo_c)
            if s is not None and memory.get(signature.sig_key(s)):
                sig_ref = s
                break
    # Memory arrives keyed by byte layout, and only the counterexamples mined under the reference
    # layout mean anything here. The same bytes decode to a different call under another
    # signature, so mixing them in would judge candidates on gibberish. With no reference layout
    # nothing is replayed and the candidates are reported unjudged rather than ranked on noise.
    mem = memory.get(signature.sig_key(sig_ref), []) if sig_ref is not None else []
    seeds = _yardstick(sig_ref, mem, n_seeds=cfg.diff_seeds,
                       in_contract="contract_gate" in features,
                       seed=fuzz.candidate_seed(cands[-1][1]),
                       typed_decode="typed_decode" in features)

    ranked = []
    for i, src in cands:
        obj, _ = oracles.compile_candidate(src, cfg.candidate_opt, tmp / f"rank{i}",
                                           mode=cfg.compile_mode)
        if obj is None:                       # compiled during the loop but not here, cannot judge
            ranked.append(Ranked(src, i, None))
            continue
        if not seeds:                         # nothing to judge on, leave it unjudged rather than
            ranked.append(Ranked(src, i, None))   # credit it with a vacuous zero
            continue
        divs, _ = fuzz.differential_run(sig_ref, orig_obj, obj, src, features=features,
                                        seeds=seeds, timeout=cfg.per_sample_timeout,
                                        max_divergences=cfg.max_divergences)
        # divs is None when the harness would not link against this candidate. That yields no
        # divergence count, so it stays unjudged and sorts last rather than being credited with
        # zero.
        ranked.append(Ranked(src, i, None if divs is None else len(divs), len(seeds)))

    ranked.sort(key=lambda r: (not r.judged, r.n_divergences if r.judged else 0, r.index))
    return ranked


def choose(per_iter: list, **kw) -> tuple:
    """Return the retained candidate source and the ranking, or empty values when there is
    nothing to keep."""
    ranked = rank(per_iter, **kw)
    if not ranked:
        return "", []
    # prefer a judged candidate, fall back to the earliest compiled one if none could be judged
    best = ranked[0]
    return best.source, ranked
