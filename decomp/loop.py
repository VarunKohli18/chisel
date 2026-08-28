"""The refinement loop. Per function we run K rounds. Each round rebuilds a prompt, samples one
candidate, compiles it, then runs the oracles the arm enables. Cross-round state is the previous
candidate, its feedback, and with memory enabled the accumulated counterexample corpus. The loop
stops early when every enabled oracle passes, and an empty oracle set is the raw one-shot model.
The loop reads only the pseudo C, the original compiled to a reference object, and self-generated
inputs. The dataset io pairs are scored offline after the loop finishes."""

from __future__ import annotations

import tempfile
import time
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path

from decomp import fuzz, harness, llm, oracles, prompt, score, select, signature
from decomp.config import Config
from decomp.dataset.records import record_key


@dataclass
class Result:
    key: str
    opt: str
    variant: str
    arm: str
    iters: int                 # the iteration the loop accepted at, or K when exhausted
    compiled: bool
    accepted: bool = False
    reexec: bool = False
    pass_rate: float = 0.0
    n_pairs: int = 0
    iter1_compiled: bool = False
    iter1_reexec: bool = False
    seconds: float = 0.0
    in_tokens: int = 0             # total prompt tokens across all iterations
    out_tokens: int = 0            # total generated tokens across all iterations
    note: str = ""
    candidate: str = ""
    # each per_iter entry records it, outcome, compiled, n_divergences, candidate, and token counts
    per_iter: list = field(default_factory=list)


def _show(verbose: bool, title: str, body: str = "") -> None:
    """Print one labeled section of the trace."""
    if not verbose:
        return
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    if body:
        print(body)


def run_one(record: dict, cell: dict, io_pairs: list | None, cfg: Config,
            verbose: bool = False) -> Result:
    key = cell.get("key") or record_key(record)
    opt = cell.get("opt", cfg.opt)
    variant = cell.get("variant", cfg.variant)
    pseudo_c = cell["pseudo_c"]
    oracle = set(cfg.oracle)
    features = set(cfg.features)
    t0 = time.time()
    per_iter: list = []
    totals = {"in": 0, "out": 0}        # running prompt and generation token counts
    work = Path(tempfile.mkdtemp(prefix="arm_"))
    _show(verbose, f"DECOMPILE {key}  [{opt}/{variant}]  arm={cfg.arm_label()}  K={cfg.iters}")

    def emit(it, s, outcome, compiled, n_div=None, n_seeds=0, new_seeds=(),
             sig_key="-", sig_note=None):
        """Append a per-iteration record and accumulate token totals. n_div is None when the round
        produced no divergence evidence, which is distinct from 0 and retention must not confuse
        the two. new_seeds are this round's counterexamples, stored base64 so a finished run can
        rebuild M, and sig_key is the byte layout they were mined under. sig_note is diagnostic
        only."""
        totals["in"] += s.in_tokens
        totals["out"] += s.out_tokens
        per_iter.append({"it": it, "outcome": outcome, "compiled": compiled,
                         "n_divergences": n_div, "n_seeds": n_seeds, "candidate": s.text,
                         "div_seeds": [b64encode(x).decode() for x in new_seeds],
                         "sig_key": sig_key, "sig_note": sig_note,
                         "in_tokens": s.in_tokens, "out_tokens": s.out_tokens,
                         "done": s.done_reason})

    def finish(cand, compiled, iters, accepted, note) -> Result:
        res = Result(key=key, opt=opt, variant=variant, arm=cfg.arm_label(), iters=iters,
                     compiled=bool(compiled), accepted=accepted, candidate=cand,
                     per_iter=per_iter, seconds=round(time.time() - t0, 1), note=note,
                     in_tokens=totals["in"], out_tokens=totals["out"])
        if io_pairs and compiled and cand:
            sc = score.score(record, cand, io_pairs)
            res.reexec, res.pass_rate, res.n_pairs = sc.reexec, round(sc.pass_rate, 4), sc.n_pairs
            if sc.error:
                res.note = (res.note + " | " + sc.error[:120]).strip(" |")
        if per_iter and io_pairs:                  # also score the first attempt
            f0 = per_iter[0]
            res.iter1_compiled = bool(f0["compiled"])
            if f0["candidate"] == cand:
                res.iter1_reexec = res.reexec
            elif res.iter1_compiled:
                res.iter1_reexec = score.score(record, f0["candidate"], io_pairs).reexec
        return res

    # raw one-shot, no oracle, accept the first output
    if not oracle:
        p = prompt.build_prompt(pseudo_c, max_chars=llm.char_budget(), task=cfg.prompt)
        _show(verbose, "ITERATION 1  prompt", p)
        s = llm.sample(p, model=cfg.model, endpoint=cfg.endpoint, temperature=cfg.temperature)
        cand = s.text
        _show(verbose, "ITERATION 1  llm output", cand)
        obj, _ = oracles.compile_candidate(cand, cfg.candidate_opt, work / "c1",
                                           mode=cfg.compile_mode)
        emit(1, s, "llm_oneshot", obj is not None)
        return finish(cand, obj is not None, 1, True, "llm one-shot")

    orig_obj = None
    if "fuzzer" in oracle:
        orig_obj = fuzz.compile_original(record, work)
        if orig_obj is None:
            return finish("", False, 0, False, "original did not compile (no reference)")

    feedback, prev = None, None
    # M is keyed by byte layout. The same seed decodes to a different call under a different
    # signature, and the candidate's signature can change between rounds, so a flat list would
    # silently replay nonsense.
    memory: dict = {}                     # counterexamples per sig_key
    last = ""
    budget = llm.char_budget()
    prev_trunc_fail = False               # last round was a truncated non-compiling candidate
    stop_note, stopped_at = "", None      # set when the loop breaks before exhausting K
    sig_recovered = signature.parse_any_signature(pseudo_c)   # the decompiler's prototype
    for it in range(1, cfg.iters + 1):
        p = prompt.build_prompt(pseudo_c, prev_source=prev, feedback=feedback, max_chars=budget,
                                task=cfg.prompt)
        _show(verbose, f"ITERATION {it}  prompt", p)
        s = llm.sample(p, model=cfg.model, endpoint=cfg.endpoint, temperature=cfg.temperature)
        cand = s.text
        last = cand
        _show(verbose, f"ITERATION {it}  llm output", cand)
        # Compiling is unavoidable since the fuzzer needs an object, but only the compiler oracle
        # may turn the diagnostics into feedback. The candidate is built at cfg.candidate_opt,
        # matching the reference, so undefined behaviour is optimised the same way on both sides.
        obj, comp_fb = oracles.compile_candidate(cand, cfg.candidate_opt, work / f"c{it}",
                                                 mode=cfg.compile_mode,
                                                 max_diags=cfg.max_compiler_diags)
        if obj is None:
            fb = comp_fb if "compiler" in oracle else (
                "The candidate could not be built for differential testing. Produce a "
                "self-contained implementation that compiles on its own.")
            _show(verbose, f"ITERATION {it}  oracle feedback (compile)", fb)
            feedback, prev = fb, cand
            emit(it, s, "compile_fail", False)
            # Stop on repeated truncation, at temperature 0 it regenerates identically.
            if s.done_reason == "length" and prev_trunc_fail:
                # Early exhaustion, not a separate exit. Fall through to the same retention path
                # so a truncated run is treated exactly like a run that used up its budget.
                stop_note = "stopped: output exceeds num_predict (repeated truncation)"
                stopped_at = it
                break
            prev_trunc_fail = s.done_reason == "length"
            continue
        prev_trunc_fail = False
        sig = oracles.signature_for(cand, pseudo_c)
        skey = signature.sig_key(sig)
        # Recorded but deliberately not acted on. Ghidra's prototype is too unreliable on stripped
        # binaries to reject candidates with, logging it lets us test after the fact whether
        # disagreement predicts false accepts.
        sig_note = signature.incompatibility(signature.parse_signature(cand), sig_recovered)

        # The compiler was handled above, now the differential fuzzer on observable output.
        # n_div stays None unless the differential check actually ran and produced a count.
        entries, n_div, n_seeds, new_seeds = [], None, 0, []
        if "fuzzer" in oracle:
            try:
                out = oracles.fuzzer_check(cand, obj, orig_obj, sig, features=features,
                                           budget=cfg.budget, timeout=cfg.per_sample_timeout,
                                           n_seeds=cfg.diff_seeds, replay_seeds=cfg.replay_seeds,
                                           max_divergences=cfg.max_divergences,
                                           max_feedback=cfg.max_feedback_divergences,
                                           orig_dict=("orig_dict" in features),
                                           accumulated=(memory.get(skey, [])
                                                        if "memory" in features else None))
            except Exception as e:
                feedback, prev = (f"The differential check could not run: {str(e)[:160]}. "
                                  "Produce a cleaner standard implementation."), cand
                emit(it, s, "fuzz_error", True)      # no evidence, n_div stays None
                continue
            if "memory" in features:
                memory[skey] = out.corpus
            n_div, n_seeds, new_seeds = out.n_divergences, out.n_seeds_run, out.new_seeds
            if not out.passed:
                entries.append(out.feedback)

        if not entries:
            _show(verbose, f"ITERATION {it}  ACCEPTED (all oracles pass)")
            emit(it, s, "accept", True, n_div, n_seeds, sig_key=skey, sig_note=sig_note)
            return finish(cand, True, it, True, f"accepted: {cfg.arm_label()}")
        feedback, prev = "\n\n".join(entries), cand
        _show(verbose, f"ITERATION {it}  oracle feedback", feedback)
        emit(it, s, "divergence", True, n_div, n_seeds, new_seeds, sig_key=skey,
             sig_note=sig_note)

    # Budget exhausted or stopped early. Without keep_best the loop returns whatever it ended on,
    # even a candidate that does not compile. That regression is what keep_best exists to repair,
    # so the baseline has to exhibit it rather than silently fall back to the last compiling
    # candidate.
    chosen, ranked, rank_note = "", [], ""
    if "keep_best" in features and orig_obj is not None:
        try:
            chosen, ranked = select.choose(per_iter, cfg=cfg, orig_obj=orig_obj,
                                           pseudo_c=pseudo_c, memory=memory, work=work / "rank")
        except Exception as e:
            rank_note = f"retention ranking failed: {str(e)[:100]}"
            # keep_best still promises never to end on a non-compiling candidate while compiled
            # ones exist, so fall back to the last compiled candidate, just unranked.
            comp = select.compiled_candidates(per_iter)
            if comp:
                chosen = comp[-1][1]
                rank_note += " (fell back to last compiled candidate)"
    if chosen:
        final, final_compiled = chosen, True
    else:
        final = last
        final_compiled = bool(per_iter and per_iter[-1].get("compiled"))
    note = stop_note or "budget exhausted"
    if chosen and ranked:
        note += f" (retained best of {sum(1 for r in ranked if r.judged)}/{len(ranked)} judged)"
    if rank_note:
        note = (note + " | " + rank_note).strip(" |")
    _show(verbose, f"EXHAUSTED  {note}")
    return finish(final, final_compiled, stopped_at or cfg.iters, False, note)
