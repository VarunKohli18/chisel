"""Prompt construction. One template is filled each round with the pseudo C, the prior
candidate, and the oracle feedback. When the prompt would overflow the context window the
feedback is kept whole, the pseudo C is trimmed from the middle first, then the prior candidate
from the tail."""

from __future__ import annotations

_PSEUDO_FLOOR = 2000       # keep enough pseudo C to identify the function
_PREV_FLOOR = 600          # keep enough of the prior attempt to show its structure

_TASK_SIMPLE = (
    "Below is Ghidra pseudo C for one function. Rewrite it as clean, compilable C named "
    "func0 that behaves identically to the original: the same return value and the same "
    "writes through its pointer arguments. Output only the C source of func0, with no "
    "prose and no comments, and include every header and type it needs.")

_TASK_DETAILED = (
    "Below is Ghidra pseudo C for one function. Rewrite it as compilable C named func0 that "
    "reproduces the original's behavior exactly, the same return value and the same writes "
    "through its pointer arguments. Keep the computation and control flow as written: preserve "
    "the order of operations and do not restructure or simplify loops, reorder side effects, or "
    "add or remove returns. But omit the decompiler scaffolding that reflects the machine and the "
    "ABI rather than the program, such as stack-protector canary checks, the placeholders the "
    "decompiler invents for values it could not recover, and register or stack spill temporaries, "
    "and rewrite synthetic types and raw memory addressing as ordinary C. The guiding rule: keep "
    "every statement that affects the return value or the bytes written through the pointer "
    "arguments, and drop every statement that does not. Output only the C source of func0, no "
    "prose and no comments, and include every header and type it needs.")

TASKS = {"simple": _TASK_SIMPLE, "detailed": _TASK_DETAILED}


def _trim_middle(text: str, budget: int) -> str:
    if len(text) <= budget or budget <= 0:
        return text if len(text) <= budget else text[:budget]
    marker = "\n/* ... elided ... */\n"
    keep = budget - len(marker)
    if keep < 200:
        return text[:budget]
    head = keep * 2 // 3
    return text[:head] + marker + text[len(text) - (keep - head):]


def _trim_tail(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    marker = "\n/* truncated */"
    return text[:max(0, budget - len(marker))] + marker if budget > len(marker) else text[:budget]


def build_prompt(pseudo_c: str, *, prev_source: str | None = None, feedback: str | None = None,
                 max_chars: int | None = None, task: str | None = None) -> str:
    """Build one prompt. The first round shows only the task and the pseudo C, later rounds
    append the prior attempt and the feedback. task selects the instruction template."""
    instruction = TASKS.get(task or "detailed", _TASK_SIMPLE)
    refinement = ""
    if prev_source or feedback:
        refinement = "\n\nThe previous attempt is not correct. Fix the issues below.\n"
        if prev_source:
            refinement += f"\nPrevious attempt:\n```c\n{{prev}}\n```\n"
        if feedback:
            refinement += f"\n{feedback}\n"

    def assemble(pc: str, prev: str) -> str:
        body = refinement.replace("{prev}", prev) if "{prev}" in refinement else refinement
        return f"{instruction}\n\nGhidra pseudo C:\n```c\n{pc}\n```\n{body}"

    if max_chars is None:
        return assemble(pseudo_c, prev_source or "")

    # fixed cost is everything but the two trimmable bodies, feedback is kept whole
    avail = max_chars - len(assemble("", ""))
    if avail < _PSEUDO_FLOOR:               # feedback nearly fills the budget
        return assemble(_trim_middle(pseudo_c, max(0, avail)), "")
    prev_keep = min(len(prev_source or ""), max(_PREV_FLOOR, avail // 4)) if prev_source else 0
    pseudo_keep = avail - prev_keep
    if pseudo_keep < _PSEUDO_FLOOR:
        pseudo_keep = _PSEUDO_FLOOR
        prev_keep = max(0, avail - pseudo_keep)
    return assemble(_trim_middle(pseudo_c, pseudo_keep),
                    _trim_tail(prev_source, prev_keep) if prev_source else "")
