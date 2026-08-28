"""Candidate sampling against a local Ollama server. One sample per call at temperature T."""

from __future__ import annotations

import logging
import os
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)

NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "24576"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))
# Request timeout in seconds.
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1200"))
_TOKEN_CHARS = 3            # chars per token
_MARGIN = 1024             # tokens reserved for template wrapping
_FENCE = re.compile(r"```(?:c|cpp|C)?\s*(?P<body>.*?)```", re.DOTALL)


def char_budget() -> int:
    """Max prompt length in characters so prompt plus generation fit the context window."""
    return max(1024, NUM_CTX - NUM_PREDICT - _MARGIN) * _TOKEN_CHARS


def _extract_c(raw: str) -> str:
    m = _FENCE.search(raw)
    return m.group("body").strip() if m else raw.strip()


class Sample(NamedTuple):
    text: str               # extracted func0 C source
    in_tokens: int          # prompt tokens the server processed
    out_tokens: int         # tokens generated
    done_reason: str        # stop, or length when the output hit num_predict


def sample(prompt: str, *, model: str, endpoint: str, temperature: float) -> Sample:
    """Return one candidate C source for the prompt, with the server's token counts."""
    import requests

    payload = {
        "model": model.removeprefix("ollama/"),
        "prompt": prompt,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        "stream": False,
        "think": os.getenv("OLLAMA_REASONING_EFFORT", "none") not in ("none", "", None),
    }
    resp = requests.post(f"{endpoint.rstrip('/')}/api/generate", json=payload,
                         timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    done = body.get("done_reason") or ""
    if done == "length":
        logger.warning("candidate hit num_predict=%d (output truncated)", NUM_PREDICT)
    return Sample(_extract_c(body.get("response") or ""),
                  int(body.get("prompt_eval_count") or 0),
                  int(body.get("eval_count") or 0), done)
