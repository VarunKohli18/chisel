"""Run configuration. One dataclass holds every knob, loaded from a yaml file and
overridden by CLI flags. A configuration is an (oracle, features) pair. ARMS is the registry
of named configurations used to label result rows."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

ORACLES = ("compiler", "fuzzer")
# Composable capabilities. Each toggles one thing. Harness features shape the differential
# harness and seed corpus, loop features shape the refinement loop.
#   typed_decode      materialize typed arguments from the input bytes, else slice raw bytes
#   typed_seeds       add the curated boundary/edge seed corpus, else libfuzzer-mined inputs only
#   rich_observables  also compare output buffers, else the return value only
#   address_free      compare pointer results by content, else by raw address
#   contract_gate     judge only inputs on which the original is well-behaved
#   orig_dict         guide the miner with constants disassembled from the original binary
#   memory            carry divergence-inducing inputs across rounds
#   keep_best         on exhaustion keep the fewest-divergence candidate, not the last
FEATURES = ("typed_decode", "typed_seeds", "rich_observables", "address_free", "contract_gate",
            "orig_dict", "memory", "keep_best")

# Hard dependencies between features. load() auto-adds the prerequisite with a warning rather
# than reject the run.
FEATURE_REQUIRES = {
    "rich_observables": ("typed_decode",),   # output buffers exist only under typed decode
    "typed_seeds":      ("typed_decode",),   # curated seeds mirror the typed byte layout
}
# Every feature is inert without the fuzzer oracle, they all shape the differential check.
_FUZZER_FEATURES = FEATURES

_ALL_ORACLES = ["compiler", "fuzzer"]

# Named configurations, each adds one capability. arm_label maps an exact match back to its name.
_OBSERVE = ["typed_decode", "typed_seeds", "rich_observables", "address_free", "contract_gate",
            "orig_dict"]
ARMS = {
    "compiler":  dict(oracle=["compiler"], features=[]),
    "fuzzer":    dict(oracle=list(_ALL_ORACLES), features=[]),
    "observe":   dict(oracle=list(_ALL_ORACLES), features=list(_OBSERVE)),
    "memory":    dict(oracle=list(_ALL_ORACLES), features=_OBSERVE + ["memory"]),
    "retain_best": dict(oracle=list(_ALL_ORACLES), features=_OBSERVE + ["memory", "keep_best"]),
}


def normalize(oracle: list, features: list) -> list:
    """Resolve feature dependencies. Auto-add any missing prerequisite and warn when a feature is
    inert because the fuzzer oracle is off. Returns the completed feature list in FEATURES order
    so arm_label's exact match still round-trips."""
    have = set(features)
    for feat, reqs in FEATURE_REQUIRES.items():
        if feat in have:
            for req in reqs:
                if req not in have:
                    warnings.warn(f"feature {feat!r} requires {req!r}, enabling it")
                    have.add(req)
    if "fuzzer" not in oracle:
        inert = [f for f in _FUZZER_FEATURES if f in have]
        if inert:
            warnings.warn(f"features {inert} are inert without the 'fuzzer' oracle")
    return [f for f in FEATURES if f in have]


@dataclass
class Config:
    benchmark: str = "exebench_hard"
    opt: str = "O0"                       # O0 through O3
    variant: str = "stripall"             # stripall or unstripped
    oracle: list = field(default_factory=lambda: list(ORACLES))
    features: list = field(default_factory=lambda: list(FEATURES))
    budget: int = 2000                    # libfuzzer entropic runs per candidate per round
    diff_seeds: int = 200                 # signature-typed inputs in the differential corpus
    max_divergences: int = 25             # divergences collected per round before the scan stops
    max_compiler_diags: int = 5           # compiler diagnostics shown as feedback
    max_feedback_divergences: int = 10    # counterexamples shown as feedback
    iters: int = 5                        # K refinement rounds
    temperature: float = 0.0
    prompt: str = "detailed"              # instruction template, simple or detailed
    model: str = "gemma4:31b"
    endpoint: str = "http://127.0.0.1:11434"
    per_sample_timeout: float = 2.0       # cpu seconds per execution, see fuzz._run
    compile_mode: str = "object"          # object for gcc -c, link for an A4D-style standalone link
    # Candidates are built at this level regardless of the cell's opt. The reference object is
    # always O0, and the comparison is only meaningful if both sides get undefined behaviour
    # optimised the same way.
    candidate_opt: str = "O0"
    replay_seeds: bool = True             # replay the curated typed seeds directly, when off they
                                          # only steer mining and are not tested verbatim

    def arm_label(self) -> str:
        """A stable label identifying this config's arm."""
        for name, preset in ARMS.items():
            if (sorted(self.oracle) == sorted(preset["oracle"])
                    and sorted(self.features) == sorted(preset["features"])):
                return name
        oc = "+".join(o for o in ORACLES if o in self.oracle) or "llm"
        ft = ",".join(f for f in FEATURES if f in self.features) or "none"
        return f"{oc}[{ft}]"


def load(config_path: str | None, overrides: dict) -> Config:
    """Build a Config from a yaml file then apply non-None CLI overrides. An --arm
    override expands first."""
    base = {}
    if config_path and Path(config_path).exists():
        base = yaml.safe_load(Path(config_path).read_text()) or {}
    cfg = Config(**{k: v for k, v in base.items() if k in Config.__dataclass_fields__})

    arm = overrides.pop("arm", None)
    if arm:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}, choose from {list(ARMS)}")
        cfg = replace(cfg, **ARMS[arm])

    clean = {k: v for k, v in overrides.items()
             if v is not None and k in Config.__dataclass_fields__}
    cfg = replace(cfg, **clean)

    for o in cfg.oracle:
        if o not in ORACLES:
            raise SystemExit(f"unknown oracle {o!r}, allowed {ORACLES}")
    for f in cfg.features:
        if f not in FEATURES:
            raise SystemExit(f"unknown feature {f!r}, allowed {FEATURES}")
    # validate first so an unknown feature still errors, then resolve dependencies
    cfg = replace(cfg, features=normalize(cfg.oracle, cfg.features))
    return cfg


def _csv(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def add_run_args(ap: argparse.ArgumentParser) -> None:
    """Register the shared run flags on a subparser. Every flag defaults to None so it overrides
    the config only when set."""
    ap.add_argument("--config", default=None, help="yaml config; CLI flags override it")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--opt", default=None)
    ap.add_argument("--variant", default=None, choices=["stripall", "unstripped"])
    ap.add_argument("--oracle", type=_csv, default=None,
                    help=f"comma list subset of {','.join(ORACLES)}, empty means raw llm")
    ap.add_argument("--feature", dest="features", type=_csv, default=None,
                    help=f"comma list subset of {','.join(FEATURES)}")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--diff-seeds", dest="diff_seeds", type=int, default=None,
                    help="signature-typed inputs per differential round")
    ap.add_argument("--max-divergences", dest="max_divergences", type=int, default=None,
                    help="divergences collected per round before the scan stops")
    ap.add_argument("--max-compiler-diags", dest="max_compiler_diags", type=int, default=None,
                    help="compiler diagnostics shown as feedback")
    ap.add_argument("--max-feedback-divergences", dest="max_feedback_divergences", type=int,
                    default=None, help="counterexamples shown as feedback")
    ap.add_argument("--candidate-opt", dest="candidate_opt", default=None,
                    choices=["O0", "O1", "O2", "O3"],
                    help="optimisation level candidates are built at, default O0 to match the "
                         "reference object")
    ap.add_argument("--compile-mode", dest="compile_mode", default=None,
                    choices=["object", "link"],
                    help="object for gcc -c, link for an A4D-style standalone link")
    ap.add_argument("--no-replay-seeds", dest="replay_seeds", action="store_const", const=False,
                    default=None,
                    help="use the curated typed seeds only to steer mining, never test them "
                         "verbatim")
    ap.add_argument("--per-sample-timeout", dest="per_sample_timeout", type=float, default=None,
                    help="cpu seconds per execution, at least 1")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--prompt", choices=["simple", "detailed"], default=None,
                    help="instruction template, simple rewrites cleanly, detailed preserves "
                         "semantics and drops decompiler scaffolding")
    ap.add_argument("--model", default=None)
    ap.add_argument("--endpoint", default=None)


def from_args(args: argparse.Namespace) -> Config:
    keys = ["benchmark", "opt", "variant", "oracle", "features",
            "budget", "diff_seeds", "max_divergences", "max_compiler_diags",
            "max_feedback_divergences", "per_sample_timeout", "iters", "temperature",
            "prompt", "model", "endpoint", "candidate_opt", "compile_mode", "replay_seeds"]
    return load(args.config, {k: getattr(args, k, None) for k in keys})
