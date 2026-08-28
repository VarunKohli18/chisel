"""Unit tests for signature parsing, the differential harness, the seed corpus, the scorer, oracle feedback formatting, and prompt assembly."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from decomp import config, fuzz, harness, oracles, prompt, score, signature

HAVE_GCC = shutil.which("gcc") is not None


# ---- signature ----

def test_parse_scalar_and_pointer():
    sig = signature.parse_signature("int func0(int x, char *s)")
    assert sig.return_type == "int" and not sig.is_void_return
    assert [(a.c_type, a.is_array, a.is_string) for a in sig.args] == \
        [("int", False, False), ("char*", False, True)]


def test_parse_void_and_array():
    sig = signature.parse_signature("void func0(int *r, int n)")
    assert sig.is_void_return
    assert sig.args[0].is_array and sig.args[0].c_type == "int*"
    assert not sig.args[1].is_array


def test_parse_kr_star_on_name():
    sig = signature.parse_signature("char *func0(int*o)")
    assert sig.return_type == "char*"      # norm_ctype: canonical star placement
    assert sig.args[0].c_type == "int*" and sig.args[0].name == "o"


def test_parse_none():
    assert signature.parse_signature("not a function") is None


# ---- seeds ----

def test_seed_corpus_deterministic():
    sig = signature.parse_signature("int func0(int x)")
    a = harness.seed_corpus(sig, n=50, seed=1)
    b = harness.seed_corpus(sig, n=50, seed=1)
    assert a == b
    assert harness.seed_corpus(sig, n=50, seed=2) != a


def test_decode_call_roundtrips():
    sig = signature.parse_signature("int func0(int x)")
    seeds = harness.seed_corpus(sig, n=5, seed=3)
    assert harness.decode_call(sig, seeds[0]).startswith("func0(x=")


# ---- harness compiles and runs (all feature toggles) ----

@pytest.mark.skipif(not HAVE_GCC, reason="gcc required")
@pytest.mark.parametrize("feats", [
    {"typed_decode", "rich_observables", "address_free"},
    set(),
    {"typed_decode"},
])
def test_harness_links_and_runs(feats):
    sig = signature.parse_signature("void func0(int *r, int n)")
    h = harness.render_harness(sig, typed_decode="typed_decode" in feats,
                               rich_observables="rich_observables" in feats,
                               address_free="address_free" in feats)
    d = Path(tempfile.mkdtemp())
    (d / "h.c").write_text(h)
    (d / "f.c").write_text("void func0(int *r,int n){for(int i=0;i<n&&i<64;i++)r[i]=i;}")
    subprocess.run(["gcc", "-c", str(d / "f.c"), "-o", str(d / "f.o")], check=True)
    r = subprocess.run(["gcc", str(d / "h.c"), str(d / "f.o"), "-O0", "-w", "-lm",
                        "-o", str(d / "a")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---- differential run catches a wrong candidate, passes a correct one ----

@pytest.mark.skipif(not HAVE_GCC, reason="gcc required")
def test_differential_run():
    d = Path(tempfile.mkdtemp())
    sig = signature.parse_signature("int func0(int x)")
    for name, body in [("o", "x*x+1"), ("good", "x*x+1"), ("bad", "x*x")]:
        (d / f"{name}.c").write_text(f"int func0(int x){{return {body};}}")
        subprocess.run(["gcc", "-c", str(d / f"{name}.c"), "-O0", "-o", str(d / f"{name}.o")], check=True)
    feats = {"typed_decode", "rich_observables", "address_free"}
    seeds = harness.seed_corpus(sig, n=80, seed=7)
    good, _ = fuzz.differential_run(sig, d / "o.o", d / "good.o", "int func0(int x){return x*x+1;}",
                                    features=feats, seeds=seeds, timeout=2.0)
    bad, _ = fuzz.differential_run(sig, d / "o.o", d / "bad.o", "int func0(int x){return x*x;}",
                                   features=feats, seeds=seeds, timeout=2.0)
    assert good == [] and len(bad) > 0
    assert "func0(x=" in bad[0].call


# ---- scorer ----

@pytest.fixture
def sq_record():
    return {"kind": "real", "fname": "sq", "func_def": "int sq(int x){return x*x+1;}", "deps": "",
            "iospec": {"funname": "sq", "funargs": ["x"], "livein": ["x"], "liveout": [],
                       "returnvarname": ["ret"], "typemap": {"x": "int32", "ret": "int32"}},
            "io_pairs": []}


@pytest.mark.skipif(not HAVE_GCC, reason="gcc required")
def test_score_correct_and_wrong(sq_record):
    pairs = [{"input": {"x": 3}}, {"input": {"x": -5}}, {"input": {"x": 0}}]
    good = score.score(sq_record, "int func0(int x){return x*x+1;}", pairs)
    bad = score.score(sq_record, "int func0(int x){return x*x;}", pairs)
    assert good.reexec and good.pass_rate == 1.0 and good.n_pairs == 3
    assert not bad.reexec and bad.pass_rate == 0.0


# ---- oracle feedback formatting ----

@pytest.mark.skipif(not HAVE_GCC, reason="gcc required")
def test_compile_feedback():
    obj, fb = oracles.compile_candidate("int func0(int x){ return", "O0", Path(tempfile.mkdtemp()))
    assert obj is None and fb.startswith("Compilation failed:") and fb.endswith("Fix it.")


@pytest.mark.skipif(not HAVE_GCC, reason="gcc required")
def test_compile_success():
    obj, fb = oracles.compile_candidate("int func0(int x){return x;}", "O2", Path(tempfile.mkdtemp()))
    assert obj is not None and fb is None


# ---- prompt assembly ----

def test_prompt_first_round_minimal():
    p = prompt.build_prompt("int func0(int x){...}")
    assert "Ghidra pseudo C" in p and "Previous attempt" not in p and "func0" in p


def test_prompt_keeps_feedback_whole_under_tight_budget():
    feedback = "DISTINCT_TOKEN " * 50
    p = prompt.build_prompt("X" * 100000, prev_source="Y" * 100000,
                            feedback=feedback, max_chars=4000)
    assert feedback.strip() in p           # feedback never truncated
    assert len(p) <= 4000 + len(feedback)


# ---- config arms ----

def test_arm_presets():
    assert "memory" not in config.load(None, {"arm": "fuzzer"}).features
    assert "memory" not in config.load(None, {"arm": "observe"}).features
    assert "memory" in config.load(None, {"arm": "memory"}).features
    assert "keep_best" in config.load(None, {"arm": "retain_best"}).features
    assert "keep_best" not in config.load(None, {"arm": "memory"}).features
    c = config.load(None, {"arm": "compiler"})
    assert c.oracle == ["compiler"] and c.arm_label() == "compiler"
    for name in config.ARMS:
        assert config.load(None, {"arm": name}).arm_label() == name
    # observe carries the split typed-input features plus the original-constant dictionary
    assert set(config.load(None, {"arm": "observe"}).features) == {
        "typed_decode", "typed_seeds", "rich_observables", "address_free", "contract_gate",
        "orig_dict"}
    # orig_dict is a fuzzer-mining feature: on from observe up, off in the bare fuzzer arm
    assert "orig_dict" not in config.load(None, {"arm": "fuzzer"}).features
    assert "orig_dict" in config.load(None, {"arm": "retain_best"}).features


def test_feature_dependencies_and_clean_break():
    # a feature with an unmet prerequisite is auto-completed (with a warning)
    for feat in ("rich_observables", "typed_seeds"):
        feats = config.load(None, {"oracle": ["compiler", "fuzzer"], "features": [feat]}).features
        assert "typed_decode" in feats and feat in feats
    # retain_best minus typed_seeds == the diff-seeds-0 ablation, and stays valid
    ablation = config.load(None, {"oracle": ["compiler", "fuzzer"],
                                  "features": ["typed_decode", "rich_observables", "address_free",
                                               "contract_gate", "memory", "keep_best"]})
    assert "typed_seeds" not in ablation.features
    # clean break: the old overloaded flag is gone
    with pytest.raises(SystemExit):
        config.load(None, {"features": ["typed_inputs"]})
