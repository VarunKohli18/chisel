# CHISEL

**C**ontrolled **H**euristics for **I**terative **S**emantic **E**xtraction and **L**ifting of
decompiler pseudocode — without test suites.

Lift an x86_64 binary to pseudo-C with Ghidra, then let an LLM refine it into compilable,
behaviorally-equivalent C through a feedback loop. Each round an **oracle** runs the candidate,
finds how it diverges from the original, and hands the LLM a short counterexample. The loop is
stateless and test-suite-free: it sees only the pseudo-C, the original compiled to a reference
binary, and inputs it generates itself. The dataset's I/O pairs are held out and used only to
score re-executability offline.

```
binary --Ghidra--> pseudo-C --LLM--> candidate --compile--> reference vs candidate
                        ^                                          |
                        |________ short counterexample feedback ___|  (K rounds)
```

## The ablation ladder

The arm *is* the config: an oracle set and a feature set. There are no per-arm code paths — each
arm adds one capability over the previous. The LLM one-shot row is derived from the compiler arm's
first generation (no feedback), not a separate run.

| arm           | what it adds                                                        | feature set |
|---------------|---------------------------------------------------------------------|-------------|
| *LLM one-shot*| raw model output, no oracle (derived from iter 1)                   | — |
| `compiler`    | it compiles (gcc oracle)                                            | — |
| `fuzzer`      | + matches the original under differential fuzzing                   | — |
| `observe`     | + typed in-contract inputs, dump output buffers, compare pointers by content | `typed_decode, typed_seeds, rich_observables, address_free, contract_gate` |
| `memory`      | + accumulate the counterexample corpus and re-test it every round   | + `memory` |
| `retain_best` | + return the best candidate the oracle saw, not the last (full CHISEL) | + `keep_best` |

- Harness features shape the differential harness: `typed_decode` materializes typed arguments
  (vs. raw bytes) and is the prerequisite for content observation, `typed_seeds` adds the curated
  boundary/edge seed corpus (vs. libFuzzer-mined inputs only), `rich_observables` dumps output
  buffers (vs. return value only), `address_free` compares pointers by content (vs. raw address),
  `contract_gate` restricts fuzzing to in-contract inputs.
- Loop features: `memory` re-tests the accumulated counterexample corpus every round; `keep_best`
  returns the best candidate by the oracle's own signal instead of the last attempt.
- Dependencies are resolved automatically: `rich_observables` and `typed_seeds` both require
  `typed_decode`, so listing either one auto-enables it (with a warning). This is why the features
  compose freely — e.g. `retain_best` minus `typed_seeds` is the "no curated seeds" ablation.
- **Sanitizer:** a differential sanitizer is deliberately *not* a loop oracle — it would require
  building the original *source* under ASan/UBSan, which the decompilation setting (only a binary
  is given) does not have.

A configuration is just an `(oracle, features)` pair, chosen with `--oracle` and `--feature`. The
names above are labels: a run whose `(oracle, features)` matches a ladder point is labeled with its
name in the results, so report groupings stay readable. Restrict a run to a function subset with
`--keys-file`.

## Layout

```
decomp/
  config.py      Config, yaml load + CLI merge, the ARMS ladder presets
  signature.py   parse func0's signature from C
  harness.py     differential harness + libFuzzer target + signature-typed seed corpus
  fuzz.py        native differential execution + libFuzzer entropic miner
  oracles.py     compiler + fuzzer oracles -> (passed, feedback)
  llm.py         Ollama client, one sample at temperature T
  prompt.py      prompt assembly; feedback is never truncated
  loop.py        the stateless K-round refinement loop
  select.py      best-candidate retention (keep_best): re-judge compiled candidates on one shared seed set
  score.py       offline re-exec scoring vs the held-out suite (the only place io-pairs are read)
  dataset/       select -> compile -> ghidra -> testsuite, and the build orchestrator
run.py           CLI: data | run | suites
scripts/
  serve_ollama.sh                start one GPU-pinned Ollama server per detected GPU
  run_study_mp.py                process-pool study orchestrator (all arms x opts x variants)
  run_baseline_llm4decompile.py, run_baseline_a4d.py   the two baselines
  derive_retain_best.py          derive the retain_best arm offline from the memory arm
  compute_metrics.py             every numerical result (_metrics.json + pass-rate rescoring)
  figures.py                     the two paper figures (ladder grid, convergence)
config.yaml      default run configuration
study_keys/      exebench_hard_keep120.txt — the 120-function study keyset
data_cache/      the compiled 120-function dataset (binaries, Ghidra cells, held-out suites)
results/         local run output (gitignored)
paper/           artifacts_latest/ (current figures + result JSONs), artifacts/ (previous run, frozen)
```

## Setup

```bash
python3 -m venv .venv          # create a virtualenv (once)
source .venv/bin/activate      # activate it (every shell; deactivate with `deactivate`)
pip install -r requirements.txt
pytest                         # 17 core tests; needs gcc and clang (libFuzzer)
```

Activate the venv in any shell before running the commands below. The scripts under `scripts/`
assume the venv's `python` is on PATH, so activate it there too.

## Usage

```bash
# 1. regenerate the dataset (records, x86_64 binaries O0-O3 x stripped/unstripped, Ghidra pseudo-C, 1000-input test suites). Needs the ghidra_extractor docker image. Shardable.
python run.py data --benchmark exebench_hard            # add --nshards 16 --shard i to parallelize

# 2. serve the model (one Ollama server per detected GPU; NGPUS=n overrides), then export the OLLAMA_ENDPOINTS list it prints
./scripts/serve_ollama.sh

# 3a. run one function step by step and watch the loop (prompt, llm output, feedback, repeat). Defaults are the full configuration; pass --oracle/--feature (or the config.yaml) to change it.
python run.py run --key combain_sorted_array --opt O0 --variant stripall \
    --iters 3 --endpoint http://127.0.0.1:11435 --verbose

# 3b. run one configuration over all functions
python run.py run --opt O0 --variant stripall --out results/runs_latest/retain_best/O0_stripall.json
# a lower ladder point, e.g. the compiler-only arm:
# python run.py run --oracle compiler --feature '' --opt O0 --variant stripall

# 3c. run the whole study (all arms x 4 opts x 2 variants) with the process-pool orchestrator. Export the OLLAMA_ENDPOINTS line serve_ollama.sh printed (one endpoint per GPU) first — a single endpoint on a 1-GPU host works too, the study is just proportionally slower (scale --workers down with it, e.g. ~40 per endpoint).
python scripts/run_study_mp.py --out results/runs_latest --workers 160 --iters 5
# resumable and non-destructive; --only-arms compiler,fuzzer,observe,memory,retain_best to subset

# 4. metrics (ablation table + _metrics.json) and figures for a runs dir
python scripts/compute_metrics.py results/runs_latest
python scripts/figures.py results/runs_latest paper/artifacts_latest/figures

# 5. how many suite cases work per function (runs the original on each suite)
python run.py suites --dir data_cache/exebench_hard
```

Any config field is overridable: `--oracle`, `--feature` (subset of
`typed_decode,typed_seeds,rich_observables,address_free,contract_gate,memory,keep_best`),
`--prompt simple|detailed`, `--budget`, `--diff-seeds` (curated seed count, used when `typed_seeds` is on),
`--per-sample-timeout` (CPU seconds per execution, the main speed knob), `--iters`,
`--temperature`, `--model`, `--endpoint`. Select one function with `--key <substring>`, a subset
with `--keys-file`, and trace with `--verbose`. Serving knobs are environment variables read by
`llm.py`: `OLLAMA_NUM_CTX` (default 24576), `OLLAMA_NUM_PREDICT` (default 8192), `OLLAMA_TIMEOUT`.

### The benchmark: 120 hard functions

The study is **120 functions x the ladder x 4 opts x 2 variants**. `exebench_hard` is our own
curated benchmark, not an official ExeBench split: its functions are drawn from ExeBench's
`real_test` and `valid_real` splits and matched against the committed keyset. The dataset under
`data_cache/exebench_hard` is exactly these 120; the keyset is `study_keys/exebench_hard_keep120.txt`.
They are selected so the loop is sound and nothing is truncated: each has a held-out suite of at
least 100 I/O pairs (enough to score re-execution), and each fits the context and generation budget
(`num_ctx=24576`, `num_predict=8192`) with no truncation of the pseudo-C, feedback, or output — so a
function is never penalized for exceeding the serving window rather than for being wrong.
`run.py suites --dir data_cache/exebench_hard` prints each function's suite size.

## What the oracle feedback looks like

Each round the failing oracle writes one short message that becomes the next prompt's correction. A
compile failure reads `Compilation failed: <first errors>. Fix it.` A fuzzer round that finds **N**
behavioral counterexamples renders a random sample (from the current and remembered divergences) as
concrete decoded calls, each **localized to where the outputs first differ** rather than dumping two
long, mostly-identical streams:

```
Your func0 diverges from the original on 25 input(s). It must reproduce the original on all of them:
On func0(p=[0, 1, 2147483647, -1, ...], q=[...], s=[...]): outputs agree on the first 133 values, then differ at index 133: original `64 0 -2147483648 [255] 1 -562 0` vs candidate `64 0 -2147483648 [1] -562 0 -1` (5 of 192 values differ; original has 192 values, candidate 192).
... (more shown)
(and 15 more)
Fix it.
```

The full divergence count is reported (a score), a sample is shown localized to the first differing
index (actionable detail), and the feedback is never truncated to fit the context window. Crashes
and timeouts fall back to describing both sides.

## Reproducing the paper

The result JSONs and figures behind the paper live in `paper/artifacts_latest/` — `jsons/` holds
one dir per arm and baseline (`compiler/ fuzzer/ observe/ memory/ retain_best/ llm4decompile/
a4d_gemma/`) plus the derived numerical artifacts, `figures/` the two plots. Everything numerical
and every plot is regenerated by exactly two scripts (the paper's tables are transcribed from
their output by the authors; the LaTeX itself is not in this repo):

```bash
python scripts/compute_metrics.py paper/artifacts_latest/jsons
# -> _metrics.json (all per-arm metrics, prints the ablation table)
# -> _llm_passrate.txt, _iter_passrate.json (suite rescoring; reused if present, --rescore to force)
python scripts/figures.py paper/artifacts_latest/jsons paper/artifacts_latest/figures
# -> ladder_grid.pdf/.png, convergence.pdf/.png
```

The architecture diagram is authored by hand and has no generation script.

## Requirements

x86_64 host, gcc and clang (libFuzzer), `setarch` for reproducible runs, Docker with the
`ghidra_extractor` image for lifting, and an Ollama server for the model. The reporting path
(`suites`, `compute_metrics.py`, `figures.py`) works offline against the committed dataset and the
result JSONs — only `run.py data` needs Docker/Ghidra and `run.py run` needs Ollama.
