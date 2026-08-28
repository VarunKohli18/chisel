#!/usr/bin/env python3
"""Generate the paper figures from the run JSONs.

Usage: figures.py [runs_dir] [fig_dir]
Defaults: results/runs_latest -> paper/artifacts_latest/figures
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "runs_latest"
FIG = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper" / "artifacts_latest" / "figures"


def ladder_grid():
    """3x2 grid across the six-arm ladder."""
    M = json.loads((RUNS / "_metrics.json").read_text())["overall"]
    ARMS = ["llm", "compiler", "fuzzer", "observe", "memory", "retain_best"]
    LABELS = ["LLM", "+compiler", "+fuzzer", "+observe", "+memory", "+best (full)"]
    # (metrics key, y-axis label, is_percent, y-limits or None)
    PANELS = [
        ("RE", "RE (%)", True, (50, 90)),
        ("iters", "$\\bar{k}$", False, None),
        ("R_tot", "$R_{tot}$ (%)", True, None),
        ("R_exec", "$R_{exec}$ (%)", True, None),
        ("FA", "FA (%)", True, None),
        ("FR", "FR (%)", True, None),
    ]
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    })
    fig, axes = plt.subplots(3, 2, figsize=(3.33, 4.35))
    x = list(range(len(ARMS)))
    bar_c, hi_c, line_c = "#4C78A8", "#F58518", "#222222"
    colors = [bar_c] * (len(ARMS) - 1) + [hi_c]

    for ax, (key, ylabel, pct, ylim) in zip(axes.flat, PANELS):
        vals = [M[a][key] * (100 if pct else 1) for a in ARMS]
        ax.bar(x, vals, color=colors, width=0.62, zorder=2)
        ax.plot(x, vals, ":", color=line_c, marker="o", markersize=2.4, linewidth=0.9, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=40, ha="right")
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            top = max(vals) * 1.28 if max(vals) > 0 else 1.0   # headroom for value labels
            ax.set_ylim(0, top)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.tick_params(length=2, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lo, hi = ax.get_ylim()
        dec = 1 if (not pct or max(vals) < 10) else 0   # small-valued panels need a decimal
        for xi, v in zip(x, vals):
            inside = v > hi - 0.12 * (hi - lo)          # keep label on-panel near the top
            ax.annotate(f"{v:.{dec}f}", (xi, min(v, hi)),
                        textcoords="offset points", xytext=(0, -5 if inside else 1.4),
                        ha="center", va="top" if inside else "bottom", fontsize=5.2)

    fig.tight_layout(pad=0.4, h_pad=0.7, w_pad=0.8)
    fig.savefig(FIG / "ladder_grid.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIG / "ladder_grid.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", FIG / "ladder_grid.pdf", "and .png")


def convergence():
    """Left: Pass@k for observe vs memory. Right: acceptances per iteration."""
    pk = json.loads((RUNS / "_iter_passrate.json").read_text())
    rb = []
    for f in glob.glob(str(RUNS / "retain_best" / "*.json")):
        rb += json.loads(Path(f).read_text())
    acc = [sum(1 for x in rb if x["accepted"] and x["iters"] == k) for k in range(1, 6)]
    exh = sum(1 for x in rb if not x["accepted"])

    COL = {"observe": "#9467bd", "memory": "#e7298a"}
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7.5,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.0})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.33, 1.55))
    x = [1, 2, 3, 4, 5]

    for arm in ["observe", "memory"]:
        a1.plot(x, pk[arm]["mean_passrate@k"], marker="o", ms=2.6, lw=1.2,
                color=COL[arm], label={"observe": "+observe", "memory": "+memory"}[arm])
    a1.set_xlabel("Iteration ($k$)"); a1.set_ylabel("Pass@$k$ (%)"); a1.set_xticks(x)
    a1.grid(True, alpha=0.25, lw=0.4)
    for s in ("top", "right"): a1.spines[s].set_visible(False)
    a1.legend(frameon=False, loc="upper right", handlelength=1.0, labelspacing=0.25, borderpad=0.1)

    heights = acc + [exh]
    colors = ["#2A9D8F"] * 5 + ["#bbbbbb"]
    xs = list(range(6))
    a2.bar(xs, heights, color=colors, width=0.72)
    a2.set_xticks(xs); a2.set_xticklabels(["1", "2", "3", "4", "5", "none"])
    a2.set_xlabel("Accepted at iteration"); a2.set_ylabel("# samples")
    a2.set_ylim(0, max(heights) * 1.15)
    a2.grid(True, axis="y", alpha=0.25, lw=0.4)
    for s in ("top", "right"): a2.spines[s].set_visible(False)
    for xi, h in zip(xs, heights):
        a2.annotate(str(h), (xi, h), textcoords="offset points", xytext=(0, 1.2), ha="center", fontsize=5.4)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(FIG / "convergence.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIG / "convergence.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", FIG / "convergence.pdf", "and .png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    ladder_grid()
    convergence()
