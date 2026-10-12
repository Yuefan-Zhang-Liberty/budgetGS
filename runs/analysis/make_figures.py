"""Make the three Week-1 reproducibility figures for BudgetGS.

1. fig1_bytes_n.png  - bytes vs N, 3 scenes, ledger line + real-encode points
2. fig2_landing.png  - landing table bar: landed bytes per scene vs 10 MB budget
3. fig3_psnr_n.png   - val PSNR vs N, 3 scenes, 30 points each (training baseline)

Data sources:
  k1-all.json   (runs/analysis, from budgetgs.evaluation.k1)
  runs.jsonl    (runs/{scene}/runs.jsonl)
  landing fit   (parsed from fit logs, budgetgs fit)
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs"
ANALYSIS = os.path.join(RUNS, "analysis")
FIGDIR = os.path.join(ANALYSIS, "figures")
SCENES = ["bicycle", "garden", "room"]
BUDGET_MB = 10.0
BUDGET_BYTES = BUDGET_MB * 1_000_000

# -- palette (matches the Week-1 review deck: deep charcoal / forest green) --
C_MAIN = "#2A2D30"
C_ACC = "#4C7A5A"
C_ACC2 = "#8AA68F"
C_MUTED = "#9BA69D"
C_SCENES = {"bicycle": "#4C7A5A", "garden": "#2A2D30", "room": "#A06A3D"}


def load_runs(scene: str) -> list[dict]:
    with open(os.path.join(RUNS, scene, "runs.jsonl"), encoding="utf-8") as f:
        row = json.loads(f.readline())
    return row["checkpoints"]


def load_k1() -> dict | None:
    path = os.path.join(ANALYSIS, "k1-all.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {r["ckpt_dir"].split("/")[1]: r for r in json.load(f)}


def fig1_bytes_n(k1: dict | None) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=150)
    if k1 is not None:
        for scene in SCENES:
            rep = k1.get(scene)
            if rep is None:
                continue
            f = rep["fit"]
            xs = np.asarray(sorted(r["n_gaussians"] for r in load_runs(scene)))
            ys_pred = f["slope"] * xs + f["intercept"]
            color = C_SCENES[scene]
            ax.plot(xs, ys_pred / 1e6, "-", color=color, lw=1.6,
                    label=f"{scene}: {f['slope']:.2f}B/N + {f['intercept']/1e6:.2f}MB, R\u00b2={f['r2']:.4f}")
            for s in rep["samples"]:
                ax.plot(s["n_gaussians"], s["actual"] / 1e6, "o", ms=4.5,
                        color=color, mfc="none", mew=1.2)
    else:
        ax.text(0.5, 0.5, "k1-all.json not ready", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=C_MUTED)
    ax.axhline(BUDGET_MB, color=C_MAIN, ls="--", lw=1.0, alpha=0.55)
    ax.text(0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]) + ax.get_xlim()[0],
            BUDGET_MB * 1.008, "10 MB budget", fontsize=8, color=C_MAIN, alpha=0.7)
    ax.set_xlabel("Gaussian count N")
    ax.set_ylabel("Serialized bytes (MB)")
    ax.set_title("Bytes vs N: ledger line and real-encode points", fontsize=11, color=C_MAIN)
    if k1 is not None:
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=C_MUTED, lw=0.4, alpha=0.35)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig1_bytes_n.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def load_fit_logs() -> dict:
    """Re-run budgetgs fit is preferred; fall back to logs if present."""
    out = {}
    for scene in SCENES:
        log = os.path.join(ANALYSIS, f"fit-{scene}.log")
        if os.path.exists(log):
            text = open(log, encoding="utf-8").read()
            m = re.search(r"levels=(\d+): (\d+) bytes", text)
            if m:
                out[scene] = {"levels": int(m.group(1)), "bytes": int(m.group(2))}
    return out


def fig2_landing(fits: dict) -> str:
    scenes = [s for s in SCENES if s in fits]
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=150)
    x = np.arange(len(scenes))
    colors = [C_SCENES[s] for s in scenes]
    bars = ax.bar(x, [fits[s]["bytes"] / 1e6 for s in scenes], width=0.52,
                  color=colors, alpha=0.85)
    ax.axhline(BUDGET_MB, color=C_MAIN, ls="--", lw=1.0)
    ax.text(len(scenes) - 0.38, BUDGET_MB * 1.012, "10 MB budget",
            fontsize=8, color=C_MAIN, alpha=0.7, ha="right")
    for xi, s in zip(x, scenes):
        b = fits[s]["bytes"]
        err = (b - BUDGET_BYTES) / BUDGET_BYTES * 100.0
        ax.text(xi, b / 1e6 + 0.25, f"levels={fits[s]['levels']}\n{b/1e6:.2f} MB ({err:+.1f}%)",
                ha="center", fontsize=8, color=C_MAIN)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=10)
    ax.set_ylabel("Landed bytes (MB)")
    ax.set_title("Landing table: closest block-precision step per scene", fontsize=11, color=C_MAIN)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=C_MUTED, lw=0.4, alpha=0.35)
    ax.set_ylim(0, max(fits[s]["bytes"] for s in scenes) / 1e6 * 1.22)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2_landing.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_psnr_n() -> str:
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=150)
    for scene in SCENES:
        cks = load_runs(scene)
        ns = [c["n_gaussians"] for c in cks]
        ps = [c["val_psnr"] for c in cks]
        ax.plot(ns, ps, "-o", ms=3, lw=1.2, color=C_SCENES[scene], label=scene)
    ax.set_xlabel("Gaussian count N")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title("Training baseline: val PSNR over the run", fontsize=11, color=C_MAIN)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=C_MUTED, lw=0.4, alpha=0.35)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig3_psnr_n.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    k1 = load_k1()
    fits = load_fit_logs()
    print("fig1:", fig1_bytes_n(k1))
    print("fig2:", fig2_landing(fits))
    print("fig3:", fig3_psnr_n())
    print("fit logs found:", {s: fits[s] for s in fits})
    print("k1 ready:", k1 is not None)


if __name__ == "__main__":
    main()
