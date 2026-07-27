#!/usr/bin/env python3
"""Render the five evidence figures from the published CSV."""

from pathlib import Path
import csv
import math
import statistics

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)

with (ROOT / "results.csv").open() as f:
    rows = list(csv.DictReader(f))
for row in rows:
    row["seed"] = int(row["seed"])
    for key in list(row):
        if key not in {"variant", "seed"}:
            row[key] = float(row[key])


def values(variant, metric):
    return [r[metric] for r in rows if r["variant"] == variant]


def mean_ci(xs):
    return statistics.mean(xs), 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


COLORS = ["#34495e", "#5b8ff9", "#61d9a5", "#f6bd16", "#e8684a"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. Headline clean NDCG.
fig, ax = plt.subplots(figsize=(7.2, 4.0))
variants = ["autoregressive", "full_diffusion_3step", "one_step_diffusion"]
labels = ["Autoregressive", "Diffusion · 3 steps", "Diffusion · 1 step"]
stats = [mean_ci(values(v, "ndcg_at_10")) for v in variants]
ax.bar(labels, [x[0] for x in stats], yerr=[x[1] for x in stats], color=COLORS[:3], capsize=5)
ax.set_ylabel("NDCG@10 (full catalog)")
ax.set_title("One-pass denoising gives the clearest matched-budget gain")
ax.grid(axis="y", alpha=0.2)
for i, (mean, _) in enumerate(stats):
    ax.text(i, mean + 0.0007, f"{mean:.4f}", ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "headline_ndcg.png", dpi=180)
plt.close(fig)

# 2. Sampling depth.
fig, ax = plt.subplots(figsize=(7.2, 4.0))
step_variants = ["one_step_diffusion", "full_diffusion_3step", "five_step_diffusion"]
steps = [1, 3, 5]
stats = [mean_ci(values(v, "ndcg_at_10")) for v in step_variants]
ax.errorbar(steps, [x[0] for x in stats], yerr=[x[1] for x in stats], marker="o", lw=2.5, capsize=5, color=COLORS[1])
ax.set_xticks(steps)
ax.set_xlabel("Denoising passes")
ax.set_ylabel("NDCG@10")
ax.set_title("Extra refinement hurts this compact reconstruction")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "sampling_steps.png", dpi=180)
plt.close(fig)

# 3. Paired mechanism effects.
fig, ax = plt.subplots(figsize=(7.2, 4.2))
full = {r["seed"]: r["ndcg_at_10"] for r in rows if r["variant"] == "full_diffusion_3step"}
ablations = [
    ("Collaborative\ntokens", "no_collaborative_tokens"),
    ("Progressive\ncurriculum", "no_progressive_curriculum"),
    ("Stability\nvoting", "no_stability_voting"),
]
for x, (label, variant) in enumerate(ablations):
    ablated = {r["seed"]: r["ndcg_at_10"] for r in rows if r["variant"] == variant}
    diffs = [full[s] - ablated[s] for s in sorted(full)]
    ax.scatter([x - 0.12, x - 0.04, x + 0.04, x + 0.12], diffs, color=COLORS[x + 1], s=42)
    ax.scatter([x], [statistics.mean(diffs)], marker="D", color="black", s=48, zorder=4)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(range(3), [x[0] for x in ablations])
ax.set_ylabel("Paired Δ NDCG@10 (full − ablated)")
ax.set_title("Only tiny or seed-sensitive mechanism gains")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "mechanism_effects.png", dpi=180)
plt.close(fig)

# 4. Forced early-decoding error.
fig, ax = plt.subplots(figsize=(7.2, 4.0))
variants = ["autoregressive", "full_diffusion_3step", "five_step_diffusion"]
labels = ["Autoregressive", "Diffusion · 3", "Diffusion · 5"]
stats = [mean_ci(values(v, "downstream_retention")) for v in variants]
ax.bar(labels, [x[0] for x in stats], yerr=[x[1] for x in stats], color=[COLORS[0], COLORS[1], COLORS[3]], capsize=5)
ax.axhline(1, color="black", lw=1, ls="--")
ax.set_ylim(0.92, 1.10)
ax.set_ylabel("Downstream NDCG retention")
ax.set_title("The forced first error does not separate the decoders")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "early_error.png", dpi=180)
plt.close(fig)

# 5. Stochastic recommendation consistency.
fig, ax = plt.subplots(figsize=(7.2, 4.0))
variants = ["autoregressive", "full_diffusion_3step", "no_stability_voting"]
labels = ["Autoregressive", "Full diffusion", "Diffusion · no vote"]
stats = [mean_ci(values(v, "top10_jaccard")) for v in variants]
ax.bar(labels, [x[0] for x in stats], yerr=[x[1] for x in stats], color=[COLORS[0], COLORS[1], COLORS[4]], capsize=5)
ax.set_ylim(0, 0.75)
ax.set_ylabel("Top-10 Jaccard across noisy decodes")
ax.set_title("Voting does not improve stochastic consistency")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "consistency.png", dpi=180)
plt.close(fig)
