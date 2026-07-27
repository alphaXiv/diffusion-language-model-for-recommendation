# Reproducing Diffusion Language Models for Recommendation

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/diffusion-language-model-for-recommendation/blob/main/notebooks/dlmrec_reproduction.py)

This repository tests the central claim of [arXiv:2607.21519](https://arxiv.org/abs/2607.21519): can bidirectional masked denoising recover recommendations better than matched left-to-right generation, and do collaborative tokens, progressive corruption, and stability voting explain the gain?

**Assessment: partially reproduced.** On a full-catalog chronological MovieLens-1M task, a one-pass 893,568-parameter denoiser reached Recall@10 **0.03150** and NDCG@10 **0.01528**, versus **0.01665** and **0.00787** for the matched autoregressive model across four seeds. The full three-step reconstruction retained only the NDCG gain (0.00841; +6.8%) and lowered Recall (0.01538; −7.6%); collaborative tokens were modestly positive, but curriculum, voting, and early-error correction were inconclusive.

For scale—not a like-for-like target—the paper reports MovieLens-1M DLMRec Recall@10 **0.0828** and NDCG@10 **0.1113** using LLaDA-8B, learned CAST tokens, text, and a different filtered timepoint split. This reproduction uses a two-layer width-128 Transformer, two-hop graph/k-means codes, reconstructed corruption and voting rules, all 1,000,209 public ratings, and each user’s final three interactions as test targets.

Read the tutorial-style [detailed report](reports/dlmrec-reproduction/report.md), inspect the [self-contained marimo notebook](notebooks/dlmrec_reproduction.py), or open the notebook directly in [Molab](https://molab.marimo.io/github/alphaXiv/diffusion-language-model-for-recommendation/blob/main/notebooks/dlmrec_reproduction.py). The machine-readable measurements are in [results.csv](reports/dlmrec-reproduction/results.csv).

## Compute and provenance

All scientific results are fresh post-cutoff OpenResearch **Kubernetes** runs on **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPUs. Each successful job used two GPUs for two concurrent seeds; peak concurrency was **16 GPUs** and the campaign’s actual end-to-end wall time was **0.1153 hours (6m55s)**. The fixed command below is copied verbatim from `orx exp status`. One shell-quoting scout failed before Python and contributes no evidence; accidental same-seed duplicate launches were consistency checks and are excluded from four-seed aggregates.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Polished public landing page, report, figures, notebook | Not run as an experiment (publication surface) | Presentation only | — |
| [failed root scout](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/matched-autoregressive-control) | Initial matched AR root | `python run_experiment.py` | Shell quoting failed before Python; no evidence | Kubernetes, 2 GPUs, 21 s |
| [AR seeds 0–1](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/autoregressive-control-valid-launcher) + [2–3](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/ar-seed-cohort-2-3) | Matched causal control | `python run_experiment.py` | R@10 0.01665; N@10 0.00787 | Kubernetes, 2 GPUs/job |
| [full diffusion seeds 0–1](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/full-masked-diffusion) + [2–3](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/full-diffusion-seed-cohort-2-3) | Graph tokens + curriculum + 3-step voting | `python run_experiment.py` | R@10 0.01538; N@10 0.00841; mixed | Kubernetes, 2 GPUs/job |
| [no graph tokens](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/ablation-no-collaborative-tokens) + [seeds 2–3](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/no-collaborative-tokens-seeds-2-3) | Ablate two-hop collaborative codes | `python run_experiment.py` | N@10 0.00806; modest 4.2% effect | Kubernetes, 2 GPUs/job |
| [no curriculum](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/ablation-no-progressive-curriculum) + [seeds 2–3](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/no-curriculum-seeds-2-3) | Static rather than progressive corruption | `python run_experiment.py` | N@10 0.00849; seed-sensitive, inconclusive | Kubernetes, 2 GPUs/job |
| [no voting](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/ablation-no-stability-voting) + [seeds 2–3](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/no-voting-seeds-2-3) | Final logits instead of trajectory voting | `python run_experiment.py` | N@10 0.00833; no consistency gain | Kubernetes, 2 GPUs/job |
| [one step](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/one-step-denoising) + [seeds 2–3](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/one-step-seeds-2-3) | Fixed-compute one-pass denoising | `python run_experiment.py` | Best: R@10 0.03150; N@10 0.01528 | Kubernetes, 2 GPUs/job |
| [five steps](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/five-step-denoising) + [seeds 2–3](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/five-step-seeds-2-3) | Extra iterative refinement | `python run_experiment.py` | N@10 0.00845; no benefit | Kubernetes, 2 GPUs/job |
| [plain diffusion](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/plain-diffusion-control) + [seeds 2–3](https://github.com/alphaXiv/diffusion-language-model-for-recommendation/tree/orx/plain-diffusion-seeds-2-3) | Remove all three tailored mechanisms | `python run_experiment.py` | N@10 0.00854; mechanisms not jointly additive | Kubernetes, 2 GPUs/job |

## Reproduce or explore

Formal jobs clone the selected branch and run:

```bash
python run_experiment.py
```

The Kubernetes shape is committed in `.orx/k8s.yaml`. To regenerate the public figures locally:

```bash
python -m pip install -r requirements-report.txt
python reports/dlmrec-reproduction/make_figures.py
marimo check notebooks/dlmrec_reproduction.py
marimo edit notebooks/dlmrec_reproduction.py
```

MovieLens data is downloaded directly from GroupLens at runtime and is not redistributed in this repository.
