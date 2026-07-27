#!/usr/bin/env python3
"""Compact MovieLens-1M AR vs masked-diffusion recommendation reproduction.

The implementation deliberately uses only PyTorch and the Python standard
library.  Each Kubernetes job allocates two GPUs and runs one independent seed
per GPU.  The parent process prints a compact terminal JSON record containing
all evidence needed for cross-experiment analysis.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path("/tmp/dlmrec_movielens")
MASK = 1
PAD = 0


@dataclass
class Split:
    train: list[list[int]]
    test: list[list[int]]
    n_items: int
    n_interactions: int


def ensure_data() -> Path:
    ratings = DATA_DIR / "ml-1m" / "ratings.dat"
    if ratings.exists():
        return ratings
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "ml-1m.zip"
    if not archive.exists():
        print(f"DATASET_DOWNLOAD url={DATA_URL}", flush=True)
        urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)
    return ratings


def load_split(path: Path, target_len: int) -> Split:
    raw: dict[int, list[tuple[int, int]]] = {}
    item_ids: set[int] = set()
    with path.open("r", encoding="latin-1") as f:
        for line in f:
            user, item, _rating, timestamp = map(int, line.strip().split("::"))
            raw.setdefault(user, []).append((timestamp, item))
            item_ids.add(item)
    item_map = {item: i + 2 for i, item in enumerate(sorted(item_ids))}
    train, test = [], []
    for user in sorted(raw):
        seq = [item_map[item] for _, item in sorted(raw[user])]
        if len(seq) < target_len + 5:
            continue
        train.append(seq[:-target_len])
        test.append(seq[-target_len:])
    return Split(train, test, len(item_map) + 2, sum(map(len, raw.values())))


def mean_aggregate(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(size, values.shape[1], device=values.device)
    counts = torch.zeros(size, 1, device=values.device)
    out.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones(index.numel(), 1, device=values.device))
    return out / counts.clamp_min(1)


@torch.no_grad()
def kmeans(x: torch.Tensor, k: int, seed: int, iterations: int = 20) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    centers = x[torch.randperm(x.shape[0], generator=generator, device=x.device)[:k]].clone()
    for _ in range(iterations):
        labels = torch.cdist(x, centers).argmin(1)
        new = mean_aggregate(x, labels, k)
        empty = new.norm(dim=1) == 0
        if empty.any():
            new[empty] = x[torch.randperm(x.shape[0], generator=generator, device=x.device)[: int(empty.sum())]]
        centers = F.normalize(new, dim=1)
    return torch.cdist(x, centers).argmin(1)


@torch.no_grad()
def collaborative_codes(split: Split, device: torch.device, seed: int, codebook: int = 64) -> torch.Tensor:
    """Two-hop LightGCN-style random-feature propagation plus hop-wise quantization."""
    rng = torch.Generator(device=device).manual_seed(1000 + seed)
    n_users, n_items = len(split.train), split.n_items
    user_idx, item_idx = [], []
    for u, seq in enumerate(split.train):
        user_idx.extend([u] * len(seq))
        item_idx.extend(seq)
    users = torch.tensor(user_idx, dtype=torch.long, device=device)
    items = torch.tensor(item_idx, dtype=torch.long, device=device)
    item_feat = F.normalize(torch.randn(n_items, 32, generator=rng, device=device), dim=1)
    hops = []
    for _ in range(2):
        user_feat = mean_aggregate(item_feat[items], users, n_users)
        item_feat = F.normalize(mean_aggregate(user_feat[users], items, n_items), dim=1)
        hops.append(item_feat[2:].clone())
    codes = torch.zeros(n_items, 2, dtype=torch.long, device=device)
    for hop, feat in enumerate(hops):
        codes[2:, hop] = kmeans(feat, codebook, seed + 97 * hop) + 1
    return codes


class CompactRec(nn.Module):
    def __init__(self, n_items: int, config: dict, codes: torch.Tensor):
        super().__init__()
        d = config["d_model"]
        self.model_type = config["model_type"]
        self.use_collab = config["collaborative_tokens"]
        self.register_buffer("codes", codes)
        self.item = nn.Embedding(n_items, d, padding_idx=PAD)
        self.collab = nn.ModuleList([nn.Embedding(65, d, padding_idx=0) for _ in range(2)])
        self.pos = nn.Embedding(config["context_length"] + config["target_length"], d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config["n_heads"],
            dim_feedforward=4 * d,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, config["n_layers"])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_items, bias=False)
        self.out.weight = self.item.weight

    def forward(self, tokens: torch.Tensor, causal: bool) -> torch.Tensor:
        bsz, length = tokens.shape
        x = self.item(tokens)
        if self.use_collab:
            item_codes = self.codes[tokens]
            x = x + 0.5 * sum(emb(item_codes[..., h]) for h, emb in enumerate(self.collab))
        x = x + self.pos(torch.arange(length, device=tokens.device))[None]
        causal_mask = None
        if causal:
            causal_mask = torch.full((length, length), float("-inf"), device=tokens.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
        hidden = self.encoder(x, mask=causal_mask, src_key_padding_mask=tokens.eq(PAD))
        return self.out(self.norm(hidden))


def make_batch(
    split: Split, users: list[int], config: dict, epoch: int, rng: random.Random, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx_len, target_len = config["context_length"], config["target_length"]
    contexts, targets = [], []
    for u in users:
        seq = split.train[u]
        end = rng.randint(target_len + 1, len(seq))
        target = seq[end - target_len : end]
        context = seq[max(0, end - target_len - ctx_len) : end - target_len]
        contexts.append([PAD] * (ctx_len - len(context)) + context)
        targets.append(target)
    context = torch.tensor(contexts, dtype=torch.long, device=device)
    target = torch.tensor(targets, dtype=torch.long, device=device)
    if config["model_type"] == "ar":
        inputs = torch.cat([context, target[:, :-1]], dim=1)
        positions = torch.arange(ctx_len - 1, ctx_len + target_len - 1, device=device)
        return inputs, target, positions
    progress = (epoch + 1) / config["epochs"]
    if config["progressive_curriculum"]:
        token_prob = 0.25 + 0.75 * progress
        history_prob = 0.02 + 0.18 * progress
    else:
        token_prob = rng.uniform(0.1, 1.0)
        history_prob = 0.10
    corrupt_target = torch.rand(target.shape, device=device) < token_prob
    corrupt_target[:, -1] = True
    noisy_target = target.masked_fill(corrupt_target, MASK)
    noisy_context = context.masked_fill(
        (torch.rand(context.shape, device=device) < history_prob) & context.ne(PAD), MASK
    )
    inputs = torch.cat([noisy_context, noisy_target], dim=1)
    positions = torch.arange(ctx_len, ctx_len + target_len, device=device)
    labels = target.masked_fill(~corrupt_target, -100)
    return inputs, labels, positions


def rank_metrics(logits: torch.Tensor, targets: torch.Tensor, histories: torch.Tensor, k: int = 10) -> dict:
    logits = logits.clone()
    logits[..., :2] = -torch.inf
    for row in range(logits.shape[0]):
        seen = histories[row][histories[row] >= 2]
        logits[row, :, seen] = -torch.inf
    top = logits.topk(k, dim=-1).indices
    matches = top.eq(targets[..., None])
    hit = matches.any(-1).float()
    rank = torch.where(
        matches,
        torch.arange(1, k + 1, device=logits.device).view(1, 1, -1),
        k + 1,
    ).min(-1).values
    ndcg = torch.where(hit.bool(), 1.0 / torch.log2(rank.float() + 1), torch.zeros_like(hit))
    rr = torch.where(hit.bool(), 1.0 / rank.float(), torch.zeros_like(hit))
    return {
        "recall@10": float(hit.mean()),
        "ndcg@10": float(ndcg.mean()),
        "mrr@10": float(rr.mean()),
        "position_recall@10": [float(x) for x in hit.mean(0)],
        "position_ndcg@10": [float(x) for x in ndcg.mean(0)],
    }


@torch.no_grad()
def decode_ar(model, context, target_len, force_early_error=False):
    tokens = context
    step_logits = []
    for step in range(target_len):
        logits = model(tokens, causal=True)[:, -1]
        logits[:, :2] = -torch.inf
        step_logits.append(logits)
        if force_early_error and step == 0:
            nxt = logits.topk(10, dim=-1).indices[:, -1]
        else:
            nxt = logits.argmax(-1)
        if step < target_len - 1:
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
    return torch.stack(step_logits, 1), None


@torch.no_grad()
def decode_diffusion(
    model,
    context,
    target_len,
    steps,
    voting,
    force_early_error=False,
    noise_scale=0.0,
):
    target = torch.full((context.shape[0], target_len), MASK, dtype=torch.long, device=context.device)
    trajectory, prev, stable = [], None, torch.zeros_like(target, dtype=torch.bool)
    for step in range(steps):
        logits = model(torch.cat([context, target], 1), causal=False)[:, -target_len:]
        logits[..., :2] = -torch.inf
        if noise_scale:
            logits = logits + torch.randn_like(logits) * noise_scale
        trajectory.append(F.log_softmax(logits, -1))
        pred = logits.argmax(-1)
        conf = logits.softmax(-1).amax(-1)
        if force_early_error and step == 0:
            pred[:, 0] = logits[:, 0].topk(10, dim=-1).indices[:, -1]
        if prev is not None:
            stable |= pred.eq(prev) & (conf > 0.35)
        target = torch.where(stable, target, pred)
        prev = pred
    if voting and len(trajectory) > 1:
        aggregate = torch.stack(trajectory).sum(0)
        final = torch.where(stable[..., None], trajectory[-1], aggregate)
    else:
        final = trajectory[-1]
    return final, target


@torch.no_grad()
def evaluate(model: CompactRec, split: Split, config: dict, device: torch.device) -> dict:
    model.eval()
    ctx_len, target_len = config["context_length"], config["target_length"]
    all_clean, all_forced, all_target, all_context = [], [], [], []
    corrected, total = 0, 0
    for start in range(0, len(split.train), config["eval_batch_size"]):
        train_rows = split.train[start : start + config["eval_batch_size"]]
        test_rows = split.test[start : start + config["eval_batch_size"]]
        contexts = [[PAD] * (ctx_len - min(ctx_len, len(x))) + x[-ctx_len:] for x in train_rows]
        context = torch.tensor(contexts, dtype=torch.long, device=device)
        target = torch.tensor(test_rows, dtype=torch.long, device=device)
        if config["model_type"] == "ar":
            clean, _ = decode_ar(model, context, target_len, False)
            forced, _ = decode_ar(model, context, target_len, True)
        else:
            clean, _ = decode_diffusion(
                model, context, target_len, config["sampling_steps"], config["stability_voting"], False
            )
            forced, forced_tokens = decode_diffusion(
                model, context, target_len, config["sampling_steps"], config["stability_voting"], True
            )
            corrected += int(forced_tokens[:, 0].eq(clean[:, 0].argmax(-1)).sum())
            total += context.shape[0]
        all_clean.append(clean.cpu())
        all_forced.append(forced.cpu())
        all_target.append(target.cpu())
        all_context.append(context.cpu())
    clean = torch.cat(all_clean)
    forced = torch.cat(all_forced)
    targets = torch.cat(all_target)
    histories = torch.cat(all_context)
    clean_metrics = rank_metrics(clean, targets, histories)
    forced_metrics = rank_metrics(forced, targets, histories)
    clean_downstream = rank_metrics(clean[:, 1:], targets[:, 1:], histories)
    forced_downstream = rank_metrics(forced[:, 1:], targets[:, 1:], histories)

    # Stochastic top-10 overlap across repeated decodes on a fixed user subset.
    n = min(config["consistency_users"], len(split.train))
    contexts = [[PAD] * (ctx_len - min(ctx_len, len(x))) + x[-ctx_len:] for x in split.train[:n]]
    context = torch.tensor(contexts, dtype=torch.long, device=device)
    tops = []
    for _ in range(config["consistency_repeats"]):
        if config["model_type"] == "ar":
            logits, _ = decode_ar(model, context, target_len, False)
            logits = logits + torch.randn_like(logits) * 0.20
        else:
            logits, _ = decode_diffusion(
                model,
                context,
                target_len,
                config["sampling_steps"],
                config["stability_voting"],
                False,
                noise_scale=0.20,
            )
        tops.append(logits[:, 0].topk(10, -1).indices.cpu())
    overlaps = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            for a, b in zip(tops[i].tolist(), tops[j].tolist()):
                sa, sb = set(a), set(b)
                overlaps.append(len(sa & sb) / len(sa | sb))
    return {
        "clean": clean_metrics,
        "forced_early_error": forced_metrics,
        "clean_downstream": clean_downstream,
        "forced_downstream": forced_downstream,
        "downstream_ndcg_retention": forced_downstream["ndcg@10"]
        / max(clean_downstream["ndcg@10"], 1e-12),
        "early_token_correction_rate": corrected / total if total else 0.0,
        "stochastic_top10_jaccard": float(np.mean(overlaps)),
    }


def train_seed(seed: int, gpu: int, config: dict, ratings_path: str, output: str) -> None:
    started = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    split = load_split(Path(ratings_path), config["target_length"])
    codes = collaborative_codes(split, device, seed)
    model = CompactRec(split.n_items, config, codes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    rng = random.Random(10_000 + seed)
    curve = []
    users = list(range(len(split.train)))
    for epoch in range(config["epochs"]):
        model.train()
        rng.shuffle(users)
        losses = []
        for start in range(0, len(users), config["batch_size"]):
            batch_users = users[start : start + config["batch_size"]]
            inputs, labels, positions = make_batch(split, batch_users, config, epoch, rng, device)
            logits = model(inputs, causal=config["model_type"] == "ar")[:, positions]
            loss = F.cross_entropy(logits.reshape(-1, split.n_items), labels.reshape(-1), ignore_index=-100)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        epoch_loss = float(np.mean(losses))
        curve.append(epoch_loss)
        print(
            f"TRAIN variant={config['name']} seed={seed} epoch={epoch + 1}/{config['epochs']} "
            f"loss={epoch_loss:.6f} device={device}",
            flush=True,
        )
    metrics = evaluate(model, split, config, device)
    result = {
        "variant": config["name"],
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "dataset": {
            "name": "MovieLens-1M",
            "url": DATA_URL,
            "users": len(split.train),
            "items_including_special": split.n_items,
            "interactions": split.n_interactions,
            "split": "per-user chronological train[:-3]/test[-3:]",
        },
        "model": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": config,
            "collaborative_code_hops": 2 if config["collaborative_tokens"] else 0,
        },
        "training_loss": curve,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    Path(output).write_text(json.dumps(result))
    print("SEED_RESULT " + json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    started = time.time()
    config = json.loads(Path("experiment.json").read_text())
    ratings = ensure_data()
    gpu_count = torch.cuda.device_count()
    if gpu_count < len(config["seeds"]):
        raise RuntimeError(f"Expected {len(config['seeds'])} GPUs, found {gpu_count}")
    print(
        "RUN_CONFIG "
        + json.dumps(
            {
                "config": config,
                "backend": "kubernetes",
                "requested_gpus": len(config["seeds"]),
                "dataset": DATA_URL,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ctx = mp.get_context("spawn")
    output_dir = Path("/tmp/dlmrec_results")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    processes = []
    for gpu, seed in enumerate(config["seeds"]):
        output = output_dir / f"seed_{seed}.json"
        process = ctx.Process(target=train_seed, args=(seed, gpu, config, str(ratings), str(output)))
        process.start()
        processes.append((process, output))
    for process, _ in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Seed worker failed with exit code {process.exitcode}")
    results = [json.loads(output.read_text()) for _, output in processes]
    metric_keys = [
        ("clean", "recall@10"),
        ("clean", "ndcg@10"),
        ("forced_downstream", "ndcg@10"),
    ]
    aggregate = {}
    for group, metric in metric_keys:
        values = [r["metrics"][group][metric] for r in results]
        aggregate[f"{group}.{metric}"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    for metric in ["downstream_ndcg_retention", "early_token_correction_rate", "stochastic_top10_jaccard"]:
        values = [r["metrics"][metric] for r in results]
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    terminal = {
        "schema": "dlmrec-reproduction-v1",
        "status": "success",
        "variant": config["name"],
        "backend": "kubernetes",
        "gpu_model_expected": "NVIDIA RTX PRO 6000 Blackwell",
        "allocated_gpu_count": len(config["seeds"]),
        "elapsed_seconds": time.time() - started,
        "aggregate": aggregate,
        "seeds": results,
    }
    print("FINAL_RESULT " + json.dumps(terminal, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
