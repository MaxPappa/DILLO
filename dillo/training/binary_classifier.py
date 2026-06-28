#!/usr/bin/env python3
"""
Train a lightweight binary success/failure classifier on LIBERO latents.

This is a non-LLM baseline for the policy explainer stage-3 signal. It uses
the same success_mask.npy labeling convention as LIBEROSuccFailDataset:

    label = 1 if success_mask[i + 1] == 1.0 else 0

By default the classifier consumes the policy latent and the flattened action
chunk. Pass --no_actions to train a pure latent-only ablation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from dillo.policy.act_policy import ActionChunkingPolicy
from dillo.policy.obs import OBS_MODALITY
from dillo.training.behavior_dataset import (
    LIBEROLatentBinaryCollator,
    LIBEROSuccFailLatentDataset,
)


class OfflineACTLatentAgent:
    """
    Minimal ACT agent for offline latent extraction.

    It only loads the trained ACT encoder and task embeddings.  No LIBERO
    environment is constructed for this offline classifier baseline.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from libero.lifelong.utils import get_task_embs
        from robomimic.utils import obs_utils as ObsUtils

        self._get_task_embs = get_task_embs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        cfg = EasyDict(checkpoint["args"])
        shape_meta = checkpoint["shape_meta"]

        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": OBS_MODALITY})

        language_input_size = 768
        self.policy = ActionChunkingPolicy(
            shape_meta=shape_meta,
            embed_size=cfg.get("embed_size", 64),
            language_input_size=language_input_size,
            language_hidden_size=128,
            chunk_size=cfg.get("chunk_size", 20),
            decoder_num_layers=cfg.get("decoder_layers", 2),
            decoder_num_heads=cfg.get("decoder_heads", 4),
            decoder_ff_dim=cfg.get("decoder_ff_dim", 256),
            decoder_dropout=cfg.get("decoder_dropout", 0.1),
            gmm_hidden_size=cfg.get("gmm_hidden", 1024),
            gmm_num_layers=2,
            gmm_num_modes=cfg.get("gmm_modes", 5),
            gmm_min_std=1e-4,
            use_joint=True,
            use_gripper=True,
            use_ee=False,
            use_augmentation=False,
            img_input_shape=shape_meta["all_shapes"].get("agentview_rgb", (3, 128, 128)),
            translation=8,
            temporal_decay=cfg.get("temporal_decay", 0.01),
        )
        self.policy.load_state_dict(checkpoint["state_dict"])
        self.policy.to(self.device)
        self.policy.eval()

        self.chunk_size = cfg.get("chunk_size", 20)
        img_shape = shape_meta["all_shapes"].get("agentview_rgb", (3, 128, 128))
        self._img_h, self._img_w = img_shape[1], img_shape[2]
        self._task_emb_cache: dict[str, torch.Tensor] = {}
        self._task_emb_cfg = EasyDict(
            task_embedding_format=cfg.get("task_embedding_format", "bert"),
            task_embedding_one_hot_offset=1,
            data=EasyDict(max_word_len=25),
            policy=EasyDict(
                language_encoder=EasyDict(
                    network_kwargs=EasyDict(input_size=language_input_size)
                )
            ),
        )

    def _get_task_emb(self, language_instruction: str) -> torch.Tensor:
        if language_instruction not in self._task_emb_cache:
            embs = self._get_task_embs(self._task_emb_cfg, [language_instruction])
            self._task_emb_cache[language_instruction] = embs[0:1].cpu()
        return self._task_emb_cache[language_instruction].to(self.device)


class LatentBinaryClassifier(nn.Module):
    """Small binary classifier over latent observation and optional action."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_actions: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.use_actions = use_actions
        input_dim = latent_dim + (action_dim if use_actions else 0)

        if num_layers <= 0:
            self.net = nn.Linear(input_dim, 1)
        else:
            layers: list[nn.Module] = []
            dim = input_dim
            for _ in range(num_layers):
                layers.extend(
                    [
                        nn.Linear(dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                dim = hidden_dim
            layers.append(nn.Linear(dim, 1))
            self.net = nn.Sequential(*layers)

    def forward(self, latent_obs: torch.Tensor, actions: torch.Tensor | None = None):
        x = latent_obs.reshape(latent_obs.shape[0], -1)
        if self.use_actions:
            if actions is None:
                raise ValueError("actions must be provided when use_actions=True")
            x = torch.cat([x, actions.reshape(actions.shape[0], -1)], dim=-1)
        return self.net(x).squeeze(-1)


def _load_config(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.config is None:
        if args.use_raw_obs is None:
            args.use_raw_obs = True
        if args.is_oracular is None:
            args.is_oracular = False
        if args.require_success_mask is None:
            args.require_success_mask = False
        return args
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)
    if args.use_raw_obs is None:
        args.use_raw_obs = True
    if args.is_oracular is None:
        args.is_oracular = False
    if args.require_success_mask is None:
        args.require_success_mask = False
    return args


def _make_datasets(args: argparse.Namespace, agent):
    dataset_kwargs = dict(
        agent=agent,
        use_raw_obs=args.use_raw_obs,
        is_oracular=args.is_oracular,
        chunk_size=args.chunk_size,
        num_chunks=args.num_chunks,
        latent_source=args.latent_source,
    )
    full_train = LIBEROSuccFailLatentDataset(
        data_dirs=args.train_data,
        require_success_mask=False,
        **dataset_kwargs,
    )
    if len(full_train) == 0:
        raise ValueError("Training dataset produced zero labeled examples")

    if args.val_data is not None:
        val_dataset = LIBEROSuccFailLatentDataset(
            data_dirs=args.val_data,
            require_success_mask=args.require_success_mask,
            **dataset_kwargs,
        )
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset produced zero labeled examples")
        return full_train, val_dataset

    if len(full_train) < 2:
        raise ValueError("Need at least two labeled examples for an automatic train/val split")
    val_size = max(1, int(len(full_train) * args.val_ratio))
    train_size = len(full_train) - val_size
    return torch.utils.data.random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )


def _infer_dims(dataset) -> tuple[int, int]:
    sample = dataset[0]
    latent_dim = int(sample["latent_obs"].numel())
    action_dim = int(sample["actions"].numel())
    return latent_dim, action_dim


def _collect_label_counts(dataset) -> tuple[int, int]:
    labels = []
    for i in range(len(dataset)):
        labels.append(int(dataset[i]["label"].item()))
    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos
    return n_pos, n_neg


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n = 0
    tp = fp = tn = fn = 0

    for batch in dataloader:
        latent_obs = batch["latent_obs"].to(device)
        actions = batch["actions"].to(device)
        labels = batch["labels"].to(device)

        logits = model(latent_obs, actions)
        loss = loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        gold = labels.long()

        total_loss += loss.item() * labels.numel()
        n += labels.numel()
        tp += int(((preds == 1) & (gold == 1)).sum().item())
        fp += int(((preds == 1) & (gold == 0)).sum().item())
        tn += int(((preds == 0) & (gold == 0)).sum().item())
        fn += int(((preds == 0) & (gold == 1)).sum().item())

    acc = (tp + tn) / max(1, n)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "loss": total_loss / max(1, n),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a lightweight LIBERO latent binary classifier baseline"
    )
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--train_data", type=str, default="../collected/libero_goal_chunks/*/*")
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--num_chunks", type=int, default=10)
    parser.add_argument("--use_raw_obs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--is_oracular", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require_success_mask", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--latent_source", type=str, default="visual",
                        choices=["visual", "context"],
                        help="visual uses only image encoder tokens; context also includes language/proprioception")
    parser.add_argument("--act_checkpoint", type=str, default=None)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2,
                        help="0 gives logistic regression; >=1 gives an MLP")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_actions", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)

    args = _load_config(parser.parse_args(), parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.use_raw_obs and args.act_checkpoint is None:
        raise ValueError("Must provide --act_checkpoint when use_raw_obs=False")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = None
    if not args.use_raw_obs:
        agent = OfflineACTLatentAgent(args.act_checkpoint, device=str(device))

    train_dataset, val_dataset = _make_datasets(args, agent)
    latent_dim, action_dim = _infer_dims(train_dataset)
    n_pos, n_neg = _collect_label_counts(train_dataset)
    print(f"[Train] {len(train_dataset)} examples ({n_pos} success, {n_neg} failure)")
    print(f"[Val]   {len(val_dataset)} examples")
    print(f"[Dims] latent_dim={latent_dim}, action_dim={action_dim}, "
          f"use_actions={not args.no_actions}")

    collator = LIBEROLatentBinaryCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = LatentBinaryClassifier(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_actions=not args.no_actions,
    ).to(device)

    pos_weight = torch.tensor(
        [n_neg / max(1, n_pos)],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        obs_type = "rawobs" if args.use_raw_obs else "latentobs"
        action_str = "latent_action" if not args.no_actions else "latent_only"
        args.output_dir = f"../checkpoints/libero_binary_classifier/{action_str}/{obs_type}/{timestamp}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            latent_obs = batch["latent_obs"].to(device)
            actions = batch["actions"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(latent_obs, actions)
            loss = loss_fn(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += loss.item() * labels.numel()
            seen += labels.numel()
            pbar.set_postfix(loss=f"{running / max(1, seen):.4f}")

        val_metrics = evaluate(model, val_loader, device, loss_fn)
        row = {"epoch": epoch, "train_loss": running / max(1, seen), **val_metrics}
        history.append(row)
        print(
            f"[Epoch {epoch}] train_loss={row['train_loss']:.4f} "
            f"val_loss={row['loss']:.4f} acc={row['accuracy']:.4f} "
            f"f1={row['f1']:.4f} precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": latent_dim,
                    "action_dim": action_dim,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                out_dir / "best.pt",
            )

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"[Saved] Best checkpoint: {out_dir / 'best.pt'}")
    print(f"[Saved] History: {out_dir / 'history.json'}")


if __name__ == "__main__":
    main()
