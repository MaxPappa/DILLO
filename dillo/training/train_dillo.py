#!/usr/bin/env python3
"""
Training script for the DILLO policy explainer on LIBERO datasets.

Usage:
    python -m dillo.training.train_dillo --config configs/dillo_stage1.yaml

    Or with overrides:
    python -m dillo.training.train_dillo \
        --stage stage1 \
        --model_name google/gemma-3-1b-it \
        --train_data data/libero_goal_video_and_obs/*/*
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from accelerate import Accelerator

from dillo.training.policy_explainer import LIBEROPolicyExplainer
from dillo.training.behavior_dataset import (
    LIBEROBehaviorDataset,
    LIBEROSuccFailDataset,
    LIBEROCombinedDataset,
    LIBEROCollatorWithLatents,
)
from dillo.evaluation.eval_utils import evaluate_text_generation
from dillo.libero_imports import prepare_libero_imports


prepare_libero_imports()


# ─────────────────────────────────────────────────────────────────────
# Optional: ACTAgent for latent extraction (only needed if use_raw_obs=False)
# ─────────────────────────────────────────────────────────────────────

def load_act_agent(checkpoint_path: str, device: str = "cuda"):
    """
    Load the ACTAgent wrapper (ActionChunkingPolicy / GMM head) for latent
    extraction. This is the same policy used in dillo.data_generation.collect_dataset.
    Only needed when use_raw_obs=False.
    """
    from dillo.policy.act_agent import ACTAgent
    return ACTAgent(checkpoint_path, device=device)


# ─────────────────────────────────────────────────────────────────────
# Validation metrics computation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_metrics(
    model, val_dataloader, device, sample_k=8, compute_slow=False,
):
    """
    Generate descriptions for the validation set and compute metrics.
    """
    model.eval()
    all_preds = []
    all_refs = []
    all_obs = []
    verdict_correct = 0
    verdict_total = 0
    sample_rows = []
    metric_model = model.module if hasattr(model, "module") else model

    for val_batch in tqdm(val_dataloader, desc="Validation", leave=False):
        latent_obs = val_batch["latent_obs"].to(device)
        actions = val_batch["actions"].to(device)
        input_ids = val_batch["input_ids"].to(device)
        labels = val_batch["labels"].to(device)
        attention_mask = val_batch["attention_mask"].to(device)
        prompt_ids = val_batch["prompt_ids_val"].to(device)
        pixel_values = val_batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        token_type_ids = val_batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        obs_cpu = val_batch["obs"].detach().cpu()

        if getattr(metric_model, "verdict_head", None) is not None and "verdict_labels" in val_batch:
            prefix = metric_model._fuse_embeds(latent_obs, actions)
            verdict_logits = metric_model.verdict_head(prefix.mean(dim=1)).float()
            verdict_preds = (torch.sigmoid(verdict_logits) >= 0.5).float()
            verdict_labels = val_batch["verdict_labels"].to(
                device=verdict_preds.device, dtype=verdict_preds.dtype
            )
            verdict_correct += (verdict_preds == verdict_labels).sum().item()
            verdict_total += verdict_labels.numel()

        generated_ids = metric_model.gen_from_batch(
            latent_obs=latent_obs,
            actions=actions,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            prompt_ids=prompt_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            do_sample=False,
        )

        pred_texts = metric_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        lab_for_decode = labels.clone()
        lab_for_decode[lab_for_decode == -100] = metric_model.tokenizer.pad_token_id
        ref_texts = metric_model.tokenizer.batch_decode(lab_for_decode, skip_special_tokens=True)

        all_preds.extend(pred_texts)
        all_refs.extend(ref_texts)
        all_obs.extend(obs_cpu)

        # Collect sample examples
        if len(sample_rows) < sample_k:
            for i in range(len(pred_texts)):
                if len(sample_rows) < sample_k:
                    sample_rows.append({"pred": pred_texts[i], "ref": ref_texts[i]})

    obs_np = np.stack([o.numpy() for o in all_obs], axis=0) if all_obs else None

    metrics = evaluate_text_generation(
        predictions=all_preds,
        references=all_refs,
        obs=obs_np,
        compute_slow_metrics=compute_slow,
    )
    if verdict_total > 0:
        metrics["verdict_accuracy"] = verdict_correct / verdict_total

    return metrics, sample_rows


# ─────────────────────────────────────────────────────────────────────
# Validation Callback (runs at epoch end)
# ─────────────────────────────────────────────────────────────────────

class ValCallback(TrainerCallback):
    """
    Custom callback to run validation + W&B logging at the end of every
    N epochs.
    """

    def __init__(
        self,
        tokenizer,
        val_dataset,
        collate_fn,
        sample_k=8,
        batch_size=8,
        eval_every_epochs=3,
        compute_slow_metrics=False,
    ):
        self.tok = tokenizer
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        self.sample_k = sample_k
        self.eval_every_epochs = eval_every_epochs
        self.compute_slow = compute_slow_metrics

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        e = state.epoch
        if e is None:
            return
        epoch_idx = int(e)
        if epoch_idx % self.eval_every_epochs != 0:
            return

        model.eval()
        device = next(model.parameters()).device
        metrics, sample_rows = compute_val_metrics(
            model, self.val_loader, device,
            sample_k=self.sample_k,
            compute_slow=self.compute_slow,
        )

        log_dict = {
            f"val/{k}": v for k, v in metrics.items()
        }
        log_dict["epoch"] = state.epoch
        log_dict["global_step"] = state.global_step

        print(f"\n[Epoch {epoch_idx}] Validation metrics:")
        for k, v in log_dict.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        if sample_rows:
            print(f"\n  Sample predictions (first {len(sample_rows)}):")
            for i, row in enumerate(sample_rows[:3]):
                print(f"    [{i}] PRED: {row['pred'][:200]}")
                print(f"    [{i}]  REF: {row['ref'][:200]}")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(log_dict, step=state.global_step)
        except Exception:
            pass


@torch.no_grad()
def save_val_outputs_after_training(
    model,
    val_dataset,
    tokenizer,
    output_dir: str,
    device,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    use_image_obs: bool = False,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collator = LIBEROCollatorWithLatents(
        tokenizer, is_val=True, use_image_obs=use_image_obs
    )
    loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collator,
    )
    metric_model = model.module if hasattr(model, "module") else model
    metric_model.eval()
    records = []

    for batch in tqdm(loader, desc="Saving val outputs"):
        latent_obs = batch["latent_obs"].to(device)
        actions = batch["actions"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_ids = batch["prompt_ids_val"].to(device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        generated_ids = metric_model.gen_from_batch(
            latent_obs=latent_obs,
            actions=actions,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            prompt_ids=prompt_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        pred_texts = metric_model.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        lab_for_decode = labels.clone()
        lab_for_decode[lab_for_decode == -100] = metric_model.tokenizer.pad_token_id
        ref_texts = metric_model.tokenizer.batch_decode(
            lab_for_decode, skip_special_tokens=True
        )

        for i, pred_raw in enumerate(pred_texts):
            pred = re.sub(
                r"^<(success|failure)>\s*", "", pred_raw,
                flags=re.IGNORECASE,
            ).strip()
            cls_match = re.match(r"^<(success|failure)>", pred_raw, re.IGNORECASE)
            idx = len(records)
            record = {
                "index": idx,
                "model": getattr(metric_model, "model_name", ""),
                "stage": getattr(metric_model, "stage", ""),
                "use_image_obs": use_image_obs,
                "folder": batch.get("folder", [None])[i],
                "chunk_idx": batch.get("chunk_idx", [None])[i],
                "task_instruction": batch.get("task_instruction", [""])[i],
                "description_gt": ref_texts[i],
                "description_pred": pred,
                "description_pred_raw": pred_raw,
                "success_pred": cls_match.group(1).lower() if cls_match else None,
                "image_path": batch.get("image_path", [None])[i],
                "success_label": (
                    float(batch["success_label"][i].item())
                    if "success_label" in batch else None
                ),
            }
            for key in (
                "eef_pos_before",
                "eef_pos_after",
                "gripper_before",
                "gripper_after",
            ):
                if key in batch:
                    record[key] = batch[key][i].detach().cpu().tolist()
            records.append(record)
            with open(out_dir / f"{idx:06d}.json", "w") as f:
                json.dump(record, f, indent=2)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "total_items": len(records),
                "use_image_obs": use_image_obs,
                "model": getattr(metric_model, "model_name", ""),
                "stage": getattr(metric_model, "stage", ""),
            },
            f,
            indent=2,
        )
    print(f"[Saved] Validation outputs to {out_dir}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train LIBERO PolicyExplainer (ActionDescriber)"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file")

    # Core params (can be overridden on CLI)
    parser.add_argument("--stage", type=str, default="stage1",
                        choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--projector_type", type=str, default="mlp2x_gelu")
    parser.add_argument("--obs_act_pair_fusion", type=str, default="sum",
                        choices=["sum", "concat", "mlp"])
    parser.add_argument("--use_raw_obs", action="store_true", default=True,
                        help="Use raw robot state (12-dim) instead of ACT latents")
    parser.add_argument("--is_oracular", action="store_true", default=False)
    parser.add_argument("--use_eos_token", action="store_true", default=True)
    parser.add_argument("--suite", type=str, default=None,
                        choices=["goal", "spatial", "object", "10", "90"],
                        help="LIBERO suite name used to namespace the checkpoint directory "
                             "(e.g. 'goal' -> checkpoints/dillo_goal_<model>)")

    # Data
    parser.add_argument("--train_data", type=str,
                        default="data/libero_goal_video_and_obs/*/*",
                        help="Glob pattern for training examples")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Glob pattern for validation examples "
                             "(if None, uses last 10%% of train)")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="Number of atomic actions per chunk")
    parser.add_argument("--num_chunks", type=int, default=10,
                        help="Expected number of chunks per episode")
    parser.add_argument("--single_obs_act", action="store_true", default=True,
                        help="One training example per chunk transition")

    # ACT agent (for latent extraction)
    parser.add_argument("--act_checkpoint", type=str, default=None,
                        help="ACT policy checkpoint (needed if use_raw_obs=False)")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_every_epochs", type=int, default=3)
    parser.add_argument("--sample_k", type=int, default=8)
    parser.add_argument("--compute_slow_metrics", action="store_true", default=False,
                        help="Compute BLEU/ROUGE/BERTScore during validation")
    parser.add_argument("--reuse_stage2_trainset", action="store_true", default=False,
                        help="Stage 3: use stage-2 behavioral data instead of "
                             "SuccFail-only data for training")
    parser.add_argument("--require_success_mask", action="store_true", default=False,
                        help="Stage 3 validation: skip folders without success_mask.npy")
    parser.add_argument("--stage3_scratch", action="store_true", default=False,
                        help="Stage 3: initialize fresh LoRA/projector/verdict-head "
                             "weights instead of loading the latest stage-2 checkpoint")
    parser.add_argument("--use_verdict_head", action="store_true", default=False,
                        help="Stage 3: predict success/failure with a binary head "
                             "instead of text tokens")
    parser.add_argument("--use_image_obs", action="store_true", default=False,
                        help="Use the dataset image as the observation input "
                             "instead of raw state or ACT latent.")
    parser.add_argument("--disable_val_callback", action="store_true", default=False,
                        help="Skip epoch-end validation generation/logging.")
    parser.add_argument("--description_loss_weight", type=float, default=1.0,
                        help="Stage 3 weight for description language-model loss")
    parser.add_argument("--verdict_loss_weight", type=float, default=1.0,
                        help="Stage 3 weight for binary verdict-head loss")
    parser.add_argument("--device", type=str, default="cuda")
    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_val_outputs_dir", type=str, default=None,
                        help="Generate validation outputs after training into this directory.")
    parser.add_argument("--save_val_outputs_batch_size", type=int, default=1)
    parser.add_argument("--save_val_outputs_max_new_tokens", type=int, default=128)
    parser.add_argument("--no_timestamp_subdir", action="store_true", default=False,
                        help="Save the final checkpoint directly in output_dir "
                             "instead of output_dir/YYYYMMDD_HHMMSS")

    # WandB / Logging
    parser.add_argument("--wandb_project", type=str, default="Libero-Dillo")
    parser.add_argument("--wandb_entity", type=str, default="pinlab-sapienza", 
                        help="WandB team or user entity")

    args = parser.parse_args()

    # Load YAML config if provided (CLI args override)
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)

    # ─── ACT Agent (for latent extraction) ─────────────────────────
    # Load first so we can read the actual embed_size before building the model.
    agent = None
    if not args.use_raw_obs and not args.use_image_obs:
        if args.act_checkpoint is None:
            raise ValueError(
                "Must provide --act_checkpoint when use_raw_obs=False"
            )
        agent = load_act_agent(args.act_checkpoint, device=args.device)

    # ─── Derived dimensions ────────────────────────────────────────
    # LIBERO raw obs: eef_pos(3) + joint_pos(7) + gripper(2) = 12
    raw_obs_dim = 12
    # ACT latent dim: read from the loaded ActionChunkingPolicy checkpoint
    # rather than hard-coding it.
    if agent is not None:
        act_latent_dim = agent.policy.embed_size
    else:
        act_latent_dim = 512  # fallback (never reached when use_raw_obs=False)
    action_dim = args.chunk_size * 7  # flattened chunk

    latent_dim = 1 if args.use_image_obs else (raw_obs_dim if args.use_raw_obs else act_latent_dim)
    if args.is_oracular:
        latent_dim *= 2  # doubled for (before, after) pair

    # ─── Model ─────────────────────────────────────────────────────
    model = LIBEROPolicyExplainer(
        latent_dim=latent_dim if args.obs_act_pair_fusion == "sum" else (
            raw_obs_dim if args.use_raw_obs else act_latent_dim
        ),
        action_dim=action_dim,
        projector_type=args.projector_type,
        stage=args.stage,
        model_name=args.model_name,
        obs_act_pair_fusion=args.obs_act_pair_fusion,
        is_oracular=args.is_oracular,
        description_loss_weight=args.description_loss_weight,
        verdict_loss_weight=args.verdict_loss_weight,
        use_verdict_head=args.use_verdict_head,
        use_image_obs=args.use_image_obs,
    )

    if args.stage == "stage3" and args.stage3_scratch:
        model.add_lora()
        print(
            "[Train] Stage 3 scratch mode: initialized fresh LoRA, "
            "projector, and verdict-head weights"
        )

    # ─── Output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obs_type = "imageobs" if args.use_image_obs else ("rawobs" if args.use_raw_obs else "latentobs")
    oracular_str = "_oracular" if args.is_oracular else ""

    if args.output_dir is None:
        suite_str = f"_{args.suite}" if args.suite else ""
        args.output_dir = (
            f"checkpoints/dillo{suite_str}_{model.model_name}"
            f"{oracular_str}/{args.stage}/{obs_type}"
        )

    # ─── Load previous-stage checkpoints ───────────────────────────
    if args.stage == "stage2":
        load_dir = args.output_dir.replace("stage2", "stage1")
        _load_latest_checkpoint(model, load_dir)
    elif args.stage == "stage3" and not args.stage3_scratch:
        load_dir = args.output_dir.replace("stage3", "stage2")
        _load_latest_checkpoint(model, load_dir)

    # ─── WandB ─────────────────────────────────────────────────────
    run_name = (
        f"libero-{model.model_name}{oracular_str}-{args.stage}"
        f"-{args.projector_type}-{timestamp}"
    )
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity



    # ─── Datasets ──────────────────────────────────────────────────
    tokenizer = model.tokenizer

    _succ_fail_kwargs = dict(
        agent=agent,
        tokenizer=tokenizer,
        use_raw_obs=args.use_raw_obs,
        is_oracular=args.is_oracular,
        use_eos_token=args.use_eos_token,
        chunk_size=args.chunk_size,
        num_chunks=args.num_chunks,
    )
    _behav_kwargs = dict(
        **_succ_fail_kwargs,
        single_obs_act=args.single_obs_act,
    )

    if args.stage == "stage3":
        # Stage 3: combined verdict + description fine-tuning.
        # With --use_verdict_head the assistant turn is only the description;
        # the binary target is carried separately as verdict_labels.
        _combined_kwargs = dict(
            **_succ_fail_kwargs,
            min_description_tokens=5,
            require_success_mask=False,
            use_text_verdict=not args.use_verdict_head,
            use_image_obs=args.use_image_obs,
        )
        _combined_dataset = LIBEROCombinedDataset(
            data_dirs=args.train_data,
            **_combined_kwargs,
        )

        if args.val_data is not None:
            train_dataset = _combined_dataset
            val_dataset = LIBEROCombinedDataset(
                data_dirs=args.val_data,
                **dict(
                    **_succ_fail_kwargs,
                    min_description_tokens=5,
                    require_success_mask=getattr(args, "require_success_mask", False),
                    use_text_verdict=not args.use_verdict_head,
                    use_image_obs=args.use_image_obs,
                ),
            )
        else:
            # Split 90 / 10
            total    = len(_combined_dataset)
            val_size = max(1, int(total * 0.1))
            train_size_combined = total - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                _combined_dataset, [train_size_combined, val_size],
                generator=torch.Generator().manual_seed(42),
            )
    else:
        # Stage 1 / Stage 2: behavioral description data
        train_dataset = LIBEROBehaviorDataset(
            data_dirs=args.train_data, validation=False, **_behav_kwargs
        )

        if args.val_data is not None:
            val_dataset = LIBEROBehaviorDataset(
                data_dirs=args.val_data, validation=True, **_behav_kwargs
            )
        else:
            # Split last 10% as validation
            total = len(train_dataset)
            val_size = max(1, int(total * 0.1))
            train_size = total - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

    print(f"[Train] {len(train_dataset)} examples")
    print(f"[Val]   {len(val_dataset)} examples")

    # ─── Training Arguments ────────────────────────────────────────
    training_args = TrainingArguments(
        optim="adamw_torch",
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="no",
        do_eval=False,
        bf16=torch.cuda.is_available(),
        report_to=["wandb"],
        run_name=run_name,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # ─── Collator ──────────────────────────────────────────────────
    data_collator = LIBEROCollatorWithLatents(
        tokenizer, use_image_obs=args.use_image_obs
    )

    tok = AutoTokenizer.from_pretrained(args.model_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tok,
    )

    # ─── Validation Callback ───────────────────────────────────────
    if args.stage != "stage1" and not args.disable_val_callback:
        val_collator = LIBEROCollatorWithLatents(
            tokenizer, is_val=True, use_image_obs=args.use_image_obs
        )
        trainer.add_callback(
            ValCallback(
                tokenizer=tokenizer,
                val_dataset=val_dataset,
                collate_fn=val_collator,
                sample_k=args.sample_k,
                batch_size=args.batch_size,
                eval_every_epochs=args.eval_every_epochs,
                compute_slow_metrics=args.compute_slow_metrics,
            )
        )

    # ─── Train ─────────────────────────────────────────────────────
    trainer.train()

    # ─── Save final checkpoint ─────────────────────────────────────
    try:
        unwrapped = trainer.model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        out_dir = (
            training_args.output_dir
            if args.no_timestamp_subdir
            else os.path.join(training_args.output_dir, timestamp)
        )
        os.makedirs(out_dir, exist_ok=True)
        unwrapped.save_checkpoints(
            out_dir, save_function=Accelerator.save, epoch=trainer.state.epoch
        )
        print(f"[Saved] Checkpoints to {out_dir}")
        if args.save_val_outputs_dir is not None:
            device = next(unwrapped.parameters()).device
            save_val_outputs_after_training(
                unwrapped,
                val_dataset,
                tokenizer,
                args.save_val_outputs_dir,
                device,
                batch_size=args.save_val_outputs_batch_size,
                max_new_tokens=args.save_val_outputs_max_new_tokens,
                use_image_obs=args.use_image_obs,
            )
    except Exception as e:
        print(f"[Warning] Failed to save checkpoints: {e}")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _load_latest_checkpoint(model, ckpt_root_dir: str):
    """Load the latest checkpoint from either timestamped or flat output dirs."""
    ckpt_root = Path(ckpt_root_dir)
    timestamped_dirs = []
    for path in ckpt_root.glob("*"):
        if not path.is_dir():
            continue
        try:
            datetime.strptime(path.name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        timestamped_dirs.append(path)

    if timestamped_dirs:
        latest = max(timestamped_dirs, key=lambda p: datetime.strptime(p.name, "%Y%m%d_%H%M%S"))
    elif list(ckpt_root.glob("*_policy_explainer.pth")):
        latest = ckpt_root
    else:
        print(f"[Warning] No checkpoints found in {ckpt_root_dir}, skipping load")
        return

    model.load_checkpoints(str(latest))
    print(f"[Loaded] Checkpoint from {latest}")


if __name__ == "__main__":
    main()
