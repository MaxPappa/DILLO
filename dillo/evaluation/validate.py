#!/usr/bin/env python3
"""
Evaluate a trained LIBEROPolicyExplainer on the validation split.

Loads the saved model checkpoint (stage2 or stage3), runs generation
on each validation item, and saves per-item predictions under
outputs/validation by default.

Usage:
    # Stage-3 checkpoint, latent obs (default)
    CUDA_VISIBLE_DEVICES=0 python -m dillo.evaluation.validate \
        --suite 90 \
        --stage stage3 \
        --checkpoint_dir checkpoints/dillo_90_gemma-3-4b-it/stage3/latentobs/latest \
        --act_checkpoint checkpoints/act/libero_90/best_model.pth \
        --model_name google/gemma-3-4b-it \
        --chunk_size 20

    # Stage-3, raw obs (no ACT agent needed)
    python -m dillo.evaluation.validate \
        --suite 10 \
        --stage stage3 \
        --checkpoint_dir ... \
        --use_raw_obs

Output structure:
    outputs/validation/libero_{suite}/{model_name}/{stage}_{ckpt_ts}/
        {i:06d}.json
        summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from dillo.training.policy_explainer import LIBEROPolicyExplainer
from dillo.training.behavior_dataset import (
    build_robot_state,
    extract_act_policy_latent,
    extract_latent_act,
)
from dillo.training.binary_classifier import OfflineACTLatentAgent

DEFAULT_OUTPUT_ROOT = Path("outputs/validation")
DEFAULT_VAL_SPLITS_DIR = Path("val_splits")


# ── Loading helpers ────────────────────────────────────────────────────

def load_val_items(suite: str, val_splits_dir: str | Path) -> List[dict]:
    path = Path(val_splits_dir) / f"libero_{suite}_val.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Val split not found: {path}. Run split_dataset.py --suite {suite} first."
        )
    with open(path) as f:
        return json.load(f)


def load_policy_explainer(
    checkpoint_dir: str,
    stage: str,
    model_name: str,
    chunk_size: int,
    latent_dim: int,
    obs_act_pair_fusion: str,
    projector_type: str,
    is_oracular: bool,
    use_verdict_head: bool,
    device: torch.device,
) -> LIBEROPolicyExplainer:
    """
    Instantiate and load a LIBEROPolicyExplainer from a checkpoint directory.

    The checkpoint_dir should contain:
      - e=N_policy_explainer.pth  (projector weights)
      - lora-<model_name>/        (LoRA adapter weights, for stage2/3)
    """
    # Determine evaluation stage string that matches load_checkpoints logic
    eval_stage = {
        "stage2": "eval_stage2",
        "stage3": "eval_stage3",
    }.get(stage, "eval")

    action_dim = chunk_size * 7

    model = LIBEROPolicyExplainer(
        latent_dim=latent_dim,
        action_dim=action_dim,
        projector_type=projector_type,
        stage=eval_stage,
        model_name=model_name,
        obs_act_pair_fusion=obs_act_pair_fusion,
        is_oracular=is_oracular,
        use_verdict_head=use_verdict_head,
    )
    model.load_checkpoints(checkpoint_dir)
    model.to(device)
    model.eval()
    print(f"[eval_trained_model] Loaded PolicyExplainer from {checkpoint_dir}")
    return model


def load_act_agent(act_checkpoint: str, device: torch.device):
    """Load the minimal offline ACT wrapper for latent extraction."""
    agent = OfflineACTLatentAgent(act_checkpoint, device=str(device))
    print(f"[eval_trained_model] Loaded offline ACT latent agent from {act_checkpoint}")
    return agent


# ── Prompt builder (mirrors LIBEROBehaviorDataset) ───────────────────

def build_user_prompt(task_instruction: str) -> str:
    return (
        "You are a robot performing the task: '{task}'. "
        "You are given the current observation and the action chunk that was executed. "
        "Describe in one sentence what the robot did."
    ).format(task=task_instruction)


def tokenize_prompt(tokenizer, user_text: str) -> torch.Tensor:
    """Tokenize the user prompt (without generation prompt, matching LIBEROBehaviorDataset)."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(out, "input_ids"):
        ids = out["input_ids"]
    elif isinstance(out, list) and out and isinstance(out[0], int):
        ids = out
    else:
        ids = out
    return torch.tensor(ids, dtype=torch.long)


# ── Single-item inference ─────────────────────────────────────────────

@torch.no_grad()
def generate_description(
    model: LIBEROPolicyExplainer,
    latent_obs: torch.Tensor,   # (latent_dim,) or (2, latent_dim) for oracular
    actions: torch.Tensor,      # (chunk_size, 7)
    prompt_ids: torch.Tensor,   # (seq_len,)
    device: torch.device,
    max_new_tokens: int = 128,
) -> str:
    """Generate a description for a single validation item."""
    # Add batch dimension
    lo   = latent_obs.unsqueeze(0).to(device)
    acts = actions.unsqueeze(0).to(device)
    pids = prompt_ids.unsqueeze(0).to(device)

    # gen_from_batch needs input_ids and labels too (even though they're only
    # used to build the labels prefix mask in gen_from_batch's embed path).
    # We pass dummy tensors for the unused args.
    dummy = pids  # reuse prompt ids as placeholder for input_ids / labels

    out_ids = model.gen_from_batch(
        latent_obs=lo,
        actions=acts,
        input_ids=dummy,
        labels=dummy,
        attention_mask=torch.ones_like(dummy),
        prompt_ids=pids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text.strip()


# ── Main loop ────────────────────────────────────────────────────────

def run_eval(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load val items ─────────────────────────────────────────────
    val_items = load_val_items(args.suite, args.val_splits_dir)
    if args.max_items is not None:
        val_items = val_items[: args.max_items]
    print(f"[eval_trained_model] {len(val_items)} val items for libero_{args.suite}")

    # ── Load ACT agent (optional) ──────────────────────────────────
    agent = None
    if not args.use_raw_obs:
        if not args.act_checkpoint:
            raise ValueError("--act_checkpoint required when use_raw_obs=False")
        agent = load_act_agent(args.act_checkpoint, device)
        # Read latent dim from the loaded policy
        latent_dim = agent.policy.embed_size
    else:
        latent_dim = 12  # eef(3) + joint(7) + gripper(2)

    if args.is_oracular and args.obs_act_pair_fusion == "sum":
        model_latent_dim = latent_dim * 2
    else:
        model_latent_dim = latent_dim

    # ── Load PolicyExplainer ────────────────────────────────────────
    model = load_policy_explainer(
        checkpoint_dir=args.checkpoint_dir,
        stage=args.stage,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        latent_dim=model_latent_dim,
        obs_act_pair_fusion=args.obs_act_pair_fusion,
        projector_type=args.projector_type,
        is_oracular=args.is_oracular,
        use_verdict_head=args.use_verdict_head,
        device=device,
    )
    tokenizer = model.tokenizer

    # ── Output dir ─────────────────────────────────────────────────
    model_short = Path(args.model_name).name
    ckpt_ts = Path(args.checkpoint_dir).name  # e.g. "20260225_020021"
    obs_type = "rawobs" if args.use_raw_obs else "latentobs"
    run_tag = f"{args.stage}_{obs_type}_{ckpt_ts}"
    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.output_root) / f"libero_{args.suite}" / model_short / run_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_trained_model] Output dir: {out_dir}")

    records = []
    for i, item in enumerate(val_items):
        out_path = out_dir / f"{i:06d}.json"
        if args.resume and out_path.exists():
            with open(out_path) as f:
                records.append(json.load(f))
            continue

        # ── Observations ───────────────────────────────────────────
        robot_state_before = np.array(item["robot_state_before"], dtype=np.float32)
        robot_state_after  = np.array(item["robot_state_after"],  dtype=np.float32)

        if args.use_raw_obs:
            if args.is_oracular:
                lat = torch.from_numpy(
                    np.stack([robot_state_before, robot_state_after])
                )  # (2, 12)
            else:
                lat = torch.from_numpy(robot_state_before)  # (12,)
        else:
            img_path_b = item.get("image_before_path")
            img_path_a = item.get("image_after_path")
            task_instr = item.get("task_instruction", "")

            _latent_fn = (
                extract_act_policy_latent if hasattr(agent, "policy") else extract_latent_act
            )
            if args.is_oracular:
                lat = _latent_fn(
                    agent,
                    np.stack([robot_state_before, robot_state_after]),
                    device,
                    image_paths=[img_path_b, img_path_a],
                    task_instruction=task_instr,
                )  # (2, embed_dim)
            else:
                lat = _latent_fn(
                    agent,
                    robot_state_before,
                    device,
                    image_paths=[img_path_b] if img_path_b else None,
                    task_instruction=task_instr,
                )  # (embed_dim,)

        # ── Actions ────────────────────────────────────────────────
        acts = torch.from_numpy(
            np.array(item["actions"], dtype=np.float32)
        )  # (chunk_size, 7)

        # ── Prompt ─────────────────────────────────────────────────
        user_text  = build_user_prompt(item["task_instruction"])
        prompt_ids = tokenize_prompt(tokenizer, user_text)  # (seq_len,)

        # ── Generate ───────────────────────────────────────────────
        try:
            prediction_raw = generate_description(
                model, lat, acts, prompt_ids, device,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(val_items)}] ERROR: {e}")
            prediction_raw = ""

        # Stage 3 outputs "<success>/<failure> description"; strip classification
        # token so the description can be compared fairly with the GT VLM text.
        prediction = re.sub(r"^<(success|failure)>\s*", "", prediction_raw,
                            flags=re.IGNORECASE).strip()
        # Also extract success/failure prediction if present
        clf_match = re.match(r"^<(success|failure)>", prediction_raw,
                             flags=re.IGNORECASE)
        success_pred = clf_match.group(1).lower() if clf_match else None

        record = {
            "index": i,
            "suite": args.suite,
            "model": args.model_name,
            "stage": args.stage,
            "checkpoint_dir": args.checkpoint_dir,
            "use_raw_obs": args.use_raw_obs,
            "folder": item["folder"],
            "chunk_idx": item["chunk_idx"],
            "task_instruction": item["task_instruction"],
            "description_gt": item["description"],
            "description_pred": prediction,          # stripped of <success>/<failure>
            "description_pred_raw": prediction_raw,  # full model output
            "success_pred": success_pred,            # "success" / "failure" / None
            "eef_pos_before": item["eef_pos_before"],
            "eef_pos_after": item["eef_pos_after"],
            "gripper_before": item["gripper_before"],
            "gripper_after": item["gripper_after"],
            "success_label": item.get("success_label"),
        }
        records.append(record)

        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        if (i + 1) % 10 == 0 or i == 0:
            succ_str = f" [{success_pred}]" if success_pred else ""
            print(f"  [{i+1}/{len(val_items)}]{succ_str} pred: {prediction[:80]!r}")

    # ── Summary ────────────────────────────────────────────────────
    summary = {
        "suite": args.suite,
        "model": args.model_name,
        "stage": args.stage,
        "checkpoint_dir": args.checkpoint_dir,
        "use_raw_obs": args.use_raw_obs,
        "total_items": len(records),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[eval_trained_model] Done. {len(records)} predictions → {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LIBEROPolicyExplainer on the val split"
    )
    parser.add_argument("--suite", required=True,
                        choices=["spatial", "goal", "object", "10", "90"])
    parser.add_argument("--stage", required=True,
                        choices=["stage2", "stage3"],
                        help="Which training stage checkpoint to load")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to the checkpoint directory (containing "
                             "e=N_policy_explainer.pth and optionally lora-*/)")
    parser.add_argument("--act_checkpoint", default=None,
                        help="ACT policy checkpoint (required if use_raw_obs=False)")
    parser.add_argument("--model_name", default="google/gemma-3-1b-it",
                        help="HuggingFace model name for the LLM backbone")
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--use_raw_obs", action="store_true", default=False,
                        help="Use 12-dim raw obs instead of ACT latent")
    parser.add_argument("--is_oracular", action="store_true", default=False)
    parser.add_argument("--obs_act_pair_fusion", default="sum",
                        choices=["sum", "concat", "mlp"])
    parser.add_argument("--projector_type", default="mlp2x_gelu")
    parser.add_argument("--use_verdict_head", action="store_true", default=False,
                        help="Checkpoint was trained with a binary verdict head; "
                             "generate description text without constrained verdict tokens")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-written output files")
    parser.add_argument("--output_dir", default=None,
                        help="Explicit directory for numbered prediction JSONs")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="Root used when --output_dir is omitted")
    parser.add_argument("--val_splits_dir", default=str(DEFAULT_VAL_SPLITS_DIR),
                        help="Directory containing libero_<suite>_val.json files")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Evaluate only the first N validation items")
    return parser.parse_args()


def main():
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
