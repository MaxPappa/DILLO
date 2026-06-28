#!/usr/bin/env python3
"""Evaluate ACT policy steering with a trained DILLO model.

The evaluator runs LIBERO simulation tasks with the ACT policy as the
action proposer and either a DILLO stage-3 explainer or a standalone MLP as
the accept/refuse steering model. It reports baseline success, steered
success, refusal statistics, and optional rollout videos.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m dillo.evaluation.steering \
        --suite libero_goal \
        --act-checkpoint checkpoints/act/libero_goal/best_model.pth \
        --explainer-ckptdir checkpoints/dillo_goal_gemma-3-1b-it/stage3/latentobs/latest \
        --explainer-model-name google/gemma-3-1b-it \
        --n-episodes 20 \
        --proposal-batch-size 8 \
        --max-refusal-attempts 8 \
        --outdir outputs/steering/libero_goal
"""



from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    import imageio
except ImportError:
    imageio = None


from dillo.libero_imports import prepare_libero_imports

prepare_libero_imports()


from peft import PeftModel

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.bddl_utils as BDDLUtils

from dillo.policy.act_agent import ACTAgent
from dillo.training.policy_explainer import LIBEROPolicyExplainer
from dillo.training.binary_classifier import LatentBinaryClassifier


SUITE_CHOICES = ["libero_10", "libero_spatial", "libero_goal", "libero_object", "libero_90"]
SUITE_TO_BENCHMARK = {
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_10": "LIBERO_10",
    "libero_90": "LIBERO_90",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RolloutStats:
    success: bool
    steps: int
    refused_count: int = 0
    accepted_attempts: List[int] = field(default_factory=list)
    forced_accepts: int = 0
    chunks_executed: int = 0
    decision_logs: List[Dict] = field(default_factory=list)


def get_bddl_root() -> Path:
    config_path = Path.home() / ".libero" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        benchmark_root = cfg["benchmark_root"]
    else:
        from libero.libero import get_default_path_dict
        benchmark_root = get_default_path_dict()["benchmark_root"]
    return Path(benchmark_root) / "bddl_files"


def resolve_act_checkpoint(args) -> str:
    if args.act_checkpoint is None:
        raise ValueError("Must provide --act-checkpoint")
    return str(resolve_existing_path(args.act_checkpoint))


def resolve_existing_path(path_str: str) -> Path:
    """Resolve path robustly across cwd/repo-root invocation styles."""
    p = Path(path_str).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.append((Path.cwd() / p))
        candidates.append((REPO_ROOT / p))
        candidates.append((REPO_ROOT.parent / p))

    for c in candidates:
        if c.exists():
            return c.resolve()

    tried = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Path not found: {path_str}. Tried:{tried}")


def parse_explainer_paths(args) -> Tuple[str, str]:
    # Derive LoRA folder name from --explainer-model-name (e.g. "google/gemma-3-4b-it" → "lora-gemma-3-4b-it")
    model_short = args.explainer_model_name.split("/")[-1]
    default_lora_name = f"lora-{model_short}"

    if args.explainer_ckptdir:
        ckpt_dir = resolve_existing_path(args.explainer_ckptdir)
        if args.explainer_projector is None:
            projs = sorted(ckpt_dir.glob("e=*policy_explainer.pth"))
            if not projs:
                raise FileNotFoundError(f"No projector ckpt found in {ckpt_dir}")
            projector = str(projs[-1])
        else:
            projector = str(resolve_existing_path(args.explainer_projector))
        lora_dir = (
            str(resolve_existing_path(args.explainer_lora_dir))
            if args.explainer_lora_dir is not None
            else str(ckpt_dir / default_lora_name)
        )
    else:
        if args.explainer_projector is None or args.explainer_lora_dir is None:
            raise ValueError(
                "Provide either --explainer-ckptdir OR both --explainer-projector and --explainer-lora-dir"
            )
        projector = str(resolve_existing_path(args.explainer_projector))
        lora_dir = str(resolve_existing_path(args.explainer_lora_dir))
    return projector, lora_dir


def infer_projector_input_dim(state_dict: Dict[str, torch.Tensor], fallback: int) -> int:
    for key, value in state_dict.items():
        if key.endswith("weight") and value.ndim == 2:
            return int(value.shape[1])
    return fallback


def load_explainer(
    projector_ckpt_path: str,
    lora_dir: str,
    model_name: str,
    projector_type: str,
    obs_act_pair_fusion: str,
    is_oracular: bool,
    device: torch.device,
) -> Tuple[LIBEROPolicyExplainer, int]:
    ckpt = torch.load(projector_ckpt_path, map_location="cpu")
    latent_sd = ckpt["latent_projector_state_dict"]
    action_sd = ckpt["action_projector_state_dict"]

    latent_dim = infer_projector_input_dim(latent_sd, fallback=512)
    action_dim = infer_projector_input_dim(action_sd, fallback=70)
    use_verdict_head = "verdict_head_state_dict" in ckpt

    model = LIBEROPolicyExplainer(
        latent_dim=latent_dim,
        action_dim=action_dim,
        projector_type=projector_type,
        stage="eval_stage3",
        #stage="eval",
        model_name=model_name,
        obs_act_pair_fusion=obs_act_pair_fusion,
        is_oracular=is_oracular,
        use_verdict_head=use_verdict_head,
    )
    model.base = PeftModel.from_pretrained(model.base, lora_dir, is_trainable=False)
    model._load_projector_state_dict(model.latent_projector, latent_sd)
    model._load_projector_state_dict(model.action_projector, action_sd)
    if use_verdict_head:
        model.verdict_head.load_state_dict(ckpt["verdict_head_state_dict"])
    model.to(device)
    model.eval()

    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    print(
        f"[Explainer] Loaded projector={projector_ckpt_path} lora={lora_dir} "
        f"(latent_dim={latent_dim}, action_dim={action_dim}, "
        f"use_verdict_head={use_verdict_head})"
    )
    return model, action_dim


def load_mlp_classifier(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[LatentBinaryClassifier, int, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    latent_dim = int(ckpt["latent_dim"])
    action_dim = int(ckpt["action_dim"])
    model = LatentBinaryClassifier(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=int(saved_args.get("hidden_dim", 256)),
        num_layers=int(saved_args.get("num_layers", 2)),
        dropout=float(saved_args.get("dropout", 0.1)),
        use_actions=not bool(saved_args.get("no_actions", False)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    latent_source = str(saved_args.get("latent_source", "visual"))
    print(
        f"[MLP] Loaded {ckpt_path} "
        f"(latent_dim={latent_dim}, action_dim={action_dim}, latent_source={latent_source})"
    )
    return model, action_dim, latent_source


def make_env(bddl_file: str, camera_size: int):
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": camera_size,
        "camera_widths": camera_size,
    }
    return OffScreenRenderEnv(**env_args)


def get_frame(obs: Dict[str, np.ndarray]) -> np.ndarray:
    a = obs["agentview_image"]
    b = obs.get("robot0_eye_in_hand_image", a)
    return np.concatenate([a, b], axis=1)


def deep_copy_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    copied = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            copied[k] = v.copy()
        else:
            copied[k] = v
    return copied


def warmup_env(env, warmup_steps: int):
    obs = env.reset()
    zero = np.zeros(7, dtype=np.float32)
    for _ in range(warmup_steps):
        obs, _, _, _ = env.step(zero)
    return obs


def reset_to_init_state(env, init_states, episode_idx: int, warmup_steps: int):
    env.reset()
    init_idx = episode_idx % int(init_states.shape[0])
    obs = env.set_init_state(init_states[init_idx])
    zero = np.zeros(7, dtype=np.float32)
    for _ in range(warmup_steps):
        obs, _, _, _ = env.step(zero)
    return obs


def env_snapshot(env):
    return env.sim.get_state().flatten().copy()


def env_restore(env, snapshot):
    env.sim.set_state_from_flattened(snapshot)
    env.sim.forward()


def env_success(env) -> bool:
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    if hasattr(env, "check_success"):
        return bool(env.check_success())
    return False


def adapt_chunk_for_explainer(chunk_actions: np.ndarray, target_chunk_size: int) -> np.ndarray:
    curr = chunk_actions.shape[0]
    if curr == target_chunk_size:
        return chunk_actions
    if curr > target_chunk_size:
        return chunk_actions[:target_chunk_size]
    pad_n = target_chunk_size - curr
    tail = np.repeat(chunk_actions[-1:, :], pad_n, axis=0)
    return np.concatenate([chunk_actions, tail], axis=0)


def get_policy_latent(policy, obs_dict, task_emb, latent_source: str = "context") -> torch.Tensor:
    if latent_source == "context":
        return policy.get_context_latent(obs_dict, task_emb, pool="mean")
    if latent_source == "visual":
        image_tokens = []
        first_rgb_key = next(iter(policy.image_encoders.keys()))
        img = obs_dict[first_rgb_key]
        for cam_key, encoder in policy.image_encoders.items():
            # Match offline LIBEROSuccFailLatentDataset/extract_latent_ale:
            # the saved agentview frame is fed through each camera encoder.
            image_tokens.append(encoder(img, langs=task_emb))
        return torch.stack(image_tokens, dim=1).mean(dim=1)
    raise ValueError(f"Unknown latent_source: {latent_source}")


@torch.no_grad()
def propose_chunks_and_latent(
    agent: ACTAgent,
    obs: Dict[str, np.ndarray],
    task_instruction: str,
    num_samples: int,
    latent_source: str = "context",
) -> Tuple[np.ndarray, torch.Tensor]:
    """Sample multiple candidate chunks from the same observation in one forward pass."""
    data = agent.preprocess_obs(obs)
    task_emb = agent._get_task_emb(task_instruction)
    data["task_emb"] = task_emb
    dist = agent.policy.forward(data["obs"], data["task_emb"])
    samples = dist.sample((num_samples,))  # (N, B=1, K, 7)
    chunk_actions = samples[:, 0].detach().cpu().numpy().astype(np.float32)  # (N, K, 7)
    latent = get_policy_latent(agent.policy, data["obs"], data["task_emb"], latent_source)
    latent = latent[0].detach()  # (E,)
    return chunk_actions, latent


@torch.no_grad()
def propose_chunk_and_latent(
    agent: ACTAgent,
    obs: Dict[str, np.ndarray],
    task_instruction: str,
    latent_source: str = "context",
) -> Tuple[np.ndarray, torch.Tensor]:
    chunks, latent = propose_chunks_and_latent(
        agent=agent,
        obs=obs,
        task_instruction=task_instruction,
        num_samples=1,
        latent_source=latent_source,
    )
    return chunks[0], latent


def build_stage3_prompt(task_instruction: str) -> str:
    # Must match the LIBEROCombinedDataset user-prompt template used at stage-3 training.
    return (
        f"You are a robot performing the task: '{task_instruction}'. "
        f"You are given the current observation and the action chunk that was executed. "
        f"First, classify whether this was a good action chunk that makes progress "
        f"toward completing the task using <success> or <failure>. "
        f"Then describe in one sentence what the robot did."
    )


def _chat_prompt_to_ids(
    tokenizer,
    messages,
    device: torch.device,
    add_generation_prompt: bool = True,
) -> torch.Tensor:
    """Robustly convert chat messages into (1, L) token ids across tokenizer variants."""
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )

    if hasattr(out, "input_ids"):
        ids = out["input_ids"]
    else:
        ids = out

    if isinstance(ids, str):
        ids = tokenizer(ids, add_special_tokens=False)["input_ids"]

    if isinstance(ids, torch.Tensor):
        tid = ids.to(device=device, dtype=torch.long)
        if tid.ndim == 1:
            tid = tid.unsqueeze(0)
        return tid

    tid = torch.tensor(ids, dtype=torch.long, device=device)
    if tid.ndim == 1:
        tid = tid.unsqueeze(0)
    return tid


def _decode_generated_only(
    tokenizer,
    out_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
) -> List[str]:
    """Decode only generated continuation.

    When generate() is called with inputs_embeds (as we do), HuggingFace returns
    ONLY the new tokens — the prompt cannot be echoed back from embeddings.
    So out_ids is already just the generated tokens; decode it directly.

    If generate() ever returns the full (prompt + generated) sequence (i.e. its
    length equals prompt_ids length + new tokens), we strip the prompt prefix.
    We detect this by checking if out_ids[i] starts with the same ids as
    prompt_ids[i] up to min(len). To keep it simple and robust: since we always
    call generate() with inputs_embeds, just decode the full out_ids tensor.
    """
    decoded: List[str] = []
    for i in range(out_ids.shape[0]):
        gen = out_ids[i]
        # Keep special tokens so <success>/<failure> labels are preserved in the text.
        txt = tokenizer.decode(gen, skip_special_tokens=False).strip()
        decoded.append(txt)
    return decoded


def _parse_label(text: str) -> str:
    """Extract 'success' or 'failure' from generated text.

    Handles both <success>/<failure> special-token format (CombinedDataset)
    and bare 'success'/'failure' word format (SuccFailDataset).
    Returns the FIRST match so the label token (always generated first) wins.
    """
    t = text.lower()
    # Match <success>, <failure>, or bare word
    matches = re.findall(r"<(success|failure)>|\b(success|failure)\b", t)
    for m in matches:
        label = m[0] or m[1]
        if label:
            return label
    return "failure"


@torch.no_grad()
def classify_chunk_success(
    explainer: LIBEROPolicyExplainer,
    latent_obs: torch.Tensor,
    chunk_actions: np.ndarray,
    task_instruction: str,
    max_new_tokens: int = 1,
) -> Tuple[str, str]:
    tok = explainer.tokenizer
    user_text = build_stage3_prompt(task_instruction)
    prompt = [{"role": "user", "content": user_text}]
    prompt_ids = _chat_prompt_to_ids(
        tok,
        prompt,
        device=latent_obs.device,
        add_generation_prompt=True,
    )

    actions_tensor = torch.from_numpy(chunk_actions).to(latent_obs.device).unsqueeze(0)
    latent_tensor = latent_obs.unsqueeze(0)

    if getattr(explainer, "verdict_head", None) is not None:
        prefix = explainer._fuse_embeds(latent_tensor, actions_tensor)
        logit = explainer.verdict_head(prefix.mean(dim=1)).float()
        prob = torch.sigmoid(logit)[0].item()
        label = "success" if prob >= 0.5 else "failure"
        return label, f"binary_head_success_prob={prob:.4f}"

    dummy = prompt_ids
    dummy_attn = torch.ones_like(dummy)
    out_ids = explainer.gen_from_batch(
        latent_obs=latent_tensor,
        actions=actions_tensor,
        input_ids=dummy,
        labels=dummy,
        attention_mask=dummy_attn,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = _decode_generated_only(tok, out_ids, prompt_ids)[0]
    label = _parse_label(decoded)
    return label, decoded


@torch.no_grad()
def classify_chunk_success_batch(
    explainer: LIBEROPolicyExplainer,
    latent_obs: torch.Tensor,
    chunk_actions_batch: np.ndarray,
    task_instruction: str,
    max_new_tokens: int = 4,
) -> Tuple[List[str], List[str]]:
    tok = explainer.tokenizer
    num_samples = int(chunk_actions_batch.shape[0])
    if num_samples == 0:
        return [], []

    user_text = build_stage3_prompt(task_instruction)
    prompt = [{"role": "user", "content": user_text}]
    prompt_ids_single = _chat_prompt_to_ids(
        tok,
        prompt,
        device=latent_obs.device,
        add_generation_prompt=True,
    )
    prompt_ids = prompt_ids_single.repeat(num_samples, 1)

    actions_tensor = torch.from_numpy(chunk_actions_batch).to(latent_obs.device)
    latent_tensor = latent_obs.unsqueeze(0).repeat(num_samples, 1)

    if getattr(explainer, "verdict_head", None) is not None:
        prefix = explainer._fuse_embeds(latent_tensor, actions_tensor)
        logits = explainer.verdict_head(prefix.mean(dim=1)).float()
        probs = torch.sigmoid(logits)
        labels = ["success" if p.item() >= 0.5 else "failure" for p in probs]
        texts = [f"binary_head_success_prob={p.item():.4f}" for p in probs]
        return labels, texts

    dummy_attn = torch.ones_like(prompt_ids)
    out_ids = explainer.gen_from_batch(
        latent_obs=latent_tensor,
        actions=actions_tensor,
        input_ids=prompt_ids,
        labels=prompt_ids,
        attention_mask=dummy_attn,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = _decode_generated_only(tok, out_ids, prompt_ids)
    labels = [_parse_label(d) for d in decoded]
    return labels, decoded


def execute_chunk(env, obs, chunk_actions, writer=None, max_steps_left=400):
    steps = 0
    success = False
    curr_obs = obs

    for action in chunk_actions:
        if steps >= max_steps_left:
            break
        curr_obs, _, done, _ = env.step(action)
        steps += 1
        if writer is not None:
            writer.append_data(get_frame(curr_obs))
        if bool(done) or env_success(env):
            success = True
            break

    return curr_obs, steps, success


def rollout_baseline(
    env,
    start_snapshot,
    start_obs,
    agent,
    task_instruction,
    max_steps,
    writer=None,
) -> RolloutStats:
    env_restore(env, start_snapshot)
    obs = deep_copy_obs(start_obs)
    agent.reset()

    if writer is not None:
        writer.append_data(get_frame(obs))

    steps = 0
    success = False
    while steps < max_steps:
        action = agent.act(obs, task_instruction)
        obs, _, done, _ = env.step(action)
        steps += 1
        if writer is not None:
            writer.append_data(get_frame(obs))
        success = bool(done) or env_success(env)
        if success:
            break

    return RolloutStats(success=success, steps=steps)


def save_refused_video(path: Path, frames: List[np.ndarray]):
    if imageio is None:
        raise ImportError("imageio is required for video writing. Install with: pip install imageio")
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=20) as writer:
        for fr in frames:
            writer.append_data(fr)


def rollout_gated(
    env,
    start_snapshot,
    start_obs,
    agent,
    explainer,
    explainer_chunk_size,
    task_instruction,
    max_steps,
    max_refusal_attempts,
    writer=None,
    refused_video_dir: Optional[Path] = None,
    task_name: str = "",
    episode_idx: int = 0,
    proposal_batch_size: int = 8,
    print_decision_logs: bool = True,
    intervene_every_n: int = 1,
) -> RolloutStats:
    env_restore(env, start_snapshot)
    obs = deep_copy_obs(start_obs)

    if writer is not None:
        writer.append_data(get_frame(obs))

    steps = 0
    success = False
    refused_total = 0
    accepted_attempts: List[int] = []
    forced_accepts = 0
    decision_idx = 0
    decision_logs: List[Dict] = []

    while steps < max_steps and not success:
        proposal_snapshot = env_snapshot(env)

        # Non-intervention step: propose 1 chunk, execute without gating
        if intervene_every_n > 1 and (decision_idx % intervene_every_n != 0):
            chunk, _ = propose_chunk_and_latent(agent, obs, task_instruction)
            remaining = max_steps - steps
            obs, took, succ = execute_chunk(
                env, obs, chunk, writer=writer, max_steps_left=remaining,
            )
            steps += took
            success = succ
            accepted_attempts.append(1)
            decision_logs.append({
                "task": task_name,
                "episode": episode_idx,
                "decision": decision_idx,
                "attempt": 1,
                "label": "skipped",
                "accepted": True,
                "classifier_text": "no_intervention",
            })
            if print_decision_logs:
                print(
                    f"[NoIntervention] task={task_name} ep={episode_idx} "
                    f"decision={decision_idx}"
                )
            decision_idx += 1
            continue

        accepted_chunk = None
        fallback_chunk = None
        accepted_attempt = None
        attempt = 1
        while attempt <= max_refusal_attempts:
            remaining = max_refusal_attempts - attempt + 1
            batch_n = min(max(1, proposal_batch_size), remaining)

            candidate_chunks, latent = propose_chunks_and_latent(
                agent=agent,
                obs=obs,
                task_instruction=task_instruction,
                num_samples=batch_n,
            )
            chunks_for_explainer = np.stack(
                [adapt_chunk_for_explainer(c, explainer_chunk_size) for c in candidate_chunks],
                axis=0,
            )
            candidate_exec_chunks = candidate_chunks
            latent_dev = latent.to(next(explainer.parameters()).device)
            labels, raw_texts = classify_chunk_success_batch(
                explainer=explainer,
                latent_obs=latent_dev,
                chunk_actions_batch=chunks_for_explainer,
                task_instruction=task_instruction,
            )

            accepted_idx = None
            for i, lab in enumerate(labels):
                if lab == "success":
                    accepted_idx = i
                    break

            last_considered = accepted_idx if accepted_idx is not None else (batch_n - 1)
            for i in range(last_considered + 1):
                curr_attempt = attempt + i
                label = labels[i]
                raw_text = raw_texts[i]
                is_accept = accepted_idx is not None and i == accepted_idx

                decision_logs.append(
                    {
                        "task": task_name,
                        "episode": episode_idx,
                        "decision": decision_idx,
                        "attempt": curr_attempt,
                        "label": label,
                        "accepted": bool(is_accept),
                        "classifier_text": raw_text,
                    }
                )

                if is_accept:
                    accepted_chunk = candidate_exec_chunks[i]
                    accepted_attempt = curr_attempt
                    if print_decision_logs:
                        print(
                            f"[Accepted] task={task_name} ep={episode_idx} decision={decision_idx} "
                            f"attempt={curr_attempt} label={label} text='{raw_text[:120]}'"
                        )
                    break

                refused_total += 1
                fallback_chunk = candidate_exec_chunks[i]
                if refused_video_dir is not None:
                    env_restore(env, proposal_snapshot)
                    obs_tmp = deep_copy_obs(obs)
                    frames = [get_frame(obs_tmp)]
                    for action in candidate_exec_chunks[i]:
                        obs_tmp, _, done_tmp, _ = env.step(action)
                        frames.append(get_frame(obs_tmp))
                        if bool(done_tmp):
                            break
                    refused_name = (
                        f"{task_name}_ep{episode_idx:03d}_decision{decision_idx:03d}_"
                        f"attempt{curr_attempt:02d}_refused_{label}.mp4"
                    )
                    save_refused_video(refused_video_dir / refused_name, frames)
                    env_restore(env, proposal_snapshot)

                if print_decision_logs:
                    print(
                        f"[Refused] task={task_name} ep={episode_idx} decision={decision_idx} "
                        f"attempt={curr_attempt} label={label} text='{raw_text[:120]}'"
                    )

            if accepted_chunk is not None:
                break
            attempt += batch_n

        if accepted_chunk is None:
            forced_accepts += 1
            if fallback_chunk is None:
                fallback_chunk, _ = propose_chunk_and_latent(agent, obs, task_instruction)
            accepted_chunk = fallback_chunk
            accepted_attempt = max_refusal_attempts + 1
            if print_decision_logs:
                print(
                    f"[ForcedAccept] task={task_name} ep={episode_idx} decision={decision_idx} "
                    f"attempt={accepted_attempt} reason=max_refusal_attempts"
                )

        accepted_attempts.append(accepted_attempt)

        env_restore(env, proposal_snapshot)
        remaining = max_steps - steps
        obs, took, succ = execute_chunk(
            env,
            obs,
            accepted_chunk,
            writer=writer,
            max_steps_left=remaining,
        )
        steps += took
        success = succ
        decision_idx += 1

    return RolloutStats(
        success=success,
        steps=steps,
        refused_count=refused_total,
        accepted_attempts=accepted_attempts,
        forced_accepts=forced_accepts,
        chunks_executed=len(accepted_attempts),
        decision_logs=decision_logs,
    )


# ══════════════════════════════════════════════════════════════════════
#  Parallel / Vectorised rollout helpers
# ══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def batch_propose_chunks_and_latents(
    agent: ACTAgent,
    obs_list: List[Dict[str, np.ndarray]],
    task_instruction: str,
    num_samples: int,
    latent_source: str = "context",
) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
    """Propose candidate chunks for *multiple* environments in one batched forward pass.

    Returns:
        per_env_chunks:  list of (num_samples, K, 7) numpy arrays
        per_env_latents: list of (E,) tensors
    """
    M = len(obs_list)
    if M == 0:
        return [], []

    data_list = [agent.preprocess_obs(obs) for obs in obs_list]
    task_emb = agent._get_task_emb(task_instruction)  # (1, 768)

    batch_obs: Dict[str, torch.Tensor] = {}
    for key in data_list[0]["obs"]:
        batch_obs[key] = torch.cat([d["obs"][key] for d in data_list], dim=0)
    batch_task_emb = task_emb.repeat(M, 1)  # (M, 768)

    dist = agent.policy.forward(batch_obs, batch_task_emb)
    samples = dist.sample((num_samples,))  # (num_samples, M, K, 7)
    samples_np = samples.detach().cpu().numpy().astype(np.float32)

    latent = get_policy_latent(agent.policy, batch_obs, batch_task_emb, latent_source)  # (M, E)

    per_env_chunks = [samples_np[:, j] for j in range(M)]
    per_env_latents = [latent[j].detach() for j in range(M)]
    return per_env_chunks, per_env_latents


@torch.no_grad()
def classify_multi_env_batch(
    explainer: LIBEROPolicyExplainer,
    latents_batch: torch.Tensor,
    chunks_batch: np.ndarray,
    task_instruction: str,
    max_new_tokens: int = 4,
) -> Tuple[List[str], List[str]]:
    """Like classify_chunk_success_batch but with pre-batched (heterogeneous) latents."""
    tok = explainer.tokenizer
    N = int(latents_batch.shape[0])
    if N == 0:
        return [], []

    user_text = build_stage3_prompt(task_instruction)
    prompt = [{"role": "user", "content": user_text}]
    prompt_ids_single = _chat_prompt_to_ids(
        tok, prompt, device=latents_batch.device, add_generation_prompt=True
    )
    prompt_ids = prompt_ids_single.repeat(N, 1)

    actions_tensor = torch.from_numpy(chunks_batch).to(latents_batch.device)
    if getattr(explainer, "verdict_head", None) is not None:
        prefix = explainer._fuse_embeds(latents_batch, actions_tensor)
        logits = explainer.verdict_head(prefix.mean(dim=1)).float()
        probs = torch.sigmoid(logits)
        labels = ["success" if p.item() >= 0.5 else "failure" for p in probs]
        texts = [f"binary_head_success_prob={p.item():.4f}" for p in probs]
        return labels, texts

    dummy_attn = torch.ones_like(prompt_ids)
    out_ids = explainer.gen_from_batch(
        latent_obs=latents_batch,
        actions=actions_tensor,
        input_ids=prompt_ids,
        labels=prompt_ids,
        attention_mask=dummy_attn,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = _decode_generated_only(tok, out_ids, prompt_ids)
    labels = [_parse_label(d) for d in decoded]
    return labels, decoded


@torch.no_grad()
def classify_mlp_multi_env_batch(
    classifier: LatentBinaryClassifier,
    latents_batch: torch.Tensor,
    chunks_batch: np.ndarray,
) -> Tuple[List[str], List[str]]:
    device = next(classifier.parameters()).device
    actions_tensor = torch.from_numpy(chunks_batch).to(device)
    logits = classifier(latents_batch.to(device), actions_tensor)
    probs = torch.sigmoid(logits)
    labels = ["success" if float(p) >= 0.5 else "failure" for p in probs]
    texts = [f"mlp_success_prob={float(p):.4f}" for p in probs]
    return labels, texts


def rollout_baseline_parallel(
    envs: List,
    init_states,
    episode_indices: List[int],
    agent: ACTAgent,
    task_instruction: str,
    max_steps: int,
    warmup_steps: int,
    writers: Optional[List] = None,
) -> List[RolloutStats]:
    """Run baseline ACT rollout on *multiple* environments with batched GPU inference."""
    N = len(envs)

    # Initialise all environments
    obs_list: List[Dict[str, np.ndarray]] = []
    for i in range(N):
        obs = reset_to_init_state(envs[i], init_states, episode_indices[i], warmup_steps)
        obs_list.append(deep_copy_obs(obs))
        if writers and writers[i]:
            writers[i].append_data(get_frame(obs))

    steps = [0] * N
    success = [False] * N
    active = [True] * N

    while any(active):
        active_indices = [i for i in range(N) if active[i]]
        if not active_indices:
            break

        # Batched chunk proposal for all active envs
        obs_batch = [obs_list[i] for i in active_indices]
        per_env_chunks, _ = batch_propose_chunks_and_latents(
            agent, obs_batch, task_instruction, num_samples=1,
        )

        # Execute one chunk per env (sequential env stepping – fast CPU ops)
        for j, i in enumerate(active_indices):
            chunk = per_env_chunks[j][0]  # (K, 7)
            remaining = max_steps - steps[i]
            obs_list[i], took, succ = execute_chunk(
                envs[i], obs_list[i], chunk,
                writer=(writers[i] if writers else None),
                max_steps_left=remaining,
            )
            steps[i] += took
            if succ:
                success[i] = True
                active[i] = False
            elif steps[i] >= max_steps:
                active[i] = False

    return [RolloutStats(success=success[i], steps=steps[i]) for i in range(N)]


def rollout_gated_parallel(
    envs: List,
    init_states,
    episode_indices: List[int],
    agent: ACTAgent,
    steering_model,
    steering_chunk_size: int,
    steering_model_type: str,
    latent_source: str,
    task_instruction: str,
    max_steps: int,
    max_refusal_attempts: int,
    warmup_steps: int,
    proposal_batch_size: int = 8,
    writers: Optional[List] = None,
    refused_video_dir: Optional[Path] = None,
    task_name: str = "",
    print_decision_logs: bool = True,
    intervene_every_n: int = 1,
) -> List[RolloutStats]:
    """Run gated ACT rollout on *multiple* environments with batched GPU inference.

    GPU calls (ACT forward, explainer classify) are batched across all active
    environments so the GPU is kept busy even when individual episodes finish at
    different times.  Environment stepping remains sequential (fast CPU work).
    """
    N = len(envs)
    steering_device = next(steering_model.parameters()).device

    # ── initialise all environments ──────────────────────────────────
    obs_list: List[Dict[str, np.ndarray]] = []
    for i in range(N):
        obs = reset_to_init_state(envs[i], init_states, episode_indices[i], warmup_steps)
        obs_list.append(deep_copy_obs(obs))
        if writers and writers[i]:
            writers[i].append_data(get_frame(obs))

    # per-env bookkeeping
    steps = [0] * N
    success_flag = [False] * N
    active = [True] * N
    refused_total = [0] * N
    forced_accepts_count = [0] * N
    accepted_attempts_per_env: List[List[int]] = [[] for _ in range(N)]
    decision_logs_per_env: List[List[Dict]] = [[] for _ in range(N)]
    decision_idx = [0] * N

    # ── main loop ────────────────────────────────────────────────────
    while any(active):
        active_indices = [i for i in range(N) if active[i]]
        if not active_indices:
            break

        # ── separate active envs into intervention vs no-intervention ──
        intervene_indices = []
        no_intervene_indices = []
        for i in active_indices:
            if intervene_every_n > 1 and (decision_idx[i] % intervene_every_n != 0):
                no_intervene_indices.append(i)
            else:
                intervene_indices.append(i)

        # ── handle no-intervention envs: propose 1 chunk, execute directly ──
        if no_intervene_indices:
            obs_ni = [obs_list[i] for i in no_intervene_indices]
            ni_chunks, _ = batch_propose_chunks_and_latents(
                agent, obs_ni, task_instruction, num_samples=1,
            )
            for j, i in enumerate(no_intervene_indices):
                chunk = ni_chunks[j][0]  # (K, 7)
                remaining = max_steps - steps[i]
                obs_list[i], took, succ = execute_chunk(
                    envs[i], obs_list[i], chunk,
                    writer=(writers[i] if writers else None),
                    max_steps_left=remaining,
                )
                steps[i] += took
                accepted_attempts_per_env[i].append(1)
                decision_logs_per_env[i].append({
                    "task": task_name,
                    "episode": episode_indices[i],
                    "decision": decision_idx[i],
                    "attempt": 1,
                    "label": "skipped",
                    "accepted": True,
                    "classifier_text": "no_intervention",
                })
                if print_decision_logs:
                    print(
                        f"[NoIntervention] task={task_name} ep={episode_indices[i]} "
                        f"decision={decision_idx[i]}"
                    )
                if succ:
                    success_flag[i] = True
                    active[i] = False
                elif steps[i] >= max_steps:
                    active[i] = False
                decision_idx[i] += 1

        # If no envs need intervention this iteration, continue
        if not intervene_indices:
            continue

        # From here on, only process envs that need intervention
        active_indices = intervene_indices

        # snapshot each active env at the start of this decision
        proposal_snapshots: Dict[int, np.ndarray] = {
            i: env_snapshot(envs[i]) for i in active_indices
        }

        # refusal-loop state per env
        env_attempt: Dict[int, int] = {i: 1 for i in active_indices}
        env_fallback: Dict[int, Optional[np.ndarray]] = {i: None for i in active_indices}
        env_accepted: Dict[int, Optional[np.ndarray]] = {i: None for i in active_indices}
        env_accepted_at: Dict[int, Optional[int]] = {i: None for i in active_indices}

        need_proposal = list(active_indices)  # env indices still seeking a chunk

        # ── refusal loop (batched across envs) ───────────────────────
        while need_proposal:
            # how many proposals each env should try this round
            per_env_batch_n = []
            for i in need_proposal:
                remaining_attempts = max_refusal_attempts - env_attempt[i] + 1
                per_env_batch_n.append(min(max(1, proposal_batch_size), remaining_attempts))
            max_batch_n = max(per_env_batch_n)

            # ---- batched ACT forward --------------------------------
            obs_for_proposal = [obs_list[i] for i in need_proposal]
            per_env_chunks, per_env_latents = batch_propose_chunks_and_latents(
                agent, obs_for_proposal, task_instruction, max_batch_n,
                latent_source=latent_source,
            )

            # ---- build flat arrays for batched classification -------
            all_latents_parts: List[torch.Tensor] = []
            all_chunks_parts: List[np.ndarray] = []
            all_exec_chunks: List[np.ndarray] = []  # raw chunks per env for execution
            env_chunk_counts: List[int] = []

            for j, i in enumerate(need_proposal):
                n = per_env_batch_n[j]
                exec_chunks_j = per_env_chunks[j][:n]       # (n, K, 7)
                steering_chunks_j = np.stack(
                    [adapt_chunk_for_explainer(c, steering_chunk_size) for c in exec_chunks_j]
                )
                lat = per_env_latents[j].to(steering_device)
                all_latents_parts.append(lat.unsqueeze(0).expand(n, -1))
                all_chunks_parts.append(steering_chunks_j)
                all_exec_chunks.append(exec_chunks_j)
                env_chunk_counts.append(n)

            cat_latents = torch.cat(all_latents_parts, dim=0)
            cat_chunks = np.concatenate(all_chunks_parts, axis=0)

            # ---- batched steering classify --------------------------
            if steering_model_type == "explainer":
                labels_all, texts_all = classify_multi_env_batch(
                    steering_model, cat_latents, cat_chunks, task_instruction,
                )
            elif steering_model_type == "mlp":
                labels_all, texts_all = classify_mlp_multi_env_batch(
                    steering_model, cat_latents, cat_chunks,
                )
            else:
                raise ValueError(f"Unknown steering_model_type: {steering_model_type}")

            # ---- per-env accept / refuse logic ----------------------
            offset = 0
            still_need: List[int] = []

            for j, i in enumerate(need_proposal):
                n = env_chunk_counts[j]
                labels = labels_all[offset : offset + n]
                texts = texts_all[offset : offset + n]
                exec_chunks = all_exec_chunks[j]

                # find first accepted
                accepted_idx = None
                for k, lab in enumerate(labels):
                    if lab == "success":
                        accepted_idx = k
                        break

                last_considered = accepted_idx if accepted_idx is not None else (n - 1)

                for k in range(last_considered + 1):
                    curr_attempt = env_attempt[i] + k
                    is_accept = (accepted_idx is not None and k == accepted_idx)

                    decision_logs_per_env[i].append({
                        "task": task_name,
                        "episode": episode_indices[i],
                        "decision": decision_idx[i],
                        "attempt": curr_attempt,
                        "label": labels[k],
                        "accepted": bool(is_accept),
                        "classifier_text": texts[k],
                    })

                    if is_accept:
                        env_accepted[i] = exec_chunks[k]
                        env_accepted_at[i] = curr_attempt
                        if print_decision_logs:
                            print(
                                f"[Accepted] task={task_name} ep={episode_indices[i]} "
                                f"decision={decision_idx[i]} attempt={curr_attempt} "
                                f"label={labels[k]} text='{texts[k][:120]}'"
                            )
                        break

                    refused_total[i] += 1
                    env_fallback[i] = exec_chunks[k]

                    # optional: record refused rollout video
                    if refused_video_dir is not None:
                        env_restore(envs[i], proposal_snapshots[i])
                        obs_tmp = deep_copy_obs(obs_list[i])
                        frames = [get_frame(obs_tmp)]
                        for action in exec_chunks[k]:
                            obs_tmp, _, done_tmp, _ = envs[i].step(action)
                            frames.append(get_frame(obs_tmp))
                            if bool(done_tmp):
                                break
                        refused_name = (
                            f"{task_name}_ep{episode_indices[i]:03d}_"
                            f"decision{decision_idx[i]:03d}_"
                            f"attempt{curr_attempt:02d}_refused_{labels[k]}.mp4"
                        )
                        save_refused_video(refused_video_dir / refused_name, frames)
                        env_restore(envs[i], proposal_snapshots[i])

                    if print_decision_logs:
                        print(
                            f"[Refused] task={task_name} ep={episode_indices[i]} "
                            f"decision={decision_idx[i]} attempt={curr_attempt} "
                            f"label={labels[k]} text='{texts[k][:120]}'"
                        )

                # decide next step for this env
                if env_accepted[i] is not None:
                    pass  # done proposing
                else:
                    env_attempt[i] += n
                    if env_attempt[i] > max_refusal_attempts:
                        forced_accepts_count[i] += 1
                        if env_fallback[i] is None:
                            fb, _ = propose_chunk_and_latent(
                                agent, obs_list[i], task_instruction,
                            )
                            env_fallback[i] = fb
                        env_accepted[i] = env_fallback[i]
                        env_accepted_at[i] = max_refusal_attempts + 1
                        if print_decision_logs:
                            print(
                                f"[ForcedAccept] task={task_name} ep={episode_indices[i]} "
                                f"decision={decision_idx[i]} attempt={env_accepted_at[i]} "
                                f"reason=max_refusal_attempts"
                            )
                    else:
                        still_need.append(i)

                offset += n

            need_proposal = still_need

        # ── execute accepted chunks (sequential env stepping) ────────
        for i in active_indices:
            chunk = env_accepted[i]
            if chunk is None:
                continue
            accepted_attempts_per_env[i].append(env_accepted_at[i])
            env_restore(envs[i], proposal_snapshots[i])
            remaining = max_steps - steps[i]
            obs_list[i], took, succ = execute_chunk(
                envs[i], obs_list[i], chunk,
                writer=(writers[i] if writers else None),
                max_steps_left=remaining,
            )
            steps[i] += took
            if succ:
                success_flag[i] = True
                active[i] = False
            elif steps[i] >= max_steps:
                active[i] = False
            decision_idx[i] += 1

    # ── build results ────────────────────────────────────────────────
    results: List[RolloutStats] = []
    for i in range(N):
        results.append(RolloutStats(
            success=success_flag[i],
            steps=steps[i],
            refused_count=refused_total[i],
            accepted_attempts=accepted_attempts_per_env[i],
            forced_accepts=forced_accepts_count[i],
            chunks_executed=len(accepted_attempts_per_env[i]),
            decision_logs=decision_logs_per_env[i],
        ))
    return results


def discover_tasks_for_suite(
    suite: str,
    task_limit: Optional[int],
    task_order_index: int,
) -> List[Dict[str, str]]:
    benchmark_name = SUITE_TO_BENCHMARK[suite]
    benchmark = get_benchmark(benchmark_name)(task_order_index)
    bddl_root = Path(get_libero_path("bddl_files"))
    init_states_root = Path(get_libero_path("init_states"))

    task_ids = list(range(benchmark.n_tasks))
    if task_limit is not None:
        task_ids = task_ids[:task_limit]

    tasks: List[Dict[str, str]] = []
    for task_idx in task_ids:
        bt = benchmark.get_task(task_idx)
        bddl_file = bddl_root / bt.problem_folder / bt.bddl_file
        init_states_file = init_states_root / bt.problem_folder / bt.init_states_file
        info = BDDLUtils.get_problem_info(str(bddl_file))
        tasks.append(
            {
                "task_id": bddl_file.stem,
                "benchmark_task_id": task_idx,
                "env_problem_name": info["problem_name"],
                "bddl_file": str(bddl_file),
                "init_states_file": str(init_states_file),
                "language_instruction": bt.language,
            }
        )
    return tasks


def safe_mean(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def write_results_csv(csv_path: Path, task_results: Dict[str, Dict], summary: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "suite",
        "task_id",
        "env_problem_name",
        "episodes",
        "gated_success_rate",
        "gated_avg_steps",
        "avg_refused_actions_per_chunk",
        "total_chunks_done",
        "avg_gating_metric",
        "gated_avg_refused_per_episode",
        "gated_total_forced_accepts",
        "baseline_success_rate",
        "baseline_avg_steps",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task_id, task_res in task_results.items():
            writer.writerow(
                {
                    "row_type": "task",
                    "suite": summary["suite"],
                    "task_id": task_id,
                    "env_problem_name": task_res.get("env_problem_name", ""),
                    "episodes": task_res.get("episodes", 0),
                    "gated_success_rate": round(task_res.get("gated_success_rate", 0.0), 3),
                    "gated_avg_steps": round(task_res.get("gated_avg_steps", 0.0), 3),
                    "avg_refused_actions_per_chunk": round(task_res.get("avg_refused_actions_per_chunk", 0.0), 3),
                    "total_chunks_done": task_res.get("total_chunks_done", 0),
                    "avg_gating_metric": round(task_res.get("avg_gating_metric", 0.0), 3),
                    "gated_avg_refused_per_episode": round(task_res.get("gated_avg_refused_per_episode", 0.0), 3),
                    "gated_total_forced_accepts": task_res.get("gated_total_forced_accepts", 0),
                    "baseline_success_rate": round(task_res.get("baseline_success_rate", 0.0), 3) if task_res.get("baseline_success_rate", "") != "" else "",
                    "baseline_avg_steps": round(task_res.get("baseline_avg_steps", 0.0), 3) if task_res.get("baseline_avg_steps", "") != "" else "",
                }
            )

        writer.writerow(
            {
                "row_type": "suite",
                "suite": summary["suite"],
                "task_id": "__suite_avg__",
                "env_problem_name": "",
                "episodes": summary.get("episodes_per_task", 0),
                "gated_success_rate": round(summary.get("gated_avg_suite_success_rate", 0.0), 3),
                "gated_avg_steps": round(summary.get("gated_avg_suite_steps", 0.0), 3),
                "avg_refused_actions_per_chunk": round(summary.get("gated_avg_refused_actions_per_chunk", 0.0), 3),
                "total_chunks_done": summary.get("gated_total_chunks_done", 0),
                "avg_gating_metric": round(summary.get("gated_avg_gating_metric", 0.0), 3),
                "gated_avg_refused_per_episode": round(summary.get("gated_avg_refused_per_episode", 0.0), 3),
                "gated_total_forced_accepts": summary.get("gated_total_forced_accepts", 0),
                "baseline_success_rate": round(summary.get("baseline_avg_suite_success_rate", 0.0), 3) if summary.get("baseline_avg_suite_success_rate", "") != "" else "",
                "baseline_avg_steps": round(summary.get("baseline_avg_suite_steps", 0.0), 3) if summary.get("baseline_avg_suite_steps", "") != "" else "",
            }
        )


def main():
    parser = argparse.ArgumentParser(description="LIBERO action-refusal evaluation")
    parser.add_argument("--suite", type=str, required=True, choices=SUITE_CHOICES)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--task-limit", type=int, default=None,
                        help="Evaluate only first N tasks in suite")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--task-order-index", type=int, default=0,
                        help="Benchmark task ordering index (matches pipeline evaluator)")
    parser.add_argument("--warmup-steps", type=int, default=40)

    parser.add_argument("--act-checkpoint", type=str, required=True,
                        help="ACT checkpoint path for this run")

    parser.add_argument("--steering-model-type", type=str, default="explainer",
                        choices=["explainer", "mlp"],
                        help="Classifier used to accept/refuse proposed chunks")
    parser.add_argument("--mlp-ckpt", type=str, default=None,
                        help="Path to MLP classifier best.pt when --steering-model-type=mlp")

    parser.add_argument("--explainer-ckptdir", type=str, default=None,
                        help="Directory containing e=*_policy_explainer.pth and lora-* folder")
    parser.add_argument("--explainer-projector", type=str, default=None,
                        help="Path to e=*_policy_explainer.pth")
    parser.add_argument("--explainer-lora-dir", type=str, default=None,
                        help="Path to lora-* directory")
    parser.add_argument("--explainer-model-name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--projector-type", type=str, default="mlp2x_gelu")
    parser.add_argument("--obs-act-pair-fusion", type=str, default="sum", choices=["sum", "concat", "mlp"])
    parser.add_argument("--is-oracular", action="store_true", default=False)

    parser.add_argument("--max-refusal-attempts", type=int, default=8,
                        help="Max chunk proposals before forced accept")
    parser.add_argument("--intervene-every-n", type=int, default=1,
                        help="Apply explainer gating every N decision steps. "
                             "On non-intervention steps, a single chunk is proposed and executed without gating. "
                             "Default 1 means gating at every step.")

    parser.add_argument("--record-video", action="store_true", default=False,
                        help="Record rollout videos")
    parser.add_argument("--record-refused-rollouts", action="store_true", default=False,
                        help="Record hypothetical videos for each refused candidate")
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--proposal-batch-size", type=int, default=4,
                        help="How many chunk proposals to classify in parallel per refusal round")
    parser.add_argument("--num-parallel", type=int, default=1,
                        help="Number of environments to run in parallel per task (batches GPU inference)")
    parser.add_argument("--print-decision-logs", action=argparse.BooleanOptionalAction, default=True,
                        help="Print per-decision [Refused]/[Accepted]/[ForcedAccept] logs")
    parser.add_argument("--compare-baseline", action=argparse.BooleanOptionalAction, default=None,
                        help="If true, also run baseline ACT from same initial state")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="runs/refusal_eval")
    args = parser.parse_args()

    compare_baseline = args.compare_baseline
    if compare_baseline is None:
        compare_baseline = not args.record_video

    if args.record_refused_rollouts and not args.record_video:
        raise ValueError("--record-refused-rollouts requires --record-video")
    if args.record_video and imageio is None:
        raise ImportError("--record-video requested but imageio is not installed. Install with: pip install imageio")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    act_ckpt = resolve_act_checkpoint(args)
    if not Path(act_ckpt).exists():
        raise FileNotFoundError(f"ACT checkpoint not found: {act_ckpt}")
    print(f"[ACT] checkpoint={act_ckpt}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir = outdir / "videos"
    refused_dir = outdir / "refused_rollouts"
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    if args.record_refused_rollouts:
        refused_dir.mkdir(parents=True, exist_ok=True)

    agent = ACTAgent(act_ckpt, device=str(device)) # TODO: read this
    projector_ckpt = ""
    lora_dir = ""
    mlp_ckpt = ""
    if args.steering_model_type == "explainer":
        projector_ckpt, lora_dir = parse_explainer_paths(args)
        if not Path(projector_ckpt).exists():
            raise FileNotFoundError(f"Explainer projector not found: {projector_ckpt}")
        if not Path(lora_dir).exists():
            raise FileNotFoundError(f"Explainer LoRA dir not found: {lora_dir}")
        steering_model, steering_action_dim = load_explainer(
            projector_ckpt_path=projector_ckpt,
            lora_dir=lora_dir,
            model_name=args.explainer_model_name,
            projector_type=args.projector_type,
            obs_act_pair_fusion=args.obs_act_pair_fusion,
            is_oracular=args.is_oracular,
            device=device,
        )
        latent_source = "context"
    else:
        if args.mlp_ckpt is None:
            raise ValueError("--mlp-ckpt is required when --steering-model-type=mlp")
        mlp_ckpt = str(resolve_existing_path(args.mlp_ckpt))
        steering_model, steering_action_dim, latent_source = load_mlp_classifier(
            mlp_ckpt, device=device,
        )

    steering_chunk_size = steering_action_dim // 7
    if steering_action_dim % 7 != 0:
        raise ValueError(
            f"Steering action_dim={steering_action_dim} is not divisible by 7"
        )
    print(
        f"[Steering] type={args.steering_model_type} "
        f"expects chunk_size={steering_chunk_size} latent_source={latent_source}"
    )

    tasks = discover_tasks_for_suite(args.suite, args.task_limit, args.task_order_index)
    print(f"[Suite] {args.suite}: {len(tasks)} tasks")

    all_task_results = {}
    decision_logs_all: List[Dict] = []
    t0 = time.time()

    for task in tasks:
        task_id = task["task_id"]
        env_problem_name = task["env_problem_name"]
        bddl_file = task["bddl_file"]
        instruction = task["language_instruction"]

        num_parallel = min(args.num_parallel, args.n_episodes)
        print(f"\n[Task] {task_id} (env={env_problem_name})  [{num_parallel} parallel envs]")
        envs = [make_env(bddl_file, camera_size=args.camera_size) for _ in range(num_parallel)]
        init_states = torch.load(task["init_states_file"], weights_only=False)

        baseline_successes = 0
        baseline_steps = []

        gated_successes = 0
        gated_steps = []
        gated_refused = []
        gated_forced_accepts = []
        gated_chunks_done = []
        gated_avg_accept_attempt = []
        gated_refused_per_chunk = []

        n_batches = (args.n_episodes + num_parallel - 1) // num_parallel
        for batch_idx in tqdm(range(n_batches), desc=f"{task_id}", leave=False):
            batch_start = batch_idx * num_parallel
            batch_end = min(batch_start + num_parallel, args.n_episodes)
            batch_eps = list(range(batch_start, batch_end))
            batch_size = len(batch_eps)

            writers_baseline: List[Optional[object]] = [None] * batch_size
            writers_gated: List[Optional[object]] = [None] * batch_size
            try:
                if args.record_video:
                    for j, ep in enumerate(batch_eps):
                        if compare_baseline:
                            p = video_dir / f"{task_id}_ep{ep:03d}_baseline.mp4"
                            writers_baseline[j] = imageio.get_writer(p, fps=20)
                        p = video_dir / f"{task_id}_ep{ep:03d}_gated.mp4"
                        writers_gated[j] = imageio.get_writer(p, fps=20)

                if compare_baseline:
                    stats_list_base = rollout_baseline_parallel(
                        envs=envs[:batch_size],
                        init_states=init_states,
                        episode_indices=batch_eps,
                        agent=agent,
                        task_instruction=instruction,
                        max_steps=args.max_steps,
                        warmup_steps=args.warmup_steps,
                        writers=writers_baseline,
                    )
                    for stats_b in stats_list_base:
                        baseline_successes += int(stats_b.success)
                        baseline_steps.append(stats_b.steps)

                stats_list_gated = rollout_gated_parallel(
                    envs=envs[:batch_size],
                    init_states=init_states,
                    episode_indices=batch_eps,
                    agent=agent,
                    steering_model=steering_model,
                    steering_chunk_size=steering_chunk_size,
                    steering_model_type=args.steering_model_type,
                    latent_source=latent_source,
                    task_instruction=instruction,
                    max_steps=args.max_steps,
                    max_refusal_attempts=args.max_refusal_attempts,
                    warmup_steps=args.warmup_steps,
                    proposal_batch_size=args.proposal_batch_size,
                    writers=writers_gated,
                    refused_video_dir=(refused_dir if args.record_refused_rollouts else None),
                    task_name=task_id,
                    print_decision_logs=args.print_decision_logs,
                    intervene_every_n=args.intervene_every_n,
                )
                for stats_g in stats_list_gated:
                    gated_successes += int(stats_g.success)
                    gated_steps.append(stats_g.steps)
                    gated_refused.append(stats_g.refused_count)
                    gated_forced_accepts.append(stats_g.forced_accepts)
                    gated_chunks_done.append(stats_g.chunks_executed)
                    if stats_g.accepted_attempts:
                        gated_avg_accept_attempt.append(safe_mean(stats_g.accepted_attempts))
                    if stats_g.chunks_executed > 0:
                        gated_refused_per_chunk.append(
                            stats_g.refused_count / float(stats_g.chunks_executed)
                        )
                    decision_logs_all.extend(stats_g.decision_logs)

            finally:
                for w in writers_baseline + writers_gated:
                    if w is not None:
                        w.close()

        for env in envs:
            env.close()

        task_res = {
            "task_id": task_id,
            "env_problem_name": env_problem_name,
            "episodes": args.n_episodes,
            "gated_success_rate": gated_successes / args.n_episodes,
            "gated_avg_steps": safe_mean(gated_steps),
            "gated_avg_refused_per_episode": safe_mean(gated_refused),
            "gated_total_forced_accepts": int(np.sum(gated_forced_accepts)) if gated_forced_accepts else 0,
            "total_chunks_done": int(np.sum(gated_chunks_done)) if gated_chunks_done else 0,
            "avg_refused_actions_per_chunk": safe_mean(gated_refused_per_chunk),
            "avg_gating_metric": safe_mean(gated_avg_accept_attempt),
        }
        if compare_baseline:
            task_res["baseline_success_rate"] = baseline_successes / args.n_episodes
            task_res["baseline_avg_steps"] = safe_mean(baseline_steps)

        all_task_results[task_id] = task_res
        print(
            f"  gated_sr={task_res['gated_success_rate']*100:.1f}% "
            f"avg_refused={task_res['gated_avg_refused_per_episode']:.2f}" +
            (
                f" | baseline_sr={task_res['baseline_success_rate']*100:.1f}%"
                if compare_baseline else ""
            )
        )

    gated_srs = [v["gated_success_rate"] for v in all_task_results.values()]
    gated_steps_suite = [v["gated_avg_steps"] for v in all_task_results.values()]
    gated_refused_per_chunk_suite = [v["avg_refused_actions_per_chunk"] for v in all_task_results.values()]
    gated_metric_suite = [v["avg_gating_metric"] for v in all_task_results.values()]
    summary = {
        "suite": args.suite,
        "act_checkpoint": act_ckpt,
        "steering_model_type": args.steering_model_type,
        "explainer_projector": projector_ckpt,
        "explainer_lora_dir": lora_dir,
        "mlp_checkpoint": mlp_ckpt,
        "steering_latent_source": latent_source,
        "episodes_per_task": args.n_episodes,
        "compare_baseline": compare_baseline,
        "record_video": args.record_video,
        "record_refused_rollouts": args.record_refused_rollouts,
        "gated_avg_suite_success_rate": safe_mean(gated_srs),
        "gated_avg_suite_steps": safe_mean(gated_steps_suite),
        "gated_total_chunks_done": int(np.sum([v.get("total_chunks_done", 0) for v in all_task_results.values()])),
        "gated_avg_refused_actions_per_chunk": safe_mean(gated_refused_per_chunk_suite),
        "gated_avg_gating_metric": safe_mean(gated_metric_suite),
        "gated_avg_refused_per_episode": safe_mean([v["gated_avg_refused_per_episode"] for v in all_task_results.values()]),
        "gated_total_forced_accepts": int(np.sum([v.get("gated_total_forced_accepts", 0) for v in all_task_results.values()])),
        "tasks": all_task_results,
        "elapsed_sec": time.time() - t0,
    }

    if compare_baseline:
        base_srs = [
            v["baseline_success_rate"]
            for v in all_task_results.values()
            if "baseline_success_rate" in v
        ]
        summary["baseline_avg_suite_success_rate"] = safe_mean(base_srs)
        summary["baseline_avg_suite_steps"] = safe_mean([
            v["baseline_avg_steps"] for v in all_task_results.values() if "baseline_avg_steps" in v
        ])
        summary["suite_success_delta_gated_minus_baseline"] = (
            summary["gated_avg_suite_success_rate"] - summary["baseline_avg_suite_success_rate"]
        )

    out_json = outdir / f"summary_{args.suite}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_csv = outdir / f"summary_{args.suite}.csv"
    write_results_csv(out_csv, all_task_results, summary)

    out_decisions = outdir / f"decision_logs_{args.suite}.jsonl"
    with open(out_decisions, "w", encoding="utf-8") as f:
        for row in decision_logs_all:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"Suite: {args.suite}")
    if compare_baseline:
        print(f"Baseline SR: {summary['baseline_avg_suite_success_rate']*100:.2f}%")
    print(f"Gated SR:    {summary['gated_avg_suite_success_rate']*100:.2f}%")
    if compare_baseline:
        delta = summary["suite_success_delta_gated_minus_baseline"]
        print(f"Delta:       {delta*100:+.2f}%")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_decisions}")
    print("=" * 70)


if __name__ == "__main__":
    main()
