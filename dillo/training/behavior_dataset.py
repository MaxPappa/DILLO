"""
Behavioral Description Datasets for LIBERO.

Loads examples produced by the chunk-based data collection pipeline
(dillo.data_generation.collect_dataset) and prepares (latent_obs, action_chunk, description)
tuples for training the PolicyExplainer model.

Dataset layout expected per example folder:
    <save_dir>/<task_name>/<episode_idx>/
        response.txt          – VLM descriptions with <frame_i_to_frame_j> tags
        task_instruction.txt  – LIBERO language instruction
        eef_pos.npy           – (11, 3) end-effector positions
        joint_pos.npy         – (11, 7) joint positions
        gripper_states.npy    – (11, 2) gripper finger positions
        actions.npy           – (num_chunks * chunk_size, 7) actions
        success_mask.npy      – optional (11,) binary mask
        images/               – 11 JPEG frame images
"""
from __future__ import annotations

import re
import math
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from transformers import DataCollatorWithPadding
from transformers.models.gemma3.image_processing_gemma3 import Gemma3ImageProcessor


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

# Regex to extract per-transition descriptions from the response file.
# Handles multi-line descriptions between <frame_i_to_frame_j>...</frame_i_to_frame_j>
TRANSITION_RE = re.compile(
    r"<frame_(\d+)_to_frame_(\d+)>(.*?)</frame_\1_to_frame_\2>",
    re.DOTALL,
)


def parse_response(text: str) -> List[str]:
    """
    Parse the VLM response.txt into a list of per-transition descriptions.

    Returns:
        descriptions: list of strings (one per chunk transition), stripped
    """
    matches = TRANSITION_RE.findall(text)
    descriptions = [m[2].strip() for m in matches]
    return descriptions


def build_robot_state(eef_pos: np.ndarray, joint_pos: np.ndarray,
                      gripper_states: np.ndarray) -> np.ndarray:
    """
    Concatenate observation arrays into a single state vector per frame.

    Args:
        eef_pos: (T, 3)
        joint_pos: (T, 7)
        gripper_states: (T, 2)

    Returns:
        state: (T, 12) float32
    """
    return np.concatenate([eef_pos, joint_pos, gripper_states], axis=1).astype("float32")


# Image transform matching ACTAgent.image_transform
_ACT_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_frame_image(image_path: str | Path) -> torch.Tensor:
    """Load a JPEG frame and apply the ACT image transform. Returns (3,224,224)."""
    img = Image.open(image_path).convert("RGB")
    return _ACT_IMG_TRANSFORM(img)


def extract_latent_act(
    agent,
    robot_state: np.ndarray,
    device: torch.device,
    image_paths: Optional[List[str | Path]] = None,
    task_instruction: Optional[str] = None,
    latent_source: str = "context",
) -> torch.Tensor:
    """
    Extract the policy's latent representation by running the *full*
    ACT encoder: state + images + language → decoder-memory tokens → pooled.

    This is analogous to the SAC ``actor.trunk[0:-2](obs)`` used in MetaWorld:
    the output is a single hidden_dim vector that encodes the policy's
    internal view of the current state.

    Memory layout (per sample):
        [state_tok, img_tok, lang_tok, z_tok]  →  4 × hidden_dim
        Pooled (mean) → 1 × hidden_dim

    When images/language are not available the function falls back to only
    the state encoder (same as before, for backward compat), but the
    resulting latent is much less informative.

    Args:
        agent: ACTAgent wrapper (has .model, .norm_stats, .device)
        robot_state: (12,) or (T, 12) raw state (eef + joint + gripper)
        device: torch device
        image_paths: list of paths to JPEG frames (one per sample in batch).
                     If None, falls back to state-only encoding.
        task_instruction: LIBERO language instruction string.
                          If None, a zero vector is used for lang_tok.

    Returns:
        latent: (hidden_dim,) or (T, hidden_dim) tensor on CPU
    """
    model = agent.model

    # ── Prepare state tensor (B, 9) ──────────────────────────────
    state_for_encoder = robot_state[..., 3:]  # skip eef_pos → joint(7)+gripper(2)
    single = state_for_encoder.ndim == 1
    if single:
        state_for_encoder = state_for_encoder[np.newaxis, :]
    B = state_for_encoder.shape[0]

    mean = agent.norm_stats["robot_state_mean"]
    std  = agent.norm_stats["robot_state_std"]
    state_norm = (state_for_encoder - mean) / std
    state_tensor = torch.from_numpy(state_norm.astype("float32")).to(device)

    with torch.no_grad():
        # State token  (B, 1, H)
        state_tok = model.state_encoder(state_tensor).unsqueeze(1)
        H = state_tok.shape[-1]

        # Image token  (B, 1, H)
        if image_paths is not None and len(image_paths) == B:
            imgs = torch.stack([_load_frame_image(p) for p in image_paths])  # (B,3,224,224)
            imgs = imgs.to(device)
            img_dict = {cam: imgs for cam in model.camera_names}
            img_tok = model.encode_images(img_dict).unsqueeze(1)  # (B,1,H)
        else:
            img_tok = torch.zeros(B, 1, H, device=device)

        # Language token  (B, 1, H)
        if task_instruction is not None and model.use_language:
            prompts = [task_instruction] * B
            lang_tok = model.encode_language(prompts, device=device, batch_size=B).unsqueeze(1)
        else:
            lang_tok = torch.zeros(B, 1, H, device=device)

        # z token (prior z = 0 at inference time, same as ACT inference)
        z_tok = model.latent_proj(
            torch.zeros(B, model.latent_dim, device=device)
        ).unsqueeze(1)  # (B,1,H)

        if latent_source == "visual":
            latent = img_tok.squeeze(1)  # (B, H)
        elif latent_source == "context":
            # Decoder memory: [state, img, lang, z]  →  (B, 4, H)
            memory = torch.cat([state_tok, img_tok, lang_tok, z_tok], dim=1)
            latent = memory.mean(dim=1)  # (B, H)
        else:
            raise ValueError(f"Unknown latent_source: {latent_source}")

    latent = latent.cpu()
    if single:
        latent = latent.squeeze(0)  # (H,)
    return latent


def extract_act_policy_latent(
    agent,
    robot_state: np.ndarray,
    device: torch.device,
    image_paths: Optional[List[str | Path]] = None,
    task_instruction: Optional[str] = None,
    latent_source: str = "context",
) -> torch.Tensor:
    """
    Extract the policy's latent representation using
    ActionChunkingPolicy.get_context_latent() from the released ACT policy.

    This is the counterpart of extract_latent_act() for data collected with
    dillo.data_generation.collect_dataset / ACTAgent.

    Memory layout (per sample):
        _spatial_encode: [lang_tok, img_tok, proprio_toks…]  →  (B, N, E)
        Pooled (mean) → 1 × embed_size

    Args:
        agent: ACTAgent wrapper — must have
               .policy (ActionChunkingPolicy), ._img_h, ._img_w,
               ._get_task_emb(instruction)
        robot_state: (12,) or (T, 12) raw state:
                     eef_pos(3) + joint_pos(7) + gripper(2)2
        device: torch device
        image_paths: list of paths to JPEG frames (one per sample in batch).
                     If None, zero images are used.
        task_instruction: LIBERO language instruction string.
                          If None, a zero language embedding is used.

    Returns:
        latent: (embed_size,) or (T, embed_size) tensor on CPU
    """
    from robomimic.utils import obs_utils as ObsUtils
    import torch.nn.functional as F_nn

    policy = agent.policy
    single = robot_state.ndim == 1
    if single:
        robot_state = robot_state[np.newaxis, :]
    B = robot_state.shape[0]

    # ── Proprioception ──────────────────────────────────────────────
    # robot_state layout: eef_pos(3) | joint_pos(7) | gripper(2)
    joint_pos = robot_state[:, 3:10].astype("float32")    # (B, 7)
    gripper_st = robot_state[:, 10:12].astype("float32")  # (B, 2)
    obs_dict = {
        "joint_states":  torch.from_numpy(joint_pos).to(device),
        "gripper_states": torch.from_numpy(gripper_st).to(device),
    }

    # ── Images ─────────────────────────────────────────────────────
    img_h, img_w = agent._img_h, agent._img_w
    if image_paths is not None and len(image_paths) == B:
        raw_imgs = np.stack([
            np.array(Image.open(p).convert("RGB")) for p in image_paths
        ]).astype("uint8")  # (B, H_orig, W_orig, 3)
        processed = []
        for i in range(B):
            # ObsUtils: (H, W, 3) uint8 → (3, H, W) float32 in [0, 1]
            t = ObsUtils.process_obs(
                torch.from_numpy(raw_imgs[i]), obs_key="agentview_rgb"
            ).float()  # (3, H_orig, W_orig)
            processed.append(t)
        imgs_tensor = torch.stack(processed).to(device)  # (B, 3, H_orig, W_orig)
        if imgs_tensor.shape[2] != img_h or imgs_tensor.shape[3] != img_w:
            imgs_tensor = F_nn.interpolate(
                imgs_tensor, size=(img_h, img_w),
                mode="bilinear", align_corners=False,
            )
    else:
        imgs_tensor = torch.zeros(B, 3, img_h, img_w, device=device)

    # All image encoder keys get the same (agentview) frame when only one
    # camera path is provided; cameras lacking a path receive zeros.
    for cam_key in policy.image_encoders:
        obs_dict[cam_key] = imgs_tensor

    # ── Task embedding ──────────────────────────────────────────────
    if task_instruction is not None:
        task_emb = agent._get_task_emb(task_instruction)  # (1, lang_dim)
        task_emb = task_emb.to(device).expand(B, -1)      # (B, lang_dim)
    else:
        # Infer lang_dim from the first language_encoder parameter
        lang_in = next(policy.language_encoder.parameters()).shape[-1]
        task_emb = torch.zeros(B, lang_in, device=device)

    # ── Run encoder ────────────────────────────────────────────────
    with torch.no_grad():
        if latent_source == "visual":
            image_tokens = []
            for cam_key, encoder in policy.image_encoders.items():
                image_tokens.append(
                    encoder(obs_dict[cam_key], langs=task_emb)
                )  # each (B, E)
            latent = torch.stack(image_tokens, dim=1).mean(dim=1)  # (B, E)
        elif latent_source == "context":
            latent = policy.get_context_latent(obs_dict, task_emb,
                                               pool="mean")  # (B, E)
        else:
            raise ValueError(f"Unknown latent_source: {latent_source}")

    latent = latent.cpu()
    if single:
        latent = latent.squeeze(0)  # (E,)
    return latent


# ─────────────────────────────────────────────────────────────────────
# Main Dataset
# ─────────────────────────────────────────────────────────────────────

class LIBEROBehaviorDataset(Dataset):
    """
    Dataset for training the PolicyExplainer on LIBERO behavioral descriptions.

    In ``single_obs_act=True`` mode (recommended), each chunk transition
    becomes a separate training example with:
      - One (or two for oracular) latent observation(s)
      - One action chunk
      - One text description

    In ``single_obs_act=False`` mode, only the last transition of the episode
    is used (legacy, not recommended for LIBERO).
    """

    def __init__(
        self,
        data_dirs: str | List[str],
        agent=None,
        tokenizer=None,
        validation: bool = False,
        use_raw_obs: bool = True,
        single_obs_act: bool = True,
        is_oracular: bool = False,
        use_eos_token: bool = True,
        chunk_size: int = 10,
        num_chunks: int = 10,
        min_description_tokens: int = 5,
    ):
        """
        Args:
            data_dirs: glob pattern(s) or list of directories containing
                       collected examples. E.g. "data/libero_goal_video_and_obs/*/*"
            agent: ACTAgent instance used for latent extraction when
                   use_raw_obs=False.
            tokenizer: HuggingFace tokenizer for the LLM backbone
            validation: if True, load validation split
            use_raw_obs: if True, use raw (eef+joint+gripper) as observation;
                         if False, extract latent via ACT state encoder
            single_obs_act: if True, one example per transition;
                            if False, one example per episode (last transition only)
            is_oracular: if True, include the next observation as well
            use_eos_token: append EOS token to full_ids
            chunk_size: number of atomic actions per chunk
            num_chunks: expected number of chunk transitions per episode
            min_description_tokens: skip descriptions shorter than this
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.agent = agent
        self.use_raw_obs = use_raw_obs
        self.single_obs_act = single_obs_act
        self.is_oracular = is_oracular
        self.use_eos_token = use_eos_token
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

        # Resolve paths
        if isinstance(data_dirs, str):
            self.paths = sorted(glob(data_dirs))
        else:
            self.paths = []
            for d in data_dirs:
                self.paths.extend(sorted(glob(d)))

        if not self.paths:
            raise FileNotFoundError(f"No examples found for patterns: {data_dirs}")

        # Storage
        self.descriptions: List[str] = []
        self.latent_observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.raw_obs: List[torch.Tensor] = []
        self.task_instructions: List[str] = []
        self.user_prompts: List[str] = []
        self.skipped: List[str] = []

        device = torch.device("cpu")
        if agent is not None and hasattr(agent, "device"):
            device = agent.device

        # User-prompt template (task inserted per example in _load_single_example)
        if self.is_oracular:
            self._user_prompt_template = (
                "You are a robot performing the task: '{task}'. "
                "You are given two consecutive observations (before and after the action) "
                "and the action chunk that was executed. "
                "Describe in one sentence what the robot did."
            )
        else:
            self._user_prompt_template = (
                "You are a robot performing the task: '{task}'. "
                "You are given the current observation and the action chunk that was executed. "
                "Describe in one sentence what the robot did."
            )

        self._load_examples(device, min_description_tokens)

        print(f"[LIBEROBehaviorDataset] Loaded {len(self.descriptions)} examples "
              f"(skipped {len(self.skipped)} folders)")

    def _load_examples(self, device: torch.device, min_desc_tokens: int):
        """Iterate over all example folders and populate the internal lists."""
        with torch.no_grad():
            for p in self.paths:
                p = Path(p)
                try:
                    self._load_single_example(p, device, min_desc_tokens)
                except Exception as e:
                    self.skipped.append(str(p))
                    # Uncomment for debugging:
                    # print(f"  Skipping {p}: {e}")

    def _load_single_example(self, folder: Path, device: torch.device,
                             min_desc_tokens: int):
        """Load one example folder and append to internal lists."""
        # Read response
        response_path = folder / "response.txt"
        if not response_path.exists():
            self.skipped.append(str(folder))
            return

        with open(response_path) as f:
            response_text = f.read()

        descriptions = parse_response(response_text)
        if len(descriptions) < 1:
            self.skipped.append(str(folder))
            return

        # Read task instruction
        task_instr_path = folder / "task_instruction.txt"
        task_instruction = ""
        if task_instr_path.exists():
            with open(task_instr_path) as f:
                task_instruction = f.read().strip()

        # Read observations
        eef_pos = np.load(folder / "eef_pos.npy").astype("float32")       # (11, 3)
        joint_pos = np.load(folder / "joint_pos.npy").astype("float32")   # (11, 7)
        gripper_st = np.load(folder / "gripper_states.npy").astype("float32")  # (11, 2)
        robot_state = build_robot_state(eef_pos, joint_pos, gripper_st)    # (11, 12)

        # Read actions: (num_chunks * chunk_size, 7)
        all_actions = np.load(folder / "actions.npy").astype("float32")
        # Reshape into (num_chunks, chunk_size, 7) — one chunk per transition
        n_total = all_actions.shape[0]
        actual_chunks = n_total // self.chunk_size
        action_chunks = all_actions[: actual_chunks * self.chunk_size].reshape(
            actual_chunks, self.chunk_size, -1
        )  # (num_chunks, chunk_size, action_dim)

        if self.single_obs_act:
            # Each transition → one training example
            for i in range(min(len(descriptions), actual_chunks)):
                desc = descriptions[i]

                # Quick token-length check to filter empty/garbage
                tok_ids = self.tokenizer(desc, return_tensors="np")["input_ids"]
                if tok_ids.shape[1] < min_desc_tokens:
                    continue
                if len(desc.strip()) == 0:
                    continue

                self.descriptions.append(desc)
                self.task_instructions.append(task_instruction)
                self.user_prompts.append(
                    self._user_prompt_template.format(task=task_instruction)
                )

                # Observation: frame i (before) and optionally i+1 (after, for
                # oracular and for fidelity evaluation)
                obs_before = robot_state[i]        # (12,)
                obs_after = robot_state[i + 1]     # (12,)

                if self.use_raw_obs:
                    if self.is_oracular:
                        lat = torch.from_numpy(
                            np.stack([obs_before, obs_after])
                        )  # (2, 12)
                    else:
                        lat = torch.from_numpy(obs_before)  # (12,)
                else:
                    # Latent mode: run the ACT policy encoder.
                    _latent_fn = (
                        extract_act_policy_latent
                        if hasattr(self.agent, "policy")
                        else extract_latent_act
                    )
                    img_before = str(folder / "images" / f"{i}.jpeg")
                    img_after  = str(folder / "images" / f"{i + 1}.jpeg")
                    if self.is_oracular:
                        lat = _latent_fn(
                            self.agent,
                            np.stack([obs_before, obs_after]),
                            device,
                            image_paths=[img_before, img_after],
                            task_instruction=task_instruction,
                        )  # (2, embed_dim)
                    else:
                        lat = _latent_fn(
                            self.agent, obs_before, device,
                            image_paths=[img_before],
                            task_instruction=task_instruction,
                        )  # (embed_dim,)

                self.latent_observations.append(lat)

                # Raw obs for fidelity: always keep (before, after) pair
                self.raw_obs.append(
                    torch.from_numpy(np.stack([obs_before, obs_after]))
                )  # (2, 12)

                # Action chunk for this transition: (chunk_size, action_dim)
                act_chunk = action_chunks[i]  # (chunk_size, 7)
                self.actions.append(torch.from_numpy(act_chunk))

        else:
            # Legacy: use the last transition only
            if len(descriptions) < 1:
                self.skipped.append(str(folder))
                return
            desc = descriptions[-1]
            tok_ids = self.tokenizer(desc, return_tensors="np")["input_ids"]
            if tok_ids.shape[1] < min_desc_tokens:
                self.skipped.append(str(folder))
                return

            self.descriptions.append(desc)
            self.task_instructions.append(task_instruction)
            self.user_prompts.append(
                self._user_prompt_template.format(task=task_instruction)
            )

            if self.use_raw_obs:
                self.latent_observations.append(torch.from_numpy(robot_state[:-1]))
            else:
                # Legacy non-single mode: extract latents for all frames
                _latent_fn = (
                    extract_act_policy_latent
                    if hasattr(self.agent, "policy")
                    else extract_latent_act
                )
                img_paths = [str(folder / "images" / f"{j}.jpeg")
                             for j in range(robot_state.shape[0] - 1)]
                lat = _latent_fn(
                    self.agent, robot_state[:-1], device,
                    image_paths=img_paths,
                    task_instruction=task_instruction,
                )
                self.latent_observations.append(lat)

            self.raw_obs.append(torch.from_numpy(robot_state))
            self.actions.append(torch.from_numpy(action_chunks.reshape(-1, action_chunks.shape[-1])))

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, index):
        user_text = self.user_prompts[index]
        assistant_text = self.descriptions[index]

        # Build chat-template token ids
        # apply_chat_template returns a BatchEncoding; extract input_ids list.
        def _chat_to_ids(messages, add_gen_prompt):
            out = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_gen_prompt,
            )
            # BatchEncoding → list[int], or already list[int]
            if hasattr(out, "input_ids"):
                ids = out["input_ids"]
            elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], int):
                ids = out
            else:
                ids = out
            return torch.tensor(ids, dtype=torch.long)

        full_ids = _chat_to_ids(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            add_gen_prompt=False,
        )

        if self.use_eos_token:
            full_ids = torch.cat([full_ids, torch.tensor([self.tokenizer.eos_token_id])])

        prompt_ids = _chat_to_ids(
            [{"role": "user", "content": user_text}],
            add_gen_prompt=True,
        )

        prompt_ids_val = _chat_to_ids(
            [{"role": "user", "content": user_text}],
            add_gen_prompt=False,
        )

        labels = full_ids.clone()
        labels[: prompt_ids.size(0)] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels": labels,
            "latent_obs": self.latent_observations[index].squeeze(),
            "actions": self.actions[index].squeeze(),
            "obs": self.raw_obs[index].squeeze(),
            "description": assistant_text,
            "prompt_ids_val": prompt_ids_val,
        }


# ─────────────────────────────────────────────────────────────────────
# Stage 3 Dataset: success / failure classification
# ─────────────────────────────────────────────────────────────────────

# Text labels used as the assistant turn for classification
_SUCC_TEXT = "success"
_FAIL_TEXT  = "failure"


class LIBEROSuccFailDataset(Dataset):
    """
    Dataset for stage-3 fine-tuning: success/failure classification.

    Each example is one chunk transition labelled ``"success"`` or
    ``"failure"`` based on the ``success_mask.npy`` file saved during
    data collection.

    success_mask layout (saved by dillo.data_generation.collect_dataset):
        (n+1,) float32  —  1.0 at position idx  if  idx > success_chunk
        The file is absent for failed episodes (treated as all-failure).

    For chunk transition i  (frame i → frame i+1):
        label = "success"  if  success_mask[i+1] == 1.0
        label = "failure"  otherwise (including absent mask)

    Args:
        data_dirs: glob pattern(s) pointing to example folders
        agent: ACTAgent or ACTAgent (only needed if use_raw_obs=False)
        tokenizer: HuggingFace tokenizer for the LLM backbone
        validation: unused (kept for API parity with LIBEROBehaviorDataset)
        use_raw_obs: if True use 12-dim raw state; else extract latent
        is_oracular: if True include the next observation as well
        use_eos_token: append EOS to full_ids
        chunk_size: atomic actions per chunk
        num_chunks: expected chunk transitions per episode
        require_success_mask: if True, skip folders that have no
                              success_mask.npy (useful to build a
                              validation set from successful episodes only)
    """

    def __init__(
        self,
        data_dirs: str | List[str],
        agent=None,
        tokenizer=None,
        validation: bool = False,
        use_raw_obs: bool = True,
        is_oracular: bool = False,
        use_eos_token: bool = True,
        chunk_size: int = 10,
        num_chunks: int = 10,
        require_success_mask: bool = False,
    ):
        super().__init__()

        self.tokenizer      = tokenizer
        self.agent          = agent
        self.use_raw_obs    = use_raw_obs
        self.is_oracular    = is_oracular
        self.use_eos_token  = use_eos_token
        self.chunk_size     = chunk_size
        self.num_chunks     = num_chunks
        self.require_mask   = require_success_mask

        # Resolve paths
        if isinstance(data_dirs, str):
            self.paths = sorted(glob(data_dirs))
        else:
            self.paths = []
            for d in data_dirs:
                self.paths.extend(sorted(glob(d)))

        if not self.paths:
            raise FileNotFoundError(f"No examples found for patterns: {data_dirs}")

        # Storage
        self.labels:              List[str]          = []  # "success" | "failure"
        self.latent_observations: List[torch.Tensor] = []
        self.actions:             List[torch.Tensor] = []
        self.raw_obs:             List[torch.Tensor] = []
        self.task_instructions:   List[str]          = []
        self.user_prompts:        List[str]          = []  # per-example (task varies)
        self.skipped:             List[str]          = []

        device = torch.device("cpu")
        if agent is not None and hasattr(agent, "device"):
            device = agent.device

        with torch.no_grad():
            for p in self.paths:
                try:
                    self._load_single_example(Path(p), device)
                except Exception:
                    self.skipped.append(str(p))

        n_succ = self.labels.count(_SUCC_TEXT)
        n_fail = self.labels.count(_FAIL_TEXT)
        print(
            f"[LIBEROSuccFailDataset] Loaded {len(self.labels)} examples "
            f"({n_succ} success, {n_fail} failure) | "
            f"skipped {len(self.skipped)} folders"
        )

    # ------------------------------------------------------------------

    def _load_single_example(self, folder: Path, device: torch.device):
        """Load one episode folder and append per-transition examples."""
        # Task instruction
        task_instr_path = folder / "task_instruction.txt"
        task_instruction = ""
        if task_instr_path.exists():
            with open(task_instr_path) as f:
                task_instruction = f.read().strip()

        # Observations
        eef_pos   = np.load(folder / "eef_pos.npy").astype("float32")      # (T+1, 3)
        joint_pos = np.load(folder / "joint_pos.npy").astype("float32")    # (T+1, 7)
        gripper   = np.load(folder / "gripper_states.npy").astype("float32") # (T+1, 2)
        robot_state = build_robot_state(eef_pos, joint_pos, gripper)         # (T+1, 12)

        # Actions: (num_chunks * chunk_size, 7)  →  (num_chunks, chunk_size, 7)
        all_actions  = np.load(folder / "actions.npy").astype("float32")
        actual_chunks = all_actions.shape[0] // self.chunk_size
        action_chunks = all_actions[: actual_chunks * self.chunk_size].reshape(
            actual_chunks, self.chunk_size, -1
        )

        # Success mask
        mask_path = folder / "success_mask.npy"
        if mask_path.exists():
            success_mask = np.load(mask_path).astype("float32")  # (T+1,)
        else:
            if self.require_mask:
                self.skipped.append(str(folder))
                return
            # Failed episode: treat all transitions as failure
            success_mask = np.zeros(robot_state.shape[0], dtype="float32")

        n_transitions = min(actual_chunks, robot_state.shape[0] - 1,
                            success_mask.shape[0] - 1)

        # Build user prompt template for this task
        user_prompt = (
            f"You are a robot performing the task: '{task_instruction}'. "
            f"You are given the current observation and an action chunk. "
            f"Was this a good action chunk that makes progress toward completing the task? "
            f"Answer with exactly one word: 'success' or 'failure'."
        )

        for i in range(n_transitions):
            label = _SUCC_TEXT if success_mask[i + 1] == 1.0 else _FAIL_TEXT

            obs_before = robot_state[i]      # (12,)
            obs_after  = robot_state[i + 1]  # (12,)

            # Latent observation
            if self.use_raw_obs:
                if self.is_oracular:
                    lat = torch.from_numpy(np.stack([obs_before, obs_after]))
                else:
                    lat = torch.from_numpy(obs_before)
            else:
                _latent_fn = (
                    extract_act_policy_latent
                    if hasattr(self.agent, "policy")
                    else extract_latent_act
                )
                img_before = str(folder / "images" / f"{i}.jpeg")
                img_after  = str(folder / "images" / f"{i + 1}.jpeg")
                if self.is_oracular:
                    lat = _latent_fn(
                        self.agent, np.stack([obs_before, obs_after]), device,
                        image_paths=[img_before, img_after],
                        task_instruction=task_instruction,
                    )
                else:
                    lat = _latent_fn(
                        self.agent, obs_before, device,
                        image_paths=[img_before],
                        task_instruction=task_instruction,
                    )

            self.labels.append(label)
            self.latent_observations.append(lat)
            self.raw_obs.append(
                torch.from_numpy(np.stack([obs_before, obs_after]))
            )
            self.actions.append(torch.from_numpy(action_chunks[i]))
            self.task_instructions.append(task_instruction)
            self.user_prompts.append(user_prompt)

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        user_text      = self.user_prompts[index]
        assistant_text = self.labels[index]

        def _chat_to_ids(messages, add_gen_prompt):
            out = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_gen_prompt,
            )
            if hasattr(out, "input_ids"):
                ids = out["input_ids"]
            elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], int):
                ids = out
            else:
                ids = out
            return torch.tensor(ids, dtype=torch.long)

        full_ids = _chat_to_ids(
            [
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            add_gen_prompt=False,
        )
        if self.use_eos_token:
            full_ids = torch.cat(
                [full_ids, torch.tensor([self.tokenizer.eos_token_id])]
            )

        prompt_ids = _chat_to_ids(
            [{"role": "user", "content": user_text}],
            add_gen_prompt=True,
        )
        prompt_ids_val = _chat_to_ids(
            [{"role": "user", "content": user_text}],
            add_gen_prompt=False,
        )

        labels = full_ids.clone()
        labels[: prompt_ids.size(0)] = -100

        return {
            "input_ids":       full_ids,
            "attention_mask":  torch.ones_like(full_ids),
            "labels":          labels,
            "latent_obs":      self.latent_observations[index].squeeze(),
            "actions":         self.actions[index].squeeze(),
            "obs":             self.raw_obs[index].squeeze(),
            "description":     assistant_text,
            "prompt_ids_val":  prompt_ids_val,
        }


# ─────────────────────────────────────────────────────────────────────
# Lightweight numeric dataset for non-LLM success/failure baselines
# ─────────────────────────────────────────────────────────────────────

class LIBEROSuccFailLatentDataset(Dataset):
    """
    Numeric success/failure dataset for lightweight binary classifiers.

    This follows the same folder parsing and labeling convention as
    ``LIBEROSuccFailDataset`` but returns tensors only:

      - ``latent_obs``: raw robot state or extracted ACT latent
      - ``actions``: action chunk for the transition
      - ``label``: 1 for success, 0 for failure

    No tokenizer or chat template is required, so it can be used as a simple
    baseline against the LLM-based stage-3 classifier.
    """

    def __init__(
        self,
        data_dirs: str | List[str],
        agent=None,
        use_raw_obs: bool = True,
        is_oracular: bool = False,
        chunk_size: int = 10,
        num_chunks: int = 10,
        require_success_mask: bool = False,
        latent_source: str = "context",
    ):
        super().__init__()

        self.agent = agent
        self.use_raw_obs = use_raw_obs
        self.is_oracular = is_oracular
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.require_mask = require_success_mask
        self.latent_source = latent_source

        if isinstance(data_dirs, str):
            self.paths = sorted(glob(data_dirs))
        else:
            self.paths = []
            for d in data_dirs:
                self.paths.extend(sorted(glob(d)))

        if not self.paths:
            raise FileNotFoundError(f"No examples found for patterns: {data_dirs}")

        self.labels: List[int] = []
        self.latent_observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.raw_obs: List[torch.Tensor] = []
        self.task_instructions: List[str] = []
        self.skipped: List[str] = []

        device = torch.device("cpu")
        if agent is not None and hasattr(agent, "device"):
            device = agent.device

        with torch.no_grad():
            for p in self.paths:
                try:
                    self._load_single_example(Path(p), device)
                except Exception:
                    self.skipped.append(str(p))

        n_succ = int(sum(self.labels))
        n_fail = len(self.labels) - n_succ
        print(
            f"[LIBEROSuccFailLatentDataset] Loaded {len(self.labels)} examples "
            f"({n_succ} success, {n_fail} failure) | "
            f"skipped {len(self.skipped)} folders"
        )

    def _load_single_example(self, folder: Path, device: torch.device):
        task_instr_path = folder / "task_instruction.txt"
        task_instruction = ""
        if task_instr_path.exists():
            with open(task_instr_path) as f:
                task_instruction = f.read().strip()

        eef_pos = np.load(folder / "eef_pos.npy").astype("float32")
        joint_pos = np.load(folder / "joint_pos.npy").astype("float32")
        gripper = np.load(folder / "gripper_states.npy").astype("float32")
        robot_state = build_robot_state(eef_pos, joint_pos, gripper)

        all_actions = np.load(folder / "actions.npy").astype("float32")
        actual_chunks = all_actions.shape[0] // self.chunk_size
        action_chunks = all_actions[: actual_chunks * self.chunk_size].reshape(
            actual_chunks, self.chunk_size, -1
        )

        mask_path = folder / "success_mask.npy"
        if mask_path.exists():
            success_mask = np.load(mask_path).astype("float32")
        else:
            if self.require_mask:
                self.skipped.append(str(folder))
                return
            success_mask = np.zeros(robot_state.shape[0], dtype="float32")

        n_transitions = min(
            actual_chunks,
            robot_state.shape[0] - 1,
            success_mask.shape[0] - 1,
        )

        for i in range(n_transitions):
            obs_before = robot_state[i]
            obs_after = robot_state[i + 1]

            if self.use_raw_obs:
                if self.is_oracular:
                    lat = torch.from_numpy(np.stack([obs_before, obs_after]))
                else:
                    lat = torch.from_numpy(obs_before)
            else:
                _latent_fn = (
                    extract_act_policy_latent
                    if hasattr(self.agent, "policy")
                    else extract_latent_act
                )
                img_before = str(folder / "images" / f"{i}.jpeg")
                img_after = str(folder / "images" / f"{i + 1}.jpeg")
                if self.is_oracular:
                    lat = _latent_fn(
                        self.agent,
                        np.stack([obs_before, obs_after]),
                        device,
                        image_paths=[img_before, img_after],
                        task_instruction=task_instruction,
                        latent_source=self.latent_source,
                    )
                else:
                    lat = _latent_fn(
                        self.agent,
                        obs_before,
                        device,
                        image_paths=[img_before],
                        task_instruction=task_instruction,
                        latent_source=self.latent_source,
                    )

            self.labels.append(1 if success_mask[i + 1] == 1.0 else 0)
            self.latent_observations.append(lat)
            self.actions.append(torch.from_numpy(action_chunks[i]))
            self.raw_obs.append(torch.from_numpy(np.stack([obs_before, obs_after])))
            self.task_instructions.append(task_instruction)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "latent_obs": self.latent_observations[index].squeeze().float(),
            "actions": self.actions[index].squeeze().float(),
            "obs": self.raw_obs[index].squeeze().float(),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "task_instruction": self.task_instructions[index],
        }


class LIBEROLatentBinaryCollator:
    """Stack tensors for the lightweight binary classifier."""

    def __call__(self, features):
        return {
            "latent_obs": torch.stack([f["latent_obs"] for f in features]),
            "actions": torch.stack([f["actions"] for f in features]),
            "obs": torch.stack([f["obs"] for f in features]),
            "labels": torch.stack([f["label"] for f in features]),
            "task_instruction": [f["task_instruction"] for f in features],
        }


# ─────────────────────────────────────────────────────────────────────
# Stage 3 Combined Dataset: classification token + description
# ─────────────────────────────────────────────────────────────────────

class LIBEROCombinedDataset(Dataset):
    """
    Dataset for stage-3 fine-tuning that trains on BOTH success/failure
    classification AND action description simultaneously.

    Each chunk-transition example produces an assistant turn of the form::

        <success> The robot moved its arm toward the cup…
        <failure> The robot failed to grasp the handle…

    The `<success>` / `<failure>` special token appears **first**, followed by
    the per-transition natural-language description from response.txt.  The
    next-token loss therefore covers both the classification token and the
    description, giving the model a combined supervision signal.

    Dataset layout (same as LIBEROBehaviorDataset)::

        <save_dir>/<task_name>/<episode_idx>/
            response.txt           – VLM descriptions with <frame_i_to_frame_j> tags
            task_instruction.txt   – LIBERO language instruction
            eef_pos.npy            – (T+1, 3)
            joint_pos.npy          – (T+1, 7)
            gripper_states.npy     – (T+1, 2)
            actions.npy            – (num_chunks * chunk_size, 7)
            success_mask.npy       – optional (T+1,) binary mask
            images/                – JPEG frames

    Folders without success_mask.npy are treated as all-failure episodes.
    Folders without response.txt (or with unparsable descriptions) are
    skipped.

    Args:
        data_dirs: glob pattern(s) pointing to example folders.
        agent: ACTAgent or ACTAgent (only needed when use_raw_obs=False).
        tokenizer: HuggingFace tokenizer; must already have <success>/<failure>
                   added as special tokens (done by LIBEROPolicyExplainer).
        validation: unused, kept for API parity.
        use_raw_obs: if True use 12-dim raw state; else extract ACT latent.
        is_oracular: if True include the next observation as well.
        use_eos_token: append EOS to full_ids.
        chunk_size: atomic actions per chunk.
        num_chunks: expected chunk transitions per episode.
        min_description_tokens: skip descriptions shorter than this.
        require_success_mask: if True, skip folders without success_mask.npy.
    """

    def __init__(
        self,
        data_dirs: str | List[str],
        agent=None,
        tokenizer=None,
        validation: bool = False,
        use_raw_obs: bool = True,
        is_oracular: bool = False,
        use_eos_token: bool = True,
        chunk_size: int = 10,
        num_chunks: int = 10,
        min_description_tokens: int = 5,
        require_success_mask: bool = False,
        use_text_verdict: bool = True,
        use_image_obs: bool = False,
    ):
        super().__init__()

        self.tokenizer          = tokenizer
        self.agent              = agent
        self.use_raw_obs        = use_raw_obs
        self.is_oracular        = is_oracular
        self.use_eos_token      = use_eos_token
        self.chunk_size         = chunk_size
        self.num_chunks         = num_chunks
        self.require_mask       = require_success_mask
        self.use_text_verdict   = use_text_verdict
        self.use_image_obs      = use_image_obs

        # Resolve paths
        if isinstance(data_dirs, str):
            self.paths = sorted(glob(data_dirs))
        else:
            self.paths = []
            for d in data_dirs:
                self.paths.extend(sorted(glob(d)))

        if not self.paths:
            raise FileNotFoundError(f"No examples found for patterns: {data_dirs}")

        # Storage
        self.labels:              List[str]          = []
        self.descriptions:        List[str]          = []
        self.latent_observations: List[torch.Tensor] = []
        self.actions:             List[torch.Tensor] = []
        self.raw_obs:             List[torch.Tensor] = []
        self.task_instructions:   List[str]          = []
        self.user_prompts:        List[str]          = []
        self.image_paths:         List[str]           = []
        self.folders:             List[str]           = []
        self.chunk_indices:       List[int]           = []
        self.eef_pos_before:      List[torch.Tensor]  = []
        self.eef_pos_after:       List[torch.Tensor]  = []
        self.gripper_before:      List[torch.Tensor]  = []
        self.gripper_after:       List[torch.Tensor]  = []
        self.skipped:             List[str]          = []

        device = torch.device("cpu")
        if agent is not None and hasattr(agent, "device"):
            device = agent.device

        # Shared user-prompt template — must be set BEFORE the loading loop
        # because _load_single_example uses it to build per-example prompts.
        if is_oracular:
            self._user_prompt_template = (
                "You are a robot performing the task: '{task}'. "
                "You are given two consecutive observations (before and after the action) "
                "and the action chunk that was executed. "
            )
        else:
            self._user_prompt_template = (
                "You are a robot performing the task: '{task}'. "
                "You are given the current observation and the action chunk that was executed. "
            )
        if use_text_verdict:
            self._user_prompt_template += (
                "First, classify whether this was a good action chunk that makes progress "
                "toward completing the task using <success> or <failure>. "
                "Then describe in one sentence what the robot did."
            )
        else:
            self._user_prompt_template += "Describe in one sentence what the robot did."

        with torch.no_grad():
            for p in self.paths:
                try:
                    self._load_single_example(
                        Path(p), device, min_description_tokens
                    )
                except Exception as e:
                    self.skipped.append(str(p))

        n_succ = self.labels.count(_SUCC_TEXT)
        n_fail = self.labels.count(_FAIL_TEXT)
        print(
            f"[LIBEROCombinedDataset] Loaded {len(self.labels)} examples "
            f"({n_succ} success, {n_fail} failure) | "
            f"skipped {len(self.skipped)} folders"
        )

    # ------------------------------------------------------------------

    def _load_single_example(
        self,
        folder: Path,
        device: torch.device,
        min_desc_tokens: int,
    ):
        """Load one episode folder and append per-transition combined examples."""
        # ── Descriptions ─────────────────────────────────────────────
        response_path = folder / "response.txt"
        if not response_path.exists():
            self.skipped.append(str(folder))
            return
        with open(response_path) as f:
            response_text = f.read()
        descriptions = parse_response(response_text)
        if len(descriptions) < 1:
            self.skipped.append(str(folder))
            return

        # ── Task instruction ─────────────────────────────────────────
        task_instr_path = folder / "task_instruction.txt"
        task_instruction = ""
        if task_instr_path.exists():
            with open(task_instr_path) as f:
                task_instruction = f.read().strip()

        # ── Observations ─────────────────────────────────────────────
        eef_pos    = np.load(folder / "eef_pos.npy").astype("float32")       # (T+1, 3)
        joint_pos  = np.load(folder / "joint_pos.npy").astype("float32")     # (T+1, 7)
        gripper    = np.load(folder / "gripper_states.npy").astype("float32") # (T+1, 2)
        robot_state = build_robot_state(eef_pos, joint_pos, gripper)           # (T+1, 12)

        # ── Actions ──────────────────────────────────────────────────
        all_actions   = np.load(folder / "actions.npy").astype("float32")
        actual_chunks = all_actions.shape[0] // self.chunk_size
        action_chunks = all_actions[: actual_chunks * self.chunk_size].reshape(
            actual_chunks, self.chunk_size, -1
        )  # (num_chunks, chunk_size, action_dim)

        # ── Success mask ─────────────────────────────────────────────
        mask_path = folder / "success_mask.npy"
        if mask_path.exists():
            success_mask = np.load(mask_path).astype("float32")  # (T+1,)
        else:
            if self.require_mask:
                self.skipped.append(str(folder))
                return
            # Failed episode: all transitions are failure
            success_mask = np.zeros(robot_state.shape[0], dtype="float32")

        n_transitions = min(
            len(descriptions),
            actual_chunks,
            robot_state.shape[0] - 1,
            success_mask.shape[0] - 1,
        )

        user_prompt = self._user_prompt_template.format(task=task_instruction)

        for i in range(n_transitions):
            desc = descriptions[i].strip()
            # Skip very short descriptions
            if self.tokenizer is not None:
                tok_ids = self.tokenizer(desc, return_tensors="np")["input_ids"]
                if tok_ids.shape[1] < min_desc_tokens:
                    continue

            label = _SUCC_TEXT if success_mask[i + 1] == 1.0 else _FAIL_TEXT

            obs_before = robot_state[i]       # (12,)
            obs_after  = robot_state[i + 1]   # (12,)
            img_before = str(folder / "images" / f"{i}.jpeg")
            img_after  = str(folder / "images" / f"{i + 1}.jpeg")

            # Latent observation
            if self.use_image_obs:
                if not Path(img_before).exists():
                    self.skipped.append(str(folder))
                    continue
                lat = torch.zeros(1, dtype=torch.float32)
            elif self.use_raw_obs:
                if self.is_oracular:
                    lat = torch.from_numpy(np.stack([obs_before, obs_after]))
                else:
                    lat = torch.from_numpy(obs_before)
            else:
                _latent_fn = (
                    extract_act_policy_latent
                    if hasattr(self.agent, "policy")
                    else extract_latent_act
                )
                if self.is_oracular:
                    lat = _latent_fn(
                        self.agent, np.stack([obs_before, obs_after]), device,
                        image_paths=[img_before, img_after],
                        task_instruction=task_instruction,
                    )
                else:
                    lat = _latent_fn(
                        self.agent, obs_before, device,
                        image_paths=[img_before],
                        task_instruction=task_instruction,
                    )

            self.labels.append(label)
            self.descriptions.append(desc)
            self.latent_observations.append(lat)
            self.actions.append(torch.from_numpy(action_chunks[i]))
            self.raw_obs.append(
                torch.from_numpy(np.stack([obs_before, obs_after]))
            )
            self.task_instructions.append(task_instruction)
            self.user_prompts.append(user_prompt)
            self.image_paths.append(img_before)
            self.folders.append(str(folder))
            self.chunk_indices.append(i)
            self.eef_pos_before.append(torch.from_numpy(eef_pos[i]))
            self.eef_pos_after.append(torch.from_numpy(eef_pos[i + 1]))
            self.gripper_before.append(torch.from_numpy(gripper[i]))
            self.gripper_after.append(torch.from_numpy(gripper[i + 1]))

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        user_text = self.user_prompts[index]
        if self.use_text_verdict:
            cls_token = f"<{self.labels[index]}>"  # "<success>" or "<failure>"
            assistant_text = f"{cls_token} {self.descriptions[index]}"
        else:
            assistant_text = self.descriptions[index]
        verdict_label = 1.0 if self.labels[index] == _SUCC_TEXT else 0.0

        def _expand_image_tokens(text: str) -> str:
            image_token = getattr(self.tokenizer, "image_token", "<image_soft_token>")
            boi_token = getattr(self.tokenizer, "boi_token", "<start_of_image>")
            eoi_token = getattr(self.tokenizer, "eoi_token", "<end_of_image>")
            full_image_sequence = f"\n\n{boi_token}{image_token * 256}{eoi_token}\n\n"
            return text.replace(boi_token, full_image_sequence)

        def _chat_to_ids(messages, add_gen_prompt):
            if self.use_image_obs:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_gen_prompt,
                )
                rendered = _expand_image_tokens(rendered)
                ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
                return torch.tensor(ids, dtype=torch.long)
            out = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_gen_prompt,
            )
            if hasattr(out, "input_ids"):
                ids = out["input_ids"]
            elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], int):
                ids = out
            else:
                ids = out
            return torch.tensor(ids, dtype=torch.long)

        user_content = (
            [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]
            if self.use_image_obs else user_text
        )

        full_ids = _chat_to_ids(
            [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_text},
            ],
            add_gen_prompt=False,
        )
        if self.use_eos_token:
            full_ids = torch.cat(
                [full_ids, torch.tensor([self.tokenizer.eos_token_id])]
            )

        prompt_ids = _chat_to_ids(
            [{"role": "user", "content": user_content}],
            add_gen_prompt=True,
        )
        prompt_ids_val = _chat_to_ids(
            [{"role": "user", "content": user_content}],
            add_gen_prompt=False,
        )

        labels = full_ids.clone()
        labels[: prompt_ids.size(0)] = -100  # mask the user / system prompt

        return {
            "input_ids":      full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels":         labels,
            "latent_obs":     self.latent_observations[index].squeeze(),
            "actions":        self.actions[index].squeeze(),
            "obs":            self.raw_obs[index].squeeze(),
            "description":    assistant_text,
            "verdict_labels": torch.tensor(verdict_label, dtype=torch.float32),
            "prompt_ids_val": prompt_ids_val,
            "image_path":      self.image_paths[index],
            "folder":          self.folders[index],
            "chunk_idx":       self.chunk_indices[index],
            "task_instruction": self.task_instructions[index],
            "eef_pos_before":  self.eef_pos_before[index],
            "eef_pos_after":   self.eef_pos_after[index],
            "gripper_before":  self.gripper_before[index],
            "gripper_after":   self.gripper_after[index],
            "success_label":   torch.tensor(verdict_label, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────

class LIBEROCollatorWithLatents:
    """
    Custom collator that pads text sequences and stacks latent/action tensors.
    Mirrors CollatorWithLatentsAndPadding from the MetaWorld codebase.
    """

    def __init__(self, tokenizer, is_val: bool = False, use_image_obs: bool = False):
        self.tokenizer = tokenizer
        self.text_pad = DataCollatorWithPadding(
            tokenizer, padding="longest", return_tensors="pt"
        )
        self.pad_id = tokenizer.pad_token_id
        self.is_val = is_val
        self.use_image_obs = use_image_obs
        self.image_processor = (
            Gemma3ImageProcessor(size={"height": 896, "width": 896})
            if use_image_obs else None
        )

    def __call__(self, features):
        # Pop side tensors before text padding
        lats = torch.stack([f.pop("latent_obs") for f in features])
        acts = torch.stack([f.pop("actions") for f in features])
        obses = torch.stack([f.pop("obs") for f in features])
        descriptions = [f.pop("description") for f in features]
        image_paths = [f.pop("image_path", None) for f in features]
        folders = [f.pop("folder", None) for f in features]
        chunk_indices = [f.pop("chunk_idx", None) for f in features]
        task_instructions = [f.pop("task_instruction", "") for f in features]
        eef_pos_before = [f.pop("eef_pos_before", None) for f in features]
        eef_pos_after = [f.pop("eef_pos_after", None) for f in features]
        gripper_before = [f.pop("gripper_before", None) for f in features]
        gripper_after = [f.pop("gripper_after", None) for f in features]
        success_labels = [f.pop("success_label", None) for f in features]
        verdict_labels = [
            f.pop("verdict_labels") for f in features if "verdict_labels" in f
        ]
        labels_list = [f.pop("labels") for f in features]
        # Always pop prompt_ids_val — only keep it in the batch for val
        prompt_ids_val_list = [f.pop("prompt_ids_val") for f in features]

        # Pad text
        batch = self.text_pad(features)

        # Pad labels to match input length
        B, L = batch["input_ids"].shape
        lab_pad = torch.full((B, L), -100, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            lab = torch.as_tensor(lab, dtype=torch.long)
            lab_pad[i, : lab.shape[0]] = lab
        lab_pad[batch["input_ids"] == self.pad_id] = -100
        batch["labels"] = lab_pad

        if self.is_val:
            batch["prompt_ids_val"] = pad_sequence(
                prompt_ids_val_list, batch_first=True,
                padding_value=self.pad_id,
            )

        if self.use_image_obs:
            images = []
            for p in image_paths:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))
            image_inputs = self.image_processor(images=images, return_tensors="pt")
            batch["pixel_values"] = image_inputs["pixel_values"]
            token_type_ids = torch.zeros_like(batch["input_ids"])
            image_token_id = getattr(self.tokenizer, "image_token_id", None)
            if image_token_id is not None:
                token_type_ids[batch["input_ids"] == image_token_id] = 1
            batch["token_type_ids"] = token_type_ids

        batch["latent_obs"] = lats
        batch["actions"] = acts
        batch["obs"] = obses
        batch["description_text"] = descriptions
        batch["image_path"] = image_paths
        batch["folder"] = folders
        batch["chunk_idx"] = chunk_indices
        batch["task_instruction"] = task_instructions
        if all(x is not None for x in eef_pos_before):
            batch["eef_pos_before"] = torch.stack(eef_pos_before)
            batch["eef_pos_after"] = torch.stack(eef_pos_after)
            batch["gripper_before"] = torch.stack(gripper_before)
            batch["gripper_after"] = torch.stack(gripper_after)
        if all(x is not None for x in success_labels):
            batch["success_label"] = torch.stack(success_labels)
        if verdict_labels:
            batch["verdict_labels"] = torch.stack(verdict_labels)

        return batch
