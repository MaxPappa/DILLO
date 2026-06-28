"""
LIBERO Validation Dataset

A lightweight dataset that mirrors the exact item-loading and split logic used
during DILLO training (dillo.training.train_dillo / LIBEROBehaviorDataset),
but additionally tracks:
  - source episode folder
  - chunk index within the episode
  - raw before/after EEF and gripper positions (for metric computation)
  - paths to before/after frame images (for VLM baselines)

The val split is reproduced with the same seed (42) and ratio (10%) that was
used in training so that the evaluation set is IDENTICAL to training's held-out
portion.

Usage:
    from libero_val_dataset import build_libero_val_items

    val_items = build_libero_val_items(
        data_dirs="data/libero_spatial_video_and_obs/*/*",
        split_seed=42,
        val_fraction=0.1,
        chunk_size=20,
        min_description_tokens=5,
        tokenizer_name="google/gemma-3-1b-it",
    )
    # val_items: list of ValItem dicts
"""
from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Same regex as dillo.training.behavior_dataset
# ---------------------------------------------------------------------------
TRANSITION_RE = re.compile(
    r"<frame_(\d+)_to_frame_(\d+)>(.*?)</frame_\1_to_frame_\2>",
    re.DOTALL,
)


def parse_response(text: str) -> List[str]:
    matches = TRANSITION_RE.findall(text)
    return [m[2].strip() for m in matches]


# ---------------------------------------------------------------------------
# Item dataclass
# ---------------------------------------------------------------------------
@dataclass
class ValItem:
    """A single chunk-transition example from the validation split."""
    folder: str           # absolute path to the episode folder
    chunk_idx: int        # 0-based index of this chunk transition in the episode

    task_instruction: str
    description: str      # ground-truth description (from response.txt)

    # Raw robot state (before / after the chunk)
    eef_pos_before: np.ndarray   # (3,)
    eef_pos_after: np.ndarray    # (3,)
    gripper_before: np.ndarray   # (2,)
    gripper_after: np.ndarray    # (2,)
    robot_state_before: np.ndarray  # (12,) eef+joint+gripper
    robot_state_after: np.ndarray   # (12,)

    # Actions for the chunk: (chunk_size, 7)
    actions: np.ndarray

    # Image paths (may be None if images/ folder is absent)
    image_before_path: Optional[str] = None
    image_after_path: Optional[str] = None

    # Success mask value at this transition (None if file absent)
    success_label: Optional[int] = None   # 1 = success, 0 = failure


# ---------------------------------------------------------------------------
# Internal: load one episode folder
# ---------------------------------------------------------------------------
def _load_episode(
    folder: Path,
    chunk_size: int,
    min_description_tokens: int,
    tokenizer,
) -> List[ValItem]:
    """Return a list of ValItems for every valid chunk-transition in one episode."""
    response_path = folder / "response.txt"
    if not response_path.exists():
        return []

    with open(response_path) as f:
        response_text = f.read()

    descriptions = parse_response(response_text)
    if not descriptions:
        return []

    # Task instruction
    task_instruction = ""
    ti_path = folder / "task_instruction.txt"
    if ti_path.exists():
        with open(ti_path) as f:
            task_instruction = f.read().strip()

    # Observations
    try:
        eef_pos  = np.load(folder / "eef_pos.npy").astype("float32")         # (T, 3)
        joint_pos = np.load(folder / "joint_pos.npy").astype("float32")      # (T, 7)
        gripper  = np.load(folder / "gripper_states.npy").astype("float32")  # (T, 2)
    except FileNotFoundError:
        return []

    robot_state = np.concatenate([eef_pos, joint_pos, gripper], axis=1)  # (T, 12)

    # Actions
    try:
        all_actions = np.load(folder / "actions.npy").astype("float32")
    except FileNotFoundError:
        return []

    n_total = all_actions.shape[0]
    actual_chunks = n_total // chunk_size
    if actual_chunks == 0:
        return []
    action_chunks = all_actions[: actual_chunks * chunk_size].reshape(
        actual_chunks, chunk_size, -1
    )  # (actual_chunks, chunk_size, 7)

    # Success mask (optional)
    success_mask = None
    mask_path = folder / "success_mask.npy"
    if mask_path.exists():
        success_mask = np.load(mask_path)  # (T,)

    items = []
    for i in range(min(len(descriptions), actual_chunks)):
        desc = descriptions[i]

        # Token-length filter (exact match of training filter)
        if tokenizer is not None:
            tok_ids = tokenizer(desc, return_tensors="np")["input_ids"]
            if tok_ids.shape[1] < min_description_tokens:
                continue
        else:
            # Fallback: word-count
            if len(desc.split()) < min_description_tokens:
                continue

        if not desc.strip():
            continue

        # Image paths
        img_dir = folder / "images"
        img_before = str(img_dir / f"{i}.jpeg") if (img_dir / f"{i}.jpeg").exists() else None
        img_after  = str(img_dir / f"{i + 1}.jpeg") if (img_dir / f"{i + 1}.jpeg").exists() else None

        # Success label
        succ = None
        if success_mask is not None and (i + 1) < len(success_mask):
            succ = int(success_mask[i + 1])

        items.append(ValItem(
            folder=str(folder),
            chunk_idx=i,
            task_instruction=task_instruction,
            description=desc,
            eef_pos_before=eef_pos[i].copy(),
            eef_pos_after=eef_pos[i + 1].copy(),
            gripper_before=gripper[i].copy(),
            gripper_after=gripper[i + 1].copy(),
            robot_state_before=robot_state[i].copy(),
            robot_state_after=robot_state[i + 1].copy(),
            actions=action_chunks[i].copy(),
            image_before_path=img_before,
            image_after_path=img_after,
            success_label=succ,
        ))

    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_libero_val_items(
    data_dirs: str | List[str],
    split_seed: int = 42,
    val_fraction: float = 0.1,
    chunk_size: int = 20,
    min_description_tokens: int = 5,
    tokenizer_name: str = "google/gemma-3-1b-it",
    tokenizer=None,
) -> List[ValItem]:
    """
    Build the validation split that exactly matches the one used during training.

    The procedure mirrors dillo.training.train_dillo:
      1. Collect all episode paths (sorted glob).
      2. Load items per episode in the same order.
      3. Apply torch.utils.data.random_split(seed=split_seed, val=val_fraction).
      4. Return only the validation items.

    Args:
        data_dirs: Glob pattern(s) or list matching episode folders, e.g.
                   "data/libero_spatial_video_and_obs/*/*"
        split_seed: Random seed used for the split (default 42 matching training).
        val_fraction: Fraction of items held out for validation (default 0.1).
        chunk_size: Number of atomic actions per chunk (must match training).
        min_description_tokens: Minimum token count to keep a description.
        tokenizer_name: HuggingFace tokenizer used for filtering.
        tokenizer: Pre-loaded tokenizer; if provided, tokenizer_name is ignored.

    Returns:
        List of ValItem objects.
    """
    # Load tokenizer
    if tokenizer is None and tokenizer_name is not None:
        from transformers import AutoTokenizer
        print(f"[build_libero_val_items] Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Resolve paths
    if isinstance(data_dirs, str):
        episode_paths = sorted(glob(data_dirs))
    else:
        episode_paths = []
        for d in data_dirs:
            episode_paths.extend(sorted(glob(d)))

    if not episode_paths:
        raise FileNotFoundError(f"No episodes found for patterns: {data_dirs}")

    print(f"[build_libero_val_items] Found {len(episode_paths)} episode folders")

    # Load items preserving order (same as LIBEROBehaviorDataset._load_examples)
    all_items: List[ValItem] = []
    skipped = 0
    for ep_path in episode_paths:
        try:
            items = _load_episode(
                Path(ep_path), chunk_size, min_description_tokens, tokenizer
            )
            all_items.extend(items)
        except Exception as e:
            skipped += 1

    print(f"[build_libero_val_items] Loaded {len(all_items)} items total "
          f"(skipped {skipped} folders)")

    if not all_items:
        raise RuntimeError("No valid items loaded. Check data_dirs and dataset format.")

    # Apply the same split as in training
    total    = len(all_items)
    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size

    # torch.utils.data.random_split with manual_seed is used only to obtain
    # the index mapping; we re-implement it with the same RNG so there is no
    # need to actually wrap items in a Dataset.
    indices = list(range(total))
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(total, generator=g).tolist()
    # random_split assigns the first `train_size` elements of the permutation
    # to train and the remaining `val_size` to val.
    train_indices = set(perm[:train_size])
    val_indices   = [i for i in perm[train_size:]]

    val_items = [all_items[i] for i in sorted(val_indices)]

    print(f"[build_libero_val_items] Split → train: {train_size}, val: {val_size}")

    return val_items


def val_items_to_dicts(val_items: List[ValItem]) -> List[dict]:
    """Serialise ValItem list to a JSON-friendly list of dicts."""
    def _to_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr

    return [
        {
            "folder": item.folder,
            "chunk_idx": item.chunk_idx,
            "task_instruction": item.task_instruction,
            "description": item.description,
            "eef_pos_before": _to_list(item.eef_pos_before),
            "eef_pos_after": _to_list(item.eef_pos_after),
            "gripper_before": _to_list(item.gripper_before),
            "gripper_after": _to_list(item.gripper_after),
            "robot_state_before": _to_list(item.robot_state_before),
            "robot_state_after": _to_list(item.robot_state_after),
            "actions": _to_list(item.actions),
            "image_before_path": item.image_before_path,
            "image_after_path": item.image_after_path,
            "success_label": item.success_label,
        }
        for item in val_items
    ]
