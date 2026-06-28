"""Inference wrapper for the DILLO ACT policy checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

from dillo.libero_imports import prepare_libero_imports

prepare_libero_imports()

import robomimic.utils.obs_utils as ObsUtils

from libero.lifelong.utils import get_task_embs

from dillo.policy.act_policy import ActionChunkingPolicy
from dillo.policy.obs import OBS_KEY_MAPPING, OBS_KEYS, OBS_MODALITY


class ACTAgent:
    """
    Rollout wrapper for :class:`ActionChunkingPolicy`.

    The saved checkpoint is expected to contain ``state_dict``, ``args``, and
    ``shape_meta``.  This is the format produced by
    ``python -m dillo.policy.train_act``.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
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
            img_input_shape=shape_meta["all_shapes"].get(
                "agentview_rgb", (3, 128, 128)
            ),
            translation=8,
            temporal_decay=cfg.get("temporal_decay", 0.01),
        )
        self.policy.load_state_dict(checkpoint["state_dict"])
        self.policy.to(self.device)
        self.policy.eval()

        self.chunk_size = cfg.get("chunk_size", 20)
        img_shape = shape_meta["all_shapes"].get("agentview_rgb", (3, 128, 128))
        self._img_h, self._img_w = img_shape[1], img_shape[2]
        self._task_emb_cache: Dict[str, torch.Tensor] = {}
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

    def reset(self) -> None:
        """Clear the open-loop action buffer at the start of an episode."""
        self.policy.reset()

    def task_embedding(self, language_instruction: str) -> torch.Tensor:
        """Return a cached ``(1, 768)`` task embedding for an instruction."""
        if language_instruction not in self._task_emb_cache:
            embs = get_task_embs(self._task_emb_cfg, [language_instruction])
            self._task_emb_cache[language_instruction] = embs[0:1].cpu()
        return self._task_emb_cache[language_instruction].to(self.device)

    def _get_task_emb(self, language_instruction: str) -> torch.Tensor:
        """Compatibility alias used by dataset latent extraction."""
        return self.task_embedding(language_instruction)

    def preprocess_obs(self, obs: dict) -> dict:
        """Convert one raw LIBERO observation into policy input tensors."""
        data = {"obs": {}}
        for obs_name in OBS_KEYS:
            env_key = OBS_KEY_MAPPING[obs_name]
            tensor = ObsUtils.process_obs(
                torch.from_numpy(obs[env_key]), obs_key=obs_name
            ).float()
            tensor = tensor.unsqueeze(0).to(self.device)
            if tensor.ndim == 4 and (
                tensor.shape[2] != self._img_h or tensor.shape[3] != self._img_w
            ):
                tensor = F.interpolate(
                    tensor,
                    size=(self._img_h, self._img_w),
                    mode="bilinear",
                    align_corners=False,
                )
            data["obs"][obs_name] = tensor
        return data

    def act(self, obs: dict, language_instruction: str) -> np.ndarray:
        """Predict one action from the internal chunk buffer."""
        data = self.preprocess_obs(obs)
        data["task_emb"] = self.task_embedding(language_instruction)
        action = self.policy.get_action(data)
        return action[0]
