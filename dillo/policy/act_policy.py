"""
Action Chunking Policy with GMM head for LIBERO.

Architecture:
    - Image Encoder: ResNet-18 with FiLM language conditioning (per camera)
    - Language Encoder: MLP projection of pretrained BERT embedding
    - Proprioception Encoder: per-modality MLPs (joint_states, gripper_states)
    - Chunk Decoder: Transformer decoder with cross-attention to observation context
    - Policy Head: GMM (5-mode Gaussian Mixture) per chunk step

The policy predicts a chunk of K future actions given a single observation.
At inference, temporal ensembling blends overlapping chunk predictions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from dillo.libero_imports import prepare_libero_imports

prepare_libero_imports()

from libero.lifelong.models.modules.rgb_modules import ResnetEncoder
from libero.lifelong.models.modules.data_augmentation import (
    BatchWiseImgColorJitterAug,
    TranslationAug,
    DataAugGroup,
)


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class ExtraModalityTokens(nn.Module):
    """Projects each proprioceptive modality into its own token."""

    def __init__(self, use_joint=True, use_gripper=True, use_ee=False, embed_size=64):
        super().__init__()
        self.modalities = []
        if use_joint:
            self.joint_encoder = nn.Linear(7, embed_size)
            self.modalities.append(("joint_states", self.joint_encoder))
        if use_gripper:
            self.gripper_encoder = nn.Linear(2, embed_size)
            self.modalities.append(("gripper_states", self.gripper_encoder))
        if use_ee:
            self.ee_encoder = nn.Linear(3, embed_size)
            self.modalities.append(("ee_states", self.ee_encoder))
        # register as module list so parameters are found
        self.encoders = nn.ModuleList([enc for _, enc in self.modalities])

    def forward(self, obs_dict):
        """
        Args:
            obs_dict: maps modality name → (B, dim)
        Returns:
            (B, num_modalities, embed_size)
        """
        tokens = [enc(obs_dict[name]) for name, enc in self.modalities]
        return torch.stack(tokens, dim=1)


class CrossAttentionDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with self-attn + cross-attn + FFN."""

    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm_sa = nn.LayerNorm(embed_size)
        self.self_attn = nn.MultiheadAttention(
            embed_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ca = nn.LayerNorm(embed_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_size),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, queries, context):
        """
        Args:
            queries:  (B, K, E)
            context:  (B, N, E)
        Returns:
            (B, K, E)
        """
        # self-attention
        q = self.norm_sa(queries)
        queries = queries + self.drop(self.self_attn(q, q, q)[0])
        # cross-attention
        q = self.norm_ca(queries)
        queries = queries + self.drop(self.cross_attn(q, context, context)[0])
        # feed-forward
        queries = queries + self.ffn(self.norm_ff(queries))
        return queries


class ChunkDecoder(nn.Module):
    """
    Transformer decoder that produces K action-embedding vectors
    by cross-attending learned chunk queries to observation context tokens.
    """

    def __init__(
        self,
        chunk_size: int,
        embed_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_queries = nn.Parameter(
            torch.randn(1, chunk_size, embed_size) * 0.02
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, chunk_size, embed_size) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(embed_size, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_size)

    def forward(self, context):
        """
        Args:
            context: (B, N_ctx, E)   observation context tokens
        Returns:
            (B, chunk_size, E)
        """
        B = context.shape[0]
        queries = (self.chunk_queries + self.pos_embed).expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, context)
        return self.final_norm(queries)


class GMMHead(nn.Module):
    """
    Gaussian Mixture Model head that supports both per-step and chunked output.
    Produces a MixtureSameFamily distribution.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 1024,
        num_layers: int = 2,
        num_modes: int = 5,
        min_std: float = 1e-4,
        activation: str = "softplus",
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        sizes = [input_size] + [hidden_size] * num_layers
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.share = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.actv = F.softplus if activation == "softplus" else torch.exp

    # ---- helpers ----------------------------------------------------------

    def _forward_fn(self, x):
        """x: (*, input_size)  →  means/stds/logits"""
        h = self.share(x)
        means = self.mean_layer(h).view(*h.shape[:-1], self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(h)
        logstds = self.logstd_layer(h).view(
            *h.shape[:-1], self.num_modes, self.output_size
        )
        stds = self.actv(logstds) + self.min_std
        return means, stds, logits

    def forward(self, x):
        """
        Args:
            x: (B, E) or (B, K, E)
        Returns:
            MixtureSameFamily distribution
        """
        means, stds, logits = self._forward_fn(x)
        compo = D.Independent(D.Normal(loc=means, scale=stds), 1)
        mix = D.Categorical(logits=logits)
        return D.MixtureSameFamily(mix, compo)

    def loss_fn(self, gmm, target, reduction="mean"):
        """Negative log-likelihood loss."""
        nll = -gmm.log_prob(target)
        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        return nll


# ---------------------------------------------------------------------------
#  Temporal ensembler (used at inference)
# ---------------------------------------------------------------------------

class TemporalEnsembler:
    """
    Blends overlapping action-chunk predictions with exponential weighting.
    More recent predictions receive higher weight.
    """

    def __init__(self, chunk_size: int, action_dim: int, decay: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay
        self.history = []           # list of (B, K, A) tensors

    def add(self, chunk):
        """Register a newly predicted action chunk."""
        self.history.append(chunk)
        if len(self.history) > self.chunk_size:
            self.history.pop(0)

    def get_action(self):
        """
        Return the temporally-ensembled action for the *current* step.

        The current step corresponds to position 0 of the most recent chunk,
        position 1 of the second-most-recent chunk, etc.
        """
        device = self.history[-1].device
        B = self.history[-1].shape[0]
        weighted_sum = torch.zeros(B, self.action_dim, device=device)
        w_sum = 0.0
        for age, chunk in enumerate(reversed(self.history)):
            pos = age            # position inside that chunk
            if pos < chunk.shape[1]:
                w = math.exp(-self.decay * age)
                weighted_sum += w * chunk[:, pos]
                w_sum += w
        return weighted_sum / w_sum

    def reset(self):
        self.history.clear()


# ---------------------------------------------------------------------------
#  Main policy
# ---------------------------------------------------------------------------

class ActionChunkingPolicy(nn.Module):
    """
    Multi-task action-chunking policy for LIBERO.

    Given a single observation (images + proprioception + language embedding),
    predicts a chunk of K future actions via a GMM distribution.
    At inference, temporal ensembling is applied across overlapping predictions.
    """

    def __init__(
        self,
        shape_meta: dict,
        # encoder
        embed_size: int = 64,
        language_input_size: int = 768,
        language_hidden_size: int = 128,
        # chunk
        chunk_size: int = 20,
        # chunk decoder transformer
        decoder_num_layers: int = 2,
        decoder_num_heads: int = 4,
        decoder_ff_dim: int = 256,
        decoder_dropout: float = 0.1,
        # gmm
        gmm_hidden_size: int = 1024,
        gmm_num_layers: int = 2,
        gmm_num_modes: int = 5,
        gmm_min_std: float = 1e-4,
        # proprio
        use_joint: bool = True,
        use_gripper: bool = True,
        use_ee: bool = False,
        # augmentation
        use_augmentation: bool = True,
        img_input_shape: tuple = (3, 128, 128),
        translation: int = 8,
        # temporal ensembling
        temporal_decay: float = 0.01,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.chunk_size = chunk_size
        self.use_augmentation = use_augmentation
        ac_dim = shape_meta["ac_dim"]

        # ---- image encoders (language-conditioned ResNet + FiLM) ----
        self.image_encoders = nn.ModuleDict()
        rgb_keys = [k for k in shape_meta["all_shapes"] if "rgb" in k]
        for name in rgb_keys:
            self.image_encoders[name] = ResnetEncoder(
                input_shape=shape_meta["all_shapes"][name],
                output_size=embed_size,
                pretrained=False,
                freeze=False,
                remove_layer_num=4,
                no_stride=False,
                language_dim=language_input_size,
                language_fusion="film",
            )

        # ---- language encoder ----
        self.language_encoder = nn.Sequential(
            nn.Linear(language_input_size, language_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(language_hidden_size, embed_size),
        )

        # ---- proprioception encoder ----
        self.extra_encoder = ExtraModalityTokens(
            use_joint=use_joint,
            use_gripper=use_gripper,
            use_ee=use_ee,
            embed_size=embed_size,
        )

        # ---- chunk decoder ----
        self.chunk_decoder = ChunkDecoder(
            chunk_size=chunk_size,
            embed_size=embed_size,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            ff_dim=decoder_ff_dim,
            dropout=decoder_dropout,
        )

        # ---- GMM policy head ----
        self.gmm_head = GMMHead(
            input_size=embed_size,
            output_size=ac_dim,
            hidden_size=gmm_hidden_size,
            num_layers=gmm_num_layers,
            num_modes=gmm_num_modes,
            min_std=gmm_min_std,
        )

        # ---- data augmentation (same as LIBERO default) ----
        if use_augmentation:
            color_aug = BatchWiseImgColorJitterAug(
                input_shape=img_input_shape,
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.3,
                epsilon=0.1,
            )
            translation_aug = TranslationAug(
                input_shape=img_input_shape,
                translation=translation,
            )
            self.img_aug = DataAugGroup((color_aug, translation_aug))
        else:
            self.img_aug = None

        # ---- action buffer (open-loop chunk execution at inference) ----
        self._action_buffer = None   # will hold (B, remaining_K, ac_dim)
        self._buffer_idx = 0

    # ------------------------------------------------------------------
    #  Spatial encoding
    # ------------------------------------------------------------------

    def _spatial_encode(self, obs, task_emb):
        """
        Encode a single-step observation into a set of context tokens.

        Args:
            obs:      dict of modality → (B, *shape)  (no time dim)
            task_emb: (B, lang_dim)
        Returns:
            context:  (B, N_tokens, embed_size)
        """
        tokens = []

        # language token
        lang_token = self.language_encoder(task_emb)  # (B, E)
        tokens.append(lang_token.unsqueeze(1))          # (B, 1, E)

        # image tokens (FiLM-conditioned)
        for name, encoder in self.image_encoders.items():
            img = obs[name]                              # (B, C, H, W)
            img_token = encoder(img, langs=task_emb)     # (B, E)
            tokens.append(img_token.unsqueeze(1))        # (B, 1, E)

        # proprioception tokens
        proprio_tokens = self.extra_encoder(obs)         # (B, n_extra, E)
        tokens.append(proprio_tokens)

        return torch.cat(tokens, dim=1)                  # (B, N, E)

    # ------------------------------------------------------------------
    #  Forward (training)
    # ------------------------------------------------------------------

    def forward(self, obs, task_emb):
        """
        Args:
            obs:      dict of modality → (B, *shape)  single-step observation
            task_emb: (B, lang_dim)
        Returns:
            gmm distribution over (B, chunk_size, ac_dim)
        """
        context = self._spatial_encode(obs, task_emb)   # (B, N, E)
        chunk_embed = self.chunk_decoder(context)        # (B, K, E)
        return self.gmm_head(chunk_embed)                # GMM dist

    # ------------------------------------------------------------------
    #  Latent extraction
    # ------------------------------------------------------------------

    def get_latent(self, obs, task_emb, pool="mean"):
        """
        Extract a latent representation that encodes both the current
        environment state and the predicted action intent.

        Args:
            obs:      dict of modality → (B, *shape)  single-step observation
            task_emb: (B, lang_dim)
            pool:     how to aggregate across the K chunk positions.
                      "mean"  → average over K        → (B, E)
                      "first" → first chunk position   → (B, E)
                      "none"  → no pooling             → (B, K, E)
        Returns:
            latent: (B, E) or (B, K, E) depending on *pool*
        """
        context = self._spatial_encode(obs, task_emb)   # (B, N, E)
        chunk_embed = self.chunk_decoder(context)        # (B, K, E)

        if pool == "mean":
            return chunk_embed.mean(dim=1)               # (B, E)
        elif pool == "first":
            return chunk_embed[:, 0]                     # (B, E)
        elif pool == "none":
            return chunk_embed                           # (B, K, E)
        else:
            raise ValueError(f"Unknown pool mode: {pool!r}")

    def get_context_latent(self, obs, task_emb, pool="mean"):
        """
        Extract a latent representation of the current environment state
        only (no action intent). This is the observation context *before*
        the chunk decoder.

        Args:
            obs:      dict of modality → (B, *shape)  single-step observation
            task_emb: (B, lang_dim)
            pool:     how to aggregate across the N context tokens.
                      "mean"  → average over N        → (B, E)
                      "first" → first token (language) → (B, E)
                      "none"  → no pooling             → (B, N, E)
        Returns:
            latent: (B, E) or (B, N, E) depending on *pool*
        """
        context = self._spatial_encode(obs, task_emb)   # (B, N, E)

        if pool == "mean":
            return context.mean(dim=1)                   # (B, E)
        elif pool == "first":
            return context[:, 0]                         # (B, E)
        elif pool == "none":
            return context                               # (B, N, E)
        else:
            raise ValueError(f"Unknown pool mode: {pool!r}")

    # ------------------------------------------------------------------
    #  Loss
    # ------------------------------------------------------------------

    def compute_loss(self, data):
        """
        Args:
            data: dict with keys  obs, actions, task_emb
                  obs values:     (B, T, ...)  where T = chunk_size
                  actions:        (B, T, ac_dim)
                  task_emb:       (B, lang_dim)
        Returns:
            scalar loss
        """
        # ---- extract first-step observation ----
        obs_0 = {}
        for key in data["obs"]:
            val = data["obs"][key]
            if val.ndim >= 3:          # (B, T, ...)
                obs_0[key] = val[:, 0]
            else:                      # (B, dim)  – should not happen in training
                obs_0[key] = val

        # ---- augment images ----
        if self.training and self.img_aug is not None:
            rgb_keys = list(self.image_encoders.keys())
            imgs = tuple(obs_0[k].unsqueeze(1) for k in rgb_keys)  # (B, 1, C, H, W)
            aug_imgs = self.img_aug(imgs)
            for i, k in enumerate(rgb_keys):
                obs_0[k] = aug_imgs[i].squeeze(1)                  # (B, C, H, W)

        # ---- forward pass ----
        dist = self.forward(obs_0, data["task_emb"])

        # ---- NLL loss over the full chunk ----
        actions = data["actions"]  # (B, T, ac_dim)
        return self.gmm_head.loss_fn(dist, actions)

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, data):
        """
        Open-loop chunk execution: predict a full chunk of K actions,
        then return them one at a time on successive calls.
        A new chunk is predicted only when the buffer is exhausted.

        Args:
            data: dict with keys  obs, task_emb
                  obs values: (B, *shape)    no time dim
                  task_emb:   (B, lang_dim)
        Returns:
            action: numpy (B, ac_dim)
        """
        self.eval()

        # predict a new chunk when the buffer is empty
        if self._action_buffer is None or self._buffer_idx >= self._action_buffer.shape[1]:
            obs = data["obs"]
            task_emb = data["task_emb"]
            dist = self.forward(obs, task_emb)
            self._action_buffer = dist.sample()         # (B, K, ac_dim)
            self._buffer_idx = 0

        action = self._action_buffer[:, self._buffer_idx]  # (B, ac_dim)
        self._buffer_idx += 1
        return action.cpu().numpy()

    def reset(self):
        """Call at the start of every evaluation episode."""
        self._action_buffer = None
        self._buffer_idx = 0
