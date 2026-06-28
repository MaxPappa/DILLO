"""
Policy Explainer model for LIBERO.

Adapts the Gemma3PolicyExplainer architecture from the MetaWorld codebase to
work with LIBERO observations (12-dim raw state or 512-dim ACT latent) and
LIBERO actions (chunk_size × 7-dim OSC_POSE commands).

The core idea is unchanged:
  - Project latent observations and action chunks into the LLM hidden space
  - Fuse them into a soft prefix
  - Prepend to text tokens and run the causal LM
  - Multi-stage training: stage1 → align projectors, stage2 → +LoRA,
    stage3 → fine-tune on success/failure classification
"""
from __future__ import annotations

import os
import re
import glob
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


class _SuccFailConstrainedProcessor:
    """LogitsProcessor that forces the first generated token to be one of the
    classification special tokens (<success> / <failure>) and bans them from
    all subsequent positions so they cannot appear in the description.

    A new instance must be created for every ``generate()`` call because the
    processor is stateful (step counter).
    """

    def __init__(self, cls_token_ids: list[int]):
        self.cls_ids = cls_token_ids
        self._step = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self._step == 0:
            # Force: only allow <success> / <failure>
            mask = torch.full_like(scores, float("-inf"))
            for tid in self.cls_ids:
                mask[:, tid] = 0.0
            scores = scores + mask
        else:
            # Ban: prevent <success> / <failure> in the description
            for tid in self.cls_ids:
                scores[:, tid] = float("-inf")
        self._step += 1
        return scores


class VerdictHead(nn.Module):
    """Binary classifier head that safely handles mixed-precision inputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_dtype = self.norm.weight.dtype
        x = x.to(dtype=head_dtype)
        return self.out(self.norm(x)).squeeze(-1)


def latest_projector_ckpt(ckpt_dir: str):
    """Find the projector checkpoint with the highest epoch number."""
    patterns = [
        os.path.join(ckpt_dir, "e=*_*policy_explainer.pth"),
        os.path.join(ckpt_dir, "e=*_*_policy_explainer.pth"),
        os.path.join(ckpt_dir, "e=*policy_explainer.pth"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    best, best_epoch = None, -1
    for p in paths:
        m = re.search(r"e=(\d+)", os.path.basename(p))
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best, best_epoch = p, ep
    return best, best_epoch


class LIBEROPolicyExplainer(nn.Module):
    """
    Soft-prefix causal LM that conditions on robot state latents + action chunks.

    Architecture:
        latent_obs → latent_projector → hidden_dim
        action_chunk → action_projector → hidden_dim
        fuse (sum / concat) → soft prefix
        [prefix || text_embeds] → LLM → next-token loss

    Args:
        latent_dim: Dimensionality of input observation.
                    12 for raw obs (eef3 + joint7 + gripper2),
                    or 512 for ACT state encoder latents.
        action_dim: Total flattened action dimension per chunk.
                    For chunk_size=10, action_dim=7: pass 70.
        projector_type: 'linear' or 'mlpNx_gelu' (e.g. 'mlp2x_gelu').
        stage: Training stage ('stage1', 'stage2', 'stage3', etc.).
        model_name: HuggingFace model name for the LLM backbone.
        obs_act_pair_fusion: How to combine obs and action embeddings
                             ('sum', 'concat', 'mlp').
        is_oracular: If True, the observation input includes 2 frames
                     (before + after), and latent_dim is doubled.
    """

    def __init__(
        self,
        latent_dim: int = 12,
        action_dim: int = 70,
        projector_type: str = "mlp2x_gelu",
        stage: str = "stage1",
        model_name: str = "google/gemma-3-1b-it",
        obs_act_pair_fusion: str = "sum",
        is_oracular: bool = False,
        description_loss_weight: float = 1.0,
        verdict_loss_weight: float = 1.0,
        use_verdict_head: bool = False,
        use_image_obs: bool = False,
    ):
        super().__init__()

        self.is_oracular = is_oracular
        self.stage = stage
        self.obs_act_pair_fusion = obs_act_pair_fusion
        self.description_loss_weight = description_loss_weight
        self.verdict_loss_weight = verdict_loss_weight
        self.use_verdict_head = use_verdict_head
        self.use_image_obs = use_image_obs

        self.model_name = model_name.split("/")[-1]

        # Load base LLM
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="eager"
        )
        # Resolve the text backbone once.
        # Pure LLM  (e.g. Gemma-3-1b):  self.base.model.embed_tokens
        # VLM       (e.g. Gemma-3-4b):  self.base.model.language_model.embed_tokens
        self._has_nested_language_model = hasattr(getattr(self.base, 'model', None), 'language_model')
        self.is_llm = not self._has_nested_language_model
        self.config = getattr(self.base.config, "text_config", self.base.config)
        hidden_size = self.config.hidden_size

        self.action_dim = action_dim

        # Stage-specific setup
        if stage == "stage1":
            self.freeze_base_architecture()
            print("[LIBEROPolicyExplainer] Stage 1: LLM frozen, training projectors only")
        elif stage == "stage0":
            print("[LIBEROPolicyExplainer] Stage 0: training full LLM + projectors (no LoRA)")
        elif stage in ("stage2",):
            self.add_lora()
            print("[LIBEROPolicyExplainer] Stage 2: LoRA enabled")
        elif stage in ("stage3",):
            # Freeze base first to reduce memory; LoRA will be loaded as
            # trainable via load_checkpoints → PeftModel.from_pretrained
            self.freeze_base_architecture()
            print("[LIBEROPolicyExplainer] Stage 3: base frozen, LoRA loaded later")
        elif stage in ("eval_stage2", "eval_stage3"):
            pass  # checkpoints loaded later
        elif stage == "eval":
            self.eval()
        else:
            pass

        # Optional fusion MLP (for obs_act_pair_fusion == 'mlp')
        if obs_act_pair_fusion == "mlp":
            self.fuse_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Register <success> / <failure> as single special tokens so they
        # are never split into sub-words and can be constrained during
        # generation.
        _n_new = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<success>", "<failure>"]}
        )
        if _n_new > 0:
            self.base.resize_token_embeddings(len(self.tokenizer))
        self.success_token_id = self.tokenizer.convert_tokens_to_ids("<success>")
        self.failure_token_id = self.tokenizer.convert_tokens_to_ids("<failure>")

        # Token-verdict models force <success>/<failure> as the first generated
        # token. Binary-head models generate descriptions only.
        self.constrain_cls_generation = (
            stage in ("stage3", "eval_stage3") and not use_verdict_head
        )

        # Observation projector
        if is_oracular and obs_act_pair_fusion == "sum":
            self.latent_projector = self._build_projector(
                latent_dim * 2, hidden_size, projector_type
            )
        else:
            self.latent_projector = self._build_projector(
                latent_dim, hidden_size, projector_type
            )

        # Action projector
        self.action_projector = self._build_projector(
            action_dim, hidden_size, projector_type
        )
        self.verdict_head = VerdictHead(hidden_size) if use_verdict_head else None

        self.model_embeds = self.base.get_input_embeddings()
        # Gemma3ForConditionalGeneration.get_input_embeddings() may return
        # the vision-side embedding or None.  Fall back to the text backbone.
        if self.model_embeds is None and self._has_nested_language_model:
            self.model_embeds = self.base.model.language_model.embed_tokens
        if self.model_embeds is None:
            self.model_embeds = self._get_text_embedding_layer()

        # Cache pad_token_id from the correct config level
        self._pad_token_id = getattr(self.config, 'pad_token_id', None)
        if self._pad_token_id is None:
            self._pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)

        # Whether to pass token_type_ids (Gemma3 VLM needs it, pure LLMs don't)
        self._needs_token_type_ids = (
            not self.is_llm and hasattr(self.base.config, 'model_type')
            and 'gemma3' in getattr(self.base.config, 'model_type', '')
        )

    # ─── LoRA ──────────────────────────────────────────────────────
    def add_lora(self):
        self.base.config.use_cache = False
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ],
        )
        self.base = get_peft_model(self.base, lora_cfg)
        self.model_embeds = self._get_text_embedding_layer()
        self.base.print_trainable_parameters()

    # ─── Checkpoint I/O ────────────────────────────────────────────
    def load_checkpoints(self, ckpt_dir: str):
        # Load LoRA weights only when a previous stage saved them
        # (stage2 loads from stage1 which has no LoRA; stage3 loads from stage2 which does)
        if self.stage in ("stage3", "eval_stage2", "eval_stage3", "eval"):
            lora_path = os.path.join(ckpt_dir, f"lora-{self.model_name}")
            is_trainable = self.stage == "stage3"
            self.base = PeftModel.from_pretrained(
                self.base, lora_path, is_trainable=is_trainable
            )

        proj_ckpt_path, epoch = latest_projector_ckpt(ckpt_dir)
        if proj_ckpt_path is None:
            raise FileNotFoundError(f"No projector checkpoint found in {ckpt_dir}")

        ckpt = torch.load(proj_ckpt_path, map_location="cpu")
        self._load_projector_state_dict(
            self.latent_projector, ckpt["latent_projector_state_dict"]
        )
        if "action_projector_state_dict" in ckpt:
            self._load_projector_state_dict(
                self.action_projector, ckpt["action_projector_state_dict"]
            )
        if self.verdict_head is not None and "verdict_head_state_dict" in ckpt:
            self.verdict_head.load_state_dict(ckpt["verdict_head_state_dict"])
        print(f"[LIBEROPolicyExplainer] Loaded projectors from {proj_ckpt_path} (epoch {epoch})")

    def save_checkpoints(self, output_dir: str, save_function, epoch):
        if self.stage != "stage1":
            self.base.save_pretrained(
                os.path.join(output_dir, f"lora-{self.model_name}"),
                save_function=save_function,
                safe_serialization=True,
            )
        checkpoint = {
            "epoch": epoch,
            "latent_projector_state_dict": self.latent_projector.state_dict(),
            "action_projector_state_dict": self.action_projector.state_dict(),
            "stage": self.stage,
        }
        if self.verdict_head is not None:
            checkpoint["verdict_head_state_dict"] = self.verdict_head.state_dict()
        torch.save(checkpoint, os.path.join(output_dir, f"e={int(epoch)}_policy_explainer.pth"))

    # ─── Projector builder ─────────────────────────────────────────
    @staticmethod
    def _build_projector(inp_dim: int, hidden_size: int, projector_type: str) -> nn.Module:
        if projector_type == "linear":
            return nn.Linear(inp_dim, hidden_size)

        mlp_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_match:
            depth = int(mlp_match.group(1))
            modules = [nn.Linear(inp_dim, hidden_size)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.LayerNorm(hidden_size))
            return nn.Sequential(*modules)

        raise ValueError(f"Unknown projector type: {projector_type}")

    # ─── Freeze helpers ────────────────────────────────────────────
    def freeze_base_architecture(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def freeze_all(self):
        self.freeze_base_architecture()
        for param in self.latent_projector.parameters():
            param.requires_grad = False
        for param in self.action_projector.parameters():
            param.requires_grad = False

    # ─── Fusion ────────────────────────────────────────────────────
    def _fuse_embeds(self, latent_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Project and fuse observation + action into soft prefix embeddings.

        Args:
            latent_obs: (B, latent_dim) or (B, 2, latent_dim) for oracular
            actions: (B, action_dim) — already flattened chunk

        Returns:
            fused: (B, num_prefix_tokens, hidden)
        """
        if self.use_image_obs:
            latent_proj = None
        elif self.is_oracular and self.obs_act_pair_fusion == "sum":
            latent_proj = self.latent_projector(
                latent_obs.reshape(latent_obs.shape[0], -1)
            )
        else:
            latent_proj = self.latent_projector(latent_obs)

        # Flatten action chunks if needed: (B, chunk_size, 7) → (B, chunk_size*7)
        if actions.dim() == 3:
            actions = actions.reshape(actions.shape[0], -1)

        action_proj = self.action_projector(actions)

        if self.use_image_obs:
            fused = action_proj
        elif self.obs_act_pair_fusion == "sum":
            fused = latent_proj + action_proj
        elif self.obs_act_pair_fusion == "concat":
            if self.is_oracular:
                fused = torch.stack(
                    [latent_proj[:, 0], latent_proj[:, 1], action_proj], dim=1
                )
            else:
                fused = torch.stack([latent_proj, action_proj], dim=1)
        elif self.obs_act_pair_fusion == "mlp":
            fused = self.fuse_layer(
                torch.cat([latent_proj, action_proj], dim=-1)
            )
        else:
            raise ValueError(f"Unknown fusion: {self.obs_act_pair_fusion}")

        if fused.dim() == 2:
            fused = fused.unsqueeze(1)  # (B, 1, H)
        return fused

    # ─── Forward ───────────────────────────────────────────────────
    def _get_embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings, handling PeftModel wrapping."""
        return self._get_text_embedding_layer()(input_ids)

    def _get_text_embedding_layer(self) -> nn.Module:
        """Resolve text token embeddings across generic CausalLM, PEFT, and Gemma3 VLM wrappers."""
        candidates = []
        base_model = getattr(self.base, "model", None)
        nested_model = getattr(base_model, "model", None)

        if self._has_nested_language_model:
            candidates.extend([
                getattr(getattr(nested_model, "language_model", None), "embed_tokens", None),
                getattr(getattr(base_model, "language_model", None), "embed_tokens", None),
                getattr(getattr(self.base, "model", None), "language_model", None),
            ])

        try:
            candidates.append(self.base.get_input_embeddings())
        except Exception:
            pass

        candidates.extend([
            getattr(nested_model, "embed_tokens", None),
            getattr(base_model, "embed_tokens", None),
            getattr(getattr(self.base, "language_model", None), "embed_tokens", None),
        ])

        for candidate in candidates:
            if isinstance(candidate, nn.Module):
                return candidate
        raise AttributeError("Could not resolve text embedding layer for backbone")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        latent_obs=None,
        actions=None,
        verdict_labels=None,
        pixel_values=None,
        token_type_ids=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        for key in (
            "description_text",
            "image_path",
            "folder",
            "chunk_idx",
            "task_instruction",
            "eef_pos_before",
            "eef_pos_after",
            "gripper_before",
            "gripper_after",
            "success_label",
            "prompt_ids_val",
            "obs",
        ):
            kwargs.pop(key, None)

        embed_input_ids = input_ids
        if (
            self.use_image_obs
            and input_ids is not None
            and self._has_nested_language_model
            and hasattr(self.base.config, "image_token_index")
        ):
            # Gemma3 image tokens are valid text embeddings and are later
            # replaced by vision features through masked_scatter.
            embed_input_ids = input_ids.clone()

        text_embeds = self._get_embed_tokens(embed_input_ids)
        fused = self._fuse_embeds(latent_obs, actions)

        # Cast fused prefix to match the LLM embedding dtype (e.g. bf16)
        fused = fused.to(dtype=text_embeds.dtype)

        inputs_embeds = torch.cat([fused, text_embeds], dim=1)

        if attention_mask is None:
            attention_mask = (input_ids != self._pad_token_id).long()
        B = inputs_embeds.size(0)
        prefix_mask = torch.ones(
            B, fused.size(1), dtype=attention_mask.dtype, device=attention_mask.device
        )
        attn = torch.cat([prefix_mask, attention_mask], dim=1)

        if labels is not None:
            prefix_ignore = labels.new_full((B, fused.size(1)), -100)
            labels = torch.cat([prefix_ignore, labels], dim=1)

        # Gemma3 VLM requires token_type_ids during training to build its
        # causal mask.  0 = text token.  We have no image tokens here so
        # everything is 0.
        extra_kwargs = dict(**kwargs)
        if self._needs_token_type_ids:
            if token_type_ids is None:
                token_type_ids = torch.zeros(
                    text_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
                )
            prefix_token_type_ids = torch.zeros(
                B, fused.size(1), dtype=token_type_ids.dtype, device=token_type_ids.device
            )
            extra_kwargs['token_type_ids'] = torch.cat(
                [prefix_token_type_ids, token_type_ids], dim=1
            )
        if pixel_values is not None:
            extra_kwargs["pixel_values"] = pixel_values

        if self.use_verdict_head and labels is not None:
            return self._forward_with_verdict_head(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                labels=labels,
                prefix=fused,
                verdict_labels=verdict_labels,
                extra_kwargs=extra_kwargs,
            )

        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            **extra_kwargs,
        )

    def _forward_with_verdict_head(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prefix: torch.Tensor,
        verdict_labels: Optional[torch.Tensor],
        extra_kwargs: dict,
    ) -> CausalLMOutputWithPast:
        """Compute description LM loss plus a binary verdict-head loss."""
        compute_description_loss = self.description_loss_weight != 0
        outputs = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels if compute_description_loss else None,
            **extra_kwargs,
        )

        desc_loss = outputs.loss if compute_description_loss else inputs_embeds.new_zeros(())
        verdict_logits = self.verdict_head(prefix.mean(dim=1))
        loss = self.description_loss_weight * desc_loss
        if verdict_labels is not None:
            verdict_logits = verdict_logits.float()
            verdict_labels = verdict_labels.to(
                device=verdict_logits.device, dtype=verdict_logits.dtype
            )
            verdict_loss = F.binary_cross_entropy_with_logits(
                verdict_logits, verdict_labels
            )
            loss = loss + self.verdict_loss_weight * verdict_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    # ─── Generation ────────────────────────────────────────────────
    @torch.no_grad()
    def gen_from_batch(
        self,
        latent_obs,
        actions,
        input_ids,
        labels,
        attention_mask,
        prompt_ids,
        pixel_values=None,
        token_type_ids=None,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.8,
        top_p=0.9,
        top_k=None,
    ):
        """
        Build [soft_prefix || prompt] and call generate().
        Returns generated token ids (new tokens only).
        """
        tok = self.tokenizer

        # Embed prompt tokens.
        prompt_embeds = self._get_embed_tokens(prompt_ids)

        prefix = self._fuse_embeds(latent_obs, actions)
        # Cast prefix to match LLM embedding dtype (e.g. bf16)
        prefix = prefix.to(dtype=prompt_embeds.dtype)
        inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)

        # Build generation kwargs
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            temperature=temperature,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            use_cache=True,
        )
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values

        # Constrained generation: force <success>/<failure> as the first
        # token and ban them from the rest of the sequence.
        if self.constrain_cls_generation:
            gen_kwargs["logits_processor"] = [
                _SuccFailConstrainedProcessor(
                    [self.success_token_id, self.failure_token_id]
                )
            ]
        if self._needs_token_type_ids:
            if token_type_ids is None:
                token_type_ids = torch.zeros(
                    prompt_ids.shape, dtype=torch.long, device=inputs_embeds.device
                )
            prefix_token_type_ids = torch.zeros(
                prompt_ids.shape[0], prefix.size(1),
                dtype=token_type_ids.dtype, device=token_type_ids.device,
            )
            gen_kwargs['token_type_ids'] = torch.zeros(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
            )
            gen_kwargs['token_type_ids'] = torch.cat(
                [prefix_token_type_ids, token_type_ids], dim=1
            )

        out_ids = self.base.generate(**gen_kwargs)
        return out_ids

    # ─── Helpers ───────────────────────────────────────────────────
    def _load_projector_state_dict(self, projector: nn.Module, sd: dict):
        """Load projector weights, reshaping FSDP-flattened tensors if needed."""
        target = projector.state_dict()
        fixed = {}
        for k, v in sd.items():
            if k in target:
                if v.numel() == target[k].numel() and v.shape != target[k].shape:
                    fixed[k] = v.view_as(target[k])
                else:
                    fixed[k] = v
            else:
                fixed[k] = v
        projector.load_state_dict(fixed, strict=True)
