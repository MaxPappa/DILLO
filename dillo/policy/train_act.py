#!/usr/bin/env python3
"""
Multi-task action-chunking policy training for LIBERO.

Trains a single policy on *all* tasks in a LIBERO suite simultaneously.
The policy uses action chunking (predicting K future actions at once) with
a GMM head for stochastic behaviour, and temporal ensembling at inference.

Everything is logged to Weights & Biases.

Usage
-----
    python -m dillo.policy.train_act --suite libero_10 --seed 42

    # key hyper-parameters
    python -m dillo.policy.train_act \\
        --suite libero_10 \\
        --chunk_size 20 \\
        --epochs 200 \\
        --batch_size 64 \\
        --lr 1e-4 \\
        --eval_every 10 \\
        --wandb_project libero_act
"""
import warnings
import os

# ---- silence noisy third-party warnings ----
warnings.filterwarnings("ignore", message=".*robosuite.*")            # robosuite macros
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")  # old gym
warnings.filterwarnings("ignore", message=".*distutils Version.*")     # thop / packaging
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")  # torchvision
warnings.filterwarnings("ignore", message=".*weights.*deprecated.*")   # torchvision
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")        # meshgrid indexing
warnings.filterwarnings("ignore", message=".*np.bool.*")               # numpy deprecation
warnings.filterwarnings("ignore", message=".*pin_memory.*deprecated.*")  # torch pin_memory
warnings.filterwarnings("ignore", message=".*is_pinned.*deprecated.*")   # torch is_pinned
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="thop")
os.environ["PYTHONWARNINGS"] = "ignore"  # propagate to subprocesses


import argparse
import gc
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from tqdm import tqdm

from dillo.libero_imports import prepare_libero_imports

prepare_libero_imports()

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

# LIBERO imports
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    get_task_embs,
    NpEncoder,
)

from dillo.policy.act_policy import ActionChunkingPolicy
from dillo.policy.obs import OBS_KEY_MAPPING, OBS_MODALITY
from dillo.suites import SUITE_TO_BENCHMARK

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _dataloader_worker_init(worker_id):
    """Re-initialize robomimic obs utils in each DataLoader worker (required with spawn)."""
    import warnings
    warnings.filterwarnings("ignore")
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": OBS_MODALITY})


# ======================================================================
#  Argument parsing
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train action-chunking policy on LIBERO")
    # benchmark
    p.add_argument("--suite", type=str, default="libero_10",
                   choices=list(SUITE_TO_BENCHMARK.keys()),
                   help="LIBERO task suite")
    p.add_argument("--task_order_index", type=int, default=0,
                   help="Task ordering index (0-20 for 10-task suites)")
    p.add_argument("--datasets_dir", type=str, default=None,
                   help="Override dataset directory")
    # training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200,
                   help="Total training epochs")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=100.0)
    p.add_argument("--lr_min", type=float, default=1e-6,
                   help="Cosine annealing minimum LR")
    p.add_argument("--num_workers", type=int, default=4)
    # policy
    p.add_argument("--chunk_size", type=int, default=20,
                   help="Number of future actions to predict at once")
    p.add_argument("--embed_size", type=int, default=64)
    p.add_argument("--decoder_layers", type=int, default=2,
                   help="Transformer decoder layers in chunk decoder")
    p.add_argument("--decoder_heads", type=int, default=4)
    p.add_argument("--decoder_ff_dim", type=int, default=256)
    p.add_argument("--decoder_dropout", type=float, default=0.1)
    p.add_argument("--gmm_modes", type=int, default=5,
                   help="Number of GMM mixture components")
    p.add_argument("--gmm_hidden", type=int, default=1024)
    p.add_argument("--temporal_decay", type=float, default=0.01,
                   help="Exponential decay for temporal ensembling at inference")
    p.add_argument("--no_augmentation", action="store_true")
    # eval
    p.add_argument("--eval_every", type=int, default=10,
                   help="Evaluate success rate every N epochs")
    p.add_argument("--n_eval", type=int, default=20,
                   help="Number of evaluation episodes per task")
    p.add_argument("--max_steps", type=int, default=600,
                   help="Max steps per evaluation episode")
    p.add_argument("--eval_num_procs", type=int, default=10,
                   help="Number of parallel eval envs")
    # wandb
    p.add_argument("--wandb_project", type=str, default="libero_act")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None,
                   help="Custom wandb run name")
    p.add_argument("--no_wandb", action="store_true")
    # device / output
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--exp_dir", type=str, default="experiments_act",
                   help="Root directory for experiment outputs")
    # language
    p.add_argument("--task_embedding_format", type=str, default="bert",
                   choices=["bert", "clip", "gpt2", "roberta"])
    return p.parse_args()


def raw_obs_to_tensor_obs(obs, task_emb, device):
    """Convert raw env observations to policy-compatible tensors."""
    env_num = len(obs)
    data = {"obs": {}, "task_emb": task_emb.repeat(env_num, 1).to(device)}

    all_obs_keys = []
    for modality_list in OBS_MODALITY.values():
        all_obs_keys += modality_list

    for obs_name in all_obs_keys:
        data["obs"][obs_name] = []

    for k in range(env_num):
        for obs_name in all_obs_keys:
            env_key = OBS_KEY_MAPPING[obs_name]
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][env_key]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key]).to(device)

    return data


# ======================================================================
#  Evaluation
# ======================================================================
def evaluate_one_task(policy, benchmark, task_id, task_emb, args):
    """
    Rollout *n_eval* episodes for a single task, return success rate.
    """
    policy.eval()
    task = benchmark.get_task(task_id)

    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")

    env_args = {
        "bddl_file_name": os.path.join(
            bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": 128,
        "camera_widths": 128,
    }

    env_num = min(args.eval_num_procs, args.n_eval)
    eval_loops = (args.n_eval + env_num - 1) // env_num

    # try to create env with retries
    env = None
    for attempt in range(5):
        try:
            if env_num == 1:
                env = DummyVectorEnv(
                    [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                )
            else:
                env = SubprocVectorEnv(
                    [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                )
            break
        except Exception:
            time.sleep(3)
    if env is None:
        print(f"[warn] failed to create env for task {task_id}, returning 0.0")
        return 0.0

    init_states_path = os.path.join(
        init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)

    num_success = 0
    for loop_i in range(eval_loops):
        env.reset()
        indices = np.arange(loop_i * env_num, (loop_i + 1) * env_num) % init_states.shape[0]
        obs = env.set_init_state(init_states[indices])

        # warm-up physics
        dummy = np.zeros((env_num, 7))
        for _ in range(40):
            obs, _, _, _ = env.step(dummy)

        dones = [False] * env_num
        policy.reset()
        steps = 0

        while steps < args.max_steps:
            steps += 1
            data = raw_obs_to_tensor_obs(obs, task_emb, args.device)
            actions = policy.get_action(data)       # numpy (B, ac_dim)
            obs, reward, done, info = env.step(actions)
            for k in range(env_num):
                dones[k] = dones[k] or done[k]
            if all(dones):
                break

        for k in range(env_num):
            if loop_i * env_num + k < args.n_eval:
                num_success += int(dones[k])

    env.close()
    gc.collect()
    return num_success / args.n_eval


def evaluate_all_tasks(policy, benchmark, task_embs, args):
    """Evaluate across all tasks, return per-task success rates."""
    results = []
    for i in tqdm(range(benchmark.n_tasks), desc="Evaluating tasks", unit="task", leave=True):
        task_emb = task_embs[i]
        sr = evaluate_one_task(policy, benchmark, i, task_emb, args)
        results.append(sr)
        tqdm.write(f"  task {i:2d}: {sr:.2f}")
    return np.array(results)


# ======================================================================
#  Experiment directory
# ======================================================================
def create_exp_dir(args):
    base = os.path.join(
        args.exp_dir,
        SUITE_TO_BENCHMARK[args.suite],
        f"ACT_chunk{args.chunk_size}_seed{args.seed}",
    )
    os.makedirs(base, exist_ok=True)
    # find next run id
    run_id = 0
    for p in Path(base).glob("run_*"):
        if p.is_dir():
            try:
                rid = int(str(p).split("run_")[-1])
                run_id = max(run_id, rid)
            except ValueError:
                pass
    run_id += 1
    exp_dir = os.path.join(base, f"run_{run_id:03d}")
    os.makedirs(exp_dir)
    return exp_dir


# ======================================================================
#  Main
# ======================================================================
def main():
    args = parse_args()
    control_seed(args.seed)

    # ---- experiment directory ----
    exp_dir = create_exp_dir(args)
    print(f"[info] experiment dir: {exp_dir}")

    # ---- wandb ----
    if not args.no_wandb:
        import wandb
        run_name = args.wandb_name or f"{args.suite}_chunk{args.chunk_size}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )

    # ---- benchmark ----
    benchmark_name = SUITE_TO_BENCHMARK[args.suite]
    benchmark = get_benchmark(benchmark_name)(args.task_order_index)
    n_tasks = benchmark.n_tasks
    print(f"[info] benchmark: {benchmark_name}  ({n_tasks} tasks)")

    # ---- datasets ----
    datasets_dir = args.datasets_dir or get_libero_path("datasets")
    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")

    datasets = []
    descriptions = []
    shape_meta = None

    for i in tqdm(range(n_tasks), desc="Loading tasks", unit="task"):
        demo_path = os.path.join(datasets_dir, benchmark.get_task_demonstration(i))
        ds, shape_meta = get_dataset(
            dataset_path=demo_path,
            obs_modality=OBS_MODALITY,
            initialize_obs_utils=(i == 0),
            seq_len=args.chunk_size,     # <-- action chunk size
            frame_stack=1
        )
        task_desc = benchmark.get_task(i).language
        descriptions.append(task_desc)
        datasets.append(ds)
        tqdm.write(f"  task {i}: {task_desc}  ({len(ds)} sequences)")

    # ---- task embeddings ----
    # build a minimal cfg-like object for get_task_embs
    from easydict import EasyDict
    _cfg = EasyDict(
        task_embedding_format=args.task_embedding_format,
        task_embedding_one_hot_offset=1,
        data=EasyDict(max_word_len=25),
        policy=EasyDict(language_encoder=EasyDict(network_kwargs=EasyDict(input_size=768))),
    )
    task_embs = get_task_embs(_cfg, descriptions)
    language_input_size = task_embs.shape[-1]   # usually 768
    benchmark.set_task_embs(task_embs)

    # ---- wrap datasets with task embeddings & concat ----
    vl_datasets = [
        SequenceVLDataset(ds, emb) for ds, emb in zip(datasets, task_embs)
    ]
    concat_dataset = ConcatDataset(vl_datasets)

    n_demos = sum(ds.n_demos for ds in datasets)
    n_seqs = sum(ds.total_num_sequences for ds in datasets)
    print(f"[info] total demos: {n_demos}   total sequences: {n_seqs}")

    train_loader = DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=RandomSampler(concat_dataset),
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
        worker_init_fn=_dataloader_worker_init if args.num_workers > 0 else None,
    )

    # ---- build policy ----
    img_shape = shape_meta["all_shapes"].get(
        "agentview_rgb", (3, 128, 128)
    )
    policy = ActionChunkingPolicy(
        shape_meta=shape_meta,
        embed_size=args.embed_size,
        language_input_size=language_input_size,
        language_hidden_size=128,
        chunk_size=args.chunk_size,
        decoder_num_layers=args.decoder_layers,
        decoder_num_heads=args.decoder_heads,
        decoder_ff_dim=args.decoder_ff_dim,
        decoder_dropout=args.decoder_dropout,
        gmm_hidden_size=args.gmm_hidden,
        gmm_num_layers=2,
        gmm_num_modes=args.gmm_modes,
        gmm_min_std=1e-4,
        use_joint=True,
        use_gripper=True,
        use_ee=False,
        use_augmentation=not args.no_augmentation,
        img_input_shape=img_shape,
        translation=8,
        temporal_decay=args.temporal_decay,
    )
    policy = safe_device(policy, args.device)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[info] policy parameters: {n_params / 1e6:.2f}M")

    # ---- optimizer & scheduler ----
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min,
    )

    # ---- save config ----
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, cls=NpEncoder, indent=4)

    # ---- training loop ----
    
    best_success_rate = -1.0
    best_epoch = 0
    best_model_path = os.path.join(exp_dir, "best_model.pth")
    log_history = {"epoch": [], "train_loss": [], "mean_success": []}

    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs")
    print(f"  Chunk size: {args.chunk_size}  |  GMM modes: {args.gmm_modes}")
    print(f"  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="ep", position=0)
    for epoch in epoch_pbar:
        t0 = time.time()

        # ---------- train ----------
        policy.train()
        epoch_loss = 0.0
        n_batches = 0
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
            leave=True,
            position=1,
        )
        for batch in batch_pbar:
            batch = TensorUtils.map_tensor(
                batch, lambda x: safe_device(x, device=args.device)
            )
            optimizer.zero_grad()
            loss = policy.compute_loss(batch)
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            batch_pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        epoch_loss /= max(n_batches, 1)
        scheduler.step()

        t1 = time.time()
        current_lr = scheduler.get_last_lr()[0]

        epoch_pbar.set_postfix(
            loss=f"{epoch_loss:.4f}",
            lr=f"{current_lr:.2e}",
            time=f"{(t1-t0)/60:.1f}m",
        )

        log_dict = {
            "epoch": epoch,
            "train/loss": epoch_loss,
            "train/lr": current_lr,
            "train/epoch_time_min": (t1 - t0) / 60.0,
        }

        tqdm.write(
            f"Epoch {epoch:4d}/{args.epochs} | loss: {epoch_loss:7.4f} | "
            f"lr: {current_lr:.2e} | time: {(t1-t0)/60:.1f}m"
        )

        # ---------- save periodic checkpoint ----------
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(exp_dir, f"model_ep{epoch}.pth")
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "shape_meta": shape_meta,
                },
                ckpt_path,
            )

        # ---------- evaluate ----------
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            t2 = time.time()
            print(f"[eval] evaluating at epoch {epoch} ...")
            with torch.no_grad():
                success_rates = evaluate_all_tasks(
                    policy, benchmark, task_embs, args
                )
            mean_sr = success_rates.mean()
            t3 = time.time()

            # per-task logging
            # for i, sr in enumerate(success_rates):
            #     log_dict[f"eval/task_{i}_success"] = sr
            log_dict["eval/mean_success"] = mean_sr
            log_dict["eval/eval_time_min"] = (t3 - t2) / 60.0

            ci = 1.96 * np.sqrt(mean_sr * (1 - mean_sr) / (args.n_eval * n_tasks))
            print(
                f"  mean success: {mean_sr:.3f} ± {ci:.3f} | "
                f"eval time: {(t3-t2)/60:.1f}m"
            )

            if mean_sr > best_success_rate:
                best_success_rate = mean_sr
                best_epoch = epoch
                torch.save(
                    {
                        "state_dict": policy.state_dict(),
                        "epoch": epoch,
                        "args": vars(args),
                        "shape_meta": shape_meta,
                        "success_rates": success_rates,
                        "mean_success": mean_sr,
                    },
                    best_model_path,
                )
                print(f"  ** new best model saved (mean_sr={mean_sr:.3f}) **")

            log_dict["eval/best_mean_success"] = best_success_rate
            log_dict["eval/best_epoch"] = best_epoch

            log_history["epoch"].append(epoch)
            log_history["train_loss"].append(epoch_loss)
            log_history["mean_success"].append(mean_sr)

        # ---------- wandb ----------
        if not args.no_wandb:
            wandb.log(log_dict, step=epoch)

    # ---- final summary ----
    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best mean success rate: {best_success_rate:.3f} (epoch {best_epoch})")
    print(f"  Experiment dir: {exp_dir}")
    print(f"{'='*60}")

    # save log history
    torch.save(log_history, os.path.join(exp_dir, "log_history.pt"))

    if not args.no_wandb:
        wandb.run.summary["best_mean_success"] = best_success_rate
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
