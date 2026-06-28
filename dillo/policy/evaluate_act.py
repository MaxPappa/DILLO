#!/usr/bin/env python3
"""
Standalone evaluation script for a trained action-chunking policy on LIBERO.

Loads a saved checkpoint and evaluates success rates across all tasks in the
suite, optionally recording rollout videos.

Usage
-----
    python -m dillo.policy.evaluate_act \\
        --checkpoint experiments_act/LIBERO_10/ACT_chunk20_seed42/run_001/best_model.pth \\
        --suite libero_10 \\
        --n_eval 20
"""

import argparse
import gc
import multiprocessing
import os
import time
import warnings

# ---- silence noisy third-party warnings ----
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*robosuite.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*distutils Version.*")
warnings.filterwarnings("ignore", message=".*np.bool.*")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm

from dillo.libero_imports import prepare_libero_imports

prepare_libero_imports()

import robomimic.utils.obs_utils as ObsUtils

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.utils import control_seed, safe_device, get_task_embs

from dillo.policy.act_policy import ActionChunkingPolicy
from dillo.policy.train_act import raw_obs_to_tensor_obs
from dillo.suites import SUITE_TO_BENCHMARK
from dillo.policy.obs import OBS_MODALITY

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained action-chunking policy")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--suite", type=str, default=None,
                   help="Task suite (inferred from checkpoint if omitted)")
    p.add_argument("--task_ids", type=int, nargs="+", default=None,
                   help="Specific task IDs to evaluate (default: all)")
    p.add_argument("--n_eval", type=int, default=20,
                   help="Episodes per task")
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--eval_num_procs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--task_order_index", type=int, default=0)
    p.add_argument("--record_video", action="store_true",
                   help="Record video of evaluation episodes")
    p.add_argument("--video_dir", type=str, default=None,
                   help="Directory to save videos (default: <checkpoint_dir>/videos)")
    p.add_argument("--video_episodes", type=int, default=3,
                   help="Number of episodes to record per task (default: 3)")
    return p.parse_args()


def evaluate_one_task(policy, benchmark, task_id, task_emb, args):
    """Rollout n_eval episodes for one task, return success rate."""
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
        print(f"[warn] failed to create env for task {task_id}")
        return 0.0

    init_states_path = os.path.join(
        init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)

    num_success = 0
    ep_pbar = tqdm(
        range(eval_loops),
        desc=f"Task {task_id:2d} episodes",
        unit="batch",
        leave=False,
    )
    for loop_i in ep_pbar:
        env.reset()
        indices = np.arange(loop_i * env_num, (loop_i + 1) * env_num) % init_states.shape[0]
        obs = env.set_init_state(init_states[indices])

        dummy = np.zeros((env_num, 7))
        for _ in range(40):
            obs, _, _, _ = env.step(dummy)

        dones = [False] * env_num
        policy.reset()
        steps = 0

        with torch.no_grad():
            step_pbar = tqdm(
                total=args.max_steps,
                desc="  steps",
                unit="step",
                leave=False,
            )
            while steps < args.max_steps:
                steps += 1
                step_pbar.update(1)
                data = raw_obs_to_tensor_obs(obs, task_emb, args.device)
                actions = policy.get_action(data)
                obs, reward, done, info = env.step(actions)
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break
            step_pbar.close()

        for k in range(env_num):
            if loop_i * env_num + k < args.n_eval:
                num_success += int(dones[k])
        ep_pbar.set_postfix(success=f"{num_success}/{min((loop_i+1)*env_num, args.n_eval)}")

    env.close()
    gc.collect()
    success_rate = num_success / args.n_eval
    return success_rate


def _upscale(frame, size=256):
    """Upscale a frame to (size, size) for visual clarity."""
    import cv2
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LANCZOS4)


def record_task_videos(policy, benchmark, task_id, task_emb, args, video_dir):
    """Record video rollouts for a single task using a single env."""
    import imageio

    policy.eval()
    task = benchmark.get_task(task_id)
    task_desc = task.language.replace(" ", "_")[:60]

    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")

    env_args = {
        "bddl_file_name": os.path.join(
            bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": 128,
        "camera_widths": 128,
    }

    env = DummyVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args)]
    )

    init_states_path = os.path.join(
        init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)

    os.makedirs(video_dir, exist_ok=True)
    num_success = 0

    for ep in tqdm(range(args.video_episodes), desc=f"  Recording task {task_id}",
                   unit="ep", leave=False):
        env.reset()
        obs = env.set_init_state(init_states[[ep % init_states.shape[0]]])

        dummy = np.zeros((1, 7))
        for _ in range(40):
            obs, _, _, _ = env.step(dummy)

        frames = []
        done_flag = False
        policy.reset()

        with torch.no_grad():
            for step in range(args.max_steps):
                # capture frame from agentview (flip vertically — MuJoCo origin is bottom-left)
                frame = obs[0]["agentview_image"]
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                frames.append(_upscale(np.flipud(frame)))

                data = raw_obs_to_tensor_obs(obs, task_emb, args.device)
                actions = policy.get_action(data)
                obs, reward, done, info = env.step(actions)
                if done[0]:
                    done_flag = True
                    # capture final frame
                    frame = obs[0]["agentview_image"]
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    frames.append(_upscale(np.flipud(frame)))
                    break

        result = "success" if done_flag else "fail"
        num_success += int(done_flag)
        video_path = os.path.join(
            video_dir, f"task{task_id:02d}_ep{ep:02d}_{result}.mp4"
        )
        imageio.mimwrite(video_path, frames, fps=20, quality=8)
        tqdm.write(f"    saved: {video_path} ({len(frames)} frames, {result})")

    env.close()
    gc.collect()
    return num_success / max(args.video_episodes, 1)


def main():
    args = parse_args()
    control_seed(args.seed)

    # ---- load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    saved_args = ckpt.get("args", {})
    shape_meta = ckpt["shape_meta"]

    # infer suite from checkpoint if not provided
    suite = args.suite or saved_args.get("suite", "libero_10")
    benchmark_name = SUITE_TO_BENCHMARK[suite]
    task_order_index = args.task_order_index

    print(f"[info] loading checkpoint: {args.checkpoint}")
    print(f"[info] suite: {suite}  benchmark: {benchmark_name}")

    # ---- initialize obs utils ----
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": OBS_MODALITY})

    # ---- benchmark & task embeddings ----
    benchmark = get_benchmark(benchmark_name)(task_order_index)
    n_tasks = benchmark.n_tasks

    descriptions = [benchmark.get_task(i).language for i in range(n_tasks)]
    _cfg = EasyDict(
        task_embedding_format=saved_args.get("task_embedding_format", "bert"),
        task_embedding_one_hot_offset=1,
        data=EasyDict(max_word_len=25),
        policy=EasyDict(language_encoder=EasyDict(network_kwargs=EasyDict(input_size=768))),
    )
    task_embs = get_task_embs(_cfg, descriptions)
    language_input_size = task_embs.shape[-1]
    benchmark.set_task_embs(task_embs)

    # ---- build policy & load weights ----
    policy = ActionChunkingPolicy(
        shape_meta=shape_meta,
        embed_size=saved_args.get("embed_size", 64),
        language_input_size=language_input_size,
        language_hidden_size=128,
        chunk_size=saved_args.get("chunk_size", 20),
        decoder_num_layers=saved_args.get("decoder_layers", 2),
        decoder_num_heads=saved_args.get("decoder_heads", 4),
        decoder_ff_dim=saved_args.get("decoder_ff_dim", 256),
        decoder_dropout=saved_args.get("decoder_dropout", 0.1),
        gmm_hidden_size=saved_args.get("gmm_hidden", 1024),
        gmm_num_layers=2,
        gmm_num_modes=saved_args.get("gmm_modes", 5),
        gmm_min_std=1e-4,
        use_joint=True,
        use_gripper=True,
        use_ee=False,
        use_augmentation=False,     # no augmentation at eval
        temporal_decay=saved_args.get("temporal_decay", 0.01),
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy = safe_device(policy, args.device)
    policy.eval()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[info] policy parameters: {n_params / 1e6:.2f}M")

    # ---- evaluate ----
    task_ids = args.task_ids if args.task_ids is not None else list(range(n_tasks))

    if args.record_video:
        # -- video mode: evaluate + record only video_episodes per task --
        video_dir = args.video_dir or os.path.join(
            os.path.dirname(args.checkpoint), "videos"
        )
        n_ep = args.video_episodes
        print(f"[info] evaluating & recording {n_ep} episodes per task to {video_dir}")
        print(f"{'='*50}")

        all_success = []
        task_pbar = tqdm(task_ids, desc="Eval + record", unit="task")
        for tid in task_pbar:
            t0 = time.time()
            task_emb = task_embs[tid]
            sr = record_task_videos(policy, benchmark, tid, task_emb, args, video_dir)
            t1 = time.time()

            task_desc = benchmark.get_task(tid).language
            ci = 1.96 * np.sqrt(sr * (1 - sr) / n_ep)
            tqdm.write(f"  Task {tid:2d} | success: {sr:.2f} ± {ci:.2f} | "
                       f"time: {t1-t0:.1f}s | {task_desc}")
            all_success.append(sr)
            task_pbar.set_postfix(mean_sr=f"{np.mean(all_success):.2f}")

        all_success = np.array(all_success)
        mean_sr = all_success.mean()
        ci = 1.96 * np.sqrt(mean_sr * (1 - mean_sr) / (n_ep * len(task_ids)))
        print(f"{'='*50}")
        print(f"  Mean success rate: {mean_sr:.3f} ± {ci:.3f}  ({n_ep} eps/task)")
        print(f"  Per-task: {np.array2string(all_success, precision=2)}")
        print(f"[info] videos saved to {video_dir}")

    else:
        # -- standard evaluation: n_eval episodes, no video --
        print(f"[info] evaluating {len(task_ids)} tasks, {args.n_eval} episodes each")
        print(f"{'='*50}")

        all_success = []
        task_pbar = tqdm(task_ids, desc="Evaluating tasks", unit="task")
        for tid in task_pbar:
            t0 = time.time()
            task_emb = task_embs[tid]
            sr = evaluate_one_task(policy, benchmark, tid, task_emb, args)
            t1 = time.time()

            task_desc = benchmark.get_task(tid).language
            ci = 1.96 * np.sqrt(sr * (1 - sr) / args.n_eval)
            tqdm.write(f"  Task {tid:2d} | success: {sr:.2f} ± {ci:.2f} | "
                       f"time: {t1-t0:.1f}s | {task_desc}")
            all_success.append(sr)
            task_pbar.set_postfix(mean_sr=f"{np.mean(all_success):.2f}")

        all_success = np.array(all_success)
        mean_sr = all_success.mean()
        ci = 1.96 * np.sqrt(mean_sr * (1 - mean_sr) / (args.n_eval * len(task_ids)))
        print(f"{'='*50}")
        print(f"  Mean success rate: {mean_sr:.3f} ± {ci:.3f}")
        print(f"  Per-task: {np.array2string(all_success, precision=2)}")

    # save results
    n_ep_total = args.video_episodes if args.record_video else args.n_eval
    results_path = args.checkpoint.replace(".pth", "_eval_results.pt")
    torch.save({
        "task_ids": task_ids,
        "success_rates": all_success,
        "mean_success": mean_sr,
        "n_eval": n_ep_total,
    }, results_path)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
