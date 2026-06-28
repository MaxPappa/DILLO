#!/usr/bin/env python3
"""
Chunk-based data collection pipeline for LIBERO environments.
Uses the ActionChunkingPolicy (GMM head, ResNet+FiLM) from this package.

This is the canonical and most up-to-date collector in this folder.

Key properties:
  - Uses ActionChunkingPolicy checkpoints from dillo.policy.train_act
  - Checkpoint format: keys 'state_dict', 'args', 'shape_meta'
  - No action normalisation/denormalisation (GMM head with tanh)
  - Task embeddings computed via BERT (cached per instruction)
  - Default chunk size is 20
  - Default max_steps is 600 (matching training config)

Usage:
    python -m dillo.data_generation.collect_dataset \
                --checkpoint experiments_act/LIBERO_90/ACT_chunk20_seed42/run_001/best_model.pth \
                --suite libero_90 \
                --send_mode timestamped_frames \
                --vlm_max_frames 4 \
                --vlm_model Qwen3-VL-30B-A3B-Thinking \
                --vlm_host 127.0.0.1 \
                --vlm_port 8000 \
                --camera_size 512 \
                --resume
"""
import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
import re
from PIL import Image

from dillo.libero_imports import prepare_libero_imports

prepare_libero_imports()

from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils
from robosuite import load_controller_config

from dillo.data_generation.vllm_client import VLLMClient
from dillo.data_generation.prompts import CHUNK_PROMPT_VIDEO_AND_OBS
from dillo.policy.act_agent import ACTAgent


# ACTAgent is imported from dillo.policy.act_agent so rollout preprocessing,
# task embeddings, and checkpoint loading have one implementation.


# =============================================================================
# Environment helpers
# =============================================================================

def make_libero_env(task_name, bddl_file, camera_heights=128, camera_widths=128):
    """
    Create a LIBERO environment for a given task.

    Args:
        task_name: The BDDL problem name
        bddl_file: Path to the .bddl file
        camera_heights: Image height
        camera_widths: Image width

    Returns:
        env: LIBERO environment instance
        language_instruction: str task description
    """
    controller_config = load_controller_config(default_controller="OSC_POSE")

    problem_info = BDDLUtils.get_problem_info(bddl_file)
    language_instruction = problem_info["language_instruction"]

    env_args = {
        "bddl_file_name": bddl_file,
        "robots": ["Panda"],
        "controller_configs": controller_config,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_camera_obs": True,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_heights": camera_heights,
        "camera_widths": camera_widths,
        "control_freq": 20,
    }
    env = TASK_MAPPING[task_name](**env_args)

    return env, language_instruction


def get_agentview_image(obs):
    """Extract the agentview RGB image from a LIBERO observation dict, rotated 180°."""
    return np.rot90(obs['agentview_image'], 2)


# =============================================================================
# Observation formatting for 2-frame pairs
# =============================================================================

def format_pair_observations(obs_before, obs_after, threshold_eef=0.002, threshold_grip=0.005):
    """
    Build observation text for a pair of frames (before/after an action chunk).

    Args:
        obs_before: obs dict at the start of the chunk
        obs_after: obs dict at the end of the chunk
        threshold_eef: minimum EEF displacement to register as movement
        threshold_grip: minimum gripper change to register

    Returns:
        text: formatted observation string for the VLM prompt
    """
    eef_before = np.around(obs_before['robot0_eef_pos'], decimals=4).tolist()
    eef_after = np.around(obs_after['robot0_eef_pos'], decimals=4).tolist()
    grip_before = float(np.around(obs_before['robot0_gripper_qpos'].mean(), decimals=4))
    grip_after = float(np.around(obs_after['robot0_gripper_qpos'].mean(), decimals=4))

    text = f"Before:\n"
    text += f"- End-Effector Position: {eef_before}\n"
    text += f"- Gripper Openness: {grip_before:.4f}\n\n"

    text += f"After:\n"
    text += f"- End-Effector Position: {eef_after}\n"
    text += f"- Gripper Openness: {grip_after:.4f}\n"

    # Compute overall EEF movement direction
    diff = np.array(obs_after['robot0_eef_pos']) - np.array(obs_before['robot0_eef_pos'])
    dirs = []
    if diff[0] > threshold_eef:
        dirs.append('right')
    elif diff[0] < -threshold_eef:
        dirs.append('left')
    if diff[1] > threshold_eef:
        dirs.append('forward')
    elif diff[1] < -threshold_eef:
        dirs.append('backward')
    if diff[2] > threshold_eef:
        dirs.append('up')
    elif diff[2] < -threshold_eef:
        dirs.append('down')

    eef_movement = ', '.join(dirs) if dirs else 'no significant movement'
    text += f"- EEF Movement Direction: {eef_movement}\n"

    # Gripper state change
    grip_diff = grip_after - grip_before
    if grip_diff > threshold_grip:
        grip_change = 'opening'
    elif grip_diff < -threshold_grip:
        grip_change = 'closing'
    else:
        grip_change = 'no change'
    text += f"- Gripper State Change: {grip_change}\n"

    return text


def subsample_frames(frames, max_frames=8, always_include_endpoints=True):
    """
    Subsample a list of frames to at most max_frames, keeping temporal
    spread and always including the first and last frame.

    This follows the Qwen3-VL best practice of sending fewer, more
    distinct frames rather than many near-identical ones.

    Args:
        frames: list of (H, W, 3) numpy arrays
        max_frames: maximum number of frames to return
        always_include_endpoints: if True, always include first and last frame

    Returns:
        indices: list of selected frame indices
        selected: list of selected frames
    """
    n = len(frames)
    if n <= max_frames:
        return list(range(n)), frames

    if always_include_endpoints:
        # Pick max_frames indices spread evenly, always including 0 and n-1
        indices = np.linspace(0, n - 1, num=max_frames, dtype=int).tolist()
    else:
        indices = np.linspace(0, n - 1, num=max_frames, dtype=int).tolist()

    # Deduplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    indices = unique_indices

    selected = [frames[i] for i in indices]
    return indices, selected

# =============================================================================
# Save utilities
# =============================================================================

def save_images(directory, images):
    """Save a sequence of images as JPEG files."""
    os.makedirs(f"{directory}/images", exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(img).save(f"{directory}/images/{i}.jpeg")


def save_example(base_dir, example_idx, response_txt, reasoning_txt,
                 language_instruction, obs_list, chunk_actions, video_arr,
                 success_mask=None):
    """
    Save a single collected example.

    Args:
        base_dir: Root save directory for this task
        example_idx: Index of this example
        response_txt: Combined VLM clean responses (all transitions)
        reasoning_txt: Combined VLM reasoning traces (all transitions)
        language_instruction: The LIBERO task description
        obs_list: List of 11 observation dicts
        chunk_actions: List of 10 arrays, each (chunk_size, 7)
        video_arr: (11, H, W, 3) image array
        success_mask: Optional (11,) binary mask
    """
    folder = Path(base_dir) / f"{example_idx:03d}"
    folder.mkdir(parents=True, exist_ok=True)

    # Save VLM final answer
    with open(folder / "response.txt", 'w', encoding='utf-8') as f:
        f.write(response_txt)

    # Save VLM reasoning trace
    with open(folder / "reasoning.txt", 'w', encoding='utf-8') as f:
        f.write(reasoning_txt)

    # Save task instruction
    with open(folder / "task_instruction.txt", 'w', encoding='utf-8') as f:
        f.write(language_instruction)

    # Save numeric data
    eef_positions = np.array([obs['robot0_eef_pos'] for obs in obs_list])
    joint_positions = np.array([obs['robot0_joint_pos'] for obs in obs_list])
    gripper_states = np.array([obs['robot0_gripper_qpos'] for obs in obs_list])

    np.save(folder / "eef_pos.npy", eef_positions)
    np.save(folder / "joint_pos.npy", joint_positions)
    np.save(folder / "gripper_states.npy", gripper_states)

    # Save all chunk actions concatenated: (num_chunks * chunk_size, 7)
    all_actions = np.concatenate(chunk_actions, axis=0)
    np.save(folder / "actions.npy", all_actions)

    if success_mask is not None:
        np.save(folder / "success_mask.npy", success_mask)

    # Save images
    save_images(folder, video_arr)


# =============================================================================
# Main Pipeline
# =============================================================================

class DataCollectionPipeline:
    """
    Collects trajectory data from LIBERO environments using the fully-trained
    ActionChunkingPolicy (GMM head), recording frames at action chunk
    boundaries. Queries a VLM with exactly 2 frames per transition and saves
    structured examples.
    """

    COLLECT_CHUNKS = 10  # number of chunks to keep for VLM queries / saving
    POST_SUCCESS_CHUNKS = 2  # extra chunks to run after success for settling

    def __init__(self, args):
        self.args = args

        # VLM client
        self.sys_prompt = (
            "You are an expert at analyzing robotic manipulation behavior. "
            "Given images and observations from a robotic arm, you describe "
            "what the robot is doing and whether it is making progress on its task."
        )
        self.chat = VLLMClient(
            model=args.vlm_model,
            system_prompt=self.sys_prompt,
            host=args.vlm_host,
            port=args.vlm_port,
        )

        # Load the specified checkpoint
        self.checkpoint_path = args.checkpoint
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        print(f"[Pipeline] Using checkpoint: {self.checkpoint_path}")

        self.agent = ACTAgent(self.checkpoint_path, device=args.device)
        self.chunk_size = self.agent.chunk_size

        # Compute number of chunks from max_steps
        self.max_steps = args.max_steps
        self.num_chunks = self.max_steps // self.chunk_size
        print(f"[Pipeline] Action chunk size: {self.chunk_size}")
        print(f"[Pipeline] Max steps: {self.max_steps} → {self.num_chunks} chunks → {self.num_chunks + 1} frames")

        # Discover LIBERO tasks
        self.suite_name = args.suite
        self._discover_tasks()

        # Collection params
        self.episodes_per_task = args.episodes_per_task
        self.save_dir = args.save_dir
        self.max_concurrent_vlm = args.max_concurrent_vlm
        self.send_mode = getattr(args, 'send_mode', 'video')
        self.vlm_max_frames = getattr(args, 'vlm_max_frames', 8)
        self.resume = getattr(args, 'resume', False)

    def _discover_tasks(self):
        """Find all tasks in the specified LIBERO suite."""
        bddl_root = self._get_bddl_root()
        task_suite_folder = Path(bddl_root) / self.suite_name
        bddl_files = sorted(task_suite_folder.glob("*.bddl"))

        if not bddl_files:
            raise FileNotFoundError(f"No .bddl files found in {task_suite_folder}")

        self.tasks = []
        for bddl_file in bddl_files:
            problem_info = BDDLUtils.get_problem_info(str(bddl_file))
            self.tasks.append({
                'task_name': problem_info['problem_name'],
                'task_id': bddl_file.stem,  # unique per task (filename without .bddl)
                'bddl_file': str(bddl_file),
                'language_instruction': problem_info['language_instruction'],
            })

        print(f"[Pipeline] Found {len(self.tasks)} tasks in suite '{self.suite_name}'")
        for t in self.tasks:
            print(f"  - {t['language_instruction']}")

    def _count_existing_episodes(self, task_save_dir):
        """
        Count how many complete episodes already exist in the task directory.

        An episode folder (e.g. '000/', '001/') is considered complete if it
        contains a 'response.txt' file (the last file written by save_example).

        Returns the number of complete episodes (= the next episode index to collect).
        """
        if not os.path.isdir(task_save_dir):
            return 0

        # Find all numeric episode folders (e.g. '000', '001') that contain
        # a completed 'response.txt'. Return the next free index after the
        # highest-numbered completed episode. This is robust to missing
        # intermediate indices (gaps) and avoids restarting from an earlier
        # index when folder numbering isn't contiguous.
        indices = []
        for entry in os.listdir(task_save_dir):
            if re.match(r"^\d{3}$", entry):
                ep_folder = Path(task_save_dir) / entry
                if ep_folder.is_dir() and (ep_folder / "response.txt").exists():
                    try:
                        indices.append(int(entry))
                    except ValueError:
                        continue

        if not indices:
            return 0

        return max(indices) + 1

    def _locate_task_save_dir(self, task_id: str, language_instruction: str) -> str:
        """
        Locate the directory under self.save_dir corresponding to this task.

        Strategy:
          1. If a directory named exactly `task_id` exists under save_dir, use it.
          2. Otherwise, scan all subdirectories of save_dir and look for a
             `task_instruction.txt` whose contents match `language_instruction`.
             If found, return that directory.
          3. If still not found, return the default path join(save_dir, task_id)
             (the caller will create it when saving).
        """
        # 1) Exact match
        candidate = os.path.join(self.save_dir, task_id)
        if os.path.isdir(candidate):
            return candidate

        # 2) Search by task_instruction file content (normalized)
        def _norm(s: str) -> str:
            return " ".join(s.split()).strip().lower()

        if os.path.isdir(self.save_dir):
            for sub in os.listdir(self.save_dir):
                subp = os.path.join(self.save_dir, sub)
                if not os.path.isdir(subp):
                    continue
                instr_file = os.path.join(subp, "task_instruction.txt")
                if os.path.exists(instr_file):
                    try:
                        with open(instr_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        # Exact normalized match
                        if _norm(content) == _norm(language_instruction):
                            return subp
                        # Substring fallback (in case of extra prefixes/suffixes)
                        if _norm(content).find(_norm(language_instruction)) != -1 or _norm(language_instruction).find(_norm(content)) != -1:
                            return subp
                    except Exception:
                        continue

        # 3) Fallback
        return candidate

    def _get_bddl_root(self):
        """Get BDDL files root directory."""
        config_path = os.path.expanduser("~/.libero/config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            benchmark_root = cfg["benchmark_root"]
        else:
            from libero.libero import get_default_path_dict
            benchmark_root = get_default_path_dict()["benchmark_root"]
        return os.path.join(benchmark_root, "bddl_files")

    def get_prompt_template(self):
        """Return the only prompt used for DILLO dataset generation."""
        return CHUNK_PROMPT_VIDEO_AND_OBS

    def collect_episode(self, env, language_instruction):
        """
        Run the episode, recording frames at every chunk boundary.
        Stops early once the task is completed (after finishing the
        successful chunk).

        Returns:
            frames: list of (num_executed+1) images (H, W, 3) — boundary frames
            observations: list of (num_executed+1) obs dicts
            chunk_actions: list of num_executed arrays, each (chunk_size, 7)
            chunk_all_frames: list of num_executed lists, each containing
                              (chunk_size+1) RGB frames (first frame of chunk
                              through last frame) for video encoding
            success: bool
            success_chunk: int or None — index of the chunk during which
                           success was first detected (0-based)
        """
        self.agent.reset()
        obs = env.reset()

        # Warm-up: 40 dummy zero-action steps so physics settle.
        # Matches the warm-up used in evaluate.py (OffScreenRenderEnv + set_init_state).
        dummy = np.zeros(7)
        for _ in range(40):
            obs, _, _, _ = env.step(dummy)

        frames = []
        observations = []
        chunk_actions = []
        chunk_all_frames = []
        success = False
        success_chunk = None

        # Record initial frame (before first chunk)
        frames.append(get_agentview_image(obs).copy())
        observations.append({
            'robot0_eef_pos': obs['robot0_eef_pos'].copy(),
            'robot0_eef_quat': obs['robot0_eef_quat'].copy(),
            'robot0_joint_pos': obs['robot0_joint_pos'].copy(),
            'robot0_gripper_qpos': obs['robot0_gripper_qpos'].copy(),
        })

        for chunk_idx in range(self.num_chunks):
            actions_this_chunk = []
            # Collect all frames within this chunk (starting with the
            # frame before the first step)
            this_chunk_frames = [get_agentview_image(obs).copy()]

            for step_in_chunk in range(self.chunk_size):
                action = self.agent.act(obs, language_instruction=language_instruction)
                obs, reward, done, info = env.step(action)
                actions_this_chunk.append(action.copy())
                this_chunk_frames.append(get_agentview_image(obs).copy())

                if not success and env._check_success():
                    success = True
                    success_chunk = chunk_idx

            # Record boundary frame after this chunk
            frames.append(get_agentview_image(obs).copy())
            observations.append({
                'robot0_eef_pos': obs['robot0_eef_pos'].copy(),
                'robot0_eef_quat': obs['robot0_eef_quat'].copy(),
                'robot0_joint_pos': obs['robot0_joint_pos'].copy(),
                'robot0_gripper_qpos': obs['robot0_gripper_qpos'].copy(),
            })
            chunk_actions.append(np.array(actions_this_chunk))
            chunk_all_frames.append(this_chunk_frames)

            # After success, run a few more chunks so the VLM sees the
            # settled "task completed" state, not just the moment of success.
            if success and chunk_idx >= success_chunk + self.POST_SUCCESS_CHUNKS:
                break

        return frames, observations, chunk_actions, chunk_all_frames, success, success_chunk

    def select_window(self, total_chunks, success, success_chunk):
        """
        Choose the start index of the COLLECT_CHUNKS-sized window.

        Args:
            total_chunks: total number of chunks in the full episode
            success: whether the episode succeeded
            success_chunk: chunk index where success was first detected

        Returns:
            start: int, the first chunk index of the window
        """
        n = self.COLLECT_CHUNKS
        max_start = total_chunks - n  # e.g. 30 - 10 = 20

        if success and success_chunk is not None:
            # Window must contain success_chunk:
            #   start <= success_chunk  AND  start + n - 1 >= success_chunk
            lo = max(0, success_chunk - n + 1)
            hi = min(success_chunk, max_start)
            start = random.randint(lo, hi)
        else:
            start = random.randint(0, max_start)

        return start

    def query_vlm_pair(
        self,
        obs_before,
        obs_after,
        language_instruction,
        chunk_idx,
        chunk_video_frames=None,
    ):
        """
        Query the VLM for one action chunk transition.

        The visual input depends on self.send_mode:
          - 'video': encode all chunk frames into an mp4 video (original)
          - 'timestamped_frames': send subsampled frames as interleaved
            timestamp-image pairs (Qwen3-VL best practice for temporal
            grounding — each frame gets "<T seconds>" annotation)
          - 'frame_list': send subsampled frames as individual images
            (no timestamps, but avoids mp4 compression artefacts)

        Args:
            obs_before: observation dict before the chunk
            obs_after: observation dict after the chunk
            language_instruction: task description string
            chunk_idx: index of the chunk (for logging)
            chunk_video_frames: list of (chunk_size+1) RGB frames for this chunk

        Returns:
            (response, thinking): tuple of clean VLM response and reasoning trace
        """
        prompt_template = self.get_prompt_template()

        frame_obs_text = format_pair_observations(obs_before, obs_after)

        message_query = prompt_template.format(
            chunk_size=self.chunk_size,
            task_instruction=language_instruction,
            frame_observations=frame_obs_text,
        )

        beg = time.time()
        if self.send_mode == 'timestamped_frames':
            # Qwen3-VL interleaved timestamp-image format.
            # Subsample frames and assign real timestamps (seconds within chunk).
            fps = self.args.camera_fps
            indices, selected = subsample_frames(
                chunk_video_frames, max_frames=self.vlm_max_frames)
            timestamped = [
                (idx / fps, frame) for idx, frame in zip(indices, selected)
            ]
            response, thinking = self.chat(
                message_query=message_query,
                timestamped_frames=timestamped,
                return_thinking=True)
        elif self.send_mode == 'frame_list':
            # Send subsampled frames as individual images (no timestamps).
            _, selected = subsample_frames(
                chunk_video_frames, max_frames=self.vlm_max_frames)
            response, thinking = self.chat(
                message_query=message_query,
                frames=selected,
                return_thinking=True)
        else:
            raise ValueError(f"Unsupported send_mode: {self.send_mode}")
        elapsed = time.time() - beg
        print(f"    Chunk {chunk_idx} VLM query: {elapsed:.1f}s")

        return response, thinking

    def run(self):
        """Main collection loop."""
        os.makedirs(self.save_dir, exist_ok=True)
        total_examples = 0

        for task_info in self.tasks:
            task_name = task_info['task_name']
            bddl_file = task_info['bddl_file']
            language_instruction = task_info['language_instruction']

            print(f"\n{'='*60}")
            print(f"Task: {language_instruction}")
            print(f"{'='*60}")

            task_id = task_info['task_id']
            task_save_dir = self._locate_task_save_dir(task_id, language_instruction)

            # Determine starting episode index when resuming
            start_ep = 0
            if self.resume:
                start_ep = self._count_existing_episodes(task_save_dir)
                print(f"  Resume check: located save dir: {task_save_dir} | existing episodes: {start_ep} | requested per-task: {self.episodes_per_task}")
                if start_ep >= self.episodes_per_task:
                    print(f"  Already have {start_ep} episodes (>= {self.episodes_per_task} requested), skipping task.")
                    continue
                if start_ep > 0:
                    print(f"  Resuming from episode {start_ep} ({start_ep} already collected)")

            # Create env for this task
            env, _ = make_libero_env(
                task_name, bddl_file,
                camera_heights=self.args.camera_size,
                camera_widths=self.args.camera_size,
            )

            for ep in range(start_ep, self.episodes_per_task):
                print(f"\n  Episode {ep+1}/{self.episodes_per_task}")

                # Run full episode
                frames, observations, chunk_actions, chunk_all_frames, \
                    success, success_chunk = \
                    self.collect_episode(env, language_instruction)

                num_collected = len(chunk_actions)
                print(f"  Success: {success}"
                      + (f" (at chunk {success_chunk}), "
                         f"{num_collected} chunks collected"
                         if success else
                         f", {num_collected} chunks collected"))

                if success:
                    # ── Successful episode: use ALL collected chunks ──
                    n = num_collected
                    w_frames = frames                    # num_collected+1 frames
                    w_observations = observations
                    w_chunk_actions = chunk_actions       # num_collected arrays
                    w_chunk_videos = chunk_all_frames
                    print(f"  Using all {n} chunks (frames 0..{n})")
                else:
                    # ── Failed episode: select a 10-chunk window ──
                    n = self.COLLECT_CHUNKS
                    if num_collected < n:
                        # Episode shorter than COLLECT_CHUNKS (shouldn't
                        # happen for failures, but handle gracefully)
                        n = num_collected
                        start = 0
                    else:
                        start = self.select_window(num_collected, success, success_chunk)
                    end = start + n
                    print(f"  Window: chunks {start}..{end-1} (frames {start}..{end})")

                    w_frames = frames[start:end + 1]
                    w_observations = observations[start:end + 1]
                    w_chunk_actions = chunk_actions[start:end]
                    w_chunk_videos = chunk_all_frames[start:end]

                # Query VLM for all selected chunk transitions
                results = [None] * n
                beg_all = time.time()
                with ThreadPoolExecutor(max_workers=self.max_concurrent_vlm) as executor:
                    future_to_idx = {
                        executor.submit(
                            self.query_vlm_pair,
                            w_observations[i], w_observations[i + 1],
                            language_instruction, i,
                            chunk_video_frames=w_chunk_videos[i],
                        ): i
                        for i in range(n)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        results[idx] = future.result()  # (response, thinking)
                print(f"  All {n} VLM queries completed in {time.time() - beg_all:.1f}s")

                # Separate clean responses and reasoning traces
                descriptions = [r[0] for r in results]
                thinking_traces = [r[1] for r in results]

                # Combine descriptions with frame tags
                response_lines = []
                for i, desc in enumerate(descriptions):
                    response_lines.append(
                        f"<frame_{i}_to_frame_{i+1}>{desc}</frame_{i}_to_frame_{i+1}>"
                    )
                response_txt = "\n".join(response_lines)

                # Combine reasoning traces with frame tags
                reasoning_lines = []
                for i, trace in enumerate(thinking_traces):
                    reasoning_lines.append(
                        f"<frame_{i}_to_frame_{i+1}>{trace}</frame_{i}_to_frame_{i+1}>"
                    )
                reasoning_txt = "\n".join(reasoning_lines)

                # Build success mask
                # Build success mask
                if success:
                    success_mask = np.ones(n + 1)  # All 1s for successful episodes
                else:
                    success_mask = np.zeros(n + 1) # All 0s for failed episodes

                # Save
                save_example(
                    base_dir=task_save_dir,
                    example_idx=ep,
                    response_txt=response_txt,
                    reasoning_txt=reasoning_txt,
                    language_instruction=language_instruction,
                    obs_list=w_observations,
                    chunk_actions=w_chunk_actions,
                    video_arr=np.array(w_frames),
                    success_mask=success_mask,
                )

                total_examples += 1

            env.close()

        print(f"\n{'='*60}")
        print(f"Collection complete. Total examples: {total_examples}")
        print(f"Saved to: {self.save_dir}")
        print(f"{'='*60}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk-based behavioral data collection from LIBERO environments "
                    "(ActionChunkingPolicy / GMM head)"
    )

    # Paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a specific ActionChunkingPolicy checkpoint (.pth file)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory for collected data '
                             '(default: collected/{suite}_video_and_obs)')

    # LIBERO
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal',
                                 'libero_10', 'libero_90'],
                        help='LIBERO task suite')
    parser.add_argument('--camera_size', type=int, default=128,
                        help='Camera image dimensions (height=width)')
    parser.add_argument('--camera_fps', type=int, default=20,
                        help='FPS for chunk videos sent to VLM (default: 20, matching control_freq)')

    # Collection
    parser.add_argument('--episodes_per_task', type=int, default=10,
                        help='Number of episodes to collect per task')
    parser.add_argument('--max_steps', type=int, default=600,
                        help='Max environment steps per episode (default: 600)')
    parser.add_argument('--max_concurrent_vlm', type=int, default=10,
                        help='Max concurrent VLM queries per episode (default: all 10 chunks in parallel)')

    # VLM
    parser.add_argument('--vlm_model', type=str, required=True,
                        help='Model name served by the local vLLM OpenAI-compatible server')
    parser.add_argument('--vlm_host', type=str, default='127.0.0.1',
                    help='vLLM server host')
    parser.add_argument('--vlm_port', type=int, default=8000,
                    help='vLLM server port')
    parser.add_argument('--send_mode', type=str, default='timestamped_frames',
                        choices=['timestamped_frames', 'frame_list'],
                        help='How to send visual input to the VLM. '
                             '"timestamped_frames": subsample frames with interleaved '
                             'timestamp annotations (Qwen3-VL best practice). '
                             '"frame_list": subsample frames as individual images.')
    parser.add_argument('--vlm_max_frames', type=int, default=8,
                        help='Max frames to send per chunk when using '
                             'timestamped_frames or frame_list send_mode (default: 8)')

    # Resume
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume collection from where it left off. '
                             'Skips tasks/episodes that already have saved data.')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for policy inference')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file (overrides CLI args)')

    args = parser.parse_args()

    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    # Build default save_dir from suite if not explicitly set
    if args.save_dir is None:
        args.save_dir = f"collected/{args.suite}_video_and_obs"

    return args


def main():
    args = parse_args()
    pipeline = DataCollectionPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
