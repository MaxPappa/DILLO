# DILLO: Describe-Then-Act

[![Paper](https://img.shields.io/badge/arXiv-2603.23149-B31B1B.svg)](https://arxiv.org/abs/2603.23149)
[![Conference](https://img.shields.io/badge/ECCV_2026-Accepted-blue.svg)](https://eccv2026.ecva.net/)
[![Code Status](https://img.shields.io/badge/Code-Available-green.svg)](#)

Official repository for **"Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models"**, accepted at **ECCV 2026**.

DILLO (DIstiLLed Language-ActiOn World Model) is a fast steering layer for
robotic policies. A privileged Vision Language Model annotates offline policy
rollouts, and a latent-conditioned LLM learns to predict semantic outcomes from
the policy latent state and planned actions. At inference time, DILLO can
describe likely next-state outcomes and accept/refuse proposed action chunks
without running a visual simulator.

## Abstract

Deploying safety-critical agents requires anticipating the consequences of
actions before they are executed. While world models offer a paradigm for this
proactive foresight, current approaches relying on visual simulation incur
prohibitive latencies, often exceeding several seconds per step. In this work,
we challenge the assumption that visual processing is necessary for failure
prevention. We show that a trained policy's latent state, combined with its
planned actions, already encodes sufficient information to anticipate action
outcomes, making visual simulation redundant for failure prevention. To this
end, we introduce DILLO, a fast steering layer that shifts the paradigm from
"simulate-then-act" to "describe-then-act." DILLO is trained via cross-modal
distillation, where a privileged Vision Language Model teacher annotates
offline trajectories and a latent-conditioned Large Language Model student
learns to predict semantic outcomes. This creates a text-only inference path,
bypassing heavy visual generation entirely, achieving a 14x speedup over
baselines. Experiments on MetaWorld and LIBERO demonstrate that DILLO produces
high-fidelity descriptions of the next state and is able to steer the policy,
improving episode success rate by up to 15 pp and 9.3 pp on average across
tasks.

## Repository Layout

- `dillo/policy`: ACT action-chunking policy, rollout wrapper, training, and evaluation.
- `dillo/data_generation`: LIBERO rollout collection and vLLM-based `video_and_obs` labeling.
- `dillo/training`: DILLO datasets, model, binary verdict head, and three-stage training.
- `dillo/evaluation`: validation generation, T2O/T2T fidelity, LLM fuzzy grading, and steering.
- `configs`: example YAML configs for DILLO stages.
- `scripts`: shell entry points for common experiments.
- `tests`: publication-oriented smoke tests.

The upstream LIBERO repository is treated as an external dependency. During
development we keep it as a local symlink named `LIBERO` at the repository root.

## Released Artifacts

The released dataset and checkpoints are optional. You can either reproduce
every step from scratch with the scripts below, or download the artifacts and
run training/evaluation directly.

- Dataset: https://huggingface.co/datasets/Sapienza/DILLO-LIBERO-dataset
- Checkpoints: https://drive.google.com/drive/folders/10sKCwHCWn7quLJQCYrFr8885iK7muXGf?usp=sharing

The checkpoint release contains ACT policies and DILLO Gemma-3 1B/4B stage-3
checkpoints. The LLM base weights are not included; they are loaded from
Hugging Face through `MODEL_NAME`.

## Environment Setup

Clone the repository and create a Python environment:

```bash
git clone https://github.com/MaxPappa/DILLO.git
cd DILLO

conda create -n dillo python=3.10 -y
conda activate dillo
```

Install PyTorch for your CUDA version, then install DILLO. Use PyTorch 2.6 or
newer for Gemma-3 4B evaluation; older PyTorch versions can fail during
generation with recent `transformers`.

```bash
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

Install the external LIBERO checkout at the repository root. Keeping LIBERO as
a symlink makes it clear that it is an external dependency, not vendored code.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
pip install -e external/LIBERO
ln -s external/LIBERO LIBERO
```

The upstream LIBERO package currently does not declare its runtime
dependencies in `setup.py`. Install the LIBERO robotics stack without
downgrading DILLO's newer `transformers`, `numpy`, `wandb`, or OpenCV
dependencies:

```bash
pip install \
  hydra-core==1.2.0 \
  numpy==1.26.4 \
  opencv-python==4.6.0.66 \
  robomimic==0.2.0 \
  einops==0.4.1 \
  thop==0.1.1-2209072238 \
  robosuite==1.4.0 \
  mujoco==2.3.7 \
  bddl==1.0.1 \
  future==0.18.2 \
  matplotlib==3.5.3 \
  cloudpickle==2.1.0 \
  gym==0.25.2
```

Create the LIBERO path config. Adjust `dataset_dir` if you want the original
LIBERO demonstrations somewhere else.

```bash
python - <<'PY'
from pathlib import Path
import yaml

repo = Path.cwd()
libero_root = repo / "LIBERO" / "libero" / "libero"
dataset_dir = repo / "libero_datasets"
cfg_dir = Path.home() / ".libero"
cfg_dir.mkdir(exist_ok=True)
dataset_dir.mkdir(exist_ok=True)

cfg = {
    "benchmark_root": str(libero_root),
    "bddl_files": str(libero_root / "bddl_files"),
    "init_states": str(libero_root / "init_files"),
    "datasets": str(dataset_dir),
    "assets": str(libero_root / "assets"),
}
(cfg_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
PY
```

For rendering and robot evaluation on headless machines, these environment
variables are usually needed:

```bash
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
```

If you will generate the dataset or run LLM fuzzy grading with a local
OpenAI-compatible model server, install vLLM as well:

```bash
pip install -e ".[serve]"
```

Authenticate to Hugging Face if you use gated backbones such as Gemma:

```bash
hf auth login
```

## Artifact Setup

To use the released DILLO dataset, download it under `data/`:

```bash
mkdir -p data
hf download Sapienza/DILLO-LIBERO-dataset \
  --repo-type dataset \
  --local-dir data/DILLO-LIBERO-dataset
```

To download a single suite, add an include filter, for example:

```bash
hf download Sapienza/DILLO-LIBERO-dataset \
  --repo-type dataset \
  --include "LIBERO_SPATIAL/**" \
  --local-dir data/DILLO-LIBERO-dataset
```

The dataset is organized as:

```text
data/DILLO-LIBERO-dataset/
  LIBERO_SPATIAL/<TASK>/<EPISODE>/
  LIBERO_OBJECT/<TASK>/<EPISODE>/
  LIBERO_GOAL/<TASK>/<EPISODE>/
  LIBERO_10/<TASK>/<EPISODE>/
  LIBERO_90/<TASK>/<EPISODE>/
```

The training scripts expect the suite aliases below. Create them once:

```bash
ln -sfn DILLO-LIBERO-dataset/LIBERO_SPATIAL data/libero_spatial_video_and_obs
ln -sfn DILLO-LIBERO-dataset/LIBERO_OBJECT  data/libero_object_video_and_obs
ln -sfn DILLO-LIBERO-dataset/LIBERO_GOAL    data/libero_goal_video_and_obs
ln -sfn DILLO-LIBERO-dataset/LIBERO_10      data/libero_10_video_and_obs
ln -sfn DILLO-LIBERO-dataset/LIBERO_90      data/libero_90_video_and_obs
```

To use the released checkpoints, download the Google Drive folder and place its
contents under `checkpoints/release`:

```text
checkpoints/release/
  act_policies/
    LIBERO_SPATIAL/best_model.pth
    LIBERO_OBJECT/best_model.pth
    LIBERO_GOAL/best_model.pth
    LIBERO_10/best_model.pth
    LIBERO_90/best_model.pth
  dillo_gemma_3_1b_it/
    LIBERO_SPATIAL/
    LIBERO_OBJECT/
    LIBERO_GOAL/
    LIBERO_10/
    LIBERO_90/
  dillo_gemma_3_4b_it/
    LIBERO_SPATIAL/
    LIBERO_OBJECT/
    LIBERO_GOAL/
    LIBERO_10/
    LIBERO_90/
```

If your downloader creates an extra top-level folder, either move its contents
into `checkpoints/release` or set `RELEASE_DIR` to that folder in the commands
below.

## Train ACT From Scratch

ACT training uses the original LIBERO demonstration datasets. Download them
through LIBERO first:

```bash
python LIBERO/benchmark_scripts/download_libero_datasets.py --use-huggingface
```

Train a policy for one suite:

```bash
SUITE=libero_goal \
EXP_DIR=experiments_act \
EPOCHS=200 \
BATCH_SIZE=64 \
CHUNK_SIZE=20 \
DEVICE=cuda \
./scripts/train_act.sh
```

The script writes runs under:

```text
experiments_act/LIBERO_GOAL/ACT_chunk20_seed42/run_001/
```

The checkpoint used by later stages is `best_model.pth`. Repeat with
`libero_spatial`, `libero_object`, `libero_10`, and `libero_90` for all suites.

Evaluate an ACT checkpoint:

```bash
python -m dillo.policy.evaluate_act \
  --checkpoint experiments_act/LIBERO_GOAL/ACT_chunk20_seed42/run_001/best_model.pth \
  --suite libero_goal \
  --n_eval 20
```

## Generate The DILLO Dataset From Scratch

Start a vLLM OpenAI-compatible server for the VLM teacher:

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Thinking --host 127.0.0.1 --port 8000
```

Collect labeled chunk transitions. The public collector uses the
`video_and_obs` prompt:

```bash
ACT_CHECKPOINT=experiments_act/LIBERO_GOAL/ACT_chunk20_seed42/run_001/best_model.pth \
VLM_MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking \
SUITE=libero_goal \
SAVE_DIR=data/libero_goal_video_and_obs \
EPISODES_PER_TASK=10 \
DEVICE=cuda \
./scripts/generate_dataset.sh
```

Repeat the command with the corresponding ACT checkpoint and `SAVE_DIR` for
each suite.

## Train DILLO

DILLO is trained in three stages. Stages 1 and 2 train the description model;
stage 3 keeps the description loss and adds a specialized binary verdict head.
The backbone is selected with `MODEL_NAME`.

Using released artifacts for LIBERO Goal and Gemma-3 1B:

```bash
RELEASE_DIR=checkpoints/release

ACT_CHECKPOINT=$RELEASE_DIR/act_policies/LIBERO_GOAL/best_model.pth \
SUITE=goal \
MODEL_NAME=google/gemma-3-1b-it \
TRAIN_DATA='data/libero_goal_video_and_obs/*/*' \
RUN_DIR=checkpoints/dillo_goal_gemma-3-1b-it \
STAGE1_EPOCHS=3 \
STAGE2_EPOCHS=3 \
STAGE3_EPOCHS=3 \
./scripts/train_dillo_stage123.sh
```

Train the 4B backbone by changing:

```bash
MODEL_NAME=google/gemma-3-4b-it
```

Use another causal LLM backbone by changing `MODEL_NAME`, for example:

```bash
MODEL_NAME=nvidia/Nemotron-Mini-4B-Instruct
```

The example YAML files in `configs/` expose the same arguments for direct
`python -m dillo.training.train_dillo` use.

## Validation And Fidelity Metrics

Create validation splits from the released or generated dataset:

```bash
python -m dillo.evaluation.split_dataset --all_suites --output_dir val_splits
```

Generate validation predictions and compute T2O/T2T fidelity from a released
1B checkpoint:

```bash
RELEASE_DIR=checkpoints/release

CHECKPOINT_DIR=$RELEASE_DIR/dillo_gemma_3_1b_it/LIBERO_GOAL \
ACT_CHECKPOINT=$RELEASE_DIR/act_policies/LIBERO_GOAL/best_model.pth \
SUITE=goal \
MODEL_NAME=google/gemma-3-1b-it \
OUTPUT_ROOT=outputs/validation \
./scripts/evaluate_validation.sh
```

For the released 4B model, use:

```bash
CHECKPOINT_DIR=$RELEASE_DIR/dillo_gemma_3_4b_it/LIBERO_GOAL
MODEL_NAME=google/gemma-3-4b-it
```

Run only the metric script on an existing prediction directory:

```bash
python -m dillo.evaluation.metrics \
  --predictions_dir outputs/validation/libero_goal/gemma-3-1b-it/stage3_latentobs_LIBERO_GOAL
```

## LLM Fuzzy Grading

Run an OpenAI-compatible LLM server, then grade all prediction JSON files under
a root directory:

```bash
ROOT_DIR=outputs/validation/libero_goal \
MODEL=Qwen/Qwen2.5-32B-Instruct \
HOST=127.0.0.1 \
PORT=8000 \
./scripts/run_llm_fuzzy.sh
```

The script writes `grading_score` and `grading_output` back into each JSON and
can also export CSV summaries.

## Steering Evaluation

Evaluate action-refusal steering with a released DILLO stage-3 checkpoint:

```bash
RELEASE_DIR=checkpoints/release

ACT_CHECKPOINT=$RELEASE_DIR/act_policies/LIBERO_GOAL/best_model.pth \
EXPLAINER_CKPTDIR=$RELEASE_DIR/dillo_gemma_3_1b_it/LIBERO_GOAL \
SUITE=libero_goal \
MODEL_NAME=google/gemma-3-1b-it \
N_EPISODES=20 \
OUTDIR=outputs/steering/libero_goal/gemma-3-1b-it \
./scripts/evaluate_steering.sh
```

For an MLP steering baseline, set:

```bash
STEERING_MODEL_TYPE=mlp
MLP_CKPT=checkpoints/mlp/libero_goal/best.pt
```

## Development Checks

```bash
python -m py_compile $(find dillo -name '*.py' -print)
python -m pytest
```

## Citation

If you find our work or code useful for your research, please cite:

```bibtex
@inproceedings{pappa2026describe,
  title={Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models},
  author={Pappa, Massimiliano and Romani, Luca and Sacco, Valentino and Palma, Alessio and Lathuili{\`e}re, St{\'e}phane and Galasso, Fabio and Alameda-Pineda, Xavier and Spinelli, Indro},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026},
  eprint={2603.23149},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
