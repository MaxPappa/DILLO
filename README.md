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

## Setup

Create or activate an environment with LIBERO, robosuite, PyTorch,
Transformers, PEFT, vLLM/OpenAI client dependencies, and the usual robotics
stack. In our experiments this environment is named `luca-libero`.

```bash
conda activate luca-libero
git clone https://github.com/MaxPappa/DILLO.git
cd DILLO
pip install -e .
```

Place or link a LIBERO checkout at the repository root:

```bash
ln -s /path/to/LIBERO LIBERO
```

The code also resolves LIBERO data paths through the standard LIBERO
configuration file:

```bash
~/.libero/config.yaml
```

Large artifacts are intentionally kept out of git. Suggested local directories:

- `data/`: collected DILLO datasets.
- `checkpoints/`: ACT and DILLO checkpoints.
- `outputs/`: validation predictions, metrics, and steering summaries.
- `experiments_act/`: ACT training runs.

## Train ACT

Train an ACT action-chunking policy on a LIBERO suite:

```bash
SUITE=libero_goal \
EXP_DIR=experiments_act \
EPOCHS=200 \
BATCH_SIZE=64 \
./scripts/train_act.sh
```

Evaluate a trained ACT checkpoint:

```bash
python -m dillo.policy.evaluate_act \
  --checkpoint experiments_act/LIBERO_GOAL/ACT_chunk20_seed42/run_001/best_model.pth \
  --suite libero_goal \
  --n_eval 20
```

## Generate The DILLO Dataset

Start a vLLM OpenAI-compatible server for the VLM teacher:

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Thinking --host 127.0.0.1 --port 8000
```

Collect labeled chunk transitions. The public collector supports the
`video_and_obs` prompt path:

```bash
ACT_CHECKPOINT=checkpoints/act/libero_goal/best_model.pth \
VLM_MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking \
SUITE=libero_goal \
SAVE_DIR=data/libero_goal_video_and_obs \
./scripts/generate_dataset.sh
```

## Train DILLO

DILLO is trained in three stages. The backbone is selected with `MODEL_NAME`.
The default path uses ACT latents plus action chunks as input, and stage 3 uses
a specialized binary verdict head.

```bash
ACT_CHECKPOINT=checkpoints/act/libero_goal/best_model.pth \
SUITE=goal \
MODEL_NAME=google/gemma-3-1b-it \
TRAIN_DATA='data/libero_goal_video_and_obs/*/*' \
RUN_DIR=checkpoints/dillo_goal_gemma-3-1b-it \
STAGE1_EPOCHS=3 \
STAGE2_EPOCHS=3 \
STAGE3_EPOCHS=3 \
./scripts/train_dillo_stage123.sh
```

Use another LLM backbone by changing `MODEL_NAME`, for example:

```bash
MODEL_NAME=nvidia/Nemotron-Mini-4B-Instruct
```

The example YAML files in `configs/` expose the same arguments for direct
`python -m dillo.training.train_dillo` use.

## Validation And Fidelity Metrics

Create a validation split from collected data:

```bash
python -m dillo.evaluation.split_dataset \
  --suite goal \
  --data_dir 'data/libero_goal_video_and_obs/*/*' \
  --output_dir val_splits
```

Generate validation predictions and compute T2O/T2T fidelity:

```bash
CHECKPOINT_DIR=checkpoints/dillo_goal_gemma-3-1b-it/stage3/latentobs \
ACT_CHECKPOINT=checkpoints/act/libero_goal/best_model.pth \
SUITE=goal \
MODEL_NAME=google/gemma-3-1b-it \
./scripts/evaluate_validation.sh
```

Run only the metric script on an existing prediction directory:

```bash
python -m dillo.evaluation.metrics \
  --predictions_dir outputs/validation/libero_goal/gemma-3-1b-it/stage3_latentobs_latest
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

Evaluate action-refusal steering with a trained DILLO stage-3 checkpoint:

```bash
ACT_CHECKPOINT=checkpoints/act/libero_goal/best_model.pth \
EXPLAINER_CKPTDIR=checkpoints/dillo_goal_gemma-3-1b-it/stage3/latentobs \
SUITE=libero_goal \
MODEL_NAME=google/gemma-3-1b-it \
N_EPISODES=20 \
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
@article{pappa2026describe,
  title={Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models},
  author={Pappa, Massimiliano and Romani, Luca and Sacco, Valentino and Palma, Alessio and Lathuili{\`e}re, St{\'e}phane and Galasso, Fabio and Alameda-Pineda, Xavier and Spinelli, Indro},
  journal={arXiv preprint arXiv:2603.23149},
  year={2026}
}
```

