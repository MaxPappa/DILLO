# DILLO: Describe-Then-Act

[![Paper](https://img.shields.io/badge/arXiv-2603.23149-B31B1B.svg)](https://arxiv.org/abs/2603.23149)
[![Conference](https://img.shields.io/badge/ECCV_2026-Accepted-blue.svg)](https://eccv2026.ecva.net/)
[![Code Status](https://img.shields.io/badge/Code-Coming_Soon-orange.svg)](#)

Official repository for **"Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models"**, accepted at **ECCV 2026**.

---

## 📢 Codebase Coming Soon!

> [!NOTE]
> We are currently preparing and cleaning up our code, training scripts, and pretrained models for public release. The full codebase will be made available soon. Please **star** the repository to stay updated!

---

## 📝 Abstract

Deploying safety-critical agents requires anticipating the consequences of actions before they are executed. While world models offer a paradigm for this proactive foresight, current approaches relying on visual simulation incur prohibitive latencies, often exceeding several seconds per step. In this work, we challenge the assumption that visual processing is necessary for failure prevention. We show that a trained policy's latent state, combined with its planned actions, already encodes sufficient information to anticipate action outcomes, making visual simulation redundant for failure prevention. To this end, we introduce DILLO (DIstiLLed Language-ActiOn World Model), a fast steering layer that shifts the paradigm from "simulate-then-act" to "describe-then-act." DILLO is trained via cross-modal distillation, where a privileged Vision Language Model teacher annotates offline trajectories and a latent-conditioned Large Language Model student learns to predict semantic outcomes. This creates a text-only inference path, bypassing heavy visual generation entirely, achieving a 14x speedup over baselines. Experiments on MetaWorld and LIBERO demonstrate that DILLO produces high-fidelity descriptions of the next state and is able to steer the policy, improving episode success rate by up to 15 pp and 9.3 pp on average across tasks.


## 🎓 Citation

If you find our work or code useful for your research, please consider citing our paper:

```bibtex
@article{pappa2026describe,
  title={Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models},
  author={Pappa, Massimiliano and Romani, Luca and Sacco, Valentino and Palma, Alessio and Lathuili{\`e}re, St{\'e}phane and Galasso, Fabio and Alameda-Pineda, Xavier and Spinelli, Indro},
  journal={arXiv preprint arXiv:2603.23149},
  year={2026}
}
