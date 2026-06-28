"""
Evaluation utilities for LIBERO PolicyExplainer.

Metrics:
  - BLEU (sentence-level, smoothed)
  - ROUGE (rouge1, rouge2, rougeL)
  - BERTScore (rescaled F1)
  - EEF Directional Fidelity: checks whether directional words in the
    generated text match the actual EEF movement direction
  - Fidelity T2T: text-to-text directional consistency between
    prediction and reference
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# ───────────────────────────── directional patterns ─────────────────

DIR_PATTERNS = {
    "right":    [r"\bright\b", r"\bto the right\b"],
    "left":     [r"\bleft\b", r"\bto the left\b"],
    "forward":  [r"\bforward\b", r"\btowards? the front\b"],
    "backward": [r"\bback(?:ward|wards)?\b"],
    "up":       [r"\bup(?:ward|wards)?\b", r"\blift(?:s|ed|ing)?\b", r"\braise[sd]?\b"],
    "down":     [r"\bdown(?:ward|wards)?\b", r"\blower(?:s|ed|ing)?\b"],
}

GRIPPER_OPEN  = [r"\bopen(?:s|ed|ing)?\b"]
GRIPPER_CLOSE = [r"\bclose[sd]?\b", r"\bclosing\b", r"\bgrasp(?:s|ed|ing)?\b"]


def _any_match(text: str, patterns: list) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _extract_dirs(text: str) -> Dict[str, str]:
    """Extract mentioned directions from text (x, y, z axes)."""
    dirs = {}
    # X
    has_right = _any_match(text, DIR_PATTERNS["right"])
    has_left = _any_match(text, DIR_PATTERNS["left"])
    if has_right and not has_left:
        dirs["x"] = "right"
    elif has_left and not has_right:
        dirs["x"] = "left"
    # Y
    has_fwd = _any_match(text, DIR_PATTERNS["forward"])
    has_bwd = _any_match(text, DIR_PATTERNS["backward"])
    if has_fwd and not has_bwd:
        dirs["y"] = "forward"
    elif has_bwd and not has_fwd:
        dirs["y"] = "backward"
    # Z
    has_up = _any_match(text, DIR_PATTERNS["up"])
    has_down = _any_match(text, DIR_PATTERNS["down"])
    if has_up and not has_down:
        dirs["z"] = "up"
    elif has_down and not has_up:
        dirs["z"] = "down"
    return dirs


def _extract_gripper(text: str) -> Optional[str]:
    """Extract gripper state from text."""
    has_open = _any_match(text, GRIPPER_OPEN)
    has_close = _any_match(text, GRIPPER_CLOSE)
    if has_open and not has_close:
        return "opening"
    elif has_close and not has_open:
        return "closing"
    return None


# ───────────────────────── EEF directional fidelity ─────────────────

def _actual_eef_dirs(eef_before: np.ndarray, eef_after: np.ndarray,
                     threshold: float = 0.002) -> Dict[str, str]:
    """Compute actual EEF directional changes from positions."""
    diff = eef_after - eef_before
    dirs = {}
    if diff[0] > threshold:
        dirs["x"] = "right"
    elif diff[0] < -threshold:
        dirs["x"] = "left"
    if diff[1] > threshold:
        dirs["y"] = "forward"
    elif diff[1] < -threshold:
        dirs["y"] = "backward"
    if diff[2] > threshold:
        dirs["z"] = "up"
    elif diff[2] < -threshold:
        dirs["z"] = "down"
    return dirs


def _actual_gripper_change(grip_before: np.ndarray, grip_after: np.ndarray,
                           threshold: float = 0.005) -> Optional[str]:
    """Compute actual gripper state change."""
    diff = grip_after.mean() - grip_before.mean()
    if diff > threshold:
        return "opening"
    elif diff < -threshold:
        return "closing"
    return None


def eef_fidelity_single(
    description: str,
    eef_before: np.ndarray,
    eef_after: np.ndarray,
    gripper_before: np.ndarray = None,
    gripper_after: np.ndarray = None,
    eef_threshold: float = 0.002,
    grip_threshold: float = 0.005,
    penalize_missing: bool = False,
) -> float:
    """
    Compute fidelity score for a single transition by checking whether
    directional words in the description match the actual EEF movement.

    Returns:
        score in [0, 1]
    """
    pred_dirs = _extract_dirs(description)
    actual_dirs = _actual_eef_dirs(eef_before, eef_after, eef_threshold)

    checks = 0
    correct = 0

    # Check each mentioned direction axis
    for axis in ("x", "y", "z"):
        if axis in pred_dirs:
            checks += 1
            if axis in actual_dirs and pred_dirs[axis] == actual_dirs[axis]:
                correct += 1
        elif penalize_missing and axis in actual_dirs:
            checks += 1  # penalize for not mentioning a real movement

    # Gripper
    if gripper_before is not None and gripper_after is not None:
        pred_grip = _extract_gripper(description)
        actual_grip = _actual_gripper_change(gripper_before, gripper_after, grip_threshold)
        if pred_grip is not None:
            checks += 1
            if pred_grip == actual_grip:
                correct += 1

    if checks == 0:
        return 1.0  # no directional claims → vacuously correct

    return correct / checks


def compute_eef_fidelity(predictions: List[str], obs: np.ndarray) -> float:
    """
    Compute average EEF directional fidelity across a batch.

    Args:
        predictions: List of generated descriptions
        obs: (N, 2, 12) — pairs of (before, after) robot states
             Columns: [eef(3), joint(7), gripper(2)]
    """
    scores = []
    for i, desc in enumerate(predictions):
        eef_before = obs[i, 0, :3]
        eef_after = obs[i, 1, :3]
        grip_before = obs[i, 0, 10:12]
        grip_after = obs[i, 1, 10:12]
        s = eef_fidelity_single(
            desc, eef_before, eef_after, grip_before, grip_after
        )
        scores.append(s)
    return float(np.mean(scores))


# ───────────────────────── Text-to-text fidelity ────────────────────

def fidelity_text_to_text(prediction: str, reference: str) -> float:
    """
    Check whether the directional mentions in the prediction match
    those in the reference description.

    Returns score in [0, 1].
    """
    pred_dirs = _extract_dirs(prediction)
    ref_dirs = _extract_dirs(reference)

    checks = 0
    correct = 0

    for axis in ("x", "y", "z"):
        if axis in ref_dirs:
            checks += 1
            if axis in pred_dirs and pred_dirs[axis] == ref_dirs[axis]:
                correct += 1
        if axis in pred_dirs and axis not in ref_dirs:
            checks += 1  # extra claim not in reference

    # Gripper
    pred_grip = _extract_gripper(prediction)
    ref_grip = _extract_gripper(reference)
    if ref_grip is not None:
        checks += 1
        if pred_grip == ref_grip:
            correct += 1

    if checks == 0:
        return 1.0
    return correct / checks


def compute_fidelity_t2t(predictions: List[str], references: List[str]) -> float:
    scores = [fidelity_text_to_text(p, r) for p, r in zip(predictions, references)]
    return float(np.mean(scores))


# ───────────────────────── Standard text metrics ────────────────────

def compute_bleu(predictions: List[str], references: List[str]) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4
    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(
            sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        )
    return float(np.mean(scores))


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in results:
            results[k].append(scores[k].fmeasure)
    return {k: float(np.mean(v)) for k, v in results.items()}


def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True)
    return float(F1.mean())


# ─────────────────────────── Combined evaluator ─────────────────────

def evaluate_text_generation(
    predictions: List[str],
    references: List[str],
    obs: np.ndarray = None,
    compute_slow_metrics: bool = False,
) -> Dict[str, float]:
    """
    Run all evaluation metrics.

    Args:
        predictions: generated descriptions
        references: ground-truth descriptions
        obs: (N, 2, 12) observation pairs for EEF fidelity
        compute_slow_metrics: whether to run BLEU/ROUGE/BERTScore (slow)

    Returns:
        dict of metric name → float
    """
    metrics = {}

    # Fast: directional fidelity
    if obs is not None:
        obs_np = obs if isinstance(obs, np.ndarray) else np.stack(
            [o.detach().cpu().numpy() for o in obs]
        )
        metrics["fidelity_eef"] = compute_eef_fidelity(predictions, obs_np)

    metrics["fidelity_t2t"] = compute_fidelity_t2t(predictions, references)

    if compute_slow_metrics:
        metrics["bleu"] = compute_bleu(predictions, references)
        rouge = compute_rouge(predictions, references)
        metrics.update({f"rouge_{k}": v for k, v in rouge.items()})
        metrics["bert_f1"] = compute_bertscore(predictions, references)

    return metrics
