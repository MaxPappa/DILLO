from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]

LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|Users|media|mnt|scratch|workspace)/[^\s'\"`<>)]*"
)
LEGACY_MODULE_RE = re.compile(
    r"\b(?:policy_explainer|train_policy_explainer|eval_action_refusal)_[a-z0-9_]+\b"
)
SUFFIXED_ACT_RE = re.compile(r"\b(?:ACTPolicy|ACTAgent)[A-Z][A-Za-z0-9_]*\b")


def _public_text_files():
    suffixes = {".py", ".md", ".sh", ".yaml", ".yml", ".toml"}
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True
        ).splitlines()
        candidates = [ROOT / rel for rel in tracked]
    except Exception:
        candidates = list(ROOT.rglob("*"))

    for path in candidates:
        if path.is_dir() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        if path.suffix in suffixes:
            yield path


def test_libero_is_external_reference():
    libero = ROOT / "LIBERO"
    assert libero.is_symlink() or not libero.exists()


def test_no_private_names_or_paths():
    offenders = []
    home = Path.home().as_posix()
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        if home in text:
            offenders.append(f"{rel}: contains the current user's home path")
        for match in LOCAL_PATH_RE.finditer(text):
            offenders.append(f"{rel}: absolute local path {match.group(0)!r}")
        for match in LEGACY_MODULE_RE.finditer(text):
            offenders.append(f"{rel}: legacy module name {match.group(0)!r}")
        for match in SUFFIXED_ACT_RE.finditer(text):
            offenders.append(f"{rel}: suffixed ACT symbol {match.group(0)!r}")
    assert offenders == []


def test_dataset_generation_exposes_only_video_and_obs_prompt():
    prompt_file = ROOT / "dillo" / "data_generation" / "prompts.py"
    text = prompt_file.read_text(encoding="utf-8")
    assert "CHUNK_PROMPT_VIDEO_AND_OBS" in text
    assert "VIDEO_ONLY" not in text
    assert "OBS_ONLY" not in text
    assert "prompt" + "_mode" not in text


def test_suite_normalization():
    from dillo.suites import benchmark_name, normalize_suite

    assert normalize_suite("goal") == "libero_goal"
    assert normalize_suite("libero_10") == "libero_10"
    assert benchmark_name("spatial") == "LIBERO_SPATIAL"
