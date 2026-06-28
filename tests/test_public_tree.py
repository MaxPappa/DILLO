from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _public_text_files():
    suffixes = {".py", ".md", ".sh", ".yaml", ".yml", ".toml"}
    for path in ROOT.rglob("*"):
        if path.is_dir() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        if path.suffix in suffixes:
            yield path


def test_libero_is_external_reference():
    libero = ROOT / "LIBERO"
    assert libero.is_symlink()


def test_no_private_names_or_paths():
    forbidden = [
        "/" + "home" + "/" + "lucaromani",
        "/" + "media" + "/" + "pinas",
        "policy_explainer" + "_libero",
        "train_policy_explainer" + "_libero.py",
        "eval_action_refusal" + "_libero.py",
        "ACTPolicy" + "LIBERO",
        "ACTPolicy" + "Ale",
        "ACTAgent" + "Ale",
    ]
    offenders = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
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
