"""Shared LIBERO suite names and benchmark mappings."""

from __future__ import annotations

VALID_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)

SUITE_TO_BENCHMARK = {
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_10": "LIBERO_10",
    "libero_90": "LIBERO_90",
}

BENCHMARK_TO_SUITE = {v: k for k, v in SUITE_TO_BENCHMARK.items()}


def normalize_suite(suite: str) -> str:
    """Accept either ``goal`` or ``libero_goal`` style names."""
    suite = suite.lower()
    if suite in VALID_SUITES:
        return suite
    candidate = f"libero_{suite}"
    if candidate in VALID_SUITES:
        return candidate
    raise ValueError(f"Unknown LIBERO suite {suite!r}. Expected one of {VALID_SUITES}.")


def benchmark_name(suite: str) -> str:
    """Return the LIBERO benchmark name for a suite."""
    return SUITE_TO_BENCHMARK[normalize_suite(suite)]
