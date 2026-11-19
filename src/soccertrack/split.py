"""Match-level split, leakage guards and the test-access log.

Owner: Member 1. Priority milestone M1 (day 1): runners of Members 1 and 2 call `require_test_access`.
Contract: SHARED_CONTRACTS.md Section 14; plan Section 5.2.
The split values live in configs/project.yaml and are frozen by the approved plan.
"""

from __future__ import annotations

from typing import Any


def matches_for(split: str, cfg: dict[str, Any]) -> list[str]:
    """Return the match IDs of `split` ("train" | "val" | "test"), honouring the use_*_optional flags."""
    raise NotImplementedError("TODO(Member 1, M1)")


def split_of(match_id: str, cfg: dict[str, Any]) -> str:
    """Return the split a match belongs to; raise for unknown matches."""
    raise NotImplementedError("TODO(Member 1, M1)")


def assert_disjoint(cfg: dict[str, Any]) -> None:
    """Raise if any match appears in more than one split (including optional matches)."""
    raise NotImplementedError("TODO(Member 1, M1)")


def require_test_access(split: str, frozen_test: bool) -> None:
    """Raise PermissionError when `split == "test"` and the caller did not pass --frozen-test."""
    raise NotImplementedError("TODO(Member 1, M1)")


def log_test_access(member: str, method: str, artefacts: list[str], model_hashes: dict[str, str]) -> None:
    """Append one entry (UTC date, member, method, config_sha256, model hashes, artefacts) to outputs/test_log.md."""
    raise NotImplementedError("TODO(Member 1, M1)")
