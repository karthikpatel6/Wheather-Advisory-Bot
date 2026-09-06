"""
policy_store.py — Loads and queries the sops.yaml file.

No SOP content, category name, threshold value, or field name may appear
hardcoded in this file or anywhere else in the codebase. Everything is
read from sops.yaml at runtime.

Usage:
    store = PolicyStore()
    all_sops    = store.get_all()
    numeric     = store.get_numeric()
    semantic    = store.get_semantic()
    sop         = store.get_by_id("SOP-001")
    rank        = store.severity_rank("high")   # → 3
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Severity ordering — higher number = more severe
_SEVERITY_RANK: dict[str, int] = {
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

# Path to sops.yaml relative to this file's parent directory
_SOPS_PATH = Path(__file__).parent.parent / "sops.yaml"


class SOPValidationError(Exception):
    """Raised when a loaded SOP fails schema validation."""


class PolicyStore:
    """Loads sops.yaml once and exposes query helpers.

    Thread-safe for reads after __init__ completes (the list is never mutated).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        sops_file = Path(path) if path else _SOPS_PATH
        if not sops_file.exists():
            raise FileNotFoundError(f"sops.yaml not found at: {sops_file}")

        with sops_file.open("r", encoding="utf-8") as f:
            raw: list[dict[str, Any]] = yaml.safe_load(f)

        if not isinstance(raw, list):
            raise SOPValidationError("sops.yaml must be a YAML list at the top level")

        self._sops: list[dict[str, Any]] = []
        for sop in raw:
            self._validate(sop)
            # Normalise whitespace in multiline string fields
            sop["applies_when"] = sop["applies_when"].strip()
            sop["guidance"] = sop["guidance"].strip()
            self._sops.append(sop)

        logger.info("PolicyStore loaded %d SOPs from %s", len(self._sops), sops_file)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, sop: dict[str, Any]) -> None:
        """Raise SOPValidationError if the SOP dict is missing required keys."""
        required = {"id", "category", "severity", "match_type", "applies_when", "guidance"}
        missing = required - sop.keys()
        if missing:
            raise SOPValidationError(
                f"SOP {sop.get('id', '(unknown)')} is missing required fields: {missing}"
            )

        if sop["severity"] not in _SEVERITY_RANK:
            raise SOPValidationError(
                f"SOP {sop['id']}: unknown severity '{sop['severity']}'. "
                f"Must be one of {list(_SEVERITY_RANK)}"
            )

        if sop["match_type"] not in ("numeric", "semantic"):
            raise SOPValidationError(
                f"SOP {sop['id']}: unknown match_type '{sop['match_type']}'. "
                "Must be 'numeric' or 'semantic'"
            )

        if sop["match_type"] == "numeric":
            cond = sop.get("condition")
            if not cond or not all(k in cond for k in ("field", "operator", "value")):
                raise SOPValidationError(
                    f"SOP {sop['id']}: numeric SOP must have condition.field, "
                    "condition.operator, and condition.value"
                )
            valid_ops = {">", ">=", "<", "<=", "=="}
            if cond["operator"] not in valid_ops:
                raise SOPValidationError(
                    f"SOP {sop['id']}: condition.operator must be one of {valid_ops}, "
                    f"got '{cond['operator']}'"
                )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict[str, Any]]:
        """Return all SOPs as loaded from YAML."""
        return list(self._sops)

    def get_numeric(self) -> list[dict[str, Any]]:
        """Return only SOPs with match_type='numeric'."""
        return [s for s in self._sops if s["match_type"] == "numeric"]

    def get_semantic(self) -> list[dict[str, Any]]:
        """Return only SOPs with match_type='semantic'."""
        return [s for s in self._sops if s["match_type"] == "semantic"]

    def get_by_id(self, sop_id: str) -> dict[str, Any] | None:
        """Return the SOP with the given id, or None if not found."""
        for sop in self._sops:
            if sop["id"] == sop_id:
                return sop
        return None

    def severity_rank(self, severity: str) -> int:
        """Return a numeric rank for severity comparison.

        critical=4 > high=3 > moderate=2 > low=1.
        Returns 0 for unknown severity strings (treated as lowest priority).
        """
        return _SEVERITY_RANK.get(severity, 0)

    def get_highest_severity(self, sop_ids: list[str]) -> dict[str, Any] | None:
        """Given a list of SOP ids, return the one with the highest severity.

        Ties are broken by first appearance in sop_ids (i.e. the order they
        were matched — numeric matches come first by convention).
        """
        best: dict[str, Any] | None = None
        best_rank = -1
        for sid in sop_ids:
            sop = self.get_by_id(sid)
            if sop is None:
                logger.warning("get_highest_severity: unknown SOP id '%s' ignored", sid)
                continue
            rank = self.severity_rank(sop["severity"])
            if rank > best_rank:
                best_rank = rank
                best = sop
        return best

    def __len__(self) -> int:
        return len(self._sops)

    def __repr__(self) -> str:
        return f"PolicyStore(sops={len(self._sops)})"
