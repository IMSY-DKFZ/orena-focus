"""Hierarchical capability taxonomy for the FOCUS challenge.

The taxonomy has two levels:

- **Groups** (5 top-level categories) — used for hierarchical aggregation
  during evaluation.  They are never directly assigned to questions.
- **Leaves** (15 capabilities) — the actual capabilities assigned to questions.
  Each question has one *primary* leaf and zero or more *secondary* leaves.

Usage::

    from focus import Capability

    cap = Capability.OBJECT_IDENTIFICATION
    cap.group           # Capability.OBJECT_RECOGNITION
    cap.is_leaf         # True
    cap.is_group        # False

    Capability.groups()  # all five top-level groups
    Capability.leaves()  # all 15 assignable capabilities
    Capability.AGGREGATION.children  # (OBJECT_AGGREGATION, EVENT_AGGREGATION)
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any


class Capability(str, Enum):
    """Hierarchical VQA capability taxonomy.

    String-valued so it serialises cleanly in JSON / Pydantic models.
    """

    # ── 1. Object recognition and identity matching ──────────────────
    OBJECT_RECOGNITION = "object_recognition"
    # leaves of group one
    OBJECT_IDENTIFICATION = "object_identification"
    INSTANCE_MATCHING = "instance_matching"
    OBJECT_ATTRIBUTES = "object_attributes"
    SPATIAL_LOCALIZATION_CAMERA = "spatial_localization_camera"
    SPATIAL_LOCALIZATION_SITUS = "spatial_localization_situs"

    # ── 2. Temporal grounding ────────────────────────────────────────
    TEMPORAL_GROUNDING = "temporal_grounding"
    # leaves of group two
    TEMPORAL_LOCALIZATION = "temporal_localization"
    DURATION_ESTIMATION = "duration_estimation"

    # ── 3. Aggregation ───────────────────────────────────────────────
    AGGREGATION = "aggregation"
    # leaves of group three
    OBJECT_AGGREGATION = "object_aggregation"
    EVENT_AGGREGATION = "event_aggregation"

    # ── 4. Event and procedural understanding ────────────────────────
    EVENT_UNDERSTANDING = "event_understanding"
    # leaves of group four
    FO_INTERACTION_RECOGNITION = "fo_interaction_recognition"
    FO_USAGE_PURPOSE = "fo_usage_purpose"
    TEMPORAL_ORDERING = "temporal_ordering"

    # ── 5. Complex reasoning ─────────────────────────────────────────
    COMPLEX_REASONING = "complex_reasoning"
    # leaves of group five
    FUNCTIONAL_REASONING = "functional_reasoning"
    CAUSAL_CONSEQUENCE_REASONING = "causal_consequence_reasoning"
    MULTI_STEP_REASONING = "multi_step_reasoning"

    # ── hierarchy helpers ────────────────────────────────────────────

    @property
    def is_group(self) -> bool:
        """``True`` if this is a top-level grouping category (not assignable to questions)."""
        return self not in _PARENT_MAP

    @property
    def is_leaf(self) -> bool:
        """``True`` if this capability can be assigned to questions."""
        return self in _PARENT_MAP

    @property
    def group(self) -> Capability:
        """Return the top-level group this leaf belongs to.

        For groups themselves, returns ``self``.
        """
        return _PARENT_MAP.get(self, self)

    @property
    def children(self) -> tuple[Capability, ...]:
        """Return leaf capabilities under this group (empty for leaves)."""
        return tuple(c for c in Capability if _PARENT_MAP.get(c) is self)

    @classmethod
    def groups(cls) -> tuple[Capability, ...]:
        """Return all five top-level grouping categories."""
        return tuple(c for c in cls if c.is_group)

    @classmethod
    def leaves(cls) -> tuple[Capability, ...]:
        """Return all leaf capabilities (assignable to questions)."""
        return tuple(c for c in cls if c.is_leaf)

    @property
    def code(self) -> str | None:
        """Return canonical short code (e.g. ``1a``)."""
        return _CAPABILITY_TO_CODE.get(self)

    @property
    def label(self) -> str:
        """Human-readable label for reporting."""
        if self.is_group:
            return _GROUP_LABELS.get(self, self.value.replace("_", " ").title())
        return self.value.replace("_", " ")

    @classmethod
    def from_any(cls, value: Any) -> Capability | None:
        """Best-effort conversion from enum/name/value/code to ``Capability``.

        Supports:
        - Capability enum instances
        - canonical values (e.g. ``"object_identification"``)
        - enum names (e.g. ``"OBJECT_IDENTIFICATION"``)
        - short codes (e.g. ``"1a"``, ``"2b"``)
        """
        # handles None gracefully. Keep in mind that test cases may not provide capability info.
        if value is None:
            return None

        # in case a Capability instance has been given in the first place
        if isinstance(value, Capability):
            return value

        raw = str(value).strip()

        # empty string
        if not raw:
            return None

        # test for code, e.g. "5a"
        code_key = raw.lower().replace(" ", "")
        if code_key in _CODE_TO_CAPABILITY:
            return _CODE_TO_CAPABILITY[code_key]

        # test for enum creation
        try:
            return Capability(raw)
        except ValueError:
            pass

        # test for key representation
        name_key = raw.upper().replace("-", "_").replace(" ", "_")
        if name_key in Capability.__members__:
            return Capability[name_key]

        # test for value representation
        value_key = raw.lower().replace("-", "_").replace(" ", "_")
        for cap in Capability:
            if cap.value == value_key:
                return cap

        # still handle gracefully, but warn user about unusual input
        warnings.warn(f"Unable to convert {raw} to capability.")
        return None

    @classmethod
    def from_code(cls, code: str) -> Capability | None:
        """Resolve short code like ``1a`` to ``Capability``. Handle None gracefully."""
        if code is None:
            return None
        return _CODE_TO_CAPABILITY.get(str(code).strip().lower().replace(" ", ""))


# ── parent mapping (leaf → group) ────────────────────────────────────

_PARENT_MAP: dict[Capability, Capability] = {
    # Object recognition
    Capability.OBJECT_IDENTIFICATION: Capability.OBJECT_RECOGNITION,
    Capability.INSTANCE_MATCHING: Capability.OBJECT_RECOGNITION,
    Capability.OBJECT_ATTRIBUTES: Capability.OBJECT_RECOGNITION,
    Capability.SPATIAL_LOCALIZATION_CAMERA: Capability.OBJECT_RECOGNITION,
    Capability.SPATIAL_LOCALIZATION_SITUS: Capability.OBJECT_RECOGNITION,
    # Temporal grounding
    Capability.TEMPORAL_LOCALIZATION: Capability.TEMPORAL_GROUNDING,
    Capability.DURATION_ESTIMATION: Capability.TEMPORAL_GROUNDING,
    # Aggregation
    Capability.OBJECT_AGGREGATION: Capability.AGGREGATION,
    Capability.EVENT_AGGREGATION: Capability.AGGREGATION,
    # Event understanding
    Capability.FO_INTERACTION_RECOGNITION: Capability.EVENT_UNDERSTANDING,
    Capability.FO_USAGE_PURPOSE: Capability.EVENT_UNDERSTANDING,
    Capability.TEMPORAL_ORDERING: Capability.EVENT_UNDERSTANDING,
    # Complex reasoning
    Capability.FUNCTIONAL_REASONING: Capability.COMPLEX_REASONING,
    Capability.CAUSAL_CONSEQUENCE_REASONING: Capability.COMPLEX_REASONING,
    Capability.MULTI_STEP_REASONING: Capability.COMPLEX_REASONING,
}

_GROUP_LABELS: dict[Capability, str] = {
    Capability.OBJECT_RECOGNITION: "1. Object recognition and identity matching",
    Capability.TEMPORAL_GROUNDING: "2. Temporal grounding",
    Capability.AGGREGATION: "3. Aggregation",
    Capability.EVENT_UNDERSTANDING: "4. Event and procedural understanding",
    Capability.COMPLEX_REASONING: "5. Complex reasoning",
}

_CODE_TO_CAPABILITY: dict[str, Capability] = {
    # Group-level codes
    "1": Capability.OBJECT_RECOGNITION,
    "2": Capability.TEMPORAL_GROUNDING,
    "3": Capability.AGGREGATION,
    "4": Capability.EVENT_UNDERSTANDING,
    "5": Capability.COMPLEX_REASONING,
    # Leaf-level codes
    "1a": Capability.OBJECT_IDENTIFICATION,
    "1b": Capability.INSTANCE_MATCHING,
    "1c": Capability.OBJECT_ATTRIBUTES,
    "1d": Capability.SPATIAL_LOCALIZATION_CAMERA,
    "1e": Capability.SPATIAL_LOCALIZATION_SITUS,
    "2a": Capability.TEMPORAL_LOCALIZATION,
    "2b": Capability.DURATION_ESTIMATION,
    "3a": Capability.OBJECT_AGGREGATION,
    "3b": Capability.EVENT_AGGREGATION,
    "4a": Capability.FO_INTERACTION_RECOGNITION,
    "4b": Capability.FO_USAGE_PURPOSE,
    "4c": Capability.TEMPORAL_ORDERING,
    "5a": Capability.FUNCTIONAL_REASONING,
    "5b": Capability.CAUSAL_CONSEQUENCE_REASONING,
    "5c": Capability.MULTI_STEP_REASONING,
}

_CAPABILITY_TO_CODE: dict[Capability, str] = {
    cap: code for code, cap in _CODE_TO_CAPABILITY.items()
}
