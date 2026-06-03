"""
fit_types.py

Definitions of model-type enums, FitKey, and label<->FitKey conversion.

Label grammar (current conventions)
------------------------------------
<base> ::= "PSPL" | "FSPL" | "2L1S" | ...

For point-lens bases the modifier sequence is:

    "<base> static"
        parallax_branch = NONE, lens_orb_motion = NONE

    "<base> par <branch>"
        parallax_branch = <branch>, lens_orb_motion = NONE

    "<base> <motion> par <branch>"
        parallax_branch = <branch>, lens_orb_motion = <motion>

For binary-lens bases an optional <binary_model_type> token may appear
*immediately after* the base and before any motion/parallax suffix:

    "<base> <binary_model_type> static"
    "<base> <binary_model_type> par <branch>"
    "<base> <binary_model_type> <motion>"
    "<base> <binary_model_type> <motion> par <branch>"

As a convenience, "<base>" alone is treated as "<base> static".

Valid binary_model_type values
-------------------------------
    Wide, Close, CloseUpper, CloseLower
    Wide_alt, Close_alt, CloseUpper_alt, CloseLower_alt

Examples
--------
    "PSPL static"
    "FSPL static"
    "PSPL par u0+"
    "FSPL par u0-"
    "2L1S static"
    "2L1S Wide static"
    "2L1S Close par u0+"
    "2L1S Wide 2Dorb"
    "2L1S Close kep par u0--"
    "2L1S par u0+"
    "2L1S 2Dorb par u0+"
    "2L1S kep par u0--"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional


# ============================================================================
# Enums
# ============================================================================


class LensType(Enum):
    POINT = "point"
    BINARY = "binary"


class SourceType(Enum):
    POINT = "point"
    FINITE = "finite"


class ParallaxBranch(Enum):
    NONE = "none"
    U0_PLUS = "u0+"
    U0_MINUS = "u0-"
    U0_PP = "u0++"
    U0_MM = "u0--"
    U0_PM = "u0+-"
    U0_MP = "u0-+"


class LensOrbMotion(Enum):
    NONE = "none"
    ORB_2D = "2Dorb"
    KEPLER = "kep"


# ============================================================================
# Binary model type constant
# Must be defined before FitKey so that __post_init__ can reference it.
# ============================================================================

#: Eight recognised binary_model_type strings:
#: four base topology labels plus their ``_alt`` (degenerate-solution) variants.
BINARY_MODEL_TYPES: FrozenSet[str] = frozenset([
    "Wide",
    "Close",
    "CloseUpper",
    "CloseLower",
    "Wide_alt",
    "Close_alt",
    "CloseUpper_alt",
    "CloseLower_alt",
])


# ============================================================================
# FitKey
# ============================================================================


@dataclass(frozen=True)
class FitKey:
    """Immutable key that uniquely identifies a microlensing model fit.

    Parameters
    ----------
    lens_type : LensType
    source_type : SourceType
    parallax_branch : ParallaxBranch
    lens_orb_motion : LensOrbMotion
    locations_used : str, optional
        Observatory/satellite combination string, e.g. ``'ground+Spitzer'``.
    binary_model_type : str, optional
        Topological classification of the binary-lens solution.
        Must be one of the values in :data:`BINARY_MODEL_TYPES`.
        May only be set when ``lens_type == LensType.BINARY``;
        ``None`` is always valid regardless of lens type.
    """

    lens_type: LensType
    source_type: SourceType
    parallax_branch: ParallaxBranch
    lens_orb_motion: LensOrbMotion
    locations_used: Optional[str] = None
    binary_model_type: Optional[str] = None

    def __post_init__(self) -> None:
        # --- Existing constraint ----------------------------------------
        # Point lenses cannot have orbital motion.
        if (
            self.lens_type == LensType.POINT
            and self.lens_orb_motion is not LensOrbMotion.NONE
        ):
            raise ValueError(
                f"Point lenses must have lens_orb_motion == NONE; "
                f"got lens_orb_motion={self.lens_orb_motion!r}"
            )

        # --- New constraints for binary_model_type ----------------------
        if self.binary_model_type is None:
            return

        # Non-None binary_model_type is only meaningful for binary lenses.
        if self.lens_type != LensType.BINARY:
            raise ValueError(
                f"binary_model_type may only be set for LensType.BINARY; "
                f"got lens_type={self.lens_type!r}"
            )

        # Empty string is not a valid value.
        if self.binary_model_type == "":
            raise ValueError("binary_model_type must not be an empty string")

        # Value must be drawn from the recognised set.
        if self.binary_model_type not in BINARY_MODEL_TYPES:
            raise ValueError(
                f"Invalid binary_model_type {self.binary_model_type!r}. "
                f"Must be one of: {sorted(BINARY_MODEL_TYPES)}"
            )


# ============================================================================
# Tag mapping tables
# ============================================================================

# Base model tag → source_type.
# NOTE: "2L1S" maps to SourceType.POINT (binary lens, single point source).
#       A future "2L1S_fs" or similar tag will cover SourceType.FINITE.
SOURCE_TAGS: Dict[str, SourceType] = {
    "PSPL": SourceType.POINT,
    "FSPL": SourceType.FINITE,
    "2L1S": SourceType.POINT,
}

# Base model tag → lens_type.
LENS_TAGS: Dict[str, LensType] = {
    "PSPL": LensType.POINT,
    "FSPL": LensType.POINT,
    "2L1S": LensType.BINARY,
}

PARALLAX_BRANCH_TAGS: Dict[str, ParallaxBranch] = {
    "none": ParallaxBranch.NONE,
    "u0+":  ParallaxBranch.U0_PLUS,
    "u0-":  ParallaxBranch.U0_MINUS,
    "u0++": ParallaxBranch.U0_PP,
    "u0--": ParallaxBranch.U0_MM,
    "u0+-": ParallaxBranch.U0_PM,
    "u0-+": ParallaxBranch.U0_MP,
}

LENS_MOTION_TAGS: Dict[str, LensOrbMotion] = {
    "none":  LensOrbMotion.NONE,
    "2Dorb": LensOrbMotion.ORB_2D,
    "kep":   LensOrbMotion.KEPLER,
}


# ============================================================================
# Label <-> FitKey conversion
# ============================================================================


def label_to_model_key(label: str) -> FitKey:
    """Parse a human-readable label into a :class:`FitKey`.

    Parameters
    ----------
    label : str

    Returns
    -------
    FitKey

    Raises
    ------
    ValueError
        If the label cannot be parsed or references unknown tags.
    """
    tokens = label.strip().split()
    if not tokens:
        raise ValueError("Empty model label")

    base = tokens[0]

    # Resolve the base tag first; we need lens_type before deciding whether
    # to look for a binary_model_type token in the next position.
    lens_type = LENS_TAGS.get(base)
    source_type = SOURCE_TAGS.get(base)

    if lens_type is None or source_type is None:
        raise ValueError(
            f"Unknown base model tag in label {label!r}: {base!r}"
        )

    # Defaults
    parallax_branch = ParallaxBranch.NONE
    lens_orb_motion = LensOrbMotion.NONE
    binary_model_type = None

    tail = list(tokens[1:])

    # ------------------------------------------------------------------
    # Optionally consume the binary_model_type token.
    # This slot is only available for binary-lens bases.  For point-lens
    # bases any unrecognised token in this position is caught below.
    # ------------------------------------------------------------------
    if tail and lens_type == LensType.BINARY and tail[0] in BINARY_MODEL_TYPES:
        binary_model_type = tail[0]
        tail = tail[1:]

    # ------------------------------------------------------------------
    # Parse the remaining modifier sequence.
    # ------------------------------------------------------------------
    if not tail:
        # Bare "<base>" or "<base> <bmt>" – treat as static.
        pass

    elif tail == ["static"]:
        # Explicit "… static" suffix.
        pass

    elif tail[0] == "par":
        # "… par <branch>" – parallax, no orbital motion.
        if len(tail) != 2:
            raise ValueError(
                f"Cannot parse label {label!r}: "
                f"'par' must be followed by exactly one branch token, "
                f"got {tail!r}"
            )
        branch_token = tail[1]
        parallax_branch = PARALLAX_BRANCH_TAGS.get(branch_token)
        if parallax_branch is None:
            raise ValueError(
                f"Unknown parallax branch token in label {label!r}: "
                f"{branch_token!r}"
            )

    elif tail[0] in LENS_MOTION_TAGS:
        # "… <motion>" or "… <motion> par <branch>"
        lens_motion_token = tail[0]
        lens_orb_motion = LENS_MOTION_TAGS[lens_motion_token]

        if len(tail) == 1:
            # Motion only; no parallax.
            pass

        elif len(tail) == 3 and tail[1] == "par":
            # Motion + parallax.
            branch_token = tail[2]
            parallax_branch = PARALLAX_BRANCH_TAGS.get(branch_token)
            if parallax_branch is None:
                raise ValueError(
                    f"Unknown parallax branch token in label {label!r}: "
                    f"{branch_token!r}"
                )

        else:
            raise ValueError(
                f"Cannot parse label {label!r}: unexpected tokens after "
                f"motion token {lens_motion_token!r}: {tail[1:]!r}"
            )

    else:
        raise ValueError(
            f"Cannot parse label {label!r}: unexpected token {tail[0]!r}"
        )

    return FitKey(
        lens_type=lens_type,
        source_type=source_type,
        parallax_branch=parallax_branch,
        lens_orb_motion=lens_orb_motion,
        binary_model_type=binary_model_type,
    )


def model_key_to_label(key: FitKey) -> str:
    """Map a :class:`FitKey` back to a human-readable label.

    Parameters
    ----------
    key : FitKey

    Returns
    -------
    str
    """
    # Find a base tag whose (lens_type, source_type) matches the key.
    base = None
    for candidate, lt in LENS_TAGS.items():
        if lt == key.lens_type and SOURCE_TAGS[candidate] == key.source_type:
            base = candidate
            break

    if base is None:
        if key.binary_model_type is not None:
            base = '2L1S ' + key.binary_model_type

    assert base is not None, f"No base label mapping for FitKey {key!r}"

    # Optional binary_model_type token placed immediately after the base.
    bmt_part = f" {key.binary_model_type}" if key.binary_model_type is not None else ""

    # ------------------------------------------------------------------
    # Static case: neither parallax nor orbital motion.
    # ------------------------------------------------------------------
    if (
        key.parallax_branch == ParallaxBranch.NONE
        and key.lens_orb_motion == LensOrbMotion.NONE
    ):
        return f"{base}{bmt_part} static"

    # ------------------------------------------------------------------
    # Resolve motion and branch labels.
    # ------------------------------------------------------------------
    lens_motion_label = None
    for lbl, motion in LENS_MOTION_TAGS.items():
        if motion == key.lens_orb_motion:
            lens_motion_label = lbl
            break

    assert lens_motion_label is not None, (
        f"No lens motion label mapping for LensOrbMotion {key.lens_orb_motion!r}"
    )

    branch_label = None
    for lbl, br in PARALLAX_BRANCH_TAGS.items():
        if br == key.parallax_branch:
            branch_label = lbl
            break

    assert branch_label is not None, (
        f"No parallax branch label mapping for {key.parallax_branch!r}"
    )

    # ------------------------------------------------------------------
    # Assemble label from parts.
    # ------------------------------------------------------------------
    if lens_motion_label == "none":
        # Parallax only: "<base> [bmt] par <branch>"
        return f"{base}{bmt_part} par {branch_label}"

    if branch_label == "none":
        # Motion only: "<base> [bmt] <motion>"
        return f"{base}{bmt_part} {lens_motion_label}"

    # Motion + parallax: "<base> [bmt] <motion> par <branch>"
    return f"{base}{bmt_part} {lens_motion_label} par {branch_label}"
