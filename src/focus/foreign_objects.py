"""Foreign object type definitions for the FOCUS dataset(s) and challenge.

This module defines the :class:`FOType` descriptor and ships predefined instances for the
nine core object classes used in the dataset and challenge.

NOTE: During the test phase additional foreign object types are possible. They will be provided as
part of the available meta data.

Usage::

    from focus import SPONGE, CLIP, MESH, FOType, FO_DEFINITION

    SPONGE.name                    # "Sponge"
    SPONGE.description             # Full description
    FOType.by_name("Clip")         # Look up by display name
    FOType.by_slug("mesh")         # Look up by machine-friendly ID
    FOType.all_defaults()          # tuple of all 9 predefined types
    FOType.names()                 # display names
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

FO_DEFINITIONS_FILE: Path = Path(__file__).parent / "assets" / "FO_definitions.txt"
"""Path to the plain-text file describing all foreign object classes.

Participants can read this file at inference time to provide their model with
grounding definitions::

    from focus import FO_DEFINITIONS_FILE
    definitions = FO_DEFINITIONS_FILE.read_text()
"""

FO_DEFINITION = """
A foreign object (FO) is any object fully introduced into the patient's body
cavity during surgery that must be retrieved or accounted for. Importantly,
standard surgical instruments that remain connected to the external environment
(e.g., graspers, scissors, trocars, staplers, cameras) are not considered foreign
objects. Furthermore, we exclude detachable parts of surgical instruments,
particularly anvil components of staplers.
"""


class FOType:
    """Descriptor for a foreign-object class in the FOCUS challenge.

    Parameters
    ----------
    name : str
        Human-readable display name (e.g., ``"Sponge"``).
    description : str
        Detailed textual description of the object type and its surgical context.
    slug : str, optional
        Machine-friendly identifier used for serialisation and lookups.
        If not provided, defaults to a lowercased, underscore-separated version of *name*.

    Attributes
    ----------
    name : str
        Display name of the foreign object type.
    slug : str
        Machine-friendly identifier.
    description : str
        Detailed description of the object.

    Examples
    --------
    >>> from focus import SPONGE
    >>> SPONGE.name
    'Sponge'
    >>> SPONGE.slug
    'sponge'
    >>> FOType.by_name("Sponge") == SPONGE
    True
    """

    _registry: ClassVar[dict[str, FOType]] = {}

    def __init__(self, name: str, description: str, slug: str | None = None) -> None:
        self.name = name
        self.slug = slug or name.lower().replace(" ", "_")
        self.description = description
        FOType._registry[self.slug] = self

    # ── serialisation ────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a detailed string representation of the FOType."""
        return f"FOType({self.name!r})"

    def __str__(self) -> str:
        """Return the display name of the foreign object."""
        return self.name

    def __eq__(self, other: object) -> bool:
        """Compare two FOType instances by their slug."""
        if isinstance(other, FOType):
            return self.slug == other.slug
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on the slug for use in sets and dicts."""
        return hash(self.slug)

    # ── registry helpers ─────────────────────────────────────────────

    @classmethod
    def by_slug(cls, slug: str) -> FOType:
        """Look up a predefined FOType by its machine-friendly slug.

        Parameters
        ----------
        slug : str
            The slug identifier (e.g., ``"sponge"``, ``"clip"``).

        Returns
        -------
        FOType
            The matching foreign object type.

        Raises
        ------
        ValueError
            If the slug is not registered.
        """
        try:
            return cls._registry[slug]
        except KeyError:
            raise ValueError(
                f"Unknown FO slug {slug!r}. Available: {', '.join(sorted(cls._registry))}"
            ) from None

    @classmethod
    def by_name(cls, name: str) -> FOType:
        """Look up a predefined FOType by its display name (case-insensitive).

        Parameters
        ----------
        name : str
            The display name (e.g., ``"Sponge"``, ``"CLIP"``).

        Returns
        -------
        FOType
            The matching foreign object type.

        Raises
        ------
        ValueError
            If no matching name is found.
        """
        lower = name.lower()
        for fo in cls._registry.values():
            if fo.name.lower() == lower:
                return fo
        raise ValueError(
            f"Unknown FO name {name!r}. "
            f"Available: {', '.join(fo.name for fo in cls._registry.values())}"
        )

    @classmethod
    def all_defaults(cls) -> tuple[FOType, ...]:
        """Return all registered predefined FOType instances.

        Returns
        -------
        tuple[FOType, ...]
            Tuple of all 9 core foreign object types.
        """
        return tuple(cls._registry.values())

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return display names of all registered FOType instances.

        Returns
        -------
        tuple[str, ...]
            Tuple of human-readable names of all registered types.
        """
        return tuple(fo.name for fo in cls._registry.values())


# ── predefined foreign-object types ──────────────────────────────────

SPONGE = FOType(
    name="Sponge",
    description=(
        "A soft, absorbent material used to soak up fluids. They are typically white when fresh "
        "and can become reddish-brown when saturated with blood."
    ),
)

CLIP = FOType(
    name="Clip",
    description=(
        "A small metal or polymer device used to seal vessels or ducts. May potentially remain in "
        "the body. Clips only count as foreign objects once placed in the abdomen. They do not "
        "count as foreign objects while loaded within the clip applier instrument."
    ),
)

SPECIMEN_BAG = FOType(
    name="Specimen Bag",
    description=(
        "A sterile pouch used to collect and retrieve resected tissue or organs from the body "
        "cavity. Only consider the pouch itself as foreign object and ignore the string attached "
        "to it."
    ),
)

SILICONE_LOOP = FOType(
    name="Silicone Loop",
    description=(
        "A soft and flexible, typically white band used to encircle and control blood vessels or "
        "structures for isolation and traction."
    ),
)

EXTERNAL_DRAIN = FOType(
    name="External Drain",
    description=(
        "A clear or fluid-filled tube used to evacuate fluids from the surgical site to the "
        "outside of the body. Its tip may be temporarily visible within the surgical field "
        "during placement or adjustment."
    ),
)

NEEDLE = FOType(
    name="Needle",
    description=(
        "A sharp, pointed metal instrument, straight or curved, used for placing sutures. If only "
        "the string of the needle is visible, it does not count as “the needle is visible”. Only "
        "if the needle itself is visible in the video."
    ),
)

GALLSTONE = FOType(
    name="Gallstone",
    description=(
        "Calcified concretion originating from the gallbladder with a roundish, white/yellow "
        "appearance. Gallstones that are within a specimen bag count as retrieved. Specimen bags "
        "are annotated separately."
    ),
)

SPECIMEN = FOType(
    name="Specimen",
    description=(
        "Excised biological tissue (e.g. appendix, resected bowel segment) that must be retrieved "
        "from the body cavity before procedure completion. As soon as every connection to other "
        "body anatomy is cut, the cut off tissue/organ counts as specimen.  This excludes fat or "
        "blood. Specimens that are within a specimen bag count as retrieved. Specimen bags are "
        "annotated separately."
    ),
)

MESH = FOType(
    name="Mesh",
    description=(
        "Mesh is a screen-like patch that surgeons use to reinforce a weak area in the "
        "body's muscle wall. This differs from a sponge which appears more like a tightly "
        "woven cloth. It is an implantable foreign body that is not removed at the end of "
        "the surgery."
    ),
)
