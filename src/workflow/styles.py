from dataclasses import dataclass
from typing import Literal

Mode = Literal["render", "keep_context"]


@dataclass(frozen=True)
class ModeDefaults:
    steps: int
    control_scale: float


# render: whole scene regenerated (sky, lighting, surroundings) -> looser control, more steps
# keep_context: only house + ground regenerated, real street kept -> tighter control
MODE_DEFAULTS: dict[Mode, ModeDefaults] = {
    "render": ModeDefaults(steps=12, control_scale=0.65),
    "keep_context": ModeDefaults(steps=9, control_scale=0.8),
}

_STRUCTURE = (
    "Keep the exact building shape, floors, slab edges, roof lines and every window and door "
    "opening in the same position. "
)

STYLES: dict[str, str] = {
    "modern_dusk": (
        "Architectural visualisation of the same building as a luxury modern home at blue hour, "
        "front elevation. "
        + _STRUCTURE
        + "Charcoal and warm grey textured plaster, black "
        "marble cladding panels with white veins, slim horizontal black metal railings along "
        "the open slab edges, dark horizontal slatted main gate, warm recessed downlights under "
        "every overhang, thin warm LED strip lines, wall sconces, large glass windows with warm "
        "interior glow and sheer curtains. Lush landscaping: trees, flowering shrubs, ornamental "
        "grasses, manicured lawn and paved driveway, soft twilight sky. Photorealistic 3D "
        "render, high detail, V-Ray quality."
    ),
    "modern_white_day": (
        "The same building as a finished, high-end modern home, photographed after completion. "
        + _STRUCTURE
        + "Bright pure-white smooth plaster walls, warm teak wood cladding on the "
        "recessed areas, dark grey natural stone on the low boundary walls, crisp white "
        "projecting slab edges with warm recessed LED downlights, slim black aluminium windows "
        "with large clear reflective glass. Clean finished site: grey porcelain paving, "
        "manicured lawn, ornamental grasses and small trees, no construction debris. "
        "Golden-hour light, photorealistic architectural photography."
    ),
    "contemporary_stone": (
        "Architectural visualisation of the same building as a contemporary villa in soft "
        "daylight, front elevation. "
        + _STRUCTURE
        + "Beige limestone cladding, light sand "
        "textured plaster, bronze-framed windows with clear glass, glass balustrades on the open "
        "slab edges, wooden louvres on the top floor, warm wooden main door and gate. Mature "
        "olive trees, lavender beds, gravel and stone paving, clear blue sky. Photorealistic 3D "
        "render, high detail."
    ),
}

DEFAULT_STYLE = "modern_dusk"


def style_prompt(style: str, extra: str = "") -> str:
    """Preset prompt plus optional user wishes (e.g. 'red brick on the ground floor')."""
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}. Available: {', '.join(STYLES)}")
    return f"{STYLES[style]} {extra}".strip()
