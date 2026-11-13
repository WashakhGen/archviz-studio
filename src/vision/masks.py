from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.models.segmentaion import Detection

# Text prompts the facade workflow asks SAM 3 for.
GROUND_PROMPTS = ["ground", "driveway", "pavement"]
FACADE_PROMPTS = ["house", "building", "window", "window opening", *GROUND_PROMPTS]

MIN_TARGET_AREA = 0.02  # target house must cover >= 2% of the image
OPENING_INSIDE = 0.5  # a window counts if >= 50% of it lies on the target house
NEIGHBOUR_MIN_SCORE = 0.45
EDGE = 0.01  # box within 1% of the left/right border = cut off by the frame


def to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size = (w, h). NEAREST keeps it hard."""
    if mask.shape[::-1] == size:
        return mask
    return np.array(to_image(mask).resize(size, Image.Resampling.NEAREST)) > 127


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    return np.array(to_image(mask).filter(ImageFilter.MaxFilter(2 * px + 1))) > 127


def union(dets: list[Detection], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for d in dets:
        out |= d.mask
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill regions fully enclosed by the mask (debris, hoses, a scooter lying on the ground)."""
    # .copy(): fromarray shares numpy memory and floodfill then silently writes nothing
    img = to_image(~mask).copy()  # 255 = outside the mask
    w, h = img.size
    border = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)]
    border += [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
    for xy in border:
        if img.getpixel(xy) == 255:
            ImageDraw.floodfill(img, xy, 128)  # everything reachable from the border
    return mask | (np.array(img) == 255)


def composite(
    base: Image.Image, edited: Image.Image, mask: Image.Image, feather: int = 6
) -> Image.Image:
    """Keep base pixels outside the mask, edited pixels inside (soft edge)."""
    edited = edited.resize(base.size, Image.Resampling.LANCZOS)
    m = mask.convert("L").resize(base.size, Image.Resampling.NEAREST)
    m = m.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(edited, base.convert("RGB"), m)


def pick_target(
    candidates: list[Detection], shape: tuple[int, int]
) -> Detection | None:
    """The house the photo is about: confident, big, and horizontally central."""
    h, w = shape
    best, best_score = None, 0.0
    for d in candidates:
        area = float(d.mask.mean())
        if area < MIN_TARGET_AREA:
            continue
        cx = (d.box[0] + d.box[2]) / 2 / w
        centrality = 1.0 - min(1.0, abs(cx - 0.5) * 2)  # 1 at centre, 0 at the edge
        score = d.score * area**0.5 * (0.5 + 0.5 * centrality)
        if score > best_score:
            best, best_score = d, score
    return best


def remove_neighbours(
    target: Detection, candidates: list[Detection], shape: tuple[int, int]
) -> np.ndarray:
    """SAM sometimes merges the target with an adjacent house. If another confident
    house/building instance lies mostly inside the target AND is cut off by the left
    or right frame edge, it is a neighbour: cut it out."""
    w = shape[1]
    out = target.mask.copy()
    target_area = target.mask.sum()
    for d in candidates:
        if d is target or d.score < NEIGHBOUR_MIN_SCORE:
            continue
        area = d.mask.sum()
        if not area or area > 0.5 * target_area:
            continue
        inside = (d.mask & target.mask).sum() / area
        at_edge = d.box[0] <= EDGE * w or d.box[2] >= (1 - EDGE) * w
        if inside >= 0.5 and at_edge:
            out &= ~d.mask
    return out


@dataclass
class FacadeMasks:
    target: np.ndarray  # the house (incl. its openings)
    openings: np.ndarray  # windows / doors / openings on the house
    facade: np.ndarray  # house minus openings
    ground: np.ndarray  # site in front, minus the house
    edit: np.ndarray  # what gets regenerated (house + ground, dilated)

    def image(self, name: str) -> Image.Image:
        return to_image(getattr(self, name))

    def overlay(self, image: Image.Image) -> Image.Image:
        """Debug view: facade red, openings blue, ground green."""
        base = np.array(image.convert("RGB").resize(self.edit.shape[::-1])).astype(
            float
        )
        for mask, color in (
            (self.facade, (255, 0, 0)),
            (self.openings, (0, 90, 255)),
            (self.ground, (0, 200, 0)),
        ):
            base[mask] = base[mask] * 0.45 + np.array(color) * 0.55
        return Image.fromarray(base.astype(np.uint8))


def build_facade_masks(
    dets: dict[str, list[Detection]],
    size: tuple[int, int],
    dilate_px: int = 6,
) -> FacadeMasks | None:
    """dets come from SAM 3 at any resolution; masks are returned at size = (w, h).

    Returns None when no target building is found.
    """
    all_masks = [d.mask for ds in dets.values() for d in ds]
    if not all_masks:
        return None
    shape = all_masks[0].shape  # (h, w) of the SAM input

    candidates = dets.get("house", []) + dets.get("building", [])
    target_det = pick_target(candidates, shape)
    if target_det is None:
        return None
    target = remove_neighbours(target_det, candidates, shape)

    # openings that sit on the target house (drops the neighbours' windows)
    opening_dets = dets.get("window", []) + dets.get("window opening", [])
    openings = np.zeros(shape, dtype=bool)
    target_grown = dilate(target, 4)  # once, not per detection
    for d in opening_dets:
        area = d.mask.sum()
        if area and (d.mask & target_grown).sum() / area >= OPENING_INSIDE:
            openings |= d.mask
    target |= openings

    # objects lying on the ground (debris, hoses, vehicles) are holes in it: fill them
    ground_dets = [d for p in GROUND_PROMPTS for d in dets.get(p, [])]
    ground = fill_holes(union(ground_dets, shape) | target) & ~target
    facade = target & ~openings
    edit = fill_holes(
        dilate(target | ground, dilate_px)
    )  # dilation closes thin gaps (hoses)

    return FacadeMasks(
        target=resize_mask(target, size),
        openings=resize_mask(openings, size),
        facade=resize_mask(facade, size),
        ground=resize_mask(ground, size),
        edit=resize_mask(edit, size),
    )
