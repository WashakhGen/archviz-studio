import math
from collections.abc import Callable

from PIL import Image

ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
}

SizeFn = Callable[[float, int], tuple[int, int]]  # (aspect_ratio, area) -> (w, h)


def make_buckets(size_fn: SizeFn, base: int = 1024) -> dict[str, tuple[int, int]]:
    """Fibo's trained (w, h) per aspect ratio, ~base*base pixels."""
    return {name: size_fn(a / b, base * base) for name, (a, b) in ASPECTS.items()}


def pick_bucket(
    width: int, height: int, buckets: dict[str, tuple[int, int]]
) -> tuple[str, tuple[int, int], float]:
    """Nearest aspect ratio in log space (so 2:1 vs 1:2 are symmetric). Returns (name, size, stretch)."""
    r = math.log(width / height)
    name, size = min(
        buckets.items(), key=lambda kv: abs(r - math.log(kv[1][0] / kv[1][1]))
    )
    stretch = abs(math.exp(r - math.log(size[0] / size[1])) - 1)
    return name, size, stretch


def to_bucket(image: Image.Image, mask: Image.Image | None, size: tuple[int, int]):
    image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    if mask is not None:
        mask = mask.convert("L").resize(
            size, Image.Resampling.NEAREST
        )  # NEAREST keeps mask hard 0/255
    return image, mask


def output_size(original: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Original aspect + size, capped (upscaling 1 MP -> 12 MP with Lanczos only gets blurry)."""
    w, h = original
    s = min(1.0, max_side / max(w, h))
    return round(w * s), round(h * s)
