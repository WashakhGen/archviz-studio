from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Sam3Model,
    Sam3Processor,
)

from src.core.config import SETTINGS
from src.core.loggings import log_main, time_into_log
from src.core.memory import free_memory, log_mem, sync

CACHE = str(SETTINGS.MODEL_DIR)
DEVICE = SETTINGS.DEVICE


@dataclass
class Detection:
    mask: np.ndarray  # bool (H, W) at the input image size
    score: float
    box: tuple[float, ...]  # x0, y0, x1, y1 in pixels


class SAM3Loader:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.dtype = torch.float16 if DEVICE != "cpu" else torch.float32

        log_main(f"Loading {SETTINGS.SAM3_REPO}")
        with time_into_log(self.timings, "sam3", echo=True):
            self.processor = Sam3Processor.from_pretrained(
                SETTINGS.SAM3_REPO, cache_dir=CACHE
            )
            self.model = (
                Sam3Model.from_pretrained(
                    SETTINGS.SAM3_REPO, dtype=self.dtype, cache_dir=CACHE
                )
                .to(DEVICE)
                .eval()
            )
        log_mem("sam3 loaded")

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        prompts: list[str],
        threshold: float | None = None,
        mask_threshold: float = 0.5,
    ) -> tuple[dict[str, list[Detection]], dict[str, float]]:
        """One vision pass, then one cheap decoder pass per text prompt.

        Pass a downscaled image (e.g. 1/2) for speed; masks come back at that size.
        """
        timings: dict[str, float] = {}
        threshold = threshold or SETTINGS.SAM3_THRESHOLD
        image = image.convert("RGB")
        target = [image.size[::-1]]  # [(h, w)]
        results: dict[str, list[Detection]] = {}

        try:
            with time_into_log(timings, "sam3_vision", echo=True):
                pixel = self.processor(images=image, return_tensors="pt")
                vision = self.model.get_vision_features(
                    pixel_values=pixel["pixel_values"].to(DEVICE, self.dtype)
                )

            with time_into_log(timings, f"sam3_{len(prompts)}_prompts", echo=True):
                for prompt in prompts:
                    text = self.processor(text=prompt, return_tensors="pt").to(DEVICE)
                    out = self.model(vision_embeds=vision, **text)
                    r = self.processor.post_process_instance_segmentation(
                        out,
                        threshold=threshold,
                        mask_threshold=mask_threshold,
                        target_sizes=target,
                    )[0]
                    results[prompt] = [
                        Detection(
                            mask=m.cpu().numpy().astype(bool),
                            score=float(s),
                            box=tuple(float(v) for v in b),
                        )
                        for m, s, b in zip(r["masks"], r["scores"], r["boxes"])
                    ]
                sync()
        finally:
            free_memory()

        return results, timings


class DepthLoader:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        log_main(f"Loading {SETTINGS.DEPTH_REPO}")
        with time_into_log(self.timings, "depth", echo=True):
            self.processor = AutoImageProcessor.from_pretrained(
                SETTINGS.DEPTH_REPO, cache_dir=CACHE
            )
            self.model = (
                AutoModelForDepthEstimation.from_pretrained(
                    SETTINGS.DEPTH_REPO, cache_dir=CACHE
                )
                .to(DEVICE)
                .eval()
            )

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> tuple[Image.Image, dict[str, float]]:
        """Relative depth at the input size, 0-255, near = bright (ControlNet convention)."""
        timings: dict[str, float] = {}
        image = image.convert("RGB")
        try:
            with time_into_log(timings, "depth", echo=True):
                inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
                pred = self.model(**inputs).predicted_depth.unsqueeze(1)
                pred = torch.nn.functional.interpolate(
                    pred, size=image.size[::-1], mode="bicubic", align_corners=False
                )[0, 0]
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                arr = (pred.clamp(0, 1) * 255).byte().cpu().numpy()
                sync()
        finally:
            free_memory()
        return Image.fromarray(arr).convert("RGB"), timings
