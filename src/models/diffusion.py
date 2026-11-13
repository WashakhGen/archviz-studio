import gc
import math
from typing import Any

import torch
from diffusers import (
    ZImageControlNetModel,
    ZImageTransformer2DModel,
)
from diffusers.pipelines.z_image.pipeline_z_image_controlnet_inpaint import (
    ZImageControlNetInpaintPipeline,
)
from huggingface_hub import hf_hub_download
from PIL import Image

from src.core.buckets import make_buckets, output_size, pick_bucket, to_bucket
from src.core.config import SETTINGS
from src.core.loggings import log_main, time_into_log
from src.core.memory import free_memory, log_mem, sync
from src.models.utils import quantize_int8

CACHE = str(SETTINGS.MODEL_DIR)
DEVICE = SETTINGS.DEVICE


class ZImageLoader:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

        log_main(f"Loading {SETTINGS.ZIMAGE_REPO} + {SETTINGS.CONTROLNET_FILE}")
        self.pipe: ZImageControlNetInpaintPipeline = self._load()

        self.buckets = make_buckets(self.size_for, SETTINGS.BUCKET_BASE)
        self.warmup_sizes = [self.buckets[a] for a in SETTINGS.WARMUP_ASPECTS]
        log_main(f"[buckets] {self.buckets}")

        if SETTINGS.AUTO_COMPILE:
            with time_into_log(self.timings, "compile_register", echo=True):
                self._compile()
            self.warmup()

    @staticmethod
    def size_for(aspect: float, area: int, multiple: int = 16) -> tuple[int, int]:
        """(w, h) with the given aspect, ~area pixels, both sides a multiple of 16."""
        w = math.sqrt(area * aspect)
        h = w / aspect
        return round(w / multiple) * multiple, round(h / multiple) * multiple

    def _load(self) -> ZImageControlNetInpaintPipeline:
        with time_into_log(self.timings, "transformer", echo=True):
            transformer = ZImageTransformer2DModel.from_pretrained(
                SETTINGS.ZIMAGE_REPO,
                subfolder="transformer",
                dtype=torch.bfloat16,
                cache_dir=CACHE,
            )
            transformer = quantize_int8(transformer, self.timings, "transformer")

        with time_into_log(self.timings, "controlnet", echo=True):
            controlnet = ZImageControlNetModel.from_single_file(
                hf_hub_download(
                    SETTINGS.CONTROLNET_REPO,
                    filename=SETTINGS.CONTROLNET_FILE,
                    cache_dir=CACHE,
                ),
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE,
            )
            controlnet = quantize_int8(controlnet, self.timings, "controlnet")

        with time_into_log(self.timings, "pipeline", echo=True):
            # text encoder loads bf16 on CPU here; quantize before anything moves to device
            pipe = ZImageControlNetInpaintPipeline.from_pretrained(
                SETTINGS.ZIMAGE_REPO,
                transformer=transformer,
                controlnet=controlnet,
                dtype=torch.bfloat16,
                cache_dir=CACHE,
            )
            pipe.text_encoder = quantize_int8(
                pipe.text_encoder, self.timings, "text_encoder"
            )
            gc.collect()
            pipe.to(DEVICE)
            pipe.vae.enable_tiling()

        log_mem("zimage loaded")
        return pipe

    def _compile(self) -> None:
        self.pipe.transformer.compile_repeated_blocks(fullgraph=False, dynamic=True)
        log_main("Registered regional compile: ZImageTransformerBlock, dynamic=True")

    def warmup(self) -> None:
        for width, height in self.warmup_sizes:
            grey = Image.new("RGB", (width, height), (128, 128, 128))
            mask = Image.new("L", (width, height), 255)
            with time_into_log(self.timings, f"warmup_{width}x{height}", echo=True):
                self.pipe(
                    "a finished modern house facade",
                    image=grey,
                    mask_image=mask,
                    control_image=grey,
                    width=width,
                    height=height,
                    num_inference_steps=SETTINGS.WARMUP_STEPS,
                    guidance_scale=0.0,
                )
                sync()
            free_memory()
        log_mem("after warmup")

    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image | None = None,
        control: Image.Image | None = None,
        seed: int = 0,
        steps: int | None = None,
        control_scale: float | None = None,
        out_size: tuple[int, int] | None = None,
    ) -> tuple[Image.Image, dict[str, float]]:
        """mask: white = regenerate (None = whole image). control: depth/canny of the ORIGINAL photo.

        Returns the raw generation resized to out_size (default: original size, capped).
        Compositing onto the original is the workflow's job.
        """
        timings: dict[str, float] = {}
        original = image.size
        name, size, stretch = pick_bucket(*original, self.buckets)

        if mask is None:
            mask = Image.new("L", original, 255)
        if control is None:
            control = image
        image, mask = to_bucket(image, mask, size)
        control = control.convert("RGB").resize(size, Image.Resampling.BICUBIC)

        if stretch > SETTINGS.MAX_STRETCH:
            log_main(f"[bucket] {original} -> {name} {size}: {stretch:.0%} stretch")
        else:
            log_main(f"[bucket] {original} -> {name} {size}")

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": image,
            "mask_image": mask,
            "control_image": control,
            "controlnet_conditioning_scale": control_scale or SETTINGS.CONTROL_SCALE,
            "width": size[0],
            "height": size[1],
            "num_inference_steps": steps or SETTINGS.STEPS,
            "guidance_scale": 0.0,
            "generator": torch.Generator("cpu").manual_seed(seed),
        }

        try:
            with time_into_log(timings, f"zimage_seed{seed}", echo=True):
                out = self.pipe(**kwargs).images[0]
                sync()
        finally:
            free_memory()

        target = out_size or output_size(original, SETTINGS.MAX_OUTPUT_SIDE)
        if out.size != target:
            out = out.resize(target, Image.Resampling.LANCZOS)
        return out, timings
