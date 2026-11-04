import gc
from typing import Any

import torch
from diffusers.models.transformers.transformer_bria_fibo2 import (
    BriaFibo2Transformer2DModel,
)
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import BriaFibo2Pipeline
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

from src.core.buckets import make_buckets, output_size, pick_bucket, to_bucket
from src.core.config import SETTINGS
from src.core.loggings import log_main, time_into_log
from src.core.memory import free_memory, log_mem, sync

SENSITIVE_LAYERS: list[str] = []


class ModelLoader:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

        self.fonts: list[str] = []
        self.warmup_steps = SETTINGS.WARMUP_STEPS

        self._common: dict[str, Any] = {
            "pretrained_model_name_or_path": SETTINGS.REPO,
            "cache_dir": str(SETTINGS.MODEL_DIR),
        }

        log_main(f"Loading {SETTINGS.REPO} model")
        self.pipe: BriaFibo2Pipeline = self._load()

        self.buckets = make_buckets(
            self.pipe.image_processor.get_image_size, SETTINGS.BUCKET_BASE
        )
        self.warmup_sizes = [self.buckets[a] for a in SETTINGS.WARMUP_ASPECTS]
        log_main(f"[buckets] {self.buckets}")

        self.fonts = self._load_fonts()
        if SETTINGS.AUTO_COMPILE:
            with time_into_log(self.timings, "compile_register", echo=True):
                self._compile()
        self.warmup()

    def _compile(self) -> None:
        self.pipe.transformer.compile_repeated_blocks(fullgraph=False, dynamic=True)
        log_main("Registered regional compile: BriaFibo2TransformerBlock, dynamic=True")

    def warmup(self) -> None:
        for width, height in self.warmup_sizes:
            image = Image.new("RGB", (width, height), (128, 128, 128))
            with time_into_log(self.timings, f"warmup_{width}x{height}", echo=True):
                self.pipe(
                    prompt="finish this house with a white plaster facade",
                    image=image,
                    num_inference_steps=self.warmup_steps,
                    guidance_scale=1.0,
                    fonts=self.fonts,
                )
                sync()
            free_memory()
        log_mem("after warmup")

    def _load_fonts(self) -> list[str]:
        return [
            hf_hub_download(
                SETTINGS.REPO, n, subfolder="fonts", cache_dir=str(SETTINGS.MODEL_DIR)
            )
            for n in ("Aileron-Regular.otf", "DejaVuSans.ttf")
        ]

    def _quantize(self, model, name: str):
        """bf16 -> int8 on CPU, before moving to device. Never hold bf16 on CPU and GPU at once."""
        if SETTINGS.QUANT == "none":
            return model
        from optimum.quanto import freeze, qint8, quantize

        with time_into_log(self.timings, f"quantize_{name}", echo=True):
            quantize(model, weights=qint8, exclude=SENSITIVE_LAYERS or None)
            freeze(model)
            gc.collect()
        return model

    def _load(self):
        with time_into_log(self.timings, "Transformer", echo=True):
            transformer = BriaFibo2Transformer2DModel.from_pretrained(
                **self._common, subfolder="transformer", torch_dtype=torch.bfloat16
            )
            transformer = self._quantize(transformer, "transformer").to(SETTINGS.DEVICE)
            gc.collect()
            log_mem("transformer")

        with time_into_log(self.timings, "Text Encoder", echo=True):
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                **self._common, subfolder="text_encoder", dtype=torch.bfloat16
            )
            text_encoder = self._quantize(text_encoder, "text_encoder").to(
                SETTINGS.DEVICE
            )
            gc.collect()
            log_mem("text_encoder")

        pipe = BriaFibo2Pipeline.from_pretrained(
            **self._common,
            transformer=transformer,
            text_encoder=text_encoder,
            dtype=torch.bfloat16,
        ).to(SETTINGS.DEVICE)

        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        return pipe

    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image | None = None,
        seed: int = 0,
        steps: int = 4,
        out_size: tuple[int, int] | None = None,
    ) -> tuple[Image.Image, dict[str, float]]:
        timings: dict[str, float] = {}
        original = image.size
        name, size, stretch = pick_bucket(*original, self.buckets)
        image, mask = to_bucket(image, mask, size)

        if stretch > SETTINGS.MAX_STRETCH:
            log_main(
                f"[bucket] {original} -> {name} {size}: {stretch:.0%} stretch, consider cropping upstream"
            )
        else:
            log_main(f"[bucket] {original} -> {name} {size}")

        kwargs = {
            "prompt": prompt,
            "image": image,
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
            "fonts": self.fonts,
            "generator": torch.Generator("cpu").manual_seed(seed),
            "height": size[1],
            "width": size[0],
        }

        if mask is not None:
            kwargs["mask"] = mask

        try:
            with time_into_log(timings, f"Inferencing - Seed:{seed}", echo=True):
                out = self.pipe(**kwargs).images[0]
                sync()
        finally:
            free_memory()

        target = out_size or output_size(original, SETTINGS.MAX_OUTPUT_SIDE)

        if out.size != target:
            out = out.resize(target, Image.Resampling.LANCZOS)
        return out, timings
