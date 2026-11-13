from dataclasses import dataclass, field

from PIL import Image, ImageOps

from src.core.config import SETTINGS
from src.core.loggings import log_main, time_into_log
from src.managers.base import SAM3, BaseManager
from src.models.memory_manager import MemoryManager
from src.vision.masks import FACADE_PROMPTS, FacadeMasks, build_facade_masks, composite
from src.workflow.styles import DEFAULT_STYLE, MODE_DEFAULTS, Mode, style_prompt

SAM_MAX_SIDE = 2048  # SAM input is downscaled to this; masks are upscaled back


@dataclass
class FacadeTask:
    image: Image.Image
    style: str = DEFAULT_STYLE  # key in src/workflow/styles.py STYLES
    mode: Mode = (
        "render"  # "render" = whole scene, "keep_context" = house + ground only
    )
    extra: str = ""  # user wishes appended to the style prompt
    prompt: str | None = None  # full override; ignores style + extra
    seed: int = 0
    steps: int | None = None  # None = mode default
    control_scale: float | None = None  # None = mode default


@dataclass
class FacadeResult:
    image: Image.Image  # final output
    raw: Image.Image  # Z-Image output before compositing (same as image in render mode)
    depth: Image.Image
    masks: FacadeMasks
    timings: dict[str, float] = field(default_factory=dict)


class NoBuildingFound(ValueError):
    pass


class FacadeManager(BaseManager):
    name = "facade"

    def __init__(self, mm: MemoryManager, preload: bool = True) -> None:
        super().__init__(mm)
        if preload:  # load at startup so requests are inference-only
            self.get_depth()
            self.get_sam3()
            self.get_zimage()

    def __call__(self, task: FacadeTask) -> FacadeResult:
        timings: dict[str, float] = {}
        image = ImageOps.exif_transpose(task.image).convert("RGB")  # phone photos

        with (
            self.mm.session(self.name),
            time_into_log(timings, "facade_total", echo=True),
        ):
            depth, t = self.get_depth()(image)
            timings |= t

            dets, t = self.get_sam3()(_downscale(image, SAM_MAX_SIDE), FACADE_PROMPTS)
            timings |= t
            if not SETTINGS.KEEP_SAM_LOADED:
                self.mm.del_model(SAM3)  # make room for Z-Image; get_sam3() reloads it

            with time_into_log(timings, "masks"):
                masks = build_facade_masks(dets, image.size)
            if masks is None:
                raise NoBuildingFound("No building found in the photo")
            log_main(
                f"[facade] target {masks.target.mean():.0%} openings {masks.openings.mean():.0%} "
                f"ground {masks.ground.mean():.0%} edit {masks.edit.mean():.0%}"
            )

            defaults = MODE_DEFAULTS[task.mode]
            prompt = task.prompt or style_prompt(task.style, task.extra)
            # render: regenerate everything (consistent sky/lighting), depth holds the geometry
            # keep_context: regenerate house + ground only, real street pasted back
            if task.mode == "render":
                edit = Image.new("L", image.size, 255)
            else:
                edit = masks.image("edit")
            log_main(f"[facade] mode={task.mode} style={task.style} seed={task.seed}")

            raw, t = self.get_zimage()(
                prompt,
                image,
                mask=edit,
                control=depth,
                seed=task.seed,
                steps=task.steps or defaults.steps,
                control_scale=task.control_scale or defaults.control_scale,
            )
            timings |= t

            final = raw if task.mode == "render" else composite(image, raw, edit)

        return FacadeResult(
            image=final, raw=raw, depth=depth, masks=masks, timings=timings
        )


def _downscale(image: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(image.size)
    if scale >= 1:
        return image
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS)
