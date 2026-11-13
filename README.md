# archviz-studio

Turn photos of unfinished (grey-structure) buildings into finished architectural visualisations, keeping the original geometry.

Upload a photo of a bare concrete house and get back a rendered elevation with the same floors, slab edges, openings and camera angle, in a chosen style (for example modern at blue hour, white and teak by day, or contemporary stone).

> **Status:** early development. The facade workflow runs end to end from a script. The API, more templates (interiors, floor plans) and quality checks are in progress.

## How it works

```
photo
 → Depth Anything V2      depth map of the original photo
 → SAM 3                  text-prompted masks: house, windows/openings, ground/driveway/pavement
 → masks                  pick the target house, drop neighbours, fill debris holes
 → Z-Image Turbo          ControlNet inpaint guided by the depth map + style prompt
 → composite / render     keep the real street (keep_context) or re-render the whole scene (render)
```

Depth control is what keeps slabs, projections and openings in place while the finishes, lighting and landscaping change.

Two modes:

| Mode | What is regenerated | Use when |
|---|---|---|
| `render` (default) | the whole image: building, sky, lighting, surroundings | you want a presentation render |
| `keep_context` | only the house and the ground in front of it | the real street and neighbours must stay |

Styles live in `src/workflow/styles.py`; adding one is a new dictionary entry.

## Project layout

```
src/
  core/        settings (.env), logging, memory helpers, aspect-ratio buckets
  models/      model loaders (Z-Image, SAM 3, Depth) and the MemoryManager registry
  managers/    workflows: BaseManager, FacadeManager
  vision/      mask building from SAM 3 detections
  workflow/    style presets
main.py        end-to-end facade test
```

Models are loaded once and registered in a shared `MemoryManager`; every workflow manager receives the same instance, so adding a new workflow does not load a second copy of any model.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Apple Silicon Mac with 24 GB or more unified memory (models run in int8), or an NVIDIA GPU
- A Hugging Face account with access to the gated [facebook/sam3](https://huggingface.co/facebook/sam3) model
- About 45 GB of free disk space for the model weights

On a 24 GB Mac, close other heavy apps and raise the GPU memory limit (resets on reboot):

```bash
sudo sysctl iogpu.wired_limit_mb=20480
```

## Setup

```bash
uv sync
uv run hf auth login     # account with access to facebook/sam3
cp .env.example .env     # optional overrides
```

Settings such as `DEVICE`, `QUANT`, `STEPS`, `CONTROL_SCALE`, `MODEL_DIR` and `KEEP_SAM_LOADED` can be set in `.env` (see `src/core/config.py`). Weights download to `MODEL_DIR` on first run.

## Usage

```python
from PIL import Image

from src.managers.facade import FacadeManager, FacadeTask
from src.models.memory_manager import MemoryManager

mm = MemoryManager()
facade = FacadeManager(mm)  # loads Depth, SAM 3 and Z-Image once

result = facade(
    FacadeTask(
        image=Image.open("house.jpg"),
        style="modern_dusk",       # see src/workflow/styles.py
        mode="render",             # or "keep_context"
        extra="red brick on the ground floor",  # optional wishes
        seed=1,
    )
)
result.image.save("render.png")
result.masks.overlay(Image.open("house.jpg")).save("masks.jpg")  # debug view
```

Or run the test script: `uv run python main.py`.

## Licensing

The source code in this repository is licensed under the **PolyForm Noncommercial License 1.0.0** (see [LICENSE](LICENSE)). You may use, modify and share it for non-commercial purposes only.

This repository does **not** include or redistribute any model weights. They are downloaded from Hugging Face, and each one has its own licence:

| Model | Licence |
|---|---|
| [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Apache 2.0 |
| [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0) | Apache 2.0 |
| [facebook/sam3](https://huggingface.co/facebook/sam3) | SAM License (Meta, gated; accept the terms on the model page) |
| [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Apache 2.0 |

Check each licence before using generated images or the models commercially.

## Acknowledgements

- Tongyi-MAI for Z-Image Turbo and Alibaba PAI for the Fun ControlNet Union
- Meta AI for SAM 3
- The Depth Anything team for Depth Anything V2
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and [Transformers](https://github.com/huggingface/transformers)
- [optimum-quanto](https://github.com/huggingface/optimum-quanto) for int8 quantization
