# archviz-studio

Turn photos of unfinished (grey-structure) buildings and rooms into finished design visualisations, keeping the original geometry.

Upload a photo of a bare facade, kitchen, living room or bedroom and get back a finished version with the same walls, openings, floors and camera angle.

> **Status:** early development. The model loader and inference pipeline work. Prompt generation, masking, quality checks and the API are in progress.

## ⚠️ Non-commercial use only

This project is built on **[Bria Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)**, whose weights are released under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**. The weights may be used for **non-commercial purposes only**. Commercial use requires a separate license from [Bria AI](https://bria.ai).

The source code in this repository is licensed under the **PolyForm Noncommercial License 1.0.0** (see [LICENSE](LICENSE)). You may use, modify and share it for non-commercial purposes only.

This repository does **not** include or redistribute any model weights. You need to download them yourself from Hugging Face and accept Bria's terms there.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Apple Silicon Mac with 24 GB or more unified memory (the model runs in int8), or an NVIDIA GPU
- Access to the gated model on Hugging Face (request access on the model page)
- Optional: a Gemini API key for structured prompt generation

## Setup

```bash
uv sync
uv run hf auth login           # account with access to briaai/Fibo-2.0-edit-turbo
cp .env.example .env           # add GEMINI_API_KEY (optional)
```

Settings such as `DEVICE`, `QUANT`, `BUCKET_BASE` and `AUTO_COMPILE` can be set in `.env` (see `src/core/config.py`).

## Usage

```python
from PIL import Image
from src.model import ModelLoader

model = ModelLoader()  # loads the model, quantizes to int8, warms up
image, timings = model(
    "Transform this unfinished grey-structure house into a finished modern home ...",
    Image.open("house.jpg"),
)
image.save("out.png")
```

## Acknowledgements

- [Bria AI](https://bria.ai) for Fibo 2.0 Edit Turbo
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [optimum-quanto](https://github.com/huggingface/optimum-quanto) for int8 quantization
