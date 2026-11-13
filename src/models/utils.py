import gc

import torch

from src.core.config import SETTINGS
from src.core.loggings import time_into_log


def quantize_int8(
    model: torch.nn.Module, timings: dict[str, float], name: str
) -> torch.nn.Module:
    """bf16 -> int8 on CPU, before moving to device. Never hold bf16 on CPU and GPU at once."""
    if SETTINGS.QUANT == "none":
        return model
    from optimum.quanto import freeze, qint8, quantize

    with time_into_log(timings, f"quantize_{name}", echo=True):
        quantize(model, weights=qint8)
        freeze(model)
        gc.collect()
    return model
