import gc

import torch

from src.core.config import SETTINGS
from src.core.loggings import log_main


def log_mem(tag: str) -> None:
    if SETTINGS.DEVICE == "mps":
        used = torch.mps.driver_allocated_memory()
    elif SETTINGS.DEVICE == "cuda":
        used = torch.cuda.memory_reserved()
    else:
        return
    log_main(f"[mem] {tag:<16} {used / 1e9:5.1f} GB")


def free_memory() -> None:
    gc.collect()
    if SETTINGS.DEVICE == "mps":
        torch.mps.empty_cache()
    elif SETTINGS.DEVICE == "cuda":
        torch.cuda.empty_cache()


def sync() -> None:
    if SETTINGS.DEVICE == "mps":
        torch.mps.synchronize()
    elif SETTINGS.DEVICE == "cuda":
        torch.cuda.synchronize()
