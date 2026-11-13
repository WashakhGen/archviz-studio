import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from src.core.loggings import log_main
from src.core.memory import free_memory, log_mem


class MemoryManager:
    """Registry of loaded models. ONE instance, created at startup, passed to every manager."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._mutex = threading.Lock()  # guards _models
        self._gpu_lock = threading.Lock()  # one job on the GPU at a time

    def add_model(self, key: str, model: Any) -> None:
        """Register a loaded model. Replaces an existing entry with the same key."""
        with self._mutex:
            if key in self._models:
                log_main(f"[mm] replacing {key}")
            self._models[key] = model
        log_mem(f"added {key}")

    def get_model(self, key: str) -> Any | None:
        """The registered model, or None if it was never added (or was removed)."""
        with self._mutex:
            return self._models.get(key)

    def del_model(self, key: str) -> bool:
        with self._mutex:
            model = self._models.pop(key, None)
        if model is None:
            return False
        del model
        free_memory()
        log_mem(f"removed {key}")
        return True

    def wipe(self) -> None:
        """Drop every model (shutdown / tests)."""
        with self._mutex:
            self._models.clear()
        free_memory()
        log_mem("wiped")

    @contextmanager
    def session(self, name: str) -> Generator[None, None, None]:
        """Exclusive GPU access for one job. Frees cached memory afterwards."""
        if not self._gpu_lock.acquire(blocking=False):
            log_main(f"[gpu] {name} waiting for GPU")
            self._gpu_lock.acquire()
        log_main(f"[gpu] {name} acquired")
        try:
            yield
        finally:
            free_memory()
            log_mem(f"after {name}")
            self._gpu_lock.release()

    @property
    def busy(self) -> bool:
        return self._gpu_lock.locked()

    def status(self) -> dict[str, object]:
        """For a /health endpoint."""
        with self._mutex:
            models = sorted(self._models)
        return {"busy": self.busy, "models": models}
