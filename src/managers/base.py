from src.models.diffusion import ZImageLoader
from src.models.memory_manager import MemoryManager
from src.models.segmentation import DepthLoader, SAM3Loader

DEPTH = "DEPTH"
SAM3 = "SAM3"
ZIMAGE = "ZIMAGE"


class BaseManager:
    """Shared model access for every workflow manager (facade, interior, floor plan...).

    Models live in the MemoryManager, not on the manager, so all managers
    built with the same `mm` share one copy of each model.
    """

    def __init__(self, mm: MemoryManager) -> None:
        self.mm = mm

    def get_depth(self) -> DepthLoader:
        model = self.mm.get_model(DEPTH)
        if model is None:
            model = DepthLoader()
            self.mm.add_model(DEPTH, model)
        return model

    def get_sam3(self) -> SAM3Loader:
        model = self.mm.get_model(SAM3)
        if model is None:
            model = SAM3Loader()
            self.mm.add_model(SAM3, model)
        return model

    def get_zimage(self) -> ZImageLoader:
        model = self.mm.get_model(ZIMAGE)
        if model is None:
            model = ZImageLoader()
            self.mm.add_model(ZIMAGE, model)
        return model
