from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class Renderer(ABC):
    def open(self) -> None:
        """Optional: allocate resources (windows/writers)."""
        pass

    @abstractmethod
    def render(self, frame: np.ndarray) -> None:
        """Display or output one frame."""
        ...

    def close(self) -> None:
        """Optional: release resources."""
        pass
