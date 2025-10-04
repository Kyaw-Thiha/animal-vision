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

    def render_split_compare(
        self,
        original: np.ndarray,
        modified: np.ndarray,
        *,
        left_label: str = "Original",
        right_label: str = "Transformed",
        draw_seam: bool = True,
    ):
        pass

    def close(self) -> None:
        """Optional: release resources."""
        pass
