from typing import Optional
import numpy as np

from animals.animal import Animal


class Dog(Animal):
    def visualize(self, input: np.ndarray) -> Optional[np.ndarray]:
        pass
