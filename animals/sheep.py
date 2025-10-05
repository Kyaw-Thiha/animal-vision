from typing import Optional, Tuple
import numpy as np

from animals.animal_utils import *

from animals.animal import Animal


class Sheep(Animal):
    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pass
        """
        Simulate a simple sheep-vision rendering from an RGB image.
        """

        # ---------- 1) validate input ----------
        assert check_input_image(image)
        orig_dtype = image.dtype
        # H, W, _ = image.shape

        # ---------- 2) normalize input ----------
        normalized_image = get_normalized_image(image)

        # ---------- 3) sRGB -> linear ----------
        linear_normalized_image = srgb_to_linear(normalized_image)
        # height, width, _ = linear_normalized_image.shape
        vector_image_srgb = linear_normalized_image.reshape(-1, 3)

        # ---------- 4) linear RGB -> LMS, collapse L/M (dichromacy proxy), LMS -> linear RGB ----------
        sheep_matrix = collapse_LMS_matrix(0.74, 1.06)
        result_in_rgb = vector_image_srgb @ sheep_matrix.T
        result_in_rgb = result_in_rgb.reshape(linear_normalized_image.shape)

        # ---------- 5) apply blur ----------
        result_in_rgb = apply_anisotropic_acuity_blur_with_streak(result_in_rgb, 0.48, 0.8, 2.2, 6.0)

        # ---------- 6) linear -> sRGB and restore dtype ----------
        result_in_srgb = np.clip(linear_to_srgb(np.clip(result_in_rgb, 0.0, 1.0)), 0.0, 1.0)

        if np.issubdtype(orig_dtype, np.integer):
            out = (result_in_srgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            out = result_in_srgb.astype(orig_dtype)

        return image, out
