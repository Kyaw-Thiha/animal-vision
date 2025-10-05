from typing import Optional, Tuple
import numpy as np

from animals.animal_utils import *

from animals.animal import Animal


class Panda(Animal):
    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate a simple panda-vision rendering from an RGB image.
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
        animal_matrix = collapse_LMS_matrix(0.58, 0.74)
        result_in_rgb = vector_image_srgb @ animal_matrix.T
        result_in_rgb = result_in_rgb.reshape(linear_normalized_image.shape)

        # ---------- 5) apply blur ----------
        result_in_rgb = apply_anisotropic_acuity_blur_with_streak(result_in_rgb, 0.52, 1.0, 2.1, 4.5)


        result_in_rgb = apply_chroma_compression(result_in_rgb, 0.06)

        # ---------- 6) linear -> sRGB and restore dtype ----------
        result_in_srgb = np.clip(linear_to_srgb(np.clip(result_in_rgb, 0.0, 1.0)), 0.0, 1.0)

        if np.issubdtype(orig_dtype, np.integer):
            out = (result_in_srgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            out = result_in_srgb.astype(orig_dtype)

        return image, out
