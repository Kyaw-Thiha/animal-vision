from typing import Optional
import numpy as np
# from animals.animal_utils import srgb_to_linear
# from animals.animal_utils import linear_to_srgb
# from animals.animal_utils import M_rgb_to_lms
# from animals.animal_utils import M_lms_to_rgb

from animals.animal_utils import *

from animals.animal import Animal


class Dog(Animal):
    def visualize(self, image: np.ndarray) -> Optional[np.ndarray]:
        pass
        """
        Simulate a simple dog-vision rendering from an RGB image.

        Steps:
        1) Validate input is an RGB image.
        2) Normalize the image
        3) Convert sRGB -> linear RGB.
        4) Map linear RGB to LMS, collapse L & M to a single “LM” channel (dichromacy proxy),
           keep S as-is, then map back LMS -> linear RGB.
        5) Apply blur
        6) Convert linear RGB -> sRGB and return in original dtype.

        note that 
        - alpha = 0.6 for collapsing LMS matrix
        - gamma = 3.5 for acuity blur
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
        dog_matrix = collapse_LMS_matrix(0.58, 0.65)
        result_in_rgb = vector_image_srgb @ dog_matrix.T
        result_in_rgb = result_in_rgb.reshape(linear_normalized_image.shape)

        # ---------- 5) apply blur ----------
        result_in_rgb = apply_acuity_blur(result_in_rgb, 3.5)

        # ---------- 6) linear -> sRGB and restore dtype ----------
        result_in_srgb = np.clip(linear_to_srgb(np.clip(result_in_rgb, 0.0, 1.0)), 0.0, 1.0)

        if np.issubdtype(orig_dtype, np.integer):
            out = (result_in_srgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            out = result_in_srgb.astype(orig_dtype)

        return out
