from typing import Optional
import numpy as np
from animals.animal_utils import *
from animals.animal_utils2 import *

from animals.animal import Animal


class Cat(Animal):
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
        5) Convert linear RGB -> sRGB and return in original dtype.

        - alpha = 0.4 for collapsing LMS matrix
        - gamma = 1.0 for acuity blur

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

        # ---------- 4) deal with night vision ----------

        if not check_is_day(linear_normalized_image, linear_rgb=True):
            print("is night")
            # apply_rod_vision returns a new image — assign it back
            linear_normalized_image = apply_rod_vision(
                linear_normalized_image,
                chroma_scale=0.07,
                luminance_boost=1.8,
                gamma=0.7,
            )

        # ---------- 5) linear RGB -> LMS, collapse L/M (dichromacy proxy), LMS -> linear RGB ----------
        cat_matrix = collapse_LMS_matrix(0.45, 0.80)
        result_in_rgb = vector_image_srgb @ cat_matrix.T
        result_in_rgb = result_in_rgb.reshape(linear_normalized_image.shape)

        # ---------- 6) apply blur ----------
        result_in_rgb = apply_acuity_blur(result_in_rgb, 1.0)

        # ---------- 7) deal with FOV ----------
        H, W = result_in_rgb.shape[:2]
        enlarged = enlarge_then_crop(
                result_in_rgb,
                scale=1.6,
                out_size=(W, H),
                crop_anchor="center",
                pad_value=0.0,
            )
        result_in_rgb = animal_fov_binocular_warp(
                enlarged,
                fov_in_deg=60.0,
                per_eye_half_fov_deg=100.0,
                overlap_deg=80.0,
                out_size=(W, H),
            )

        # ---------- 8) linear -> sRGB and restore dtype ----------
        result_in_srgb = np.clip(linear_to_srgb(np.clip(result_in_rgb, 0.0, 1.0)), 0.0, 1.0)

        if np.issubdtype(orig_dtype, np.integer):
            out = (result_in_srgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            out = result_in_srgb.astype(orig_dtype)

        return out
