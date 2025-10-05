from typing import Optional
import numpy as np
import cv2 as cv

from animals.animal import Animal
from animals.animal_utils import *
from animals.cat_widevision_utils import *


class Cat(Animal):
    """
    Returns two images:
      (1) human_zoomed : center-zoomed human-like view (narrower FOV)
      (2) cat_wide     : wide-angle cat view (from ORIGINAL image)
    """
    # geometry params
    CAMERA_HFOV_DEG = 100.0
    CAT_PER_EYE_HALF_FOV_DEG = 105.0
    CAT_OVERLAP_DEG = 40.0
    CAT_TO_HUMAN_RATIO = 1.30
    ENABLE_FOV_WARP = True

    def visualize(self, image: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        assert isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3, "HxWx3 RGB"

        orig_dtype = image.dtype
        H, W = image.shape[:2]

        # ---------- A) Human branch: center zoom ----------
        scale = zoom_scale_from_cat_ratio(
            camera_hfov_deg=self.CAMERA_HFOV_DEG,
            cat_per_eye_half_fov_deg=self.CAT_PER_EYE_HALF_FOV_DEG,
            cat_to_human_ratio=self.CAT_TO_HUMAN_RATIO,
        )
        human_zoomed = center_zoom(image, scale=scale)

        # ---------- B) Cat branch: start from ORIGINAL ----------
        cat_srgb01 = get_normalized_image(image)  # use original, not zoomed
        if self.ENABLE_FOV_WARP:
            cat_srgb01 = animal_fov_binocular_warp(
                cat_srgb01.astype(np.float32),
                fov_in_deg=self.CAMERA_HFOV_DEG,
                per_eye_half_fov_deg=self.CAT_PER_EYE_HALF_FOV_DEG,
                overlap_deg=self.CAT_OVERLAP_DEG,
                out_size=(W, H),
                border_mode=0,
                border_value=0.0,
            )

        # color pipeline (L/M merge)
        lin = srgb_to_linear(cat_srgb01)
        vec = lin.reshape(-1, 3)
        lms = sRGB_to_LMS(vec)
        alpha = 0.5  # how strongly to merge L and M
        LM = alpha * lms[:, 0] + (1.0 - alpha) * lms[:, 1]
        lms_merged = np.stack([LM, LM, lms[:, 2]], axis=1)
        lin_rgb = LMS_to_RGB(lms_merged).reshape(H, W, 3)
        lin_rgb = apply_acuity_blur(lin_rgb, sigma=1.0)
        cat_srgb = np.clip(linear_to_srgb(np.clip(lin_rgb, 0.0, 1.0)), 0.0, 1.0)

        # ---------- Restore dtype ----------
        if np.issubdtype(orig_dtype, np.integer):
            human_out = human_zoomed if np.issubdtype(human_zoomed.dtype, np.integer) else (np.clip(human_zoomed,0,1)*255.0+0.5).astype(orig_dtype)
            cat_out   = (cat_srgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            human_out = human_zoomed.astype(orig_dtype)
            cat_out   = cat_srgb.astype(orig_dtype)

        return human_out, cat_out
