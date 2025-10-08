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

<<<<<<< HEAD
        # ---------- 2) pre-zoom (human view) + cat FOV ----------
        # Zoom first (narrow human FOV) and also make cat FOV from the ORIGINAL image.
        zoomed_human, cat_view = human_zoom_and_cat_view(
            image,
            camera_hfov_deg=100.0,  # camera/image HFOV
            cat_per_eye_half_fov_deg=105.0,  # cat per-eye half-FOV (phi)
            binocular_overlap_deg=40.0,  # binocular overlap
            cat_to_human_ratio=1.30,  # cat sees ~1.3× wider than human
            # zoom_scale=1.25,              # (optional) override ratio
        )

        # ---------- 3) normalize input (use zoomed human for "human" branch) ----------
        normalized_image = get_normalized_image(zoomed_human)


        # ---------- 4) sRGB -> linear ----------
        linear_normalized_image = srgb_to_linear(normalized_image)
        # height, width, _ = linear_normalized_image.shape
        vector_image_srgb = linear_normalized_image.reshape(-1, 3)

        # ---------- 5) deal with night vision ----------

        # if not check_is_day(linear_normalized_image, linear_rgb=True):
        #     print("is night")
        #     # apply_rod_vision returns a new image — assign it back
        #     linear_normalized_image = apply_rod_vision(
        #         linear_normalized_image,
        #         chroma_scale=0.07,
        #         luminance_boost=1.8,
        #         gamma=0.7,
        #     )

        # ---------- 5) linear RGB -> LMS, collapse L/M (dichromacy proxy), LMS -> linear RGB ----------
        cat_matrix = collapse_LMS_matrix(0.45, 0.80)
        result_in_rgb = vector_image_srgb @ cat_matrix.T
        result_in_rgb = result_in_rgb.reshape(linear_normalized_image.shape)

        # ---------- 6) apply blur ----------
        result_in_rgb = apply_acuity_blur(result_in_rgb, 1.0)

        # ---------- 8) linear -> sRGB and restore dtype ----------
        result_in_srgb = np.clip(linear_to_srgb(np.clip(result_in_rgb, 0.0, 1.0)), 0.0, 1.0)
=======
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
>>>>>>> Tina-animals

        # ---------- Restore dtype ----------
        if np.issubdtype(orig_dtype, np.integer):
            human_out = human_zoomed if np.issubdtype(human_zoomed.dtype, np.integer) else (np.clip(human_zoomed,0,1)*255.0+0.5).astype(orig_dtype)
            cat_out   = (cat_srgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            human_out = human_zoomed.astype(orig_dtype)
            cat_out   = cat_srgb.astype(orig_dtype)

        return human_out, cat_out
