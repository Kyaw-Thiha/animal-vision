from typing import Optional, Tuple
import numpy as np

from animals.animal import Animal
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi

from uv_helpers import (
    to_float01,
    from_float01,
    srgb_to_linear,
    linear_to_srgb,
    safe_norm,
    integrate_uv,
    integrate_band,
    gaussian_blur,
    panorama_warp,
    classic_rgb_to_hsi_scaled,
)


class Pieris(Animal):
    """
    Pieris (stylized) — prime 'flower-finder':
      - UV guide emphasis: petals with UV patterns/veins pop toward yellow-white.
      - Green foliage kept distinct; mid-green opponent contrast around UV guides.
      - Gentle brightness guidance to 'pull' gaze toward flowers.

    Returns:
      (baseline_rgb, pieris_rgb)
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,
        uv_band=(320.0, 400.0),
        blue_band=(430.0, 500.0),
        green_band=(500.0, 570.0),
        panorama_scale: float = 1.05,
        # UV guide shaping
        guide_sigma: float = 1.2,  # structure scale for UV guides
        guide_gain: float = 0.75,  # how strongly guides brighten
        # Foliage separation
        foliage_opponent_gain: float = 0.25,  # G vs (U+B) opponent lift
        # Global palette
        petal_warmth: float = 0.08,  # push UV guides toward warm-white
        clarity_unsharp_sigma: float = 0.8,
        clarity_amount: float = 0.22,
        # Attention map
        center_bias: float = 0.12,
        bias_radius: float = 0.80,
        bias_softness: float = 7.0,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = np.asarray(lambdas, np.float32) if lambdas is not None else np.linspace(300, 700, 81)
        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.blue_lo, self.blue_hi = map(float, blue_band)
        self.green_lo, self.green_hi = map(float, green_band)
        self.panorama_scale = float(panorama_scale)

        self.guide_sigma = float(guide_sigma)
        self.guide_gain = float(guide_gain)
        self.foliage_opponent_gain = float(foliage_opponent_gain)
        self.petal_warmth = float(petal_warmth)
        self.clarity_unsharp_sigma = float(clarity_unsharp_sigma)
        self.clarity_amount = float(clarity_amount)
        self.center_bias = float(center_bias)
        self.bias_radius = float(bias_radius)
        self.bias_softness = float(bias_softness)

    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3
        dtype = image.dtype

        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)
        baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale) if self.panorama_scale != 1.0 else img_lin
        baseline_out = from_float01(linear_to_srgb(np.clip(baseline_lin, 0, 1)), dtype)

        use_fast = 0.0 < self.hsi_scale < 1.0
        if use_fast:
            try:
                hsi = classic_rgb_to_hsi_scaled(baseline_lin, wavelengths=self.lambdas, scale=self.hsi_scale)
            except AssertionError:
                hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)

        U = safe_norm(integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi))
        Bv = safe_norm(integrate_band(hsi, self.lambdas, self.blue_lo, self.blue_hi))
        Gv = safe_norm(integrate_band(hsi, self.lambdas, self.green_lo, self.green_hi))

        render = baseline_lin.copy()

        # UV guide map: blur a bit to form coherent petal/vein structures
        U_s = gaussian_blur(U, self.guide_sigma)
        U_s = U_s / (np.percentile(U_s, 95.0) + 1e-8)
        U_s = np.clip(U_s, 0.0, 1.0)

        # Boost brightness & warm tint on UV guides (toward warm-white petals)
        guide_w = (self.guide_gain * U_s)[..., None]
        render = np.clip(render + guide_w * np.array([0.35, 0.35 + self.petal_warmth, 0.25], np.float32), 0.0, 1.0)

        # Foliage separation: opponent contrast (G vs U+B)
        foliage = np.clip(Gv - 0.5 * (U + Bv), 0.0, 1.0)
        render[..., 1] = np.clip(render[..., 1] + self.foliage_opponent_gain * foliage, 0.0, 1.0)

        # Gentle clarity overall
        if self.clarity_unsharp_sigma > 0.0 and self.clarity_amount > 0.0:
            blur = gaussian_blur(render, self.clarity_unsharp_sigma)
            render = np.clip(render + self.clarity_amount * (render - blur), 0.0, 1.0)

        # Center attention bias (subtle brightening near center)
        H, W = render.shape[:2]
        yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
        xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
        r = np.sqrt(xx * xx + yy * yy)
        t = 1.0 / (1.0 + np.exp(-self.bias_softness * (r - self.bias_radius)))
        att = 1.0 + self.center_bias * (1.0 - t)
        render = np.clip(render * att[..., None], 0.0, 1.0)

        out = from_float01(linear_to_srgb(np.clip(render, 0, 1)), dtype)
        return baseline_out, out
