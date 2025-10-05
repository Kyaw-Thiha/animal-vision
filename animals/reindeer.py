# animals/reindeer.py (refactored)
from typing import Optional, Tuple
import numpy as np

from animals.animal import Animal
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi

# Shared helpers
from uv_helpers import (
    to_float01,
    from_float01,
    srgb_to_linear,
    linear_to_srgb,
    safe_norm,
    integrate_uv,
    integrate_band,
    snow_glare_tone_compress,
    apply_scatter_and_blue_bias,
    panorama_warp,
    classic_rgb_to_hsi_scaled,
)


class Reindeer(Animal):
    """
    Reindeer vision simulator.

    Key visual traits (slightly exaggerated for effect):
      - UV sensitivity: integrates ~320–400 nm energy from the HSI cube and boosts it.
      - Snow-glare control: compress highlights to preserve snow texture.
      - Winter scatter: lower acuity + bluish bias from seasonal tapetum shift.
      - Panoramic bias: gentle horizontal expansion to hint at wide FOV.

    Returns:
      (baseline_rgb, reindeer_rgb) — both same HxWx3, same dtype as input.
      - baseline_rgb: original image (only FOV-warped if enabled, for alignment).
      - reindeer_rgb: simulated reindeer view (sRGB-encoded).
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,  # <1.0 uses downsample speed path
        uv_band: Tuple[float, float] = (300.0, 410.0),  # nm
        uv_boost: float = 3.5,  # boost UV contribution
        snow_glare_compression: float = 0.55,  # 0=no compression, ~0.5 gentle, <0.3 strong
        winter_mode: bool = True,  # blue scatter & softer acuity
        scatter_sigma: float = 1.2,  # px, winter blur
        blue_bias: float = 0.08,  # add to B channel in winter
        panorama_scale: float = 1.3,  # horizontal expand (>1 widens FOV)
        return_uv_heatmap: bool = True,  # (kept for API parity; not returned)
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10, "lambdas must be a 1D vector of wavelengths (nm)."

        self.uv_lo, self.uv_hi = float(uv_band[0]), float(uv_band[1])
        self.uv_boost = float(uv_boost)
        self.snow_glare_compression = float(snow_glare_compression)
        self.winter_mode = bool(winter_mode)
        self.scatter_sigma = float(scatter_sigma)
        self.blue_bias = float(blue_bias)
        self.panorama_scale = float(panorama_scale)
        self.return_uv_heatmap = bool(return_uv_heatmap)  # not returned, but kept for compatibility

    # ---------- public API ----------
    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        image : np.ndarray
            HxWx3 RGB. Integer in [0..255] or float in [0..1].

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            (baseline_rgb, reindeer_rgb). Both same dtype as input.
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."

        orig_dtype = image.dtype

        # 1) Normalize and go to scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Optional FOV/panorama warp for baseline geometry
        if self.panorama_scale and self.panorama_scale != 1.0:
            baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale)
        else:
            baseline_lin = img_lin

        baseline_srgb = linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0))
        baseline_out = from_float01(baseline_srgb, orig_dtype)

        # 3) RGB -> HSI (downsample speed path if hsi_scale < 1.0 and OpenCV available)
        use_fast = 0.0 < self.hsi_scale < 1.0
        if use_fast:
            try:
                hsi = classic_rgb_to_hsi_scaled(baseline_lin, wavelengths=self.lambdas, scale=self.hsi_scale)
            except AssertionError:
                # OpenCV missing — fall back to full-res
                hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)

        # 4) UV + visible integrations
        uv_map = integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi)
        vis_map = safe_norm(integrate_band(hsi, self.lambdas, 420.0, 680.0))

        # 5) UV saliency
        uv_saliency = safe_norm(uv_map / (1e-6 + 0.6 * vis_map))

        # 6) Reindeer rendering on top of baseline geometry
        render_lin = baseline_lin.copy()
        # UV overlay: cool/cyan-ish lift
        render_lin[..., 2] = np.clip(render_lin[..., 2] + self.uv_boost * 0.35 * uv_saliency, 0.0, 1.0)  # +B
        render_lin[..., 1] = np.clip(render_lin[..., 1] + self.uv_boost * 0.15 * uv_saliency, 0.0, 1.0)  # +G

        # Snow-glare compression
        render_lin = snow_glare_tone_compress(render_lin, strength=self.snow_glare_compression)

        # Winter scatter & blue bias
        if self.winter_mode:
            render_lin = apply_scatter_and_blue_bias(render_lin, sigma=self.scatter_sigma, blue_bias=self.blue_bias)

        # 7) Linear -> sRGB and restore dtype
        render_srgb = linear_to_srgb(np.clip(render_lin, 0.0, 1.0))
        reindeer_out = from_float01(render_srgb, orig_dtype)

        return (baseline_out, reindeer_out)
