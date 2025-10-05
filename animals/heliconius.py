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


class Heliconius(Animal):
    """
    Heliconius (stylized) — emphasize UV+red patterning (courtship/warning signals):
      - UV+R conjunction map: pattern spots/bands pop hard.
      - Background slight desaturation; signals get warm, saturated lift.
      - Moderate clarity; keep greens readable for foliage navigation.

    Returns:
      (baseline_rgb, heliconius_rgb)
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,
        uv_band=(320.0, 400.0),
        red_band=(600.0, 680.0),
        green_band=(500.0, 570.0),
        panorama_scale: float = 1.05,
        # Signal shaping
        conj_sigma_small: float = 0.8,
        conj_sigma_large: float = 2.2,
        conj_gain: float = 1.0,
        sat_boost: float = 0.45,
        red_gain: float = 0.40,
        # Background handling
        bg_desat: float = 0.20,
        bg_cool: float = 0.04,  # subtle cool push to separate warm signals
        # Clarity
        base_soft_sigma: float = 0.30,
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 0.25,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = np.asarray(lambdas, np.float32) if lambdas is not None else np.linspace(300, 700, 81)
        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.red_lo, self.red_hi = map(float, red_band)
        self.green_lo, self.green_hi = map(float, green_band)
        self.panorama_scale = float(panorama_scale)

        self.conj_sigma_small = float(conj_sigma_small)
        self.conj_sigma_large = float(conj_sigma_large)
        self.conj_gain = float(conj_gain)
        self.sat_boost = float(sat_boost)
        self.red_gain = float(red_gain)

        self.bg_desat = float(bg_desat)
        self.bg_cool = float(bg_cool)
        self.base_soft_sigma = float(base_soft_sigma)
        self.unsharp_sigma = float(unsharp_sigma)
        self.unsharp_amount = float(unsharp_amount)

    def _luma(self, x: np.ndarray) -> np.ndarray:
        return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)

    def _sat_apply(self, lin: np.ndarray, scale: np.ndarray) -> np.ndarray:
        Y = self._luma(lin)[..., None]
        return np.clip(Y + (lin - Y) * scale[..., None], 0.0, 1.0).astype(np.float32)

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
        Rb = safe_norm(integrate_band(hsi, self.lambdas, self.red_lo, self.red_hi))
        Gv = safe_norm(integrate_band(hsi, self.lambdas, self.green_lo, self.green_hi))

        # UV∧R conjunction + band-pass to target signal scales
        uv_small = gaussian_blur(U, self.conj_sigma_small)
        uv_large = gaussian_blur(U, self.conj_sigma_large)
        r_small = gaussian_blur(Rb, self.conj_sigma_small)
        r_large = gaussian_blur(Rb, self.conj_sigma_large)
        uv_dog = np.clip(uv_small - uv_large, 0.0, 1.0)
        r_dog = np.clip(r_small - r_large, 0.0, 1.0)
        conj = uv_dog * r_dog
        conj = conj / (np.percentile(conj, 95.0) + 1e-8)
        conj = np.clip(conj, 0.0, 1.0)

        render = baseline_lin.copy()
        # Global prep
        if self.base_soft_sigma > 0.0:
            render = gaussian_blur(render, self.base_soft_sigma)

        # Background: slight cool + desat where conjunction is weak
        bg_w = 1.0 - conj
        render[..., 2] = np.clip(render[..., 2] + self.bg_cool * bg_w, 0.0, 1.0)
        sat_scale = 1.0 - self.bg_desat * bg_w
        render = self._sat_apply(render, sat_scale.astype(np.float32))

        # Signals: unsharp + warm/sat lift gated by conj
        if self.unsharp_sigma > 0.0 and self.unsharp_amount > 0.0:
            blurred = gaussian_blur(render, self.unsharp_sigma)
            render = np.clip(render + (self.unsharp_amount * conj[..., None]) * (render - blurred), 0.0, 1.0)

        render[..., 0] = np.clip(render[..., 0] + self.red_gain * conj, 0.0, 1.0)  # +R on signals
        # saturation boost near signals
        render = self._sat_apply(render, (1.0 + self.sat_boost * conj).astype(np.float32))

        out = from_float01(linear_to_srgb(np.clip(render, 0, 1)), dtype)
        return baseline_out, out
