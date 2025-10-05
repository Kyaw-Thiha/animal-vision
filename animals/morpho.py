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

try:
    import cv2  # optional
except Exception:
    cv2 = None


class Morpho(Animal):
    """
    Morpho butterfly (stylized) — structural blue 'iridescence':
      - Angle-ish shimmer: blue/cyan shifts based on local orientation.
      - UV-assisted gloss: UV strong areas get extra specular lift.
      - High clarity center; subtle 'ommatidial' micro-pixelation overall.

    Returns:
      (baseline_rgb, morpho_rgb) — same HxWx3/dtype as input.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,
        uv_band=(320.0, 400.0),
        blue_band=(440.0, 500.0),
        green_band=(500.0, 570.0),
        panorama_scale: float = 1.05,  # tiny widen just to keep consistency
        # Iridescence controls
        sheen_strength: float = 0.55,  # specular-like lift
        hue_shift_strength: float = 0.45,  # cyan↔deep-blue shift
        gloss_sigma: float = 1.0,  # neighborhood for specular
        # Micro-mosaic (compound eye vibe)
        mosaic_downscale: float = 0.35,  # 0..1 fraction; smaller = coarser mosaic
        center_clarity: float = 0.25,  # extra central sharpness
        vignette_softness: float = 7.0,
        vignette_radius: float = 0.82,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = np.asarray(lambdas, np.float32) if lambdas is not None else np.linspace(300, 700, 81)

        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.blue_lo, self.blue_hi = map(float, blue_band)
        self.green_lo, self.green_hi = map(float, green_band)

        self.panorama_scale = float(panorama_scale)
        self.sheen_strength = float(sheen_strength)
        self.hue_shift_strength = float(hue_shift_strength)
        self.gloss_sigma = float(gloss_sigma)

        self.mosaic_downscale = float(np.clip(mosaic_downscale, 0.15, 1.0))
        self.center_clarity = float(center_clarity)
        self.vignette_softness = float(vignette_softness)
        self.vignette_radius = float(vignette_radius)

    def _grad(self, x: np.ndarray):
        if cv2 is not None:
            gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
            gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
        else:
            gx = np.pad(x, ((0, 0), (1, 1)), mode="reflect")
            gx = 0.5 * (gx[:, 2:] - gx[:, :-2]).astype(np.float32)
            gy = np.pad(x, ((1, 1), (0, 0)), mode="reflect")
            gy = 0.5 * (gy[2:, :] - gy[:-2, :]).astype(np.float32)
        return gx, gy

    def _mosaic(self, img: np.ndarray) -> np.ndarray:
        """Down/up scale for 'ommatidial' pixelation (keeps linear range)."""
        if self.mosaic_downscale >= 0.999 or cv2 is None:
            return img
        H, W = img.shape[:2]
        h = max(1, int(round(H * self.mosaic_downscale)))
        w = max(1, int(round(W * self.mosaic_downscale)))
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)

    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3
        dtype = image.dtype

        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale) if self.panorama_scale != 1.0 else img_lin
        baseline_out = from_float01(linear_to_srgb(np.clip(baseline_lin, 0, 1)), dtype)

        # HSI
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

        # Angle-ish iridescence: hue shift guided by local orientation of blue texture
        gx, gy = self._grad(Bv.astype(np.float32))
        ori = np.arctan2(gy, gx).astype(np.float32)  # [-pi,pi]
        align = 0.5 * (1.0 + np.cos(2.0 * ori))  # [0..1], cyan↔deep-blue selector

        # UV-assisted gloss (blurred highlight)
        gloss = gaussian_blur(U, self.gloss_sigma)
        gloss = gloss / (np.percentile(gloss, 95.0) + 1e-8)
        gloss = np.clip(gloss, 0.0, 1.0)

        # Apply hue shift & gloss in linear RGB: push B and G differently based on 'align'
        # align≈1 -> cyan tilt; align≈0 -> deep blue tilt
        shift_cyan = self.hue_shift_strength * align[..., None]
        shift_deep = self.hue_shift_strength * (1.0 - align)[..., None]
        render[..., 2] = np.clip(render[..., 2] + 0.40 * shift_deep[..., 0] + 0.25 * shift_cyan[..., 0], 0.0, 1.0)  # B
        render[..., 1] = np.clip(render[..., 1] + 0.35 * shift_cyan[..., 0], 0.0, 1.0)  # G
        # Specular-like sheen (UV-weighted)
        render = np.clip(render + self.sheen_strength * gloss[..., None] * np.array([0.10, 0.25, 0.45], np.float32), 0.0, 1.0)

        # Micro-mosaic
        render = self._mosaic(render)

        # Center clarity (inverse vignette)
        H, W = render.shape[:2]
        yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
        xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
        r = np.sqrt(xx * xx + yy * yy)
        t = 1.0 / (1.0 + np.exp(-self.vignette_softness * (r - self.vignette_radius)))  # edge→1
        # sharpen center a bit
        if cv2 is not None:
            sharp = render + 0.22 * (render - gaussian_blur(render, 1.0))
            render = np.clip((1.0 - t[..., None]) * sharp + t[..., None] * render, 0.0, 1.0)

        out = from_float01(linear_to_srgb(np.clip(render, 0, 1)), dtype)
        return baseline_out, out
