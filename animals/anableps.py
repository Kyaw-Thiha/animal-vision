from typing import Optional, Tuple
import numpy as np

from animals.animal import Animal
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi

# Shared helpers from your stack
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


class Anableps(Animal):
    """
    Four-eyed fish (Anableps anableps) vision simulator — stylized split-world.
    Top (air) and bottom (water) halves are processed with distinct pipelines.
    UV saliency enhances water-side shimmer; air-side is warmer and crisper.

    Returns:
      (baseline_rgb, anableps_rgb) — both same HxWx3 and dtype as input.
        - baseline_rgb: geometry-aligned baseline (FOV warp only).
        - anableps_rgb: simulated split-world view.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,  # downsample HSI path if <1.0
        # Bands (nm)
        uv_band: Tuple[float, float] = (320.0, 400.0),
        blue_band: Tuple[float, float] = (430.0, 500.0),
        green_band: Tuple[float, float] = (500.0, 570.0),
        red_band: Tuple[float, float] = (600.0, 680.0),
        # Geometry / FOV
        panorama_scale: float = 1.20,  # mild widen to hint lateral eyes
        # Horizon & refraction
        horizon_y: float = 0.44,  # 0..1 (relative height of air/water split)
        seam_softness_px: float = 8.0,  # vertical soft blend around horizon
        ripple_amp_px: float = 6.0,  # amplitude of horizon ripple (0 disables)
        ripple_waves: float = 2.5,  # number of waves across width
        refract_push_px: float = 3.0,  # vertical remap amplitude below horizon
        # Air (top) look
        air_warmth: Tuple[float, float, float] = (1.06, 1.03, 0.99),  # per-channel gain in linear RGB
        air_clarity_unsharp: float = 0.35,  # strength of unsharp mask (0 disables)
        air_unsharp_sigma: float = 1.0,
        # Water (bottom) look
        red_kill: float = 0.55,  # suppress long-λ
        blue_lift: float = 0.08,
        green_lift: float = 0.12,
        haze_strength: float = 0.10,  # 0..~0.25 freshwater-like haze
        haze_tint: Tuple[float, float, float] = (0.80, 0.92, 1.00),  # blue-green
        base_blur_sigma_water: float = 0.7,
        uv_boost: float = 3.4,  # UV shimmer overall strength
        uv_R_gain: float = 0.36,  # per-channel shimmer mix
        uv_G_gain: float = 0.18,
        uv_B_gain: float = 0.42,
        # Peripheral acuity
        periph_blur_sigma: float = 1.2,
        periph_radius: float = 0.70,  # 0..1 radius of sharper center
        periph_softness: float = 6.0,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10, "lambdas must be 1D wavelengths (nm)."

        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.blue_lo, self.blue_hi = map(float, blue_band)
        self.green_lo, self.green_hi = map(float, green_band)
        self.red_lo, self.red_hi = map(float, red_band)

        self.panorama_scale = float(panorama_scale)
        self.horizon_y = float(horizon_y)
        self.seam_softness_px = float(seam_softness_px)
        self.ripple_amp_px = float(ripple_amp_px)
        self.ripple_waves = float(ripple_waves)
        self.refract_push_px = float(refract_push_px)

        self.air_warmth = np.array(air_warmth, dtype=np.float32)
        self.air_clarity_unsharp = float(air_clarity_unsharp)
        self.air_unsharp_sigma = float(air_unsharp_sigma)

        self.red_kill = float(red_kill)
        self.blue_lift = float(blue_lift)
        self.green_lift = float(green_lift)
        self.haze_strength = float(haze_strength)
        self.haze_tint = np.array(haze_tint, dtype=np.float32)
        self.base_blur_sigma_water = float(base_blur_sigma_water)

        self.uv_boost = float(uv_boost)
        self.uv_R_gain = float(uv_R_gain)
        self.uv_G_gain = float(uv_G_gain)
        self.uv_B_gain = float(uv_B_gain)

        self.periph_blur_sigma = float(periph_blur_sigma)
        self.periph_radius = float(periph_radius)
        self.periph_softness = float(periph_softness)

    # ---------- helpers ----------
    def _unsharp(self, img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
        if sigma <= 0.0 or strength <= 0.0:
            return img
        blur = gaussian_blur(img, sigma)
        high = np.clip(img - blur, -1.0, 1.0)
        return np.clip(img + strength * high, 0.0, 1.0)

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
            (baseline_rgb, anableps_rgb). Both same dtype as input.
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."
        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Geometry: optional mild panorama widen
        if self.panorama_scale and self.panorama_scale != 1.0:
            baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale)
        else:
            baseline_lin = img_lin

        baseline_srgb = linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0))
        baseline_out = from_float01(baseline_srgb, orig_dtype)

        # 3) RGB -> HSI (downsample path if available)
        use_fast = 0.0 < self.hsi_scale < 1.0
        if use_fast:
            try:
                hsi = classic_rgb_to_hsi_scaled(baseline_lin, wavelengths=self.lambdas, scale=self.hsi_scale)
            except AssertionError:
                hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)

        # 4) Spectral maps
        U = integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi)  # UV
        Bv = safe_norm(integrate_band(hsi, self.lambdas, self.blue_lo, self.blue_hi))  # blue
        Gv = safe_norm(integrate_band(hsi, self.lambdas, self.green_lo, self.green_hi))  # green
        Rv = safe_norm(integrate_band(hsi, self.lambdas, self.red_lo, self.red_hi))  # red (ref)
        Un = safe_norm(U)

        # 5) Compute horizon & masks
        H, W = baseline_lin.shape[:2]
        y0 = int(np.clip(self.horizon_y * H, 0, H - 1))

        # Optional ripple along the seam (air-water interface)
        if self.ripple_amp_px > 0.0:
            x = np.linspace(0, 2.0 * np.pi * self.ripple_waves, W, dtype=np.float32)
            ripple = (self.ripple_amp_px * np.sin(x)).astype(np.float32)  # shape (W,)
        else:
            ripple = np.zeros((W,), np.float32)

        # Build soft vertical weights for blending near the seam
        yy = np.arange(H, dtype=np.float32)[:, None]  # (H,1)
        seam_soft = max(1.0, float(self.seam_softness_px))
        # Per-column horizon line with ripple
        horizon = y0 + ripple[None, :]  # (1,W)
        dist = yy - horizon  # + -> below seam, - -> above
        # Air mask (top): goes to 1 above seam, soft around 0 near seam
        air_w = 1.0 / (1.0 + np.exp(+dist / seam_soft))
        # Water mask (bottom): complementary
        water_w = 1.0 - air_w
        air_w3 = air_w[..., None]
        water_w3 = water_w[..., None]

        # 6) Process AIR (top)
        air = baseline_lin.copy()
        # Warm WB tint
        air = np.clip(air * self.air_warmth[None, None, :], 0.0, 1.0)
        # Slight clarity via unsharp
        air = self._unsharp(air, sigma=self.air_unsharp_sigma, strength=self.air_clarity_unsharp)

        # 7) Process WATER (bottom)
        water = baseline_lin.copy()
        # Attenuate reds, lift blue/green
        water[..., 0] = np.clip(water[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)
        water[..., 1] = np.clip(water[..., 1] + self.green_lift, 0.0, 1.0)
        water[..., 2] = np.clip(water[..., 2] + self.blue_lift, 0.0, 1.0)
        # Haze tint (veiling light)
        if self.haze_strength > 0.0:
            a = np.clip(self.haze_strength, 0.0, 1.0)
            water = (1.0 - a) * water + a * self.haze_tint[None, None, :]
        # Base blur (turbidity)
        if self.base_blur_sigma_water > 0.0:
            water = gaussian_blur(water, self.base_blur_sigma_water)
        # UV shimmer overlay (magenta-leaning with some blue)
        uv = Un[..., None]
        water[..., 0] = np.clip(water[..., 0] + self.uv_boost * self.uv_R_gain * uv[..., 0], 0.0, 1.0)
        water[..., 1] = np.clip(water[..., 1] + self.uv_boost * self.uv_G_gain * uv[..., 0], 0.0, 1.0)
        water[..., 2] = np.clip(water[..., 2] + self.uv_boost * self.uv_B_gain * uv[..., 0], 0.0, 1.0)
        # Reinforce blue/green visibility consistency
        water[..., 2] = np.clip(water[..., 2] + 0.20 * Bv, 0.0, 1.0)
        water[..., 1] = np.clip(water[..., 1] + 0.26 * Gv, 0.0, 1.0)

        # 8) Optional refraction / vertical remap for WATER below horizon
        if cv2 is not None and self.refract_push_px > 0.0:
            # Build per-pixel map only for below-horizon rows: push down more near seam
            y_indices = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, axis=1)  # (H,W)
            x_indices = np.repeat(np.arange(W, dtype=np.float32)[None, :], H, axis=0)
            # Distance below horizon (>0 below, <0 above)
            below = np.maximum(y_indices - horizon, 0.0)
            # Exponential decay with depth
            push = self.refract_push_px * np.exp(-below / (2.5 * self.seam_softness_px))
            map_y = np.clip(y_indices + push, 0, H - 1).astype(np.float32)
            map_x = x_indices.astype(np.float32)
            water = cv2.remap(
                water.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
            )

        # 9) Composite split-world with soft seam
        render = air * air_w3 + water * water_w3

        # 10) Peripheral softness (keep moderate; fish have wide FOV but we want center cues)
        if self.periph_blur_sigma > 0.0:
            periph = gaussian_blur(render, self.periph_blur_sigma)
            yy_n = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx_n = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx_n * xx_n + yy_n * yy_n)
            t = 1.0 / (1.0 + np.exp(-self.periph_softness * (r - self.periph_radius)))
            t = t[..., None]
            render = (1.0 - t) * render + t * periph

        # 11) Back to sRGB + dtype
        out_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        anableps_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, anableps_out)
