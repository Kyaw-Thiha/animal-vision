from typing import Optional, Tuple
import numpy as np

from animals.animal import Animal
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi

# Shared helpers (your modules)
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


class Goldfish(Animal):
    """
    Goldfish vision simulator (freshwater, tetrachromacy with UV sensitivity — stylized).

    What we emphasize (some creative liberty, grounded in aquatic optics):
      - Strong UV saliency (∼320–400 nm) → iridescent “sheen” on scales / biofilms.
      - Blue/green dominance (underwater attenuation suppresses long reds).
      - Freshwater haze / backscatter → slight low-frequency wash + greenish veiling light.
      - Very wide monocular FOV (lateral eyes) → gentle panorama widen for a “wrapped” feel.
      - Mild peripheral softness (spherical optics + water turbidity).

    Returns:
      (baseline_rgb, goldfish_rgb) — both same HxWx3 and dtype as input.
        - baseline_rgb: geometry-aligned baseline (after optional FOV warp only).
        - goldfish_rgb: simulated goldfish view (sRGB-encoded).
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,  # downsample HSI speed path if < 1.0
        uv_band: Tuple[float, float] = (320.0, 400.0),  # nm
        blue_band: Tuple[float, float] = (430.0, 500.0),  # nm
        green_band: Tuple[float, float] = (500.0, 570.0),  # nm
        red_band: Tuple[float, float] = (600.0, 680.0),  # nm (for attenuation logic)
        uv_boost: float = 3.0,  # overall UV emphasis
        panorama_scale: float = 1.45,  # widen FOV horizontally
        haze_strength: float = 0.12,  # 0..~0.3 freshwater veiling light
        haze_tint: Tuple[float, float, float] = (0.78, 0.92, 1.0),  # blue-green veiling (linear RGB)
        red_kill: float = 0.55,  # 0..1 multiplier on red (higher = less red)
        green_lift: float = 0.12,  # additive green lift post-attenuation
        blue_lift: float = 0.06,  # additive blue lift post-attenuation
        base_blur_sigma: float = 0.8,  # global softness
        periph_blur_sigma: float = 1.8,  # extra blur in periphery
        periph_radius: float = 0.65,  # 0..1 radius where image stays sharper
        periph_softness: float = 6.0,  # transition steepness for periphery blur
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

        self.uv_boost = float(uv_boost)
        self.panorama_scale = float(panorama_scale)

        self.haze_strength = float(haze_strength)
        self.haze_tint = np.array(haze_tint, dtype=np.float32)

        self.red_kill = float(red_kill)
        self.green_lift = float(green_lift)
        self.blue_lift = float(blue_lift)

        self.base_blur_sigma = float(base_blur_sigma)
        self.periph_blur_sigma = float(periph_blur_sigma)
        self.periph_radius = float(periph_radius)
        self.periph_softness = float(periph_softness)

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
            (baseline_rgb, goldfish_rgb). Both same dtype as input.
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."

        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Geometry: widen FOV (panorama) for lateral-eye vibe
        if self.panorama_scale and self.panorama_scale != 1.0:
            baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale)
        else:
            baseline_lin = img_lin

        baseline_srgb = linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0))
        baseline_out = from_float01(baseline_srgb, orig_dtype)

        # 3) RGB -> HSI via your converter (downsample path if desired)
        use_fast = 0.0 < self.hsi_scale < 1.0
        if use_fast:
            try:
                hsi = classic_rgb_to_hsi_scaled(baseline_lin, wavelengths=self.lambdas, scale=self.hsi_scale)
            except AssertionError:
                hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)

        # 4) Spectral integrations
        U = integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi)  # UV band
        Bv = safe_norm(integrate_band(hsi, self.lambdas, self.blue_lo, self.blue_hi))  # blue band (visible)
        Gv = safe_norm(integrate_band(hsi, self.lambdas, self.green_lo, self.green_hi))  # green band (visible))
        Rv = safe_norm(integrate_band(hsi, self.lambdas, self.red_lo, self.red_hi))  # red (for attenuation logic)

        # UV saliency: UV vs visible backdrop (favor G/B context)
        uv_saliency = safe_norm(U / (1e-6 + 0.45 * Gv + 0.35 * Bv + 0.15 * Rv))

        # 5) Start from baseline geometry and apply aquatic optics
        render = baseline_lin.copy()

        # 5a) Underwater red attenuation + slight blue/green lift
        render[..., 0] = np.clip(render[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)  # R suppressed strongly
        render[..., 1] = np.clip(render[..., 1] + self.green_lift, 0.0, 1.0)  # G lift
        render[..., 2] = np.clip(render[..., 2] + self.blue_lift, 0.0, 1.0)  # B lift

        # 5b) Veiling light / freshwater haze (adds a blue-green wash, reduces contrast)
        if self.haze_strength > 0.0:
            a = np.clip(self.haze_strength, 0.0, 1.0)
            render = (1.0 - a) * render + a * self.haze_tint[None, None, :]

        # 5c) Soft global blur (turbidity / spherical optics)
        if self.base_blur_sigma > 0.0:
            render = gaussian_blur(render, self.base_blur_sigma)

        # 5d) UV “sheen” overlay: emphasize scales/biofilm edges with magenta-leaning lift
        #     (Magenta: +R +B, light +G to avoid dead blacks; scaled by uv_saliency)
        uv = uv_saliency[..., None]
        render[..., 0] = np.clip(render[..., 0] + self.uv_boost * 0.42 * uv[..., 0], 0.0, 1.0)  # +R
        render[..., 2] = np.clip(render[..., 2] + self.uv_boost * 0.35 * uv[..., 0], 0.0, 1.0)  # +B
        render[..., 1] = np.clip(render[..., 1] + self.uv_boost * 0.12 * uv[..., 0], 0.0, 1.0)  # +G (subtle)

        # 5e) Reinforce blue/green visibility from spectral maps (keeps “underwater” look consistent)
        render[..., 2] = np.clip(render[..., 2] + 0.22 * Bv, 0.0, 1.0)
        render[..., 1] = np.clip(render[..., 1] + 0.30 * Gv, 0.0, 1.0)

        # 5f) Peripheral softness (radial blend with an extra-blurred copy)
        if self.periph_blur_sigma > 0.0:
            periph = gaussian_blur(render, self.periph_blur_sigma)
            H, W = render.shape[:2]
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            # Smooth step: 0 near center, →1 past periph_radius
            t = 1.0 / (1.0 + np.exp(-self.periph_softness * (r - self.periph_radius)))
            t = t[..., None]
            render = (1.0 - t) * render + t * periph

        # 6) Back to sRGB + original dtype
        render_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        goldfish_out = from_float01(render_srgb, orig_dtype)

        return (baseline_out, goldfish_out)
