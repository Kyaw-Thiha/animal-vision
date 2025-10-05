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
    gaussian_blur,
    panorama_warp,
    classic_rgb_to_hsi_scaled,
)


class Damselfish(Animal):
    """
    Reef damselfish vision simulator (stylized, UV-aware).

    Design goals (contrast with Goldfish):
      - Clear tropical water look → minimal haze, higher global contrast.
      - Strong blue & yellow chroma (common reef palette).
      - UV facial/body pattern saliency → UV-guided *edge contrast* and glossy sheen,
        not a broad magenta wash.
      - Crisper center acuity with light peripheral falloff (opposite of goldfish's murky pond vibe).
      - Slight panorama widen, but milder than goldfish.

    Returns:
      (baseline_rgb, damselfish_rgb) — both same HxWx3 and dtype as input.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,  # downsample path for HSI conversion
        uv_band: Tuple[float, float] = (320.0, 400.0),  # nm
        blue_band: Tuple[float, float] = (440.0, 500.0),  # nm
        yellow_band: Tuple[float, float] = (560.0, 600.0),  # nm (yellow proxy from G+R side)
        red_band: Tuple[float, float] = (600.0, 680.0),  # just for attenuation logic
        uv_edge_boost: float = 0.45,  # UV-guided unsharp strength (0..~0.8)
        uv_gloss_boost: float = 0.30,  # UV-driven specular-ish sheen (0..~0.6)
        blue_chroma_gain: float = 0.22,  # amplify blue where spectrum says so
        yellow_chroma_gain: float = 0.28,  # amplify yellow where spectrum says so
        red_kill: float = 0.35,  # saltwater long-λ attenuation (less than goldfish)
        base_blur_sigma: float = 0.35,  # tiny softness before sharpening (clarity)
        unsharp_sigma: float = 1.2,  # radius for UV-guided unsharp
        panorama_scale: float = 1.25,  # lighter widen than goldfish (1.2–1.35 looks good)
        periph_radius: float = 0.70,  # crisp center radius
        periph_softness: float = 7.0,  # falloff steepness
        periph_extra_blur: float = 0.8,  # extra periphery blur (small)
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10, "lambdas must be 1D wavelengths (nm)."

        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.blue_lo, self.blue_hi = map(float, blue_band)
        self.yellow_lo, self.yellow_hi = map(float, yellow_band)
        self.red_lo, self.red_hi = map(float, red_band)

        self.uv_edge_boost = float(uv_edge_boost)
        self.uv_gloss_boost = float(uv_gloss_boost)
        self.blue_chroma_gain = float(blue_chroma_gain)
        self.yellow_chroma_gain = float(yellow_chroma_gain)
        self.red_kill = float(red_kill)

        self.base_blur_sigma = float(base_blur_sigma)
        self.unsharp_sigma = float(unsharp_sigma)

        self.panorama_scale = float(panorama_scale)
        self.periph_radius = float(periph_radius)
        self.periph_softness = float(periph_softness)
        self.periph_extra_blur = float(periph_extra_blur)

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
            (baseline_rgb, damselfish_rgb). Both same dtype as input.
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."
        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Geometry: mild panorama widen (reef fish lateral eyes but not exaggerated)
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

        # 4) Spectral integrations
        U = integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi)  # UV band
        Bv = integrate_band(hsi, self.lambdas, self.blue_lo, self.blue_hi)  # blue
        Yv = integrate_band(hsi, self.lambdas, self.yellow_lo, self.yellow_hi)  # yellow proxy
        Rv = integrate_band(hsi, self.lambdas, self.red_lo, self.red_hi)  # red (attenuation ref)

        # Normalize bands for stable gains
        Bn = safe_norm(Bv)
        Yn = safe_norm(Yv)
        Rn = safe_norm(Rv)
        Un = safe_norm(U)

        # 5) Start from baseline; saltwater long-λ attenuation (lighter than goldfish)
        render = baseline_lin.copy()
        render[..., 0] = np.clip(render[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)  # reduce reds a bit

        # 6) Tiny global softness to avoid ringing, then UV-guided unsharp mask
        if self.base_blur_sigma > 0.0:
            render = gaussian_blur(render, self.base_blur_sigma)

        if self.unsharp_sigma > 0.0 and self.uv_edge_boost > 0.0:
            blurred = gaussian_blur(render, self.unsharp_sigma)
            highpass = np.clip(render - blurred, -1.0, 1.0)
            # UV-guided gain: emphasize edges where UV is salient (facial/body UV patterns)
            gain = 1.0 + self.uv_edge_boost * (Un[..., None])
            render = np.clip(render + gain * highpass, 0.0, 1.0)

        # 7) UV-driven glossy sheen: brighten highlights slightly where UV is strong
        if self.uv_gloss_boost > 0.0:
            # Per-channel lift with slight blue bias (reef blue water)
            lift = self.uv_gloss_boost * Un[..., None]
            render[..., 2] = np.clip(render[..., 2] + 0.60 * lift[..., 0], 0.0, 1.0)  # B
            render[..., 1] = np.clip(render[..., 1] + 0.30 * lift[..., 0], 0.0, 1.0)  # G
            render[..., 0] = np.clip(render[..., 0] + 0.15 * lift[..., 0], 0.0, 1.0)  # R (subtle)

        # 8) Spectral-consistent chroma shaping: punchy blues & yellows
        render[..., 2] = np.clip(render[..., 2] + self.blue_chroma_gain * Bn, 0.0, 1.0)  # B
        # Yellow ≈ mix of R+G; boost both proportionally to Y band while keeping water look
        y_boost = self.yellow_chroma_gain * Yn
        render[..., 1] = np.clip(render[..., 1] + 0.65 * y_boost, 0.0, 1.0)
        render[..., 0] = np.clip(render[..., 0] + 0.35 * y_boost, 0.0, 1.0)

        # 9) Peripheral falloff (clear water → keep it light)
        if self.periph_extra_blur > 0.0:
            periph = gaussian_blur(render, self.periph_extra_blur)
            H, W = render.shape[:2]
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            t = 1.0 / (1.0 + np.exp(-self.periph_softness * (r - self.periph_radius)))
            t = t[..., None]
            render = (1.0 - t) * render + t * periph

        # 10) Back to sRGB + original dtype
        out_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        damselfish_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, damselfish_out)
