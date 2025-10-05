from typing import Optional, Tuple
import numpy as np

from animals.animal import Animal
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi

# Your shared helpers
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


class Anchovy(Animal):
    """
    Northern anchovy (Engraulis) vision simulator — UV + polarization stylized.

    Core ideas:
      - Clear ocean optics: minimal haze, light red attenuation, crisp microcontrast.
      - UV-polarization: orientation-dependent gain (cos^2 law) on UV-guided contrast.
      - UV edge emphasis: edges aligned with E-vector get extra pop; orthogonal edges are subdued.
      - Subtle blue bias and specular-like highlights where UV is strong and aligned.

    Returns:
      (baseline_rgb, anchovy_rgb) — both same HxWx3 and dtype as input.
        - baseline_rgb: only geometry-adjusted (FOV warp) view.
        - anchovy_rgb: simulated anchovy view.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,  # downsample path for HSI conversion
        # Spectral bands (nm)
        uv_band: Tuple[float, float] = (320.0, 400.0),
        blue_band: Tuple[float, float] = (440.0, 500.0),
        green_band: Tuple[float, float] = (500.0, 570.0),
        red_band: Tuple[float, float] = (600.0, 680.0),
        # Geometry / FOV
        panorama_scale: float = 1.20,  # slight widen hinting lateral eyes
        # Ocean look (very clear)
        red_kill: float = 0.25,  # small long-λ attenuation
        base_soft_sigma: float = 0.30,  # tiny pre-sharpen softness
        unsharp_sigma: float = 1.0,  # unsharp radius
        unsharp_amount: float = 0.35,  # unsharp strength
        haze_strength: float = 0.04,  # almost clear water
        haze_tint: Tuple[float, float, float] = (0.90, 0.97, 1.00),
        # Polarization parameters
        evec_angle_deg: float = 0.0,  # global E-vector direction (degrees, 0=+x)
        pol_strength: float = 0.55,  # 0..~1: how strong orientation gain is
        pol_gamma: float = 1.2,  # sharpness of orientation selectivity
        orientation_mix: float = 0.35,  # 0=global only, 1=local UV-gradient only
        # UV-driven gloss/chroma
        uv_gloss_gain: float = 0.28,  # highlight lift where aligned UV is strong
        blue_chroma_gain: float = 0.18,  # blue saturation when UV/blue bands agree
        green_chroma_gain: float = 0.10,  # subtle green support
        # Peripheral acuity (keep light)
        periph_blur_sigma: float = 0.6,
        periph_radius: float = 0.78,
        periph_softness: float = 7.0,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10, "lambdas must be a 1D wavelength vector (nm)."

        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.blue_lo, self.blue_hi = map(float, blue_band)
        self.green_lo, self.green_hi = map(float, green_band)
        self.red_lo, self.red_hi = map(float, red_band)

        self.panorama_scale = float(panorama_scale)
        self.red_kill = float(red_kill)
        self.base_soft_sigma = float(base_soft_sigma)
        self.unsharp_sigma = float(unsharp_sigma)
        self.unsharp_amount = float(unsharp_amount)
        self.haze_strength = float(haze_strength)
        self.haze_tint = np.array(haze_tint, dtype=np.float32)

        self.evec_angle = np.deg2rad(float(evec_angle_deg))
        self.pol_strength = float(pol_strength)
        self.pol_gamma = float(pol_gamma)
        self.orientation_mix = float(np.clip(orientation_mix, 0.0, 1.0))

        self.uv_gloss_gain = float(uv_gloss_gain)
        self.blue_chroma_gain = float(blue_chroma_gain)
        self.green_chroma_gain = float(green_chroma_gain)

        self.periph_blur_sigma = float(periph_blur_sigma)
        self.periph_radius = float(periph_radius)
        self.periph_softness = float(periph_softness)

    # ---------- small helpers ----------
    def _sobel(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients (gx, gy) for a single-channel float32 image."""
        if cv2 is not None:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REFLECT101)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REFLECT101)
        else:
            # Simple finite differences with reflection padding
            x = np.pad(img, ((0, 0), (1, 1)), mode="reflect")
            y = np.pad(img, ((1, 1), (0, 0)), mode="reflect")
            gx = 0.5 * (x[:, 2:] - x[:, :-2]).astype(np.float32)
            gy = 0.5 * (y[2:, :] - y[:-2, :]).astype(np.float32)
        return gx.astype(np.float32), gy.astype(np.float32)

    def _unsharp(self, img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        if sigma <= 0.0 or amount <= 0.0:
            return img
        blur = gaussian_blur(img, sigma)
        high = np.clip(img - blur, -1.0, 1.0)
        return np.clip(img + amount * high, 0.0, 1.0)

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
            (baseline_rgb, anchovy_rgb). Both same dtype as input.
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."
        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Light geometry cue: small panorama widen
        if self.panorama_scale and self.panorama_scale != 1.0:
            baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale)
        else:
            baseline_lin = img_lin

        # Baseline output (geometry-aligned)
        baseline_srgb = linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0))
        baseline_out = from_float01(baseline_srgb, orig_dtype)

        # 3) HSI conversion (optionally downsampled)
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
        Gv = integrate_band(hsi, self.lambdas, self.green_lo, self.green_hi)  # green
        Rv = integrate_band(hsi, self.lambdas, self.red_lo, self.red_hi)  # red (reference)

        # Normalize maps for stable gains
        Un = safe_norm(U)
        Bn = safe_norm(Bv)
        Gn = safe_norm(Gv)
        Rn = safe_norm(Rv)

        # 5) Polarization orientation: local UV gradient orientation vs a global E-vector
        #    For polarization sensitivity we use cos^2(theta - phi). Use local theta from UV gradients.
        gx, gy = self._sobel(Un.astype(np.float32))
        # Local orientation of UV features (radians, [-pi, pi])
        theta = np.arctan2(gy, gx).astype(np.float32)

        # Blend global axis with local orientation (work in doubled-angle space)
        # cos(2*alpha), sin(2*alpha) representation avoids wrap-around issues.
        cos2_local = np.cos(2.0 * theta)
        sin2_local = np.sin(2.0 * theta)
        cos2_global = float(np.cos(2.0 * self.evec_angle))
        sin2_global = float(np.sin(2.0 * self.evec_angle))

        mix = float(self.orientation_mix)
        cos2_mix = (1.0 - mix) * cos2_global + mix * cos2_local
        sin2_mix = (1.0 - mix) * sin2_global + mix * sin2_local
        # Effective alignment value in [ -1 .. 1 ]
        align = cos2_mix  # cos(2*(theta-phi))

        # Orientation-dependent gain in [0..1], sharpened by pol_gamma
        align01 = np.clip(0.5 * (align + 1.0), 0.0, 1.0) ** float(self.pol_gamma)

        # UV magnitude (edge strength) as an additional factor
        uv_mag = np.sqrt(gx * gx + gy * gy)
        uv_mag = uv_mag / (np.percentile(uv_mag, 95.0) + 1e-8)
        uv_mag = np.clip(uv_mag, 0.0, 1.0)

        # Final polarization gain map (0..1+)
        pol_gain = 1.0 + self.pol_strength * (align01 * Un * uv_mag)

        # 6) Start from baseline: clear-ocean chroma/contrast
        render = baseline_lin.copy()
        # Light red attenuation
        render[..., 0] = np.clip(render[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)
        # Whisper of haze (very clear water)
        if self.haze_strength > 0.0:
            a = float(np.clip(self.haze_strength, 0.0, 1.0))
            render = (1.0 - a) * render + a * self.haze_tint[None, None, :]

        # 7) Tiny softening then polarization-guided unsharp contrast
        if self.base_soft_sigma > 0.0:
            render = gaussian_blur(render, self.base_soft_sigma)

        if self.unsharp_sigma > 0.0 and self.unsharp_amount > 0.0:
            blurred = gaussian_blur(render, self.unsharp_sigma)
            high = np.clip(render - blurred, -1.0, 1.0)
            render = np.clip(render + (self.unsharp_amount * pol_gain[..., None]) * high, 0.0, 1.0)

        # 8) UV-aligned glossy lift & chroma shaping
        gloss = self.uv_gloss_gain * (align01 * Un)[..., None]
        render[..., 2] = np.clip(render[..., 2] + 0.70 * gloss[..., 0], 0.0, 1.0)  # B highlight
        render[..., 1] = np.clip(render[..., 1] + 0.30 * gloss[..., 0], 0.0, 1.0)  # G highlight
        # Spectral-consistent chroma boosts (when blue/green agree with UV)
        render[..., 2] = np.clip(render[..., 2] + self.blue_chroma_gain * (Bn * Un), 0.0, 1.0)
        render[..., 1] = np.clip(render[..., 1] + self.green_chroma_gain * (Gn * Un), 0.0, 1.0)

        # 9) Light peripheral softness (wide-FOV hint without losing crispness)
        if self.periph_blur_sigma > 0.0:
            periph = gaussian_blur(render, self.periph_blur_sigma)
            H, W = render.shape[:2]
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            t = 1.0 / (1.0 + np.exp(-self.periph_softness * (r - self.periph_radius)))
            t = t[..., None]
            render = (1.0 - t) * render + t * periph

        # 10) Back to sRGB + dtype
        out_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        anchovy_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, anchovy_out)
