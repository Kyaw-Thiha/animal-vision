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

try:
    import cv2  # optional
except Exception:
    cv2 = None


class Guppy(Animal):
    """
    Guppy (Poecilia reticulata) vision simulator — UV 'private-channel' spot pop.

    Visual choices (stylized but grounded):
      - UV-gated spot saliency: DoG band-pass on UV map to isolate patch-scale features.
      - Private-channel chroma: boost blue-violet/green saturation near UV patches; mild background desat.
      - Shallow stream optics: tiny haze, light long-λ attenuation, warm daylight tint.
      - Soft attention/vignette to 'feature' central subjects.

    Returns:
      (baseline_rgb, guppy_rgb) — both same HxWx3 and dtype as input.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,  # downsample HSI for speed if <1
        # Spectral bands (nm)
        uv_band: Tuple[float, float] = (320.0, 400.0),
        blue_band: Tuple[float, float] = (430.0, 500.0),
        green_band: Tuple[float, float] = (500.0, 570.0),
        red_band: Tuple[float, float] = (600.0, 680.0),
        # Geometry / FOV
        panorama_scale: float = 1.22,  # shallow-water lateral eyes hint
        # Shallow-stream look
        red_kill: float = 0.28,  # light long-λ attenuation
        haze_strength: float = 0.06,  # tiny veil (clear water)
        haze_tint: Tuple[float, float, float] = (0.92, 0.98, 1.00),
        warm_tint: Tuple[float, float, float] = (1.03, 1.01, 0.99),  # daylight warmth
        base_soft_sigma: float = 0.35,  # pre-sharpen softness
        unsharp_sigma: float = 0.9,
        unsharp_amount: float = 0.28,
        # UV 'private-channel' controls
        dog_small_sigma: float = 0.8,  # DoG band-pass around spot scale
        dog_large_sigma: float = 2.4,
        dog_gain: float = 0.85,  # strength of UV spot saliency (0..~1.2)
        uv_chroma_boost: float = 0.40,  # saturation boost near UV patches
        uv_blue_gain: float = 0.55,  # per-channel gains for UV-gated lift
        uv_green_gain: float = 0.35,
        uv_red_gain: float = 0.12,
        background_desat: float = 0.18,  # desaturate non-UV areas slightly
        # Attention / vignette
        vignette_strength: float = 0.12,  # 0..~0.25
        vignette_radius: float = 0.78,
        vignette_softness: float = 7.0,
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

        self.red_kill = float(red_kill)
        self.haze_strength = float(haze_strength)
        self.haze_tint = np.array(haze_tint, dtype=np.float32)
        self.warm_tint = np.array(warm_tint, dtype=np.float32)
        self.base_soft_sigma = float(base_soft_sigma)
        self.unsharp_sigma = float(unsharp_sigma)
        self.unsharp_amount = float(unsharp_amount)

        self.dog_small_sigma = float(dog_small_sigma)
        self.dog_large_sigma = float(dog_large_sigma)
        self.dog_gain = float(dog_gain)
        self.uv_chroma_boost = float(uv_chroma_boost)
        self.uv_blue_gain = float(uv_blue_gain)
        self.uv_green_gain = float(uv_green_gain)
        self.uv_red_gain = float(uv_red_gain)
        self.background_desat = float(background_desat)

        self.vignette_strength = float(vignette_strength)
        self.vignette_radius = float(vignette_radius)
        self.vignette_softness = float(vignette_softness)

    # ---------- small helpers ----------
    def _unsharp(self, img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        if sigma <= 0.0 or amount <= 0.0:
            return img
        blur = gaussian_blur(img, sigma)
        high = np.clip(img - blur, -1.0, 1.0)
        return np.clip(img + amount * high, 0.0, 1.0)

    def _rgb_to_luma(self, lin: np.ndarray) -> np.ndarray:
        # Rec.709 luma in linear-light
        return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)

    def _saturation(self, lin: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        Y = self._rgb_to_luma(lin)
        mean_chroma = np.mean(np.abs(lin - Y[..., None]), axis=2)
        return (mean_chroma / (np.percentile(mean_chroma, 95.0) + eps)).astype(np.float32)

    def _apply_saturation_scale(self, lin: np.ndarray, scale: np.ndarray) -> np.ndarray:
        # Scale chroma around luma while preserving luminance
        Y = self._rgb_to_luma(lin)[..., None]
        return np.clip(Y + (lin - Y) * scale[..., None], 0.0, 1.0).astype(np.float32)

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
            (baseline_rgb, guppy_rgb). Both same dtype as input.
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."
        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Mild panorama (geometry cue)
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
        Bv = integrate_band(hsi, self.lambdas, self.blue_lo, self.blue_hi)  # blue
        Gv = integrate_band(hsi, self.lambdas, self.green_lo, self.green_hi)  # green
        Rv = integrate_band(hsi, self.lambdas, self.red_lo, self.red_hi)  # red (ref)

        Un = safe_norm(U)
        Bn = safe_norm(Bv)
        Gn = safe_norm(Gv)
        Rn = safe_norm(Rv)  # not used heavily but useful for gating if desired

        # 5) Shallow stream look: light red attenuation, whisper of haze, warm tint
        render = baseline_lin.copy()
        render[..., 0] = np.clip(render[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)  # attenuate red a bit
        if self.haze_strength > 0.0:
            a = float(np.clip(self.haze_strength, 0.0, 1.0))
            render = (1.0 - a) * render + a * self.haze_tint[None, None, :]
        render = np.clip(render * self.warm_tint[None, None, :], 0.0, 1.0)

        # 6) Gentle clarity, then UV spot saliency via DoG band-pass on UV map
        if self.base_soft_sigma > 0.0:
            render = gaussian_blur(render, self.base_soft_sigma)

        # DoG on UV
        uv_small = gaussian_blur(Un, self.dog_small_sigma)
        uv_large = gaussian_blur(Un, self.dog_large_sigma)
        uv_dog = np.clip(uv_small - uv_large, 0.0, 1.0)
        uv_spot = uv_dog / (np.percentile(uv_dog, 95.0) + 1e-8)
        uv_spot = np.clip(uv_spot, 0.0, 1.0)  # 0..1 spot saliency

        # 7) UV-gated contrast & chroma boosts
        # Unsharp scaled by UV spot saliency (local contrast on patches)
        if self.unsharp_sigma > 0.0 and self.unsharp_amount > 0.0:
            blurred = gaussian_blur(render, self.unsharp_sigma)
            high = np.clip(render - blurred, -1.0, 1.0)
            render = np.clip(render + (self.unsharp_amount * uv_spot[..., None]) * high, 0.0, 1.0)

        # Channel-wise lift around UV patches (blue/green >> red)
        lift = (self.uv_chroma_boost * uv_spot)[..., None]
        render[..., 2] = np.clip(render[..., 2] + self.uv_blue_gain * lift[..., 0] * (Bn), 0.0, 1.0)  # B
        render[..., 1] = np.clip(render[..., 1] + self.uv_green_gain * lift[..., 0] * (Gn), 0.0, 1.0)  # G
        render[..., 0] = np.clip(render[..., 0] + self.uv_red_gain * lift[..., 0] * (Un), 0.0, 1.0)  # R (subtle)

        # 8) Background slight desaturation where UV is weak (make patches pop)
        sat = self._saturation(render)
        # Build a desaturation factor that is lower (more desaturation) when UV is low
        desat_factor = 1.0 - self.background_desat * (1.0 - Un) * (1.0 - sat)
        render = self._apply_saturation_scale(render, desat_factor.astype(np.float32))

        # 9) Soft attention/vignette (center preservation)
        if self.vignette_strength > 0.0:
            H, W = render.shape[:2]
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            t = 1.0 / (1.0 + np.exp(-self.vignette_softness * (r - self.vignette_radius)))
            # t ~1 at edges, 0 at center. Reduce saturation and brightness slightly at edges.
            vign = 1.0 - self.vignette_strength * t
            render = np.clip(render * vign[..., None], 0.0, 1.0)

        # 10) Back to sRGB + dtype
        out_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        guppy_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, guppy_out)
