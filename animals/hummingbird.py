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


class Hummingbird(Animal):
    """
    Hummingbird (stylized) — UV-tetrachromat 'combo color' rendering.

    Visual ideas (creative but grounded):
      - Compute non-spectral combos: UV+Blue, UV+Green, UV+Red.
      - Map each combo to a distinct synthetic hue family (metamer LUT), not a plain UV overlay.
      - Combo-gated clarity & iridescent sheen to pop gorgets/nectar guides.
      - Gentle flower-guide assist (UV-rich petal patterns).

    Returns:
      (baseline_rgb, hummingbird_rgb) — both same HxWx3 and dtype as input.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,
        # Spectral bands (nm)
        uv_band: Tuple[float, float] = (320.0, 400.0),
        blue_band: Tuple[float, float] = (430.0, 500.0),
        green_band: Tuple[float, float] = (500.0, 570.0),
        red_band: Tuple[float, float] = (600.0, 680.0),
        # Geometry
        panorama_scale: float = 1.05,
        # Global palette shaping
        red_kill: float = 0.10,  # tiny red deemphasis to keep combos readable
        base_soft_sigma: float = 0.25,  # pre-clarity soften
        unsharp_sigma: float = 0.9,
        unsharp_amount: float = 0.24,
        # Combo-color rendering
        combo_opacity: float = 0.55,  # blend of combo tint into render
        combo_saturation: float = 0.45,  # saturation of combo tints
        combo_sheen: float = 0.28,  # specular-ish lift on combos
        # Per-combo target hues (sRGB values converted to linear internally)
        # Chosen to feel distinct and 'non-spectral' to humans
        tgt_uvb_srgb: Tuple[int, int, int] = (120, 150, 255),  # UV+B → electric sky-cyan
        tgt_uvg_srgb: Tuple[int, int, int] = (110, 255, 170),  # UV+G → mint/emerald
        tgt_uvr_srgb: Tuple[int, int, int] = (255, 110, 210),  # UV+R → orchid/magenta
        # Nectar-guide assist (optional)
        guide_sigma: float = 1.0,
        guide_gain: float = 0.25,  # gentle petal/vein brightening
        # Peripheral acuity (keep light; hummingbirds fixate tight)
        periph_blur_sigma: float = 0.6,
        periph_radius: float = 0.82,
        periph_softness: float = 7.0,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10

        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.b_lo, self.b_hi = map(float, blue_band)
        self.g_lo, self.g_hi = map(float, green_band)
        self.r_lo, self.r_hi = map(float, red_band)

        self.panorama_scale = float(panorama_scale)
        self.red_kill = float(red_kill)
        self.base_soft_sigma = float(base_soft_sigma)
        self.unsharp_sigma = float(unsharp_sigma)
        self.unsharp_amount = float(unsharp_amount)

        self.combo_opacity = float(np.clip(combo_opacity, 0.0, 1.0))
        self.combo_saturation = float(combo_saturation)
        self.combo_sheen = float(combo_sheen)

        # Store combo targets in linear space
        def s2l(rgb):
            v = np.array(rgb, np.float32) / 255.0
            a = 0.055
            return np.where(v <= 0.04045, v / 12.92, ((v + a) / (1 + a)) ** 2.4).astype(np.float32)

        self.tgt_uvb_lin = s2l(tgt_uvb_srgb)
        self.tgt_uvg_lin = s2l(tgt_uvg_srgb)
        self.tgt_uvr_lin = s2l(tgt_uvr_srgb)

        self.guide_sigma = float(guide_sigma)
        self.guide_gain = float(guide_gain)

        self.periph_blur_sigma = float(periph_blur_sigma)
        self.periph_radius = float(periph_radius)
        self.periph_softness = float(periph_softness)

    # ---------- helpers ----------
    def _unsharp(self, img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        if sigma <= 0.0 or amount <= 0.0:
            return img
        blur = gaussian_blur(img, sigma)
        high = np.clip(img - blur, -1.0, 1.0)
        return np.clip(img + amount * high, 0.0, 1.0)

    def _apply_saturation(self, lin: np.ndarray, s: float) -> np.ndarray:
        if s == 1.0:
            return lin
        Y = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2])[..., None]
        return np.clip(Y + (lin - Y) * s, 0.0, 1.0).astype(np.float32)

    # ---------- main ----------
    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3
        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Gentle panorama
        baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale) if self.panorama_scale != 1.0 else img_lin
        baseline_out = from_float01(linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0)), orig_dtype)

        # 3) RGB→HSI (downsample path if desired)
        use_fast = 0.0 < self.hsi_scale < 1.0
        if use_fast:
            try:
                hsi = classic_rgb_to_hsi_scaled(baseline_lin, wavelengths=self.lambdas, scale=self.hsi_scale)
            except AssertionError:
                hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)

        # 4) Spectral bands
        U = safe_norm(integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi))
        Bv = safe_norm(integrate_band(hsi, self.lambdas, self.b_lo, self.b_hi))
        Gv = safe_norm(integrate_band(hsi, self.lambdas, self.g_lo, self.g_hi))
        Rv = safe_norm(integrate_band(hsi, self.lambdas, self.r_lo, self.r_hi))

        # 5) Build non-spectral combo maps (UV+X) and normalize
        #    Multiplicative gate emphasizes co-occurrence; then normalize to [0,1]
        UxB = safe_norm(U * Bv)
        UxG = safe_norm(U * Gv)
        UxR = safe_norm(U * Rv)

        # Optional light band-pass to focus combos on patch scales (gorgets/nectar guides)
        def bandpass(m, s_small=0.8, s_large=2.0):
            m1 = gaussian_blur(m, s_small)
            m2 = gaussian_blur(m, s_large)
            d = np.clip(m1 - m2, 0.0, 1.0)
            d = d / (np.percentile(d, 95.0) + 1e-8)
            return np.clip(d, 0.0, 1.0).astype(np.float32)

        UxB_bp = bandpass(UxB)
        UxG_bp = bandpass(UxG)
        UxR_bp = bandpass(UxR)

        # 6) Start from palette-shaped baseline
        render = baseline_lin.copy()
        render[..., 0] = np.clip(render[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)
        if self.base_soft_sigma > 0.0:
            render = gaussian_blur(render, self.base_soft_sigma)

        # 7) Combo-gated clarity & sheen (iridescent pop around combo areas)
        combo_max = np.maximum.reduce([UxB_bp, UxG_bp, UxR_bp])
        if self.unsharp_sigma > 0.0 and self.unsharp_amount > 0.0:
            blurred = gaussian_blur(render, self.unsharp_sigma)
            high = np.clip(render - blurred, -1.0, 1.0)
            render = np.clip(render + (self.unsharp_amount * combo_max[..., None]) * high, 0.0, 1.0)

        if self.combo_sheen > 0.0:
            sheen = (0.55 * UxB_bp + 0.65 * UxG_bp + 0.75 * UxR_bp)[..., None]  # slightly weight toward UV+R
            render = np.clip(render + self.combo_sheen * sheen, 0.0, 1.0)

        # 8) Compose the non-spectral combo tint layer (in linear RGB)
        # Normalize weights so overlapping combos mix gracefully
        w_sum = UxB_bp + UxG_bp + UxR_bp + 1e-8
        wB = (UxB_bp / w_sum)[..., None]
        wG = (UxG_bp / w_sum)[..., None]
        wR = (UxR_bp / w_sum)[..., None]

        combo_tint = (
            wB * self.tgt_uvb_lin[None, None, :] + wG * self.tgt_uvg_lin[None, None, :] + wR * self.tgt_uvr_lin[None, None, :]
        ).astype(np.float32)

        # Boost saturation of the tint separately, then blend
        combo_tint = self._apply_saturation(combo_tint, 1.0 + self.combo_saturation)
        render = np.clip((1.0 - self.combo_opacity) * render + self.combo_opacity * combo_tint, 0.0, 1.0)

        # 9) Nectar-guide assist (subtle brightening where UV is coherent in petals)
        if self.guide_gain > 0.0:
            U_s = gaussian_blur(U, self.guide_sigma)
            U_s = U_s / (np.percentile(U_s, 95.0) + 1e-8)
            U_s = np.clip(U_s, 0.0, 1.0)
            render = np.clip(render + self.guide_gain * U_s[..., None] * np.array([0.20, 0.25, 0.10], np.float32), 0.0, 1.0)

        # 10) Light peripheral softness (small; hummers track precisely)
        if self.periph_blur_sigma > 0.0:
            H, W = render.shape[:2]
            periph = gaussian_blur(render, self.periph_blur_sigma)
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            t = 1.0 / (1.0 + np.exp(-self.periph_softness * (r - self.periph_radius)))
            render = (1.0 - t[..., None]) * render + t[..., None] * periph

        # 11) Back to sRGB + dtype
        out_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        hummingbird_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, hummingbird_out)
