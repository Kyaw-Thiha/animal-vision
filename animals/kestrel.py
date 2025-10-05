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


class Kestrel(Animal):
    """
    Kestrel (Falco) UV-trail scout — stylized aerial view.

    Emphasis:
      - UV 'trailness' map from the UV band (structure-tensor ridge measure).
      - False-color UV overlay (magenta-leaning) over ground only.
      - Sky/ground split (data-driven + vertical prior) with different tone maps.
      - UV-gated clarity (unsharp) so trails & field texture snap into focus.

    Returns:
      (baseline_rgb, kestrel_rgb) — both same HxWx3 and dtype as input.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,
        # Spectral bands (nm)
        uv_band: Tuple[float, float] = (320.0, 400.0),
        blue_band: Tuple[float, float] = (440.0, 500.0),  # sky proxy
        green_band: Tuple[float, float] = (500.0, 570.0),  # vegetation proxy
        red_band: Tuple[float, float] = (600.0, 680.0),
        # Geometry cue
        panorama_scale: float = 1.10,
        # Sky/ground look
        sky_cool_tint: Tuple[float, float, float] = (0.95, 0.98, 1.03),
        sky_haze: float = 0.10,  # 0..~0.25
        ground_warm_tint: Tuple[float, float, float] = (1.02, 1.01, 0.99),
        ground_contrast: float = 0.08,  # gentle local contrast on ground
        # UV trail rendering
        uv_overlay_strength: float = 0.55,  # 0..1 overlay opacity
        uv_magenta: Tuple[float, float, float] = (0.60, 0.12, 0.70),  # linear RGB tint
        ridge_sigma: float = 3,  # structure scale for ridge detection
        ridge_gain: float = 1.0,  # trailness amplitude
        # UV-gated clarity
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 0.30,
        # Peripheral softness (wide-FOV hint)
        periph_blur_sigma: float = 0.7,
        periph_radius: float = 0.82,
        periph_softness: float = 7.0,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10

        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.blue_lo, self.blue_hi = map(float, blue_band)
        self.green_lo, self.green_hi = map(float, green_band)
        self.red_lo, self.red_hi = map(float, red_band)

        self.panorama_scale = float(panorama_scale)

        self.sky_cool_tint = np.array(sky_cool_tint, np.float32)
        self.sky_haze = float(sky_haze)
        self.ground_warm_tint = np.array(ground_warm_tint, np.float32)
        self.ground_contrast = float(ground_contrast)

        self.uv_overlay_strength = float(np.clip(uv_overlay_strength, 0.0, 1.0))
        self.uv_magenta = np.array(uv_magenta, np.float32)
        self.ridge_sigma = float(ridge_sigma)
        self.ridge_gain = float(ridge_gain)

        self.unsharp_sigma = float(unsharp_sigma)
        self.unsharp_amount = float(unsharp_amount)

        self.periph_blur_sigma = float(periph_blur_sigma)
        self.periph_radius = float(periph_radius)
        self.periph_softness = float(periph_softness)

    # ---------- small helpers ----------
    def _sobel(self, ch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if cv2 is not None:
            gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
            gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
        else:
            x = np.pad(ch, ((0, 0), (1, 1)), mode="reflect")
            y = np.pad(ch, ((1, 1), (0, 0)), mode="reflect")
            gx = 0.5 * (x[:, 2:] - x[:, :-2]).astype(np.float32)
            gy = 0.5 * (y[2:, :] - y[:-2, :]).astype(np.float32)
        return gx.astype(np.float32), gy.astype(np.float32)

    def _ridge_measure(self, u: np.ndarray, sigma: float) -> np.ndarray:
        """
        Simple structure-tensor 'coherence' ridge detector on UV map.
        High when gradients are strong and locally 1-D (trail-like).
        """
        gx, gy = self._sobel(u)
        gxx = gaussian_blur(gx * gx, sigma)
        gyy = gaussian_blur(gy * gy, sigma)
        gxy = gaussian_blur(gx * gy, sigma)
        # eigenvalues of 2x2 structure tensor (per pixel)
        # λ1,2 = (gxx+gyy)/2 ± sqrt(((gxx-gyy)/2)^2 + gxy^2)
        trace = gxx + gyy
        diff = gxx - gyy
        root = np.sqrt(np.maximum((0.5 * diff) ** 2 + gxy * gxy, 0.0)).astype(np.float32)
        lam1 = 0.5 * trace + root  # ≥ lam2
        lam2 = 0.5 * trace - root
        # coherence in [0..1]: (lam1 - lam2) / (lam1 + lam2 + eps)
        eps = 1e-8
        coh = (lam1 - lam2) / (lam1 + lam2 + eps)
        # ridge score: coherence * normalized gradient energy
        energy = np.clip(trace, 0.0, None)
        energy /= np.percentile(energy, 95.0) + 1e-8
        ridge = np.clip(coh * energy, 0.0, 1.0)
        return ridge.astype(np.float32)

    # ---------- main ----------
    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3
        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Mild panorama (wide-FOV hint)
        baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale) if self.panorama_scale != 1.0 else img_lin
        baseline_out = from_float01(linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0)), orig_dtype)

        # 3) RGB->HSI
        use_fast = 0.0 < self.hsi_scale < 1.0
        if use_fast:
            try:
                hsi = classic_rgb_to_hsi_scaled(baseline_lin, wavelengths=self.lambdas, scale=self.hsi_scale)
            except AssertionError:
                hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)

        # 4) Spectral maps
        U = safe_norm(integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi))
        Bv = safe_norm(integrate_band(hsi, self.lambdas, self.blue_lo, self.blue_hi))
        Gv = safe_norm(integrate_band(hsi, self.lambdas, self.green_lo, self.green_hi))
        Rv = safe_norm(integrate_band(hsi, self.lambdas, self.red_lo, self.red_hi))

        # 5) Estimate sky vs ground mask
        H, W = baseline_lin.shape[:2]
        # Blue dominance + vertical prior (upper rows more likely sky)
        vert_prior = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None]  # top=1, bottom=0
        blue_dom = np.clip(Bv - 0.6 * Gv, 0.0, 1.0)
        sky_score = 0.6 * vert_prior + 0.4 * blue_dom
        # Smooth and normalize
        sky_score = gaussian_blur(sky_score, 3.0)
        sky_score = sky_score / (np.percentile(sky_score, 98.0) + 1e-8)
        sky_score = np.clip(sky_score, 0.0, 1.0)
        # Convert to soft probability with a sigmoid
        sky_w = 1.0 / (1.0 + np.exp(-6.0 * (sky_score - 0.45)))
        ground_w = 1.0 - sky_w
        sky_w3 = sky_w[..., None]
        ground_w3 = ground_w[..., None]

        # 6) UV 'trailness' detection on ground
        ridge = self._ridge_measure(U, sigma=self.ridge_sigma)
        trailness = np.clip(self.ridge_gain * ridge * ground_w, 0.0, 1.0)

        # 7) Start render
        render = baseline_lin.copy()

        # Sky: cool + haze
        if self.sky_haze > 0.0:
            a = float(np.clip(self.sky_haze, 0.0, 1.0))
            sky_tinted = np.clip(render * self.sky_cool_tint[None, None, :], 0.0, 1.0)
            render = sky_w3 * ((1.0 - a) * sky_tinted + a * np.array([0.90, 0.97, 1.00], np.float32)) + ground_w3 * render
        else:
            render = sky_w3 * np.clip(render * self.sky_cool_tint[None, None, :], 0.0, 1.0) + ground_w3 * render

        # Ground: warm tint + slight local contrast
        ground_part = render.copy()
        ground_part = np.clip(ground_part * self.ground_warm_tint[None, None, :], 0.0, 1.0)
        if self.ground_contrast > 0.0:
            blurred = gaussian_blur(ground_part, 1.2)
            ground_part = np.clip(ground_part + self.ground_contrast * (ground_part - blurred), 0.0, 1.0)
        render = sky_w3 * render + ground_w3 * ground_part

        # 8) UV false-color overlay on ground (magenta-leaning)
        # Normalize UV for stability, then overlay only on ground
        U95 = U / (np.percentile(U, 95.0) + 1e-8)
        U95 = np.clip(U95, 0.0, 1.0)
        uv_rgb = (U95[..., None]) * self.uv_magenta[None, None, :]
        render = np.clip(
            (1.0 - self.uv_overlay_strength * ground_w3) * render + (self.uv_overlay_strength * ground_w3) * uv_rgb, 0.0, 1.0
        )

        # 9) UV-gated clarity (unsharp scaled by trailness)
        if self.unsharp_sigma > 0.0 and self.unsharp_amount > 0.0:
            blur = gaussian_blur(render, self.unsharp_sigma)
            high = np.clip(render - blur, -1.0, 1.0)
            render = np.clip(render + (self.unsharp_amount * trailness[..., None]) * high, 0.0, 1.0)

        # 10) Peripheral softness (very light)
        if self.periph_blur_sigma > 0.0:
            periph = gaussian_blur(render, self.periph_blur_sigma)
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            t = 1.0 / (1.0 + np.exp(-self.periph_softness * (r - self.periph_radius)))
            render = (1.0 - t[..., None]) * render + t[..., None] * periph

        # 11) Back to sRGB + dtype
        out_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        kestrel_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, kestrel_out)
