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


class MantisShrimp(Animal):
    """
    Mantis shrimp (stylized) — hyper-spectral + polarization flair.

    Design choices (creative but grounded in their 'too many' channels & polarization):
      - Hyper-spectral slices: multiple narrow bands across UV→red produce a 'spectral barcode'.
      - Polarization strike: orientation-dependent gain (linear pol proxy) + phase-shifted twin to hint at circular pol.
      - Midband scanlines: light horizontal 'sampling rows' animation-ready (static here).
      - Channel independence: per-pixel 'winner-take-most' tint to exaggerate categorical color sense.
      - Clear tropical water look: minimal haze, crisp micro-contrast.

    Returns:
      (baseline_rgb, mantis_rgb) — both same HxWx3 and dtype as input.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,  # downsample HSI path if < 1.0
        panorama_scale: float = 1.12,  # light widen
        # Spectral slices (nm) — tune/extend freely
        bands: Tuple[Tuple[float, float], ...] = (
            (320.0, 360.0),  # UV1
            (360.0, 400.0),  # UV2
            (400.0, 430.0),  # V
            (430.0, 460.0),  # B1
            (460.0, 490.0),  # B2/C
            (490.0, 520.0),  # G1
            (520.0, 550.0),  # G2
            (550.0, 580.0),  # Y
            (580.0, 610.0),  # O
            (610.0, 680.0),  # R
        ),
        # Ocean clarity & contrast
        red_kill: float = 0.18,
        haze_strength: float = 0.03,
        haze_tint: Tuple[float, float, float] = (0.92, 0.98, 1.00),
        pre_soft_sigma: float = 0.25,
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 0.32,
        # Polarization stylization
        evec_angle_deg: float = 30.0,  # 'sun' E-vector direction (global)
        pol_linear_strength: float = 0.55,
        pol_linear_gamma: float = 1.2,
        pol_circular_strength: float = 0.35,  # phase-shifted twin (circular-ish)
        orientation_mix: float = 0.5,  # 0=global only, 1=local only
        # Spectral 'barcode' → RGB remapping
        barcode_saturation: float = 0.40,  # overlay saturation
        barcode_opacity: float = 0.55,  # blend of barcode tint into render
        winner_take_most: float = 0.35,  # how much the max-band wins locally
        # Midband scanline effect
        scan_row_freq: float = 26.0,  # waves across image height
        scan_row_gain: float = 0.08,  # contrast gain by row
        scan_soften: float = 0.8,  # blur of row mask
        # Peripheral acuity
        periph_blur_sigma: float = 0.7,
        periph_radius: float = 0.80,
        periph_softness: float = 7.0,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10, "lambdas must be a 1D wavelength vector (nm)."

        self.panorama_scale = float(panorama_scale)
        self.bands = tuple((float(a), float(b)) for (a, b) in bands)

        self.red_kill = float(red_kill)
        self.haze_strength = float(haze_strength)
        self.haze_tint = np.array(haze_tint, dtype=np.float32)
        self.pre_soft_sigma = float(pre_soft_sigma)
        self.unsharp_sigma = float(unsharp_sigma)
        self.unsharp_amount = float(unsharp_amount)

        self.evec_angle = np.deg2rad(float(evec_angle_deg))
        self.pol_linear_strength = float(pol_linear_strength)
        self.pol_linear_gamma = float(pol_linear_gamma)
        self.pol_circular_strength = float(pol_circular_strength)
        self.orientation_mix = float(np.clip(orientation_mix, 0.0, 1.0))

        self.barcode_saturation = float(barcode_saturation)
        self.barcode_opacity = float(np.clip(barcode_opacity, 0.0, 1.0))
        self.winner_take_most = float(np.clip(winner_take_most, 0.0, 1.0))

        self.scan_row_freq = float(scan_row_freq)
        self.scan_row_gain = float(scan_row_gain)
        self.scan_soften = float(scan_soften)

        self.periph_blur_sigma = float(periph_blur_sigma)
        self.periph_radius = float(periph_radius)
        self.periph_softness = float(periph_softness)

    # --------- helpers ---------
    def _sobel(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if cv2 is not None:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REFLECT101)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REFLECT101)
        else:
            x = np.pad(img, ((0, 0), (1, 1)), mode="reflect")
            y = np.pad(img, ((1, 1), (0, 0)), mode="reflect")
            gx = 0.5 * (x[:, 2:] - x[:, :-2]).astype(np.float32)
            gy = 0.5 * (y[2:, :] - y[:-2, :]).astype(np.float32)
        return gx.astype(np.float32), gy.astype(np.float32)

    def _rgb_to_luma(self, lin: np.ndarray) -> np.ndarray:
        return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)

    def _apply_saturation(self, lin: np.ndarray, s: float) -> np.ndarray:
        if s == 1.0:
            return lin
        Y = self._rgb_to_luma(lin)[..., None]
        return np.clip(Y + (lin - Y) * s, 0.0, 1.0).astype(np.float32)

    # --------- main ---------
    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3
        dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Mild panorama widen
        baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale) if self.panorama_scale != 1.0 else img_lin
        baseline_out = from_float01(linear_to_srgb(np.clip(baseline_lin, 0, 1)), dtype)

        # 3) RGB→HSI (optional downsample)
        use_fast = 0.0 < self.hsi_scale < 1.0
        if use_fast:
            try:
                hsi = classic_rgb_to_hsi_scaled(baseline_lin, wavelengths=self.lambdas, scale=self.hsi_scale)
            except AssertionError:
                hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)

        H, W = baseline_lin.shape[:2]

        # 4) Integrate many narrow spectral bands → stack into 'barcode'
        band_maps = []
        for lo, hi in self.bands:
            bm = safe_norm(integrate_band(hsi, self.lambdas, lo, hi))
            band_maps.append(bm)
        S = np.stack(band_maps, axis=2).astype(np.float32)  # (H,W,N)
        N = S.shape[2]

        # 5) Build a vivid spectral-color LUT (N hues around the circle)
        idx = np.arange(N, dtype=np.float32)
        hue = (idx / max(N, 1)).astype(np.float32)  # [0..1)

        # HSV→RGB (linear-ish), simple mapping
        def hsv2rgb(h, s, v):
            i = np.floor(h * 6.0).astype(np.int32)
            f = h * 6.0 - i
            p = v * (1.0 - s)
            q = v * (1.0 - f * s)
            t = v * (1.0 - (1.0 - f) * s)
            i = i % 6
            out = np.stack(
                [
                    np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v], default=v),
                    np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p], default=v),
                    np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q], default=v),
                ],
                axis=-1,
            ).astype(np.float32)
            return out

        lut = hsv2rgb(hue, np.full_like(hue, 0.95, np.float32), np.ones_like(hue, np.float32))  # (N,3)

        # 6) 'Winner-take-most' categorical tint + soft mixing from full spectrum
        S_norm = S / (np.percentile(S, 95.0) + 1e-8)
        S_norm = np.clip(S_norm, 0.0, 1.0)
        max_idx = np.argmax(S_norm, axis=2)  # (H,W)
        max_val = np.take_along_axis(S_norm, max_idx[..., None], axis=2)[..., 0]  # (H,W)
        # Soft mix tint = normalized weighted sum of LUT by S_norm
        weights = S_norm / (np.sum(S_norm, axis=2, keepdims=True) + 1e-8)
        soft_rgb = weights @ lut  # (H,W,3)
        hard_rgb = lut[max_idx]  # (H,W,3)
        barcode_rgb = (1.0 - self.winner_take_most) * soft_rgb + self.winner_take_most * hard_rgb
        # Add saturation
        Yb = (0.2126 * barcode_rgb[..., 0] + 0.7152 * barcode_rgb[..., 1] + 0.0722 * barcode_rgb[..., 2])[..., None]
        barcode_rgb = np.clip(Yb + (barcode_rgb - Yb) * (1.0 + self.barcode_saturation), 0.0, 1.0)

        # 7) Basic clear-water look
        render = baseline_lin.copy()
        render[..., 0] = np.clip(render[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)
        if self.haze_strength > 0.0:
            a = float(np.clip(self.haze_strength, 0.0, 1.0))
            render = (1.0 - a) * render + a * self.haze_tint[None, None, :]
        if self.pre_soft_sigma > 0.0:
            render = gaussian_blur(render, self.pre_soft_sigma)

        # 8) Polarization effects (linear + 'circular-ish')
        #    Use local orientation from the broad-spectrum energy (sum of bands) for stability.
        broad = np.mean(S_norm, axis=2).astype(np.float32)  # (H,W)
        gx, gy = self._sobel(broad)
        theta = np.arctan2(gy, gx).astype(np.float32)  # local edge/texture orientation

        # Linear pol alignment (cos^2 law in doubled-angle space), blended with global axis
        cos2_local = np.cos(2.0 * theta)
        sin2_local = np.sin(2.0 * theta)
        cos2_global = float(np.cos(2.0 * self.evec_angle))
        sin2_global = float(np.sin(2.0 * self.evec_angle))
        mix = self.orientation_mix
        cos2_mix = (1.0 - mix) * cos2_global + mix * cos2_local
        sin2_mix = (1.0 - mix) * sin2_global + mix * sin2_local
        align = cos2_mix  # cos(2*(theta-phi)) in [-1,1]
        align01 = np.clip(0.5 * (align + 1.0), 0.0, 1.0) ** self.pol_linear_gamma

        # 'Circular-ish' component: 90° phase shift (sine) to light up orthogonal/phase-rotated structure
        align_circ = np.clip(0.5 * (sin2_mix + 1.0), 0.0, 1.0)

        pol_gain = 1.0 + self.pol_linear_strength * align01 + self.pol_circular_strength * align_circ
        # Unsharp guided by polarization gain
        if self.unsharp_sigma > 0.0 and self.unsharp_amount > 0.0:
            blur = gaussian_blur(render, self.unsharp_sigma)
            high = np.clip(render - blur, -1.0, 1.0)
            render = np.clip(render + (self.unsharp_amount * pol_gain[..., None]) * high, 0.0, 1.0)

        # 9) Blend in the spectral barcode tint
        render = np.clip((1.0 - self.barcode_opacity) * render + self.barcode_opacity * barcode_rgb, 0.0, 1.0)

        # 10) Midband scanline effect (horizontal sampling rows)
        if self.scan_row_gain != 0.0:
            H, W = render.shape[:2]

            # Make a 2D mask with width W so broadcasting matches (H,W,3)
            y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]  # (H,1)
            rows = 0.5 + 0.5 * np.sin(2.0 * np.pi * self.scan_row_freq * y)  # (H,1)
            rows = rows * np.ones((1, W), dtype=np.float32)  # (H,W)

            if self.scan_soften > 0.0:
                rows = gaussian_blur(rows, self.scan_soften)  # (H,W)

            row_gain = 1.0 + self.scan_row_gain * (rows - 0.5)  # (H,W)
            render = np.clip(render * row_gain[..., None], 0.0, 1.0)  # (H,W,3)

        # 11) Light peripheral softness
        if self.periph_blur_sigma > 0.0:
            periph = gaussian_blur(render, self.periph_blur_sigma)
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            t = 1.0 / (1.0 + np.exp(-self.periph_softness * (r - self.periph_radius)))
            t = t[..., None]
            render = (1.0 - t) * render + t * periph

        # 12) Back to sRGB + dtype
        out = from_float01(linear_to_srgb(np.clip(render, 0.0, 1.0)), dtype)
        return (baseline_out, out)
