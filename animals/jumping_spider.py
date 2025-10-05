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
    panorama_warp,  # kept for parity; we won't widen much
    classic_rgb_to_hsi_scaled,
)

try:
    import cv2  # optional
except Exception:
    cv2 = None


class JumpingSpider(Animal):
    """
    Jumping spider (Salticidae) — stylized UV+Green, foveated & scanning vision.

    Emphasis:
      - UV patch saliency (Difference-of-Gaussians on UV).
      - Green↔UV opponent contrast (vegetation vs display patches).
      - Foveated acuity: sharp central 'principal-eye' region; strong peripheral blur/vignette.
      - Micro 'scanline' attention bands + a couple static attention spots to mimic retinal scanning.

    Returns:
      (baseline_rgb, spider_rgb) — both same HxWx3 and dtype as input.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,
        # Spectral bands (nm)
        uv_band: Tuple[float, float] = (320.0, 400.0),
        green_band: Tuple[float, float] = (500.0, 570.0),
        red_band: Tuple[float, float] = (600.0, 680.0),
        blue_band: Tuple[float, float] = (430.0, 500.0),
        # Geometry / FOV vibe
        panorama_scale: float = 1.02,  # spiders don't have wide panorama; keep ~1.0
        # UV saliency (patches, iridescences)
        dog_small_sigma: float = 0.9,
        dog_large_sigma: float = 2.2,
        uv_patch_gain: float = 0.95,  # strength of UV patch pop
        # Opponent contrast: G vs UV (and a touch of Blue for cool speculars)
        opponent_gain: float = 0.30,
        # Global palette shaping
        red_kill: float = 0.25,  # deemphasize long-λ a bit
        base_soft_sigma: float = 0.25,  # tiny soften before clarity
        clarity_sigma: float = 0.9,
        clarity_amount: float = 0.24,
        # Foveation (principal eyes)
        fovea_radius: float = 0.38,  # normalized radius of sharp center
        fovea_softness: float = 10.0,  # transition steepness
        periph_blur_sigma: float = 2.2,  # strong peripheral blur
        periph_vignette_strength: float = 0.22,  # darken edges a touch
        # Scanline attention (horizontal)
        scan_row_freq: float = 22.0,  # waves across height
        scan_row_gain: float = 0.08,  # ± contrast around 1
        scan_soften: float = 0.9,
        # Attention spots (static stand-ins for moveable retina)
        spots: Tuple[Tuple[float, float], ...] = ((0.50, 0.52), (0.57, 0.48)),  # (y,x) in [0..1]
        spot_sigma: float = 0.08,  # gaussian radius (normalized)
        spot_gain: float = 0.20,  # brightness/clarity lift at spots
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10

        self.uv_lo, self.uv_hi = map(float, uv_band)
        self.g_lo, self.g_hi = map(float, green_band)
        self.r_lo, self.r_hi = map(float, red_band)
        self.b_lo, self.b_hi = map(float, blue_band)

        self.panorama_scale = float(panorama_scale)

        self.dog_small_sigma = float(dog_small_sigma)
        self.dog_large_sigma = float(dog_large_sigma)
        self.uv_patch_gain = float(uv_patch_gain)
        self.opponent_gain = float(opponent_gain)

        self.red_kill = float(red_kill)
        self.base_soft_sigma = float(base_soft_sigma)
        self.clarity_sigma = float(clarity_sigma)
        self.clarity_amount = float(clarity_amount)

        self.fovea_radius = float(fovea_radius)
        self.fovea_softness = float(fovea_softness)
        self.periph_blur_sigma = float(periph_blur_sigma)
        self.periph_vignette_strength = float(periph_vignette_strength)

        self.scan_row_freq = float(scan_row_freq)
        self.scan_row_gain = float(scan_row_gain)
        self.scan_soften = float(scan_soften)

        self.spots = tuple((float(y), float(x)) for (y, x) in spots)
        self.spot_sigma = float(spot_sigma)
        self.spot_gain = float(spot_gain)

    # ---------- small helpers ----------
    def _unsharp(self, img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        if sigma <= 0.0 or amount <= 0.0:
            return img
        blur = gaussian_blur(img, sigma)
        high = np.clip(img - blur, -1.0, 1.0)
        return np.clip(img + amount * high, 0.0, 1.0)

    def _make_attention_spots(self, H: int, W: int) -> np.ndarray:
        """Return (H,W) attention mask with a few gaussian spots near center."""
        yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
        mask = np.zeros((H, W), np.float32)
        s2 = max(self.spot_sigma, 1e-4) ** 2
        for yc, xc in self.spots:
            mask += np.exp(-((yy - yc) ** 2 + (xx - xc) ** 2) / (2.0 * s2))
        # Normalize to [0,1]
        m95 = max(1e-8, float(np.percentile(mask, 95.0)))
        return np.clip(mask / m95, 0.0, 1.0).astype(np.float32)

    # ---------- main ----------
    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3
        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Geometry: spiders aren't panorama-heavy; keep ~1.0
        baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale) if self.panorama_scale != 1.0 else img_lin
        baseline_out = from_float01(linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0)), orig_dtype)

        # 3) RGB -> HSI
        use_fast = 0.0 < self.hsi_scale < 1.0
        if use_fast:
            try:
                hsi = classic_rgb_to_hsi_scaled(baseline_lin, wavelengths=self.lambdas, scale=self.hsi_scale)
            except AssertionError:
                hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(baseline_lin, wavelengths=self.lambdas)

        # 4) Spectral maps
        U = safe_norm(integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi))  # UV
        Gv = safe_norm(integrate_band(hsi, self.lambdas, self.g_lo, self.g_hi))  # Green
        Bv = safe_norm(integrate_band(hsi, self.lambdas, self.b_lo, self.b_hi))  # Blue (for cool sheens)
        Rv = safe_norm(integrate_band(hsi, self.lambdas, self.r_lo, self.r_hi))  # Red (reference)

        # 5) Start render and set overall palette
        render = baseline_lin.copy()
        render[..., 0] = np.clip(render[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)  # reduce red a bit

        if self.base_soft_sigma > 0.0:
            render = gaussian_blur(render, self.base_soft_sigma)

        # 6) UV patch saliency via DoG on U
        uv_small = gaussian_blur(U, self.dog_small_sigma)
        uv_large = gaussian_blur(U, self.dog_large_sigma)
        uv_dog = np.clip(uv_small - uv_large, 0.0, 1.0)
        uv_patch = uv_dog / (np.percentile(uv_dog, 95.0) + 1e-8)
        uv_patch = np.clip(uv_patch, 0.0, 1.0)

        # 7) Green↔UV opponent contrast
        #    Positive where green dominates (foliage/background), negative where UV dominates (display patches).
        opp = Gv - U
        opp = opp / (np.percentile(np.abs(opp), 95.0) + 1e-8)
        opp = np.clip(opp, -1.0, 1.0)

        # Apply opponent shaping to channels (push G up where opp>0; push B/R up a bit where opp<0 with UV patches)
        g_boost = np.clip(opp, 0.0, 1.0) * self.opponent_gain
        u_boost = np.clip(-opp, 0.0, 1.0) * self.opponent_gain
        render[..., 1] = np.clip(render[..., 1] + 0.40 * g_boost, 0.0, 1.0)  # +G for green-dominant
        render[..., 2] = np.clip(render[..., 2] + 0.30 * u_boost * (Bv), 0.0, 1.0)  # +B toward cool UV sheens
        render[..., 0] = np.clip(render[..., 0] + 0.12 * u_boost * (U), 0.0, 1.0)  # small +R for iridescent warmth

        # 8) UV patch pop (gated contrast & saturation)
        if self.clarity_sigma > 0.0 and self.clarity_amount > 0.0:
            blurred = gaussian_blur(render, self.clarity_sigma)
            high = np.clip(render - blurred, -1.0, 1.0)
            render = np.clip(render + (self.clarity_amount * self.uv_patch_gain * uv_patch[..., None]) * high, 0.0, 1.0)

        # 9) Scanline attention bands (horizontal, subtle)
        if self.scan_row_gain != 0.0:
            H, W = render.shape[:2]
            y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]  # (H,1)
            rows = 0.5 + 0.5 * np.sin(2.0 * np.pi * self.scan_row_freq * y)  # (H,1)
            rows = rows * np.ones((1, W), dtype=np.float32)  # (H,W)
            if self.scan_soften > 0.0:
                rows = gaussian_blur(rows, self.scan_soften)  # (H,W)
            row_gain = 1.0 + self.scan_row_gain * (rows - 0.5)  # (H,W)
            render = np.clip(render * row_gain[..., None], 0.0, 1.0)  # (H,W,3)

        # 10) Attention spots (brighten/clarify a couple of foveal targets)
        H, W = render.shape[:2]
        spots_mask = self._make_attention_spots(H, W)  # (H,W)
        if self.spot_gain > 0.0:
            # Local brightness lift
            render = np.clip(render + self.spot_gain * spots_mask[..., None], 0.0, 1.0)
            # Local clarity lift
            sharp = self._unsharp(render, sigma=0.8, amount=0.25)
            render = np.clip((1.0 - 0.6 * spots_mask[..., None]) * render + (0.6 * spots_mask[..., None]) * sharp, 0.0, 1.0)

        # 11) Foveated acuity: strong peripheral blur + vignette
        if self.periph_blur_sigma > 0.0 or self.periph_vignette_strength > 0.0:
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            # 0 at center → 1 at edge
            edge_w = 1.0 / (1.0 + np.exp(-self.fovea_softness * (r - self.fovea_radius)))
            edge_w3 = edge_w[..., None]
            if self.periph_blur_sigma > 0.0:
                periph = gaussian_blur(render, self.periph_blur_sigma)
                render = (1.0 - edge_w3) * render + edge_w3 * periph
            if self.periph_vignette_strength > 0.0:
                vign = 1.0 - self.periph_vignette_strength * edge_w
                render = np.clip(render * vign[..., None], 0.0, 1.0)

        # 12) Back to sRGB + dtype
        out_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        spider_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, spider_out)
