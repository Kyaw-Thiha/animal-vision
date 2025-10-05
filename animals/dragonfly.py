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


class Dragonfly(Animal):
    """
    Dragonfly (Odonata) — stylized UV + polarization vision with dorsal–ventral specializations.

    Visual choices (creative but grounded):
      - Sky polarization bands: orientation-dependent gain over the sky, modulating UV/blue contrast.
      - Water finder: horizontally polarized 'water' regions get a distinct UV/blue pop and specular control.
      - Dorsal (upper visual field): tuned toward sky/UV-blue patterning; higher pol selectivity.
      - Ventral (lower visual field): tuned toward water/green scene; highlights & horizon cues enhanced.
      - Mild panorama widen (wide compound eye wrap), light peripheral softness.

    Returns:
      (baseline_rgb, dragonfly_rgb) — both same HxWx3 and dtype as input.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.25,
        # Spectral bands (nm)
        uv_band: Tuple[float, float] = (320.0, 400.0),
        blue_band: Tuple[float, float] = (440.0, 500.0),
        green_band: Tuple[float, float] = (500.0, 570.0),
        red_band: Tuple[float, float] = (600.0, 680.0),
        # Geometry
        panorama_scale: float = 1.15,
        # Sky/ground split estimation
        sky_prior_strength: float = 0.6,  # weight for top-down prior (top=sky)
        sky_blue_weight: float = 0.4,  # weight for spectral cue (blue dominance)
        sky_sigmoid_mid: float = 0.46,  # mid-point for sky probability sigmoid
        sky_sigmoid_steepness: float = 6.0,
        # Polarization modelling
        sky_pol_strength: float = 0.65,  # how strongly sky pol modulates contrast
        sky_pol_gamma: float = 1.3,  # sharpness of sky pol tuning
        water_pol_strength: float = 0.55,  # modulation over 'water-like' regions (horizontal pol)
        water_pol_gamma: float = 1.2,
        sky_evec_base_deg: float = 90.0,  # reference E-vector near zenith (deg; ~perp to sun az)
        sky_evec_sweep_deg: float = -45.0,  # how E-vector rotates with image row
        # Spectral shaping
        red_kill: float = 0.22,  # deemphasize long-λ slightly
        sky_uv_blue_gain: Tuple[float, float] = (0.25, 0.20),  # (UV->B, B) gains on sky
        water_uv_blue_gain: Tuple[float, float] = (0.30, 0.24),  # (UV->B, B) gains on water
        ventral_green_gain: float = 0.12,  # subtle green support below horizon
        # Clarity / highlights
        base_soft_sigma: float = 0.30,  # pre-sharpen softness
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 0.30,  # global clarity
        highlight_knee: float = 0.85,  # soft-knee compress for specular control
        highlight_strength: float = 0.35,
        # Peripheral acuity
        periph_blur_sigma: float = 0.7,
        periph_radius: float = 0.80,
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

        self.sky_prior_strength = float(sky_prior_strength)
        self.sky_blue_weight = float(sky_blue_weight)
        self.sky_sigmoid_mid = float(sky_sigmoid_mid)
        self.sky_sigmoid_steepness = float(sky_sigmoid_steepness)

        self.sky_pol_strength = float(sky_pol_strength)
        self.sky_pol_gamma = float(sky_pol_gamma)
        self.water_pol_strength = float(water_pol_strength)
        self.water_pol_gamma = float(water_pol_gamma)
        self.sky_evec_base = np.deg2rad(float(sky_evec_base_deg))
        self.sky_evec_sweep = np.deg2rad(float(sky_evec_sweep_deg))

        self.red_kill = float(red_kill)
        self.sky_uv_blue_gain = tuple(map(float, sky_uv_blue_gain))
        self.water_uv_blue_gain = tuple(map(float, water_uv_blue_gain))
        self.ventral_green_gain = float(ventral_green_gain)

        self.base_soft_sigma = float(base_soft_sigma)
        self.unsharp_sigma = float(unsharp_sigma)
        self.unsharp_amount = float(unsharp_amount)
        self.highlight_knee = float(highlight_knee)
        self.highlight_strength = float(highlight_strength)

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

    def _soft_knee(self, lin: np.ndarray, knee: float, amount: float) -> np.ndarray:
        if amount <= 0.0:
            return lin
        x = np.clip(lin, 0.0, 1.0)
        below = x <= knee
        y = np.empty_like(x)
        y[below] = x[below]
        t = (x[~below] - knee) / (1.0 - knee + 1e-8)
        y[~below] = knee + (1.0 - knee) * (t / (1.0 + amount * t))
        return y

    # ---------- main ----------
    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3
        orig_dtype = image.dtype

        # 1) Normalize & scene-linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Mild panorama
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
        Bv = safe_norm(integrate_band(hsi, self.lambdas, self.b_lo, self.b_hi))  # Blue
        Gv = safe_norm(integrate_band(hsi, self.lambdas, self.g_lo, self.g_hi))  # Green
        Rv = safe_norm(integrate_band(hsi, self.lambdas, self.r_lo, self.r_hi))  # Red (ref)

        # 5) Sky probability (soft) using top-down prior + blue dominance
        H, W = baseline_lin.shape[:2]
        vert_prior = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None]  # top=1, bottom=0
        blue_dom = np.clip(Bv - 0.6 * Gv, 0.0, 1.0)
        score = self.sky_prior_strength * vert_prior + self.sky_blue_weight * blue_dom
        score = gaussian_blur(score, 2.5)
        score = score / (np.percentile(score, 98.0) + 1e-8)
        sky_w = 1.0 / (1.0 + np.exp(-self.sky_sigmoid_steepness * (score - self.sky_sigmoid_mid)))
        ground_w = 1.0 - sky_w
        sky_w3 = sky_w[..., None]
        ground_w3 = ground_w[..., None]

        # 6) Polarization fields
        # Local texture orientation from UV+Blue (stable for sky/water cues)
        UB = 0.6 * Bv + 0.4 * U
        gx, gy = self._sobel(UB.astype(np.float32))
        theta = np.arctan2(gy, gx).astype(np.float32)  # local orientation

        # SKY: E-vector as a function of elevation (row y)
        y_norm = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]  # top=0, bottom=1
        sky_evec = self.sky_evec_base + self.sky_evec_sweep * y_norm  # radians (H,1)
        cos2_local = np.cos(2.0 * theta)
        sin2_local = np.sin(2.0 * theta)
        cos2_sky = np.cos(2.0 * sky_evec)
        sin2_sky = np.sin(2.0 * sky_evec)
        align_sky = cos2_local * cos2_sky + sin2_local * sin2_sky  # cos(2Δ) in [-1,1]
        align_sky01 = np.clip(0.5 * (align_sky + 1.0), 0.0, 1.0) ** self.sky_pol_gamma

        # WATER: assume horizontally polarized reflections (E-vector ~ 0°)
        cos2_water = 1.0  # cos(0)
        align_water = cos2_local * cos2_water + sin2_local * 0.0  # = cos(2θ)
        align_water01 = np.clip(0.5 * (align_water + 1.0), 0.0, 1.0) ** self.water_pol_gamma

        # 7) Start render; shape global palette
        render = baseline_lin.copy()
        render[..., 0] = np.clip(render[..., 0] * (1.0 - self.red_kill), 0.0, 1.0)

        if self.base_soft_sigma > 0.0:
            render = gaussian_blur(render, self.base_soft_sigma)

        # Dorsal (sky) modulation — UV/blue contrast scaled by polarization alignment & sky mask
        sky_gain = (1.0 + self.sky_pol_strength * (align_sky01 * sky_w))[..., None]
        render = np.clip(render * (0.95 + 0.05 * sky_w3), 0.0, 1.0)  # tiny exposure balance
        render[..., 2] = np.clip(render[..., 2] + self.sky_uv_blue_gain[1] * (Bv * sky_w * align_sky01), 0.0, 1.0)  # +B
        render[..., 1] = np.clip(render[..., 1] + 0.10 * (U * sky_w * align_sky01), 0.0, 1.0)  # +G via UV scatter proxy
        render = np.clip(render * sky_gain, 0.0, 1.0)

        # Ventral (ground/water) modulation — water finder (horizontal pol) and green support
        water_gain = (1.0 + self.water_pol_strength * (align_water01 * ground_w))[..., None]
        render[..., 2] = np.clip(render[..., 2] + self.water_uv_blue_gain[1] * (Bv * ground_w * align_water01), 0.0, 1.0)  # +B
        render[..., 2] = np.clip(
            render[..., 2] + self.water_uv_blue_gain[0] * (U * ground_w * align_water01), 0.0, 1.0
        )  # +B via UV
        render[..., 1] = np.clip(render[..., 1] + self.ventral_green_gain * (Gv * ground_w), 0.0, 1.0)  # +G
        render = np.clip(render * water_gain, 0.0, 1.0)

        # 8) Global clarity (unsharp), then control highlights with a soft knee (water glint handling)
        if self.unsharp_sigma > 0.0 and self.unsharp_amount > 0.0:
            blur = gaussian_blur(render, self.unsharp_sigma)
            high = np.clip(render - blur, -1.0, 1.0)
            render = np.clip(render + self.unsharp_amount * high, 0.0, 1.0)

        render = self._soft_knee(render, knee=self.highlight_knee, amount=self.highlight_strength)

        # 9) Peripheral softness (light compound-eye edge softness)
        if self.periph_blur_sigma > 0.0:
            periph = gaussian_blur(render, self.periph_blur_sigma)
            yy = (np.linspace(-1.0, 1.0, H, dtype=np.float32))[:, None]
            xx = (np.linspace(-1.0, 1.0, W, dtype=np.float32))[None, :]
            r = np.sqrt(xx * xx + yy * yy)
            t = 1.0 / (1.0 + np.exp(-self.periph_softness * (r - self.periph_radius)))
            render = (1.0 - t[..., None]) * render + t[..., None] * periph

        # 10) Back to sRGB + dtype
        out_srgb = linear_to_srgb(np.clip(render, 0.0, 1.0))
        dragonfly_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, dragonfly_out)
