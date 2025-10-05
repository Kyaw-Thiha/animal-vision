from typing import Optional, Tuple, Literal
import numpy as np

from animals.animal import Animal
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi

# Shared helpers (from your uv_helpers.py)
from uv_helpers import (
    to_float01,
    from_float01,
    srgb_to_linear,
    linear_to_srgb,
    safe_norm,
    integrate_uv,
    integrate_band,
    snow_glare_tone_compress,
    apply_scatter_and_blue_bias,
    panorama_warp,
)

try:
    import cv2  # optional, for fast down/up sampling
except Exception:
    cv2 = None


Mode = Literal["auto", "day", "night"]


class RatUV(Animal):
    """
    Rat vision simulator (UV-aware), aligned to your Reindeer class structure.

    Highlights (intentionally a bit stylized):
      - UV sensitivity: integrates ~330–400 nm and makes it salient.
      - Panoramic bias: gentle horizontal expand to suggest wide FOV.
      - Low acuity + short-wave tilt: global blur + slight blue lift.
      - Twilight logic: day uses highlight compression; night lifts midtones.
      - Ground-focus vignette: subtly preserves lower frame luminance.

    Returns:
      (baseline_rgb, rat_rgb) — same HxWx3, same dtype as input.
        baseline_rgb : original scene geometry (after optional panorama warp)
        rat_rgb      : simulated rat POV (sRGB-encoded)
    """

    # Default wavelength grid (includes UV)
    DEFAULT_LAMBDAS = np.linspace(320.0, 700.0, 129, dtype=np.float64)  # keep float64 for exact uniform diffs

    # Soft spectral bands (nm)
    UV_BAND = (330.0, 400.0)
    B_BAND = (400.0, 500.0)
    G_BAND = (500.0, 600.0)

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 0.55,  # <1.0 uses downsample/upsample speed path
        panorama_scale: float = 1.45,  # >1 widens horizontal FOV
        uv_boost_alpha: float = 0.55,  # composite weight when blending with baseline
        day_blur_sigma: float = 0.8,
        night_blur_sigma: float = 1.25,
        blue_bias_day: float = 0.03,
        blue_bias_night: float = 0.05,
        tone_knee: float = 0.82,
        tone_strength: float = 0.65,
        ground_vignette_day: float = 0.10,
        ground_vignette_night: float = 0.14,
    ):
        super().__init__()
        # Keep wavelengths in float64 to preserve exact uniform spacing;
        # This prevents strict uniformity checks from failing.
        if lambdas is None:
            self.lambdas = self.DEFAULT_LAMBDAS.copy()
        else:
            wl = np.asarray(lambdas, dtype=np.float64).ravel()
            # Snap to exact uniform between endpoints with same length (belt & suspenders).
            self.lambdas = np.linspace(float(wl[0]), float(wl[-1]), wl.size, dtype=np.float64)

        self.hsi_scale = float(hsi_scale)
        self.panorama_scale = float(panorama_scale)
        self.uv_boost_alpha = float(np.clip(uv_boost_alpha, 0.0, 1.0))

        self.day_blur_sigma = float(day_blur_sigma)
        self.night_blur_sigma = float(night_blur_sigma)
        self.blue_bias_day = float(blue_bias_day)
        self.blue_bias_night = float(blue_bias_night)
        self.tone_knee = float(tone_knee)
        self.tone_strength = float(tone_strength)
        self.ground_vignette_day = float(ground_vignette_day)
        self.ground_vignette_night = float(ground_vignette_night)

        self._uv_band = self.UV_BAND
        self._b_band = self.B_BAND
        self._g_band = self.G_BAND

    # ---------- internals ----------
    @staticmethod
    def _choose_mode(img01: np.ndarray, mode: Mode) -> Mode:
        if mode != "auto":
            return mode
        Y = 0.2126 * img01[..., 0] + 0.7152 * img01[..., 1] + 0.0722 * img01[..., 2]
        return "night" if float(np.median(Y)) < 0.12 else "day"

    @staticmethod
    def _ground_focus_vignette(lin: np.ndarray, amount: float = 0.12) -> np.ndarray:
        H, _W = lin.shape[:2]
        yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
        mask = np.clip(1.0 - yy, 0.0, 1.0)  # 1 @ bottom → 0 @ top
        gain = 1.0 - amount * (1.0 - mask)
        return np.clip(lin * gain[..., None], 0.0, 1.0)

    def _classic_rgb_to_hsi_scaled_nocast(self, rgb_lin: np.ndarray) -> np.ndarray:
        """
        Scaled HSI conversion without down-casting wavelengths to float32.
        Mirrors uv_helpers.classic_rgb_to_hsi_scaled but keeps float64 lambdas.
        """
        if not (0.0 < self.hsi_scale < 1.0) or cv2 is None:
            return classic_rgb_to_hsi(rgb_lin, wavelengths=self.lambdas)

        H, W = rgb_lin.shape[:2]
        h_small = max(1, int(round(H * self.hsi_scale)))
        w_small = max(1, int(round(W * self.hsi_scale)))
        small = cv2.resize(rgb_lin.astype(np.float32, copy=False), (w_small, h_small), interpolation=cv2.INTER_AREA)
        hsi_small = classic_rgb_to_hsi(small, wavelengths=self.lambdas)  # keep float64 wavelengths
        hsi_full = cv2.resize(hsi_small.astype(np.float32, copy=False), (W, H), interpolation=cv2.INTER_LINEAR)
        return hsi_full

    # ---------- public API ----------
    def visualize(self, image: np.ndarray, *, mode: Mode = "auto") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        image : np.ndarray
            HxWx3 RGB. Integer [0..255] or float [0..1].
        mode : {"auto","day","night"}

        Returns
        -------
        (baseline_rgb, rat_rgb)
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."
        orig_dtype = image.dtype

        # 1) sRGB → linear
        img01 = to_float01(image)
        img_lin = srgb_to_linear(img01)

        # 2) Panorama/FOV warp (for baseline alignment)
        if self.panorama_scale and self.panorama_scale != 1.0:
            baseline_lin = panorama_warp(img_lin, scale_x=self.panorama_scale)
        else:
            baseline_lin = img_lin

        baseline_srgb = linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0))
        baseline_out = from_float01(baseline_srgb, orig_dtype)

        # 3) RGB → HSI (scaled path with NO float32 lambdas)
        #    We feed the same geometry (baseline_lin) like your Reindeer class.
        hsi = self._classic_rgb_to_hsi_scaled_nocast(baseline_lin)

        # 4) Integrate spectral bands
        U = integrate_uv(hsi, self.lambdas, *self._uv_band)  # normalized [0,1]
        B = integrate_band(hsi, self.lambdas, *self._b_band)  # raw → scaled in display blend
        G = integrate_band(hsi, self.lambdas, *self._g_band)

        # 5) Build a punchy rat false-color (linear RGB proxy)
        #    R channel leans on UV; G from G; B from B+UV to push short-wave salience.
        #    We normalize via percentiles inside safe_norm for robustness.
        def norm95(x):
            return x / max(1e-8, float(np.percentile(x, 95.0)))

        U_n, B_n, G_n = norm95(U), norm95(B), norm95(G)
        rat_false_lin = np.stack(
            [
                np.clip(0.85 * U_n + 0.10 * G_n, 0.0, 1.0),  # R
                np.clip(0.80 * G_n + 0.20 * B_n, 0.0, 1.0),  # G
                np.clip(0.70 * B_n + 0.40 * U_n, 0.0, 1.0),  # B
            ],
            axis=2,
        ).astype(np.float32)

        # 6) Composite with baseline to keep scene plausibility
        a = self.uv_boost_alpha
        render_lin = np.clip((1.0 - a) * baseline_lin + a * rat_false_lin, 0.0, 1.0)

        # 7) Rat-ish optics & luminance behavior
        mode_eff: Mode = self._choose_mode(img01, mode)
        blur_sigma = self.night_blur_sigma if mode_eff == "night" else self.day_blur_sigma
        blue_bias = self.blue_bias_night if mode_eff == "night" else self.blue_bias_day

        # (a) low acuity + short-wave tilt
        render_lin = apply_scatter_and_blue_bias(render_lin, sigma=blur_sigma, blue_bias=blue_bias)

        # (b) day: compress highlights; night: scotopic lift of mids
        if mode_eff == "day":
            render_lin = snow_glare_tone_compress(render_lin, strength=self.tone_strength, knee=self.tone_knee)
        else:
            Y = 0.2126 * render_lin[..., 0] + 0.7152 * render_lin[..., 1] + 0.0722 * render_lin[..., 2]
            lift = 0.18
            gain = (Y + lift) / (Y + 1e-6)
            render_lin = np.clip(render_lin * gain[..., None], 0.0, 1.0)

        # (c) subtle ground-focus vignette
        gv = self.ground_vignette_night if mode_eff == "night" else self.ground_vignette_day
        render_lin = self._ground_focus_vignette(render_lin, amount=gv)

        # 8) linear → sRGB, restore dtype
        out_srgb = linear_to_srgb(np.clip(render_lin, 0.0, 1.0))
        rat_out = from_float01(out_srgb, orig_dtype)

        return (baseline_out, rat_out)
