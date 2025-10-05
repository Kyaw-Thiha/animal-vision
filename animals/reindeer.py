from typing import Optional, Tuple
import numpy as np
import cv2

from ml.MST_plus_plus.predict_code.predict import predict_rgb_to_hsi
from ml.MST_plus_plus.predict_code.predict_torch import predict_rgb_to_hsi_torch
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi
from animals.animal import Animal


class Reindeer(Animal):
    """
    Reindeer vision simulator.

    Key visual traits we emulate (slightly exaggerated on purpose):
      - UV sensitivity: integrates ~320–400 nm energy from the HSI cube and boosts it.
      - Snow-glare control: compress highlights so snow doesn't blow out everything.
      - Winter scatter: lower acuity + bluish bias due to seasonal tapetum shift.
      - Panoramic bias: gentle horizontal expansion to hint at wide FOV.

    Output:
      Tuple[render_rgb, uv_debug] both same HxWx3, same dtype as input.
      - render_rgb: simulated reindeer view (sRGB).
      - uv_debug  : 3-channel heatmap (gray/false-color) showing UV saliency.
    """

    def __init__(
        self,
        *,
        lambdas: Optional[np.ndarray] = None,
        hsi_scale: float = 1.0,
        # Artistic/behavioral knobs:
        uv_band: Tuple[float, float] = (300.0, 410.0),  # nm
        uv_boost: float = 3,  # boost UV contribution
        snow_glare_compression: float = 0.55,  # 0=no compression, ~0.5 gentle, <0.3 strong
        winter_mode: bool = True,  # blue scatter & softer acuity
        scatter_sigma: float = 1.2,  # px, winter blur
        blue_bias: float = 0.08,  # add to B channel in winter
        panorama_scale: float = 1.3,  # horizontal expand (>1 widens FOV)
        return_uv_heatmap: bool = True,
    ):
        self.hsi_scale = float(hsi_scale)
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float32) if lambdas is not None else np.linspace(300.0, 700.0, 81, dtype=np.float32)
        )
        assert self.lambdas.ndim == 1 and self.lambdas.size >= 10, "lambdas must be a 1D vector of wavelengths (nm)."
        self.uv_lo, self.uv_hi = float(uv_band[0]), float(uv_band[1])
        self.uv_boost = float(uv_boost)
        self.snow_glare_compression = float(snow_glare_compression)
        self.winter_mode = bool(winter_mode)
        self.scatter_sigma = float(scatter_sigma)
        self.blue_bias = float(blue_bias)
        self.panorama_scale = float(panorama_scale)
        self.return_uv_heatmap = bool(return_uv_heatmap)

    def _resize_preserve_range(self, x: np.ndarray, out_hw: Tuple[int, int], *, interp: int) -> np.ndarray:
        """
        Resize HxWxC (or HxW) array to (H_out, W_out, C) using OpenCV, keeping dtype/range.
        Works in float32 internally for numeric stability; restores dtype if input was integer.
        """
        assert cv2 is not None, "OpenCV is required for resizing."
        H_out, W_out = out_hw
        was_float = np.issubdtype(x.dtype, np.floating)
        xf = x.astype(np.float32, copy=False)
        if x.ndim == 2:
            y = cv2.resize(xf, (W_out, H_out), interpolation=interp)
        else:
            y = cv2.resize(xf, (W_out, H_out), interpolation=interp)
        return y.astype(x.dtype, copy=False) if not was_float else y

    def _classic_rgb_to_hsi_scaled(
        self,
        rgb01: np.ndarray,
        *,
        wavelengths: np.ndarray,
        scale: float = 0.5,
    ) -> np.ndarray:
        """
        Downsample → classic_rgb_to_hsi → upsample back to original size.
        - Down: INTER_AREA for anti-aliased shrink
        - Up:   INTER_LINEAR for smooth per-band interpolation
        """
        assert cv2 is not None, "OpenCV is required for hsi_downsample fast path."
        H, W = rgb01.shape[:2]
        h_small = max(1, int(round(H * scale)))
        w_small = max(1, int(round(W * scale)))

        # 1) Downsample RGB (sRGB-encoded, float in [0,1])
        rgb_small = self._resize_preserve_range(rgb01, (h_small, w_small), interp=cv2.INTER_AREA)

        # 2) Classic converter at reduced resolution
        hsi_small = classic_rgb_to_hsi(rgb_small, wavelengths=wavelengths.astype(np.float32))
        assert hsi_small.ndim == 3 and h_small == hsi_small.shape[0] and w_small == hsi_small.shape[1]

        # 3) Upsample HSI back to original (band-wise)
        hsi_full = self._resize_preserve_range(hsi_small, (H, W), interp=cv2.INTER_LINEAR)
        return hsi_full

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
            (baseline_rgb, reindeer_rgb). Both same dtype as input.
            - baseline_rgb: original image (only FOV-warped if enabled, so alignment matches).
            - reindeer_rgb: simulated reindeer view.
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."

        orig_dtype = image.dtype

        # ---------- 1) to float32 [0,1] ----------
        img = self._to_float01(image)

        # ---------- 2) sRGB -> scene-linear ----------
        img_lin = self._srgb_to_linear(img)

        # ---------- 3) Optional FOV/panorama warp on the *baseline* ----------
        # Apply first so both baseline and processed outputs share geometry.
        if self.panorama_scale and self.panorama_scale != 1.0:
            baseline_lin = self._panorama_warp(img_lin, scale_x=self.panorama_scale)
        else:
            baseline_lin = img_lin

        # Prepare baseline output (no other changes):
        baseline_srgb = self._linear_to_srgb(np.clip(baseline_lin, 0.0, 1.0))
        baseline_out = self._from_float01(baseline_srgb, orig_dtype)

        # ---------- 4) RGB -> HSI using *your* converter ----------
        hsi = self._classic_rgb_to_hsi_scaled(
            baseline_lin, wavelengths=np.asarray(self.lambdas, dtype=np.float32), scale=0.1
        )  # (H, W, N_bands) float32

        # ---------- 5) UV + visible integrations ----------
        uv_map = self._integrate_uv(hsi, self.lambdas, self.uv_lo, self.uv_hi)
        vis_map = self._integrate_band(hsi, self.lambdas, 420.0, 680.0)
        vis_map = self._safe_norm(vis_map)

        # ---------- 6) UV saliency ----------
        uv_saliency = self._safe_norm(uv_map / (1e-6 + 0.6 * vis_map))

        # ---------- 7) Reindeer rendering on top of the baseline geometry ----------
        render_lin = baseline_lin.copy()
        # UV overlay: cool/cyan-ish lift
        render_lin[..., 2] = np.clip(render_lin[..., 2] + self.uv_boost * 0.35 * uv_saliency, 0.0, 1.0)  # +B
        render_lin[..., 1] = np.clip(render_lin[..., 1] + self.uv_boost * 0.15 * uv_saliency, 0.0, 1.0)  # +G

        # Snow-glare compression
        render_lin = self._snow_glare_tone_compress(render_lin, strength=self.snow_glare_compression)

        # Winter scatter & blue bias
        if self.winter_mode:
            render_lin = self._apply_scatter_and_blue_bias(render_lin, sigma=self.scatter_sigma, blue_bias=self.blue_bias)

        # ---------- 8) linear -> sRGB, restore dtype ----------
        render_srgb = self._linear_to_srgb(np.clip(render_lin, 0.0, 1.0))
        reindeer_out = self._from_float01(render_srgb, orig_dtype)

        return (baseline_out, reindeer_out)

    # ---------- helpers ----------
    def _integrate_uv(self, hsi: np.ndarray, lambdas: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """
        Integrate HSI over a UV band. Uses a gentle cosine window inside [lo,hi] and 0 outside.
        Returns HxW float32 in [0,1] (normalized).
        """
        weight = self._bandpass_weights(lambdas, lo, hi)
        uv = np.tensordot(hsi, weight, axes=([2], [0]))  # (H,W)
        return self._safe_norm(uv)

    def _integrate_band(self, hsi: np.ndarray, lambdas: np.ndarray, lo: float, hi: float) -> np.ndarray:
        weight = self._bandpass_weights(lambdas, lo, hi)
        band = np.tensordot(hsi, weight, axes=([2], [0]))  # (H,W)
        return band.astype(np.float32)

    def _bandpass_weights(self, lambdas: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """
        Cosine-tapered bandpass on [lo,hi]; zero elsewhere. Normalized to sum=1 when possible.
        """
        wl = lambdas.astype(np.float32)
        w = np.zeros_like(wl, dtype=np.float32)
        mask = (wl >= lo) & (wl <= hi)
        if not np.any(mask):
            # avoid zero sum; fall back to tiny uniform
            return np.ones_like(wl, dtype=np.float32) / float(wl.size)

        # cosine window inside the passband
        x = (wl[mask] - lo) / (hi - lo)
        w[mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * x))  # raised-cosine in-band
        s = float(np.sum(w))
        if s > 1e-12:
            w /= s
        else:
            w = np.ones_like(wl, dtype=np.float32) / float(wl.size)
        return w

    def _snow_glare_tone_compress(self, img_lin: np.ndarray, strength: float) -> np.ndarray:
        """
        Compress very bright values to preserve texture in snow.
        Simple soft-knee curve applied per channel in linear light.
        strength in [0..1], larger -> stronger compression of highs.
        """
        if strength <= 0.0:
            return img_lin
        x = np.clip(img_lin, 0.0, 1.0)
        knee = 0.8
        # piecewise: below knee ~identity; above knee compress with a rational curve
        below = x <= knee
        y = np.empty_like(x)
        y[below] = x[below]
        # smooth compression above knee
        t = (x[~below] - knee) / (1.0 - knee)
        y[~below] = knee + (1.0 - knee) * (t / (1.0 + strength * t))
        return y

    def _apply_scatter_and_blue_bias(self, img_lin: np.ndarray, *, sigma: float, blue_bias: float) -> np.ndarray:
        """
        Lower acuity via Gaussian blur + bias the blue channel slightly (winter tapetum 'blue').
        """
        try:
            import cv2
        except Exception:
            cv2 = None

        out = img_lin.copy()
        if cv2 is not None and sigma > 0.15:
            k = max(3, int(2.0 * round(3.0 * sigma) + 1))  # odd kernel size ~ 6*sigma
            out = cv2.GaussianBlur(out, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=4)
        out[..., 2] = np.clip(out[..., 2] + blue_bias, 0.0, 1.0)
        return out

    def _panorama_warp(self, img_lin: np.ndarray, *, scale_x: float) -> np.ndarray:
        """
        Gentle horizontal expansion to hint at panoramic FOV.
        Implemented as resize in X then center-crop back to original W.
        """
        if abs(scale_x - 1.0) < 1e-3:
            return img_lin
        try:
            import cv2
        except Exception:
            cv2 = None

        H, W = img_lin.shape[:2]
        newW = max(2, int(round(W * scale_x)))
        if cv2 is None:
            # numpy fallback: nearest neighbor
            x = np.linspace(0, W - 1, newW).round().astype(int)
            widened = img_lin[:, x]
        else:
            widened = cv2.resize(img_lin, (newW, H), interpolation=cv2.INTER_CUBIC)

        # center-crop back to W
        if newW == W:
            return widened
        start = (newW - W) // 2
        end = start + W
        return widened[:, start:end, :]

    # ---------- color/dtype helpers ----------
    def _to_float01(self, img: np.ndarray) -> np.ndarray:
        if np.issubdtype(img.dtype, np.integer):
            return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return img.astype(np.float32).clip(0.0, 1.0)

    def _from_float01(self, img01: np.ndarray, dtype) -> np.ndarray:
        if np.issubdtype(dtype, np.integer):
            return np.clip(img01 * 255.0 + 0.5, 0.0, 255.0).astype(dtype)
        return img01.astype(dtype)

    def _srgb_to_linear(self, s: np.ndarray) -> np.ndarray:
        # IEC 61966-2-1
        a = 0.055
        lin = np.where(s <= 0.04045, s / 12.92, ((s + a) / (1 + a)) ** 2.4)
        return lin.astype(np.float32)

    def _linear_to_srgb(self, l: np.ndarray) -> np.ndarray:
        a = 0.055
        srgb = np.where(l <= 0.0031308, l * 12.92, (1 + a) * np.power(l, 1 / 2.4) - a)
        return srgb.astype(np.float32)

    def _safe_norm(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-9:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)
