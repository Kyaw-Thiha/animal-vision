from typing import Optional, Literal, Tuple, Callable
import numpy as np

from ml.MST_plus_plus.predict_code.predict import predict_rgb_to_hsi
from ml.MST_plus_plus.predict_code.predict_torch import predict_rgb_to_hsi_torch
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi
from animals.animal import Animal

try:
    import cv2 as cv  # used for resizing/blur; falls back if unavailable
except Exception:
    cv = None


class HoneyBee(Animal):
    """
    Honeybee (Apis mellifera) vision simulator using a pre-trained RGB→HSI ML model.

    Pipeline per frame:
      0) Validate & normalize input dtype
      1) Predict HSI cube from the RGB input (your ML model)
      2) Choose an illuminant E(λ) (D65 by default) and compute per-pixel Radiance(λ)
         NOTE: If your model already outputs Radiance (not reflectance), you can skip the illuminant step.
      3) Integrate honeybee cone catches: U, B, G  (UV/Blue/Green)
      4) Chromatic adaptation (von Kries) — either "white_patch" or "gray_world"
      5) Optional acuity blur (honeybees have lower spatial acuity than humans)
      6) Map (U,B,G) to display sRGB using one of several visualization modes:
            - "falsecolor": interpretable mapping (UV→magenta bias, Blue→blue/cyan, Green→green/yellow)
            - "custom_matrix": user-supplied 3×3 matrix M so sRGB = M·[U B G]^T
            - "opponent": convert to 2D opponent signals then render via HSV-like mapping
         (Pick based on your demo goals; "falsecolor" is robust and simple.)
      7) Tone/gamut handling and convert back to the original dtype

    Assumptions:
      - The ML model returns an HSI cube with shape (H, W, C_hsi), e.g., 31 bands.
      - If `hsi_band_centers_nm` is not provided, we assume 31 bands uniformly from 400–700 nm inclusive.

    Parameters you can tweak in __init__:
      - onnx_path: path to your RGB→HSI ONNX model
      - hsi_band_centers_nm: 1D array of nm centers for the HSI bands
      - illuminant: function or array defining E(λ); defaults to D65
      - adaptation: "white_patch" | "gray_world" | None
      - mapping_mode: "falsecolor" | "custom_matrix" | "opponent"
      - custom_matrix: optional 3×3 for mapping [U,B,G]→sRGB (linear)
      - blur_sigma_px: optional Gaussian σ (in pixels) for acuity blur
      - assume_hsi_is_reflectance: True if model outputs reflectance, False if it already outputs radiance
      - hsi_downsample: if True, run classic_rgb_to_hsi at reduced resolution for speed
      - hsi_scale: scale factor (e.g., 0.5 halves H and W) when hsi_downsample is True
    """

    def __init__(
        self,
        onnx_path: str = "./ml/MST_plus_plus/export/mst_plus_plus.onnx",
        hsi_band_centers_nm: Optional[np.ndarray] = None,
        illuminant: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        adaptation: Optional[Literal["white_patch", "gray_world"]] = "white_patch",
        mapping_mode: Literal[
            "falsecolor", "custom_matrix", "opponent", "uv_purple_yellow", "falsecolor_uv_mixed"
        ] = "uv_purple_yellow",
        custom_matrix: Optional[np.ndarray] = None,
        blur_sigma_px: Optional[float] = 0.2,
        assume_hsi_is_reflectance: bool = True,
        *,
        hsi_downsample: bool = False,
        hsi_scale: float = 0.1,
    ):
        self.onnx_path = onnx_path
        self.adaptation = adaptation
        self.mapping_mode = mapping_mode
        self.custom_matrix = custom_matrix
        self.blur_sigma_px = blur_sigma_px or 0.0
        self.assume_hsi_is_reflectance = assume_hsi_is_reflectance

        # Fast-path controls for classic_rgb_to_hsi
        self.hsi_downsample = bool(hsi_downsample)
        self.hsi_scale = float(hsi_scale)

        # Default band centers: 31 bands from 400..700 nm inclusive
        if hsi_band_centers_nm is None:
            self.lambdas = np.linspace(400.0, 700.0, 31, dtype=np.float32)
        else:
            self.lambdas = np.asarray(hsi_band_centers_nm, dtype=np.float32)

        # Illuminant function E(λ)
        self.E = illuminant if illuminant is not None else self._D65_like

        # Precompute bee cone curves sampled to band centers (simple log-normal shapes)
        self.UV_curve, self.Blue_curve, self.Green_curve = self._honeybee_cone_curves(self.lambdas)

        # Normalize cone curves so their integrals are comparable (helps stability)
        for v in (self.UV_curve, self.Blue_curve, self.Green_curve):
            s = v.sum()
            if s > 0:
                v /= s

        # For opponent mapping, small epsilon to avoid divide-by-zero
        self._eps = 1e-8

    # ------------------- Public API -------------------

    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Run the honeybee vision pipeline on an HxWx3 RGB image."""
        # 0) Validate & normalize
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."

        orig_dtype = image.dtype
        img = self._to_float01(image)

        # 1) RGB → HSI via your ML model or classic converter
        # hsi = predict_rgb_to_hsi(img, self.onnx_path)  # shape (H,W,C_hsi), float32
        # hsi = predict_rgb_to_hsi_torch(img, "mst_plus_plus", "./ml/MST_plus_plus/model_zoo/mst_plus_plus.pth", overlap=512)
        if self.hsi_downsample and cv is not None and 0.05 <= self.hsi_scale < 1.0:
            hsi = self._classic_rgb_to_hsi_scaled(
                img, wavelengths=np.asarray(self.lambdas, dtype=np.float32), scale=self.hsi_scale
            )
        else:
            # Fallback to full-res (either by choice or if OpenCV not available)
            hsi = classic_rgb_to_hsi(img, wavelengths=np.asarray(self.lambdas, dtype=np.float32))

        assert hsi.ndim == 3 and hsi.shape[:2] == img.shape[:2], "HSI must match H and W."
        bands = hsi.shape[2]
        assert bands == len(self.lambdas), f"HSI bands ({bands}) != length of provided band centers ({len(self.lambdas)})."

        # 2) Radiance(λ): multiply by illuminant if your HSI is reflectance
        if self.assume_hsi_is_reflectance:
            E = self.E(self.lambdas).astype(hsi.dtype)  # shape (C_hsi,)
            radiance = hsi * E[None, None, :]
        else:
            radiance = hsi  # already radiance from the model

        # 3) Cone catches (per pixel): U, B, G = Σ Radiance(λ) * Cone(λ)
        U = np.tensordot(radiance, self.UV_curve, axes=([2], [0]))  # (H,W)
        B = np.tensordot(radiance, self.Blue_curve, axes=([2], [0]))
        G = np.tensordot(radiance, self.Green_curve, axes=([2], [0]))

        # 4) Chromatic adaptation (optional)
        if self.adaptation == "white_patch":
            U, B, G = self._von_kries_white_patch(U, B, G)
        elif self.adaptation == "gray_world":
            U, B, G = self._von_kries_gray_world(U, B, G)

        # 5) Optional spatial acuity blur
        if (self.blur_sigma_px or 0) > 0:
            U = self._gaussian_blur(U, self.blur_sigma_px)
            B = self._gaussian_blur(B, self.blur_sigma_px)
            G = self._gaussian_blur(G, self.blur_sigma_px)

        # 6) Map (U,B,G) → sRGB (linear), then encode and restore dtype
        if self.mapping_mode == "falsecolor":
            rgb_lin = self._map_falsecolor(U, B, G)
        elif self.mapping_mode == "custom_matrix":
            assert self.custom_matrix is not None and self.custom_matrix.shape == (3, 3), (
                "Provide custom_matrix as 3x3 for 'custom_matrix' mode."
            )
            rgb_lin = self._map_linear_matrix(U, B, G, self.custom_matrix)
        elif self.mapping_mode == "opponent":
            rgb_lin = self._map_opponent(U, B, G)
        elif self.mapping_mode == "uv_purple_yellow":
            rgb_lin = self._map_uv_purple_yellow_soft(U)
            assert rgb_lin.ndim == 3 and rgb_lin.shape[2] == 3, f"uv_purple_yellow returned {rgb_lin.shape}"
        elif self.mapping_mode == "falsecolor_uv_mixed":
            self.uv_mix_alpha = 0.45  # [0.25 - 0.45]
            rgb_lin = self._map_falsecolor_uv_mixed(U, B, G, alpha=self.uv_mix_alpha)
        else:
            raise ValueError(f"Unknown mapping_mode: {self.mapping_mode}")

        rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

        out_srgb = self._linear_to_srgb(rgb_lin)
        if np.issubdtype(orig_dtype, np.integer):
            out = (out_srgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            out = out_srgb.astype(orig_dtype)
        return image, out

    # ------------------- Fast-path helpers -------------------

    def _resize_preserve_range(self, x: np.ndarray, out_hw: Tuple[int, int], *, interp: int) -> np.ndarray:
        """
        Resize HxWxC (or HxW) array to (H_out, W_out, C) using OpenCV, keeping dtype/range.
        Works in float32 internally for numeric stability; restores dtype if input was integer.
        """
        assert cv is not None, "OpenCV is required for resizing."
        H_out, W_out = out_hw
        was_float = np.issubdtype(x.dtype, np.floating)
        xf = x.astype(np.float32, copy=False)
        if x.ndim == 2:
            y = cv.resize(xf, (W_out, H_out), interpolation=interp)
        else:
            y = cv.resize(xf, (W_out, H_out), interpolation=interp)
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
        assert cv is not None, "OpenCV is required for hsi_downsample fast path."
        H, W = rgb01.shape[:2]
        h_small = max(1, int(round(H * scale)))
        w_small = max(1, int(round(W * scale)))

        # 1) Downsample RGB (sRGB-encoded, float in [0,1])
        rgb_small = self._resize_preserve_range(rgb01, (h_small, w_small), interp=cv.INTER_AREA)

        # 2) Classic converter at reduced resolution
        hsi_small = classic_rgb_to_hsi(rgb_small, wavelengths=wavelengths.astype(np.float32))
        assert hsi_small.ndim == 3 and h_small == hsi_small.shape[0] and w_small == hsi_small.shape[1]

        # 3) Upsample HSI back to original (band-wise)
        hsi_full = self._resize_preserve_range(hsi_small, (H, W), interp=cv.INTER_LINEAR)
        return hsi_full

    # ------------------- Spectral Models -------------------

    def _D65_like(self, lambdas_nm: np.ndarray) -> np.ndarray:
        """
        Lightweight D65-ish daylight SPD over 400–700 nm.
        This is a smooth curve peaking in the blue-green; normalized to mean=1 for stability.
        Swap with a measured E(λ) if you have one.
        """
        x = (lambdas_nm - 560.0) / 50.0
        base = np.exp(-0.5 * x**2) + 0.3 * np.exp(-0.5 * ((lambdas_nm - 450.0) / 35.0) ** 2)
        base /= base.mean()
        return base.astype(np.float32)

    def _honeybee_cone_curves(self, lambdas_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple log-normal-ish cone sensitivity shapes for honeybee UV, Blue, Green.
        Peaks roughly near: UV ~350 nm, Blue ~440 nm, Green ~540 nm.
        Replace with published curves for higher fidelity.
        """

        def log_normal(λ, peak, sigma):
            return np.exp(-0.5 * ((λ - peak) / sigma) ** 2)

        UV = log_normal(lambdas_nm, 350.0, 25.0)
        B = log_normal(lambdas_nm, 440.0, 30.0)
        G = log_normal(lambdas_nm, 540.0, 35.0)
        return UV.astype(np.float32), B.astype(np.float32), G.astype(np.float32)

    # ------------------- Adaptation -------------------

    def _von_kries_white_patch(self, U: np.ndarray, B: np.ndarray, G: np.ndarray):
        """von Kries-type adaptation using the brightest pixel per channel."""
        Uw = max(U.max(), self._eps)
        Bw = max(B.max(), self._eps)
        Gw = max(G.max(), self._eps)
        return U / Uw, B / Bw, G / Gw

    def _von_kries_gray_world(self, U: np.ndarray, B: np.ndarray, G: np.ndarray):
        """von Kries-type adaptation using channel means (gray-world)."""
        Um = max(U.mean(), self._eps)
        Bm = max(B.mean(), self._eps)
        Gm = max(G.mean(), self._eps)
        return U / Um, B / Bm, G / Gm

    # ------------------- Visualization Mappings -------------------

    def _map_falsecolor(self, U: np.ndarray, B: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        FALSECOLOR mapping: UV salient as magenta/pink; Blue as blue/cyan; Green as green/yellow.
        """

        def norm95(x):
            s = np.percentile(x, 95.0)
            return x / max(s, self._eps)

        U_n = norm95(U)
        B_n = norm95(B)
        G_n = norm95(G)

        R = 0.85 * U_n + 0.10 * G_n
        Gc = 0.80 * G_n + 0.20 * B_n
        Bl = 0.70 * B_n + 0.40 * U_n
        return np.stack([R, Gc, Bl], axis=2)

    def _map_linear_matrix(self, U: np.ndarray, B: np.ndarray, G: np.ndarray, M: np.ndarray) -> np.ndarray:
        """CUSTOM 3×3 MATRIX mapping: sRGB_linear = M · [U, B, G]^T"""
        H, W = U.shape
        C = np.stack([U, B, G], axis=2).reshape(-1, 3)
        return (C @ M.T).reshape(H, W, 3)

    def _map_opponent(self, U: np.ndarray, B: np.ndarray, G: np.ndarray) -> np.ndarray:
        """OPPONENT visualization (HSV-like angle/radius mapping)."""
        O1 = G - B
        O2 = B - U
        L = (U + B + G) / 3.0

        angle = np.arctan2(O2, O1)  # [-π, π]
        hue = (angle + np.pi) / (2 * np.pi)  # [0,1]
        radius = np.sqrt(O1 * O1 + O2 * O2)
        sat = radius / (np.percentile(radius, 95.0) + self._eps)
        val = L / (np.percentile(L, 95.0) + self._eps)

        hsv = np.stack([hue, np.clip(sat, 0, 1), np.clip(val, 0, 1)], axis=2)
        return self._hsv_to_rgb(hsv)

    def _map_uv_purple_yellow(self, U: np.ndarray) -> np.ndarray:
        U = np.asarray(U)
        if U.ndim == 3 and U.shape[2] == 1:
            U = U[..., 0]
        elif U.ndim != 2:
            raise ValueError(f"U must be HxW or HxWx1, got shape {U.shape}")

        denom = float(np.percentile(U, 99.0))
        denom = max(denom, float(self._eps))
        u = (U.astype(np.float32) / denom).clip(0.0, 1.0)
        u = u**0.85

        c_purple_srgb = np.array([128, 0, 150], np.float32) / 255.0
        c_yellow_srgb = np.array([255, 225, 60], np.float32) / 255.0

        def srgb_to_linear(v: np.ndarray) -> np.ndarray:
            a = 0.055
            return np.where(v <= 0.04045, v / 12.92, ((v + a) / (1 + a)) ** 2.4).astype(np.float32)

        c0 = srgb_to_linear(c_purple_srgb)
        c1 = srgb_to_linear(c_yellow_srgb)

        u3 = u[..., None]
        rgb_lin = (1.0 - u3) * c0[None, None, :] + u3 * c1[None, None, :]
        return np.ascontiguousarray(np.clip(rgb_lin, 0.0, 1.0), dtype=np.float32)

    def _map_uv_purple_yellow_soft(
        self,
        U: np.ndarray,
        *,
        u_gamma: float = 0.90,
        accent_gamma: float = 0.85,
        accent_strength: float = 0.05,
    ) -> np.ndarray:
        """Warm, pastel UV-only visualization; returns linear sRGB (HxWx3)."""
        U = np.asarray(U)
        if U.ndim == 3 and U.shape[2] == 1:
            U = U[..., 0]
        elif U.ndim != 2:
            raise ValueError(f"U must be HxW or HxWx1, got {U.shape}")

        eps = float(self._eps)

        denom = float(np.percentile(U, 98.0))
        denom = max(denom, eps)
        u = (U.astype(np.float32) / denom).clip(0.0, 1.0)
        u = u ** float(u_gamma)

        c_purple_srgb = np.array([176, 124, 232], np.float32) / 255.0
        c_warm_srgb = np.array([255, 211, 138], np.float32) / 255.0

        def srgb_to_linear(v: np.ndarray) -> np.ndarray:
            a = 0.055
            return np.where(v <= 0.04045, v / 12.92, ((v + a) / (1 + a)) ** 2.4).astype(np.float32)

        c0 = srgb_to_linear(c_purple_srgb)
        c1 = srgb_to_linear(c_warm_srgb)

        u3 = u[..., None]
        rgb_lin = (1.0 - u3) * c0 + u3 * c1

        gray = np.array([0.5, 0.5, 0.5], np.float32)
        purple_dir = c0 - gray
        a = float(accent_strength)
        if a > 0:
            w = (u ** float(accent_gamma))[..., None]
            rgb_lin = rgb_lin + a * w * purple_dir

        Y = (0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]) + eps
        Y_target = np.clip(0.22 + 0.55 * u, 0.0, 1.0)
        gain = (Y_target / Y)[..., None]
        gain = np.clip(gain, 0.6, 1.6)
        rgb_lin = rgb_lin * gain

        rgb_lin = rgb_lin / (1.0 + 0.6 * rgb_lin)
        return np.clip(np.ascontiguousarray(rgb_lin, dtype=np.float32), 0.0, 1.0)

    def _map_falsecolor_uv_mixed(self, U: np.ndarray, B: np.ndarray, G: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """Blend falsecolor with UV purple↔yellow tint in linear sRGB."""
        base = self._map_falsecolor(U, B, G)
        uv_tint = self._map_uv_purple_yellow_soft(U)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        mixed = (1.0 - alpha) * base + alpha * uv_tint
        p95 = np.percentile(mixed, 99.0)
        if p95 > 1e-8:
            p95 = np.float32(np.percentile(mixed, 99.0))
            scale = np.maximum(np.float32(1.0), p95)
            mixed = mixed / float(scale)
        return np.clip(mixed.astype(np.float32), 0.0, 1.0)

    # ------------------- Utilities -------------------

    def _to_float01(self, x: np.ndarray) -> np.ndarray:
        """Cast image to float32 in [0,1] (assumes sRGB-encoded input)."""
        if x.dtype == np.uint8:
            y = x.astype(np.float32) / 255.0
        else:
            y = x.astype(np.float32)
            if y.max() > 1.001:
                y = np.clip(y / 255.0, 0, 1)
        return y

    def _linear_to_srgb(self, rgb_lin: np.ndarray) -> np.ndarray:
        """IEC 61966-2-1 sRGB EOTF (linear → sRGB-encoded)."""
        a = 0.055
        out = np.where(
            rgb_lin <= 0.0031308,
            12.92 * rgb_lin,
            (1 + a) * np.power(np.clip(rgb_lin, 0.0, None), 1 / 2.4) - a,
        )
        return out

    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        """Minimal HSV→RGB (values in [0,1]). Interpreted as linear RGB for our visualization."""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0).astype(np.int32)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        i_mod = i % 6
        r = np.select([i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5], [v, q, p, p, t, v], default=0)
        g = np.select([i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5], [t, v, v, q, p, p], default=0)
        b = np.select([i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5], [p, p, t, v, v, q], default=0)
        return np.stack([r, g, b], axis=2)

    def _gaussian_blur(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian blur with OpenCV if available; otherwise fallback to a separable approx."""
        if sigma <= 0:
            return img
        if cv is not None:
            k = int(2 * np.ceil(3 * sigma) + 1)
            return cv.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv.BORDER_REFLECT101)
        # naive numpy fallback (box blur approx)
        r = max(1, int(np.ceil(2 * sigma)))
        from numpy.lib.stride_tricks import sliding_window_view

        pad = ((r, r), (r, r))
        x = np.pad(img, pad, mode="reflect")
        win = sliding_window_view(x, (2 * r + 1, 2 * r + 1))
        return win.mean(axis=(-1, -2))
