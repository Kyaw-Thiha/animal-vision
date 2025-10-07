from typing import Optional, Literal, Tuple, Callable
import numpy as np

from animals.animal import Animal
from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi

# Shared helpers
from uv_helpers import (
    to_float01,
    linear_to_srgb,
    D65_like,
    von_kries_white_patch,
    von_kries_gray_world,
    gaussian_blur,
    classic_rgb_to_hsi_scaled,
    EPS_DEFAULT,
)

from uv_mappers import (
    map_falsecolor,
    map_linear_matrix,
    map_opponent,
    map_uv_purple_yellow,  # unused by default, but available
    map_uv_purple_yellow_soft,  # default for "uv_purple_yellow"
    map_falsecolor_uv_mixed,
)


class HoneyBee(Animal):
    """
    Honeybee (Apis mellifera) vision simulator using a pre-trained/fallback RGB→HSI model.

    Pipeline:
      0) Validate & normalize
      1) RGB → HSI (ML or classic); optional downsample speed path
      2) If HSI is reflectance, multiply by illuminant E(λ) to get radiance
      3) Integrate bee cone catches U/B/G
      4) Chromatic adaptation (von Kries)
      5) Optional spatial blur (acuity)
      6) Map (U,B,G) → linear sRGB via chosen visualization
      7) Encode to sRGB and restore dtype

    Returns:
      (baseline_rgb, bee_rgb) — both same dtype as input. Baseline is the *input image*.
    """

    def __init__(
        self,
        onnx_path: str = "./ml/MST_plus_plus/export/mst_plus_plus.onnx",
        hsi_band_centers_nm: Optional[np.ndarray] = None,
        illuminant: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        adaptation: Optional[Literal["white_patch", "gray_world"]] = "white_patch",
        mapping_mode: Literal[
            "falsecolor",
            "custom_matrix",
            "opponent",
            "uv_purple_yellow",
            "falsecolor_uv_mixed",
        ] = "opponent",
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
        self.blur_sigma_px = float(blur_sigma_px or 0.0)
        self.assume_hsi_is_reflectance = assume_hsi_is_reflectance

        # Speed controls for classic_rgb_to_hsi
        self.hsi_downsample = bool(hsi_downsample)
        self.hsi_scale = float(hsi_scale)

        # Band centers (default: 31 from 400..700 nm)
        self.lambdas = (
            np.linspace(400.0, 700.0, 31, dtype=np.float32)
            if hsi_band_centers_nm is None
            else np.asarray(hsi_band_centers_nm, dtype=np.float32)
        )

        # Illuminant E(λ)
        self.E = illuminant if illuminant is not None else D65_like

        # Bee cone curves (simple log-normal shapes; replace with measured data if available)
        self.UV_curve, self.Blue_curve, self.Green_curve = self._honeybee_cone_curves(self.lambdas)
        for v in (self.UV_curve, self.Blue_curve, self.Green_curve):
            s = v.sum()
            if s > 0:
                v /= s

        self._eps = EPS_DEFAULT

    # ------------------- Public API -------------------

    def visualize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Run the honeybee vision pipeline on an HxWx3 RGB image; returns (baseline, bee_render)."""
        # 0) Validate & normalize
        assert isinstance(image, np.ndarray), "Input must be a numpy ndarray."
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB."
        orig_dtype = image.dtype
        baseline = image  # baseline is the original image (unaltered)
        img01 = to_float01(image)

        # 1) RGB → HSI (choose downsample fast path if enabled)
        if self.hsi_downsample and 0.05 <= self.hsi_scale < 1.0:
            # Try fast path; if OpenCV is unavailable (assert), fall back to full-res
            try:
                hsi = classic_rgb_to_hsi_scaled(
                    img01,
                    wavelengths=self.lambdas,
                    scale=self.hsi_scale,
                )
            except AssertionError:
                hsi = classic_rgb_to_hsi(img01, wavelengths=self.lambdas)
        else:
            hsi = classic_rgb_to_hsi(img01, wavelengths=self.lambdas)

        assert hsi.ndim == 3 and hsi.shape[:2] == img01.shape[:2], "HSI must match H and W."
        assert hsi.shape[2] == len(self.lambdas), "HSI bands must match wavelength vector length."

        # 2) Radiance(λ) if HSI is reflectance
        if self.assume_hsi_is_reflectance:
            E = self.E(self.lambdas).astype(hsi.dtype)  # (C_hsi,)
            radiance = hsi * E[None, None, :]
        else:
            radiance = hsi

        # 3) Cone catches U/B/G
        U = np.tensordot(radiance, self.UV_curve, axes=([2], [0]))  # (H,W)
        B = np.tensordot(radiance, self.Blue_curve, axes=([2], [0]))
        G = np.tensordot(radiance, self.Green_curve, axes=([2], [0]))

        # 4) Chromatic adaptation
        if self.adaptation == "white_patch":
            U, B, G = von_kries_white_patch(U, B, G, eps=self._eps)
        elif self.adaptation == "gray_world":
            U, B, G = von_kries_gray_world(U, B, G, eps=self._eps)

        # 5) Spatial acuity blur (optional)
        if self.blur_sigma_px > 0:
            U = gaussian_blur(U, self.blur_sigma_px)
            B = gaussian_blur(B, self.blur_sigma_px)
            G = gaussian_blur(G, self.blur_sigma_px)

        # 6) Map (U,B,G) → linear sRGB
        if self.mapping_mode == "falsecolor":
            rgb_lin = map_falsecolor(U, B, G, eps=self._eps)
        elif self.mapping_mode == "custom_matrix":
            assert self.custom_matrix is not None and self.custom_matrix.shape == (3, 3), (
                "Provide custom_matrix as 3×3 for 'custom_matrix' mode."
            )
            rgb_lin = map_linear_matrix(U, B, G, self.custom_matrix)
        elif self.mapping_mode == "opponent":
            rgb_lin = map_opponent(U, B, G, eps=self._eps)
        elif self.mapping_mode == "uv_purple_yellow":
            rgb_lin = map_uv_purple_yellow_soft(U)  # (linear RGB)
        elif self.mapping_mode == "falsecolor_uv_mixed":
            rgb_lin = map_falsecolor_uv_mixed(U, B, G, alpha=0.45)
        else:
            raise ValueError(f"Unknown mapping_mode: {self.mapping_mode}")

        rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

        # 7) Encode sRGB and restore dtype
        out_srgb = linear_to_srgb(rgb_lin)
        if np.issubdtype(orig_dtype, np.integer):
            out = (out_srgb * 255.0 + 0.5).astype(orig_dtype)
        else:
            out = out_srgb.astype(orig_dtype)

        return baseline, out

    # ------------------- Bee spectral model (kept local) -------------------

    def _honeybee_cone_curves(self, lambdas_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Log-normal-ish cone sensitivity shapes for honeybee UV/Blue/Green.
        Peaks ~350 nm (UV), ~440 nm (Blue), ~540 nm (Green).
        Replace with measured cone fundamentals for higher fidelity.
        """

        def log_normal(λ, peak, sigma):
            return np.exp(-0.5 * ((λ - peak) / sigma) ** 2)

        UV = log_normal(lambdas_nm, 350.0, 25.0)
        Blue = log_normal(lambdas_nm, 440.0, 30.0)
        Green = log_normal(lambdas_nm, 540.0, 35.0)
        return UV.astype(np.float32), Blue.astype(np.float32), Green.astype(np.float32)
