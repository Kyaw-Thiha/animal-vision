from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np

try:
    import cv2  # optional
except Exception:
    cv2 = None

# -------------------- constants --------------------
EPS_DEFAULT: float = 1e-8


# -------------------- dtype & color transforms --------------------
def to_float01(x: np.ndarray) -> np.ndarray:
    """Cast image to float32 in [0,1] (assumes sRGB-encoded input)."""
    if x.dtype == np.uint8:
        y = x.astype(np.float32) / 255.0
    else:
        y = x.astype(np.float32)
        if y.max() > 1.001:  # tolerate 0..255 float inputs
            y = np.clip(y / 255.0, 0.0, 1.0)
    return y


def from_float01(img01: np.ndarray, dtype) -> np.ndarray:
    """Restore original dtype from float01 image."""
    if np.issubdtype(dtype, np.integer):
        return np.clip(img01 * 255.0 + 0.5, 0.0, 255.0).astype(dtype)
    return img01.astype(dtype)


def srgb_to_linear(s: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 EOCF (sRGB→linear)."""
    a = 0.055
    lin = np.where(s <= 0.04045, s / 12.92, ((s + a) / (1 + a)) ** 2.4)
    return lin.astype(np.float32)


def linear_to_srgb(l: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 OECF (linear→sRGB)."""
    a = 0.055
    srgb = np.where(l <= 0.0031308, l * 12.92, (1 + a) * np.power(np.clip(l, 0.0, None), 1 / 2.4) - a)
    return srgb.astype(np.float32)


def safe_norm(x: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] guarding tiny ranges."""
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


# -------------------- geometry / filtering --------------------
def resize_preserve_range(x: np.ndarray, out_hw: Tuple[int, int], *, interp: int) -> np.ndarray:
    """Resize HxWxC (or HxW) to (H_out, W_out, [C]); keeps dtype range."""
    assert cv2 is not None, "OpenCV is required for resizing."
    H_out, W_out = out_hw
    was_float = np.issubdtype(x.dtype, np.floating)
    xf = x.astype(np.float32, copy=False)
    y = cv2.resize(xf, (W_out, H_out), interpolation=interp)
    return y.astype(x.dtype, copy=False) if not was_float else y


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with OpenCV if available; otherwise cheap fallback."""
    if sigma <= 0:
        return img
    if cv2 is not None:
        k = int(2 * np.ceil(3 * sigma) + 1)
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
    # box-blur-ish fallback
    r = max(1, int(np.ceil(2 * sigma)))
    from numpy.lib.stride_tricks import sliding_window_view

    pad = ((r, r), (r, r)) + ((0, 0),) if img.ndim == 3 else ((r, r), (r, r))
    x = np.pad(img, pad, mode="reflect")
    win = sliding_window_view(x, (2 * r + 1, 2 * r + 1) + (() if img.ndim == 2 else (1,)))
    return win.mean(axis=(-2, -3))


def panorama_warp(img_lin: np.ndarray, *, scale_x: float) -> np.ndarray:
    """Horizontal widen followed by center-crop back to original width."""
    if abs(scale_x - 1.0) < 1e-3:
        return img_lin
    H, W = img_lin.shape[:2]
    newW = max(2, int(round(W * scale_x)))
    if cv2 is None:
        x = np.linspace(0, W - 1, newW).round().astype(int)
        widened = img_lin[:, x]
    else:
        widened = cv2.resize(img_lin, (newW, H), interpolation=cv2.INTER_CUBIC)
    if newW == W:
        return widened
    start = (newW - W) // 2
    end = start + W
    return widened[:, start:end, :]


def apply_scatter_and_blue_bias(img_lin: np.ndarray, *, sigma: float, blue_bias: float) -> np.ndarray:
    """Lower acuity via blur + add blue bias to B channel."""
    out = img_lin.copy()
    if sigma > 0.15:
        out = gaussian_blur(out, sigma)
    out[..., 2] = np.clip(out[..., 2] + float(blue_bias), 0.0, 1.0)
    return out


def snow_glare_tone_compress(img_lin: np.ndarray, *, strength: float, knee: float = 0.8) -> np.ndarray:
    """Soft-knee highlight compression in linear light."""
    if strength <= 0.0:
        return img_lin
    x = np.clip(img_lin, 0.0, 1.0)
    below = x <= knee
    y = np.empty_like(x)
    y[below] = x[below]
    t = (x[~below] - knee) / (1.0 - knee)
    y[~below] = knee + (1.0 - knee) * (t / (1.0 + strength * t))
    return y


# -------------------- spectral helpers --------------------
def bandpass_weights(lambdas: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Raised-cosine weights on [lo,hi]; normalized to sum≈1."""
    wl = lambdas.astype(np.float32)
    w = np.zeros_like(wl, dtype=np.float32)
    mask = (wl >= lo) & (wl <= hi)
    if not np.any(mask):
        return np.ones_like(wl, dtype=np.float32) / float(wl.size)
    x = (wl[mask] - lo) / (hi - lo)
    w[mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * x))
    s = float(np.sum(w))
    if s > 1e-12:
        w /= s
    else:
        w = np.ones_like(wl, dtype=np.float32) / float(wl.size)
    return w


def integrate_band(hsi: np.ndarray, lambdas: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Integrate HSI over [lo,hi] with raised-cosine weights."""
    weight = bandpass_weights(lambdas, lo, hi)
    band = np.tensordot(hsi, weight, axes=([2], [0]))  # (H,W)
    return band.astype(np.float32)


def integrate_uv(hsi: np.ndarray, lambdas: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """UV estimate in [lo,hi], normalized to [0,1]."""
    uv = integrate_band(hsi, lambdas, lo, hi)
    return safe_norm(uv)


def classic_rgb_to_hsi_scaled(
    rgb01: np.ndarray,
    *,
    wavelengths: np.ndarray,
    scale: float,
    converter: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """
    Downsample → classic_rgb_to_hsi → upsample back to original size.
    - 'converter' should be a function like: classic_rgb_to_hsi(rgb01, wavelengths)
    """
    assert cv2 is not None, "OpenCV is required for fast down/up-sample."
    assert 0.0 < scale <= 1.0, "scale must be (0,1]."
    H, W = rgb01.shape[:2]
    h_small = max(1, int(round(H * scale)))
    w_small = max(1, int(round(W * scale)))

    rgb_small = resize_preserve_range(rgb01, (h_small, w_small), interp=cv2.INTER_AREA)

    if converter is None:
        # Late import to avoid hard dependency
        from ml.classic_rgb_to_hsi.classic_rgb_to_hsi import classic_rgb_to_hsi as _conv

        hsi_small = _conv(rgb_small, wavelengths=wavelengths.astype(np.float32))
    else:
        hsi_small = converter(rgb_small, wavelengths.astype(np.float32))

    hsi_full = resize_preserve_range(hsi_small, (H, W), interp=cv2.INTER_LINEAR)
    return hsi_full


# -------------------- adaptation & illuminant --------------------
def D65_like(lambdas_nm: np.ndarray) -> np.ndarray:
    """Smooth daylight SPD; mean-normalized to 1 for stability."""
    x = (lambdas_nm - 560.0) / 50.0
    base = np.exp(-0.5 * x**2) + 0.3 * np.exp(-0.5 * ((lambdas_nm - 450.0) / 35.0) ** 2)
    base /= base.mean()
    return base.astype(np.float32)


def von_kries_white_patch(U: np.ndarray, B: np.ndarray, G: np.ndarray, eps: float = EPS_DEFAULT):
    Uw = max(U.max(), eps)
    Bw = max(B.max(), eps)
    Gw = max(G.max(), eps)
    return U / Uw, B / Bw, G / Gw


def von_kries_gray_world(U: np.ndarray, B: np.ndarray, G: np.ndarray, eps: float = EPS_DEFAULT):
    Um = max(U.mean(), eps)
    Bm = max(B.mean(), eps)
    Gm = max(G.mean(), eps)
    return U / Um, B / Bm, G / Gm
