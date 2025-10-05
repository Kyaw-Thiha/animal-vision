from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np

EPS_DEFAULT: float = 1e-8

try:
    import cv2  # optional
except Exception:
    cv2 = None


# -------------------- visualization mappings --------------------
def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Minimal HSV→RGB (values in [0,1]); interpreted as linear RGB."""
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


def map_falsecolor(U: np.ndarray, B: np.ndarray, G: np.ndarray, eps: float = EPS_DEFAULT) -> np.ndarray:
    """UV→magenta bias, Blue→blue/cyan, Green→green/yellow (linear RGB)."""

    def norm95(x):
        s = np.percentile(x, 95.0)
        return x / max(float(s), eps)

    U_n = norm95(U)
    B_n = norm95(B)
    G_n = norm95(G)
    R = 0.85 * U_n + 0.10 * G_n
    Gc = 0.80 * G_n + 0.20 * B_n
    Bl = 0.70 * B_n + 0.40 * U_n
    return np.clip(np.stack([R, Gc, Bl], axis=2), 0.0, 1.0).astype(np.float32)


def map_linear_matrix(U: np.ndarray, B: np.ndarray, G: np.ndarray, M: np.ndarray) -> np.ndarray:
    """sRGB_linear = M · [U,B,G]^T."""
    H, W = U.shape
    C = np.stack([U, B, G], axis=2).reshape(-1, 3)
    out = (C @ M.T).reshape(H, W, 3)
    return out.astype(np.float32)


def map_opponent(U: np.ndarray, B: np.ndarray, G: np.ndarray, eps: float = EPS_DEFAULT) -> np.ndarray:
    """Opponent mapping (HSV-like) → linear RGB."""
    O1 = G - B
    O2 = B - U
    L = (U + B + G) / 3.0
    angle = np.arctan2(O2, O1)
    hue = (angle + np.pi) / (2 * np.pi)
    radius = np.sqrt(O1 * O1 + O2 * O2)
    sat = radius / (np.percentile(radius, 95.0) + eps)
    val = L / (np.percentile(L, 95.0) + eps)
    hsv = np.stack([hue, np.clip(sat, 0, 1), np.clip(val, 0, 1)], axis=2)
    return hsv_to_rgb(hsv).astype(np.float32)


def map_uv_purple_yellow(U: np.ndarray, eps: float = EPS_DEFAULT) -> np.ndarray:
    """UV-only visualization between purple↔yellow (linear RGB)."""
    if U.ndim == 3 and U.shape[2] == 1:
        U = U[..., 0]
    elif U.ndim != 2:
        raise ValueError(f"U must be HxW or HxWx1, got {U.shape}")
    denom = max(float(np.percentile(U, 99.0)), eps)
    u = (U.astype(np.float32) / denom).clip(0.0, 1.0) ** 0.85

    c_purple_srgb = np.array([128, 0, 150], np.float32) / 255.0
    c_yellow_srgb = np.array([255, 225, 60], np.float32) / 255.0

    def _s2l(v):
        a = 0.055
        return np.where(v <= 0.04045, v / 12.92, ((v + a) / (1 + a)) ** 2.4).astype(np.float32)

    c0 = _s2l(c_purple_srgb)
    c1 = _s2l(c_yellow_srgb)
    u3 = u[..., None]
    rgb_lin = (1.0 - u3) * c0 + u3 * c1
    return np.clip(rgb_lin, 0.0, 1.0).astype(np.float32)


def map_uv_purple_yellow_soft(
    U: np.ndarray,
    *,
    u_gamma: float = 0.90,
    accent_gamma: float = 0.85,
    accent_strength: float = 0.05,
    eps: float = EPS_DEFAULT,
) -> np.ndarray:
    """Warm, pastel UV-only visualization (linear RGB)."""
    if U.ndim == 3 and U.shape[2] == 1:
        U = U[..., 0]
    elif U.ndim != 2:
        raise ValueError(f"U must be HxW or HxWx1, got {U.shape}")
    denom = max(float(np.percentile(U, 98.0)), eps)
    u = (U.astype(np.float32) / denom).clip(0.0, 1.0) ** float(u_gamma)

    c_purple_srgb = np.array([176, 124, 232], np.float32) / 255.0
    c_warm_srgb = np.array([255, 211, 138], np.float32) / 255.0

    def _s2l(v):
        a = 0.055
        return np.where(v <= 0.04045, v / 12.92, ((v + a) / (1 + a)) ** 2.4).astype(np.float32)

    c0 = _s2l(c_purple_srgb)
    c1 = _s2l(c_warm_srgb)

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
    return np.clip(rgb_lin, 0.0, 1.0).astype(np.float32)


def map_falsecolor_uv_mixed(U: np.ndarray, B: np.ndarray, G: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Blend falsecolor with UV purple↔yellow tint (linear RGB)."""
    base = map_falsecolor(U, B, G)
    uv_tint = map_uv_purple_yellow_soft(U)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    mixed = (1.0 - alpha) * base + alpha * uv_tint
    p99 = float(np.percentile(mixed, 99.0))
    if p99 > EPS_DEFAULT:
        mixed = mixed / max(1.0, p99)
    return np.clip(mixed.astype(np.float32), 0.0, 1.0)
