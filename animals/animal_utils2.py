import numpy as np
import cv2 as cv
from typing import Optional, Tuple, Union


import numpy as np
import cv2  # only used for fast downsample; remove if you don't want it

# assumes you already have srgb_to_linear() imported in this module
# from animals.animal_utils import srgb_to_linear

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1 + a)) ** 2.4,
    )

def _check_day_night(
    image: np.ndarray,
    threshold_lin: float = 0.10,
    linear_rgb: bool = False,
    *,
    use_percentile: float = 0.50,   # 0.50 = median (robust to highlights)
    downsample_to: int | None = 256, # speed & stability; None = full-res
    hysteresis: float | None = 0.02, # +/- band around threshold; None = off
    prev_mode: str | None = None     # "day"/"night" from previous frame if you have one
):
    """
    Used to pick day vs night.

    - Works for sRGB uint8/float or linear float.
    - If linear_rgb=False, auto-normalizes and converts sRGB->linear first.
    - Uses robust percentile luminance (default: median).
    - Optional hysteresis to avoid mode flicker near the threshold.

    Returns: (mode: "day"/"night", confidence: 0..1, y_stat)
    """
    assert image.ndim == 3 and image.shape[2] == 3, "HxWx3 image required"

    x = image
    # Normalize if needed
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)

    # sRGB -> linear (if the input is not already linear)
    if not linear_rgb:
        x = np.clip(x, 0.0, 1.0)
        x = srgb_to_linear(x)

    # Optional downsample for speed & noise robustness
    if downsample_to is not None:
        H, W = x.shape[:2]
        scale = min(1.0, downsample_to / max(H, W))
        if scale < 1.0:
            x = cv2.resize(x, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    # Linear luminance (BT.709)
    Y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    Y = np.clip(Y, 0.0, 1.0)

    # Robust statistic (median by default; can set to e.g. 0.6 for 60th percentile)
    q = float(np.percentile(Y, use_percentile * 100.0))
    y_stat = q  # keep name consistent with your original "y_med" intent

    # Hysteresis (optional, needs prev_mode if you have it)
    thr_lo = threshold_lin
    thr_hi = threshold_lin
    if hysteresis is not None:
        h = float(max(0.0, hysteresis))
        if prev_mode == "day":
            thr_lo = threshold_lin - h  # require darker to flip to night
        elif prev_mode == "night":
            thr_hi = threshold_lin + h  # require brighter to flip to day

    # Classify
    mode = "day" if y_stat >= thr_hi else "night"
    if thr_lo < y_stat < thr_hi:
        # inside the deadband: stick with previous if provided
        if prev_mode in ("day", "night"):
            mode = prev_mode

    # Confidence: distance from threshold relative to dynamic range around it
    # Use interpercentile spread for scale (25–75th), with small floor
    p25, p75 = np.percentile(Y, [25, 75])
    spread = max(1e-3, float(p75 - p25))
    conf = float(np.clip(abs(y_stat - threshold_lin) / (spread + 1e-6), 0.0, 1.0))

    return mode, conf, y_stat


def check_is_day(image: np.ndarray, **kwargs) -> bool:
    """Convenience. True if day."""
    return _check_day_night(image, **kwargs)[0] == "day"
# --- Enlarge then crop/pad ---
def enlarge_then_crop(
    image: np.ndarray,
    *,
    scale: Optional[float] = None,
    out_size: Optional[Tuple[int, int]] = None,
    crop_anchor: str = "center",        # "center","tl","tr","bl","br"
    pad_value: Union[float, int] = 0,
    interpolation: int = cv.INTER_LINEAR
) -> np.ndarray:
    """
    **Prepare for wide-FOV warps.**
    First resize by 'scale', then crop/pad to 'out_size'.
    """
    assert isinstance(image, np.ndarray) and image.ndim >= 2, "HxW or HxWxC required"

    # resize (optional)
    out = image
    if scale is not None and scale != 1.0:
        h, w = image.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        out = cv.resize(image, (new_w, new_h), interpolation=interpolation)

    if out_size is None:
        return out

    out_w, out_h = out_size
    src_h, src_w = out.shape[:2]

    # crop if larger
    dx, dy = max(0, src_w - out_w), max(0, src_h - out_h)
    if dx > 0 or dy > 0:
        if crop_anchor == "center":
            x0, y0 = dx // 2, dy // 2
        elif crop_anchor == "tl":
            x0, y0 = 0, 0
        elif crop_anchor == "tr":
            x0, y0 = dx, 0
        elif crop_anchor == "bl":
            x0, y0 = 0, dy
        elif crop_anchor == "br":
            x0, y0 = dx, dy
        else:
            raise ValueError("crop_anchor must be one of {'center','tl','tr','bl','br'}")
        x1 = x0 + min(out_w, src_w)
        y1 = y0 + min(out_h, src_h)
        out = out[y0:y1, x0:x1]
        src_h, src_w = out.shape[:2]

    # pad if smaller
    pad_l = max(0, (out_w - src_w)//2); pad_r = max(0, out_w - src_w - pad_l)
    pad_t = max(0, (out_h - src_h)//2); pad_b = max(0, out_h - src_h - pad_t)
    if pad_l or pad_r or pad_t or pad_b:
        pads = ((pad_t, pad_b), (pad_l, pad_r)) if out.ndim == 2 else ((pad_t, pad_b), (pad_l, pad_r), (0, 0))
        out = np.pad(out, pads, mode="constant", constant_values=pad_value)

    return out[:out_h, :out_w].copy()


# --- Binocular wide-FOV warp ---
def animal_fov_binocular_warp(
    img_srgb_01: np.ndarray,
    *,
    fov_in_deg: float,                      # camera/input horiz FOV
    per_eye_half_fov_deg: float,            # animal per-eye half FOV (φ)
    overlap_deg: float,                     # binocular overlap (O)
    out_size: Optional[Tuple[int, int]] = None,
    border_mode: int = cv.BORDER_CONSTANT,
    border_value: Union[float, int] = 0,
) -> np.ndarray:
    """
    **Used to emulate wide FOV + binocular blend.**
    Works on sRGB float [0,1]. Keeps size unless out_size given.
    Note: cannot invent content outside camera FOV.
    """
    assert img_srgb_01.ndim == 3 and img_srgb_01.shape[2] == 3, "HxWx3 float [0,1] required"

    H_in, W_in, _ = img_srgb_01.shape
    if out_size is None:
        out_w, out_h = W_in, H_in
    else:
        out_w, out_h = out_size

    # angles
    phi = np.deg2rad(per_eye_half_fov_deg)      # per-eye half FOV
    psi = np.deg2rad(fov_in_deg * 0.5)          # input half FOV
    O   = np.deg2rad(overlap_deg)
    alpha = max(0.0, phi - 0.5 * O)             # eye axis yaw offset

    # output grid
    u = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)
    v = np.linspace(0.0, float(out_h - 1), out_h, dtype=np.float32)
    U, _ = np.meshgrid(u, v)

    # eye-relative and world yaw
    thetaL = U * phi; thetaR = U * phi
    gammaL = thetaL - alpha
    gammaR = thetaR + alpha

    def yaw_to_xsrc(gamma: np.ndarray) -> np.ndarray:
        xsrc = (gamma / psi) * (W_in * 0.5) + (W_in * 0.5)
        return xsrc.astype(np.float32)

    xL = yaw_to_xsrc(gammaL)
    xR = yaw_to_xsrc(gammaR)
    ymap = np.repeat(np.linspace(0, H_in - 1, out_h, dtype=np.float32)[:, None], out_w, axis=1)

    # validity within input FOV
    validL = (np.abs(gammaL) <= psi).astype(np.float32)
    validR = (np.abs(gammaR) <= psi).astype(np.float32)

    left = cv.remap(img_srgb_01, xL, ymap, interpolation=cv.INTER_LINEAR,
                    borderMode=border_mode, borderValue=border_value)
    right = cv.remap(img_srgb_01, xR, ymap, interpolation=cv.INTER_LINEAR,
                     borderMode=border_mode, borderValue=border_value)

    # weights near each eye axis
    wL = (np.cos(0.5 * np.pi * (thetaL / phi)) ** 2).astype(np.float32) * validL
    wR = (np.cos(0.5 * np.pi * (thetaR / phi)) ** 2).astype(np.float32) * validR
    wsum = (wL + wR + 1e-8)[..., None]

    out = (left * wL[..., None] + right * wR[..., None]) / wsum
    return np.clip(out, 0.0, 1.0).astype(np.float32)

