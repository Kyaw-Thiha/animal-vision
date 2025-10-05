# ========= Center Zoom & FOV helpers =========
import math
import numpy as np

try:
    import cv2 as cv
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

def center_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Center-anchored zoom-in by 'scale' (>1).
    Crop center (W/scale,H/scale) and resize back to (W,H).
    """
    assert image.ndim >= 2, "HxW or HxWxC"
    if scale <= 1.0:
        return image
    H, W = image.shape[:2]
    cw = max(1, int(round(W / scale)))
    ch = max(1, int(round(H / scale)))
    x0 = (W - cw) // 2
    y0 = (H - ch) // 2
    crop = image[y0:y0+ch, x0:x0+cw]
    if _HAS_CV2:
        return cv.resize(crop, (W, H), interpolation=cv.INTER_LINEAR)
    else:
        from PIL import Image
        return np.asarray(Image.fromarray(crop).resize((W, H), Image.Resampling.LANCZOS))

def zoom_scale_from_cat_ratio(
    *, camera_hfov_deg: float, cat_per_eye_half_fov_deg: float, cat_to_human_ratio: float
) -> float:
    """
    Convert 'cat wider than human' ratio -> zoom scale.
    eff_cat_hfov = min(camera_hfov, 2*phi); human_hfov = eff_cat_hfov/ratio;
    scale = tan(cam/2)/tan(human/2)
    """
    phi = float(cat_per_eye_half_fov_deg)
    eff_cat_hfov = min(float(camera_hfov_deg), 2.0 * phi)
    ratio = max(1.01, float(cat_to_human_ratio))
    cam = math.tan(math.radians(camera_hfov_deg) * 0.5)
    hum = math.tan(math.radians(eff_cat_hfov / ratio) * 0.5)
    return float(cam / max(hum, 1e-6))

def animal_fov_binocular_warp(
    img_srgb_01: np.ndarray,
    *,
    fov_in_deg: float,
    per_eye_half_fov_deg: float,
    overlap_deg: float,
    out_size: tuple | None = None,
    border_mode: int = 0,    # cv.BORDER_CONSTANT
    border_value: float | int = 0,
) -> np.ndarray:
    """
    Wide-FOV + binocular blend on float [0,1] RGB.
    """
    if not _HAS_CV2:
        raise ImportError("cv2 is required for binocular warp. Install opencv-python.")
    assert img_srgb_01.ndim == 3 and img_srgb_01.shape[2] == 3

    H_in, W_in, _ = img_srgb_01.shape
    out_w, out_h = (W_in, H_in) if out_size is None else out_size

    phi = np.deg2rad(per_eye_half_fov_deg)
    psi = np.deg2rad(fov_in_deg * 0.5)
    O   = np.deg2rad(overlap_deg)
    alpha = max(0.0, phi - 0.5 * O)  # eye axis offset

    u = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)
    v = np.linspace(0.0, float(out_h - 1), out_h, dtype=np.float32)
    U, _ = np.meshgrid(u, v)

    thetaL = U * phi; thetaR = U * phi
    gammaL = thetaL - alpha
    gammaR = thetaR + alpha

    def yaw_to_xsrc(gamma: np.ndarray) -> np.ndarray:
        xsrc = (gamma / psi) * (W_in * 0.5) + (W_in * 0.5)
        return xsrc.astype(np.float32)

    xL = yaw_to_xsrc(gammaL); xR = yaw_to_xsrc(gammaR)
    ymap = np.repeat(np.linspace(0, H_in - 1, out_h, dtype=np.float32)[:, None], out_w, axis=1)

    validL = (np.abs(gammaL) <= psi).astype(np.float32)
    validR = (np.abs(gammaR) <= psi).astype(np.float32)

    left = cv.remap(img_srgb_01, xL, ymap, interpolation=cv.INTER_LINEAR,
                    borderMode=border_mode, borderValue=border_value)
    right = cv.remap(img_srgb_01, xR, ymap, interpolation=cv.INTER_LINEAR,
                     borderMode=border_mode, borderValue=border_value)

    wL = (np.cos(0.5 * np.pi * (thetaL / phi)) ** 2).astype(np.float32) * validL
    wR = (np.cos(0.5 * np.pi * (thetaR / phi)) ** 2).astype(np.float32) * validR
    wsum = (wL + wR + 1e-8)[..., None]

    out = (left * wL[..., None] + right * wR[..., None]) / wsum
    return np.clip(out, 0.0, 1.0).astype(np.float32)
