# --- Center zoom (no enlarge_then_crop) ---
import math
import numpy as np

# Try OpenCV; if not available, we will fall back to Pillow inside center_zoom
try:
    import cv2 as cv
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

def center_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Center-anchored zoom-in by 'scale' (>1). Crop (W/scale,H/scale) around center,
    then resize back to (W,H). If scale<=1, return input.
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
        # Pillow fallback
        from PIL import Image
        pil = Image.fromarray(crop)
        pil = pil.resize((W, H), resample=Image.Resampling.LANCZOS)
        return np.asarray(pil)

def zoom_scale_from_cat_ratio(
    *, camera_hfov_deg: float, cat_per_eye_half_fov_deg: float, cat_to_human_ratio: float
) -> float:
    """
    Convert 'cat wider than human' ratio -> zoom scale.
      eff_cat_hfov = min(camera_hfov, 2*phi)
      human_hfov   = eff_cat_hfov / ratio
      scale        = tan(cam/2) / tan(human/2)
    """
    phi = float(cat_per_eye_half_fov_deg)
    eff_cat_hfov = min(float(camera_hfov_deg), 2.0 * phi)
    ratio = max(1.01, float(cat_to_human_ratio))  # >1 means cat sees wider
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

def _to_float01(x: np.ndarray) -> np.ndarray:
    """uint8/float -> float32 [0,1]."""
    if np.issubdtype(x.dtype, np.integer):
        return (x.astype(np.float32) / 255.0).clip(0.0, 1.0)
    x = x.astype(np.float32)
    if x.max() > 1.001:
        x = x / 255.0
    return x.clip(0.0, 1.0)

def _from_float01(x: np.ndarray, dtype) -> np.ndarray:
    """float [0,1] -> original dtype."""
    x = np.clip(x, 0.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        return (x * 255.0 + 0.5).astype(dtype)
    return x.astype(dtype)

def human_zoom_and_cat_view(
    image: np.ndarray,
    *,
    camera_hfov_deg: float,
    cat_per_eye_half_fov_deg: float,
    binocular_overlap_deg: float,
    cat_to_human_ratio: float = 1.30,   # cat sees ~1.3× wider than human
    zoom_scale: float | None = None,    # set to override ratio-based scale
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      (A) zoomed human-like image (center zoom-in)
      (B) cat FOV warp from ORIGINAL image
    """
    H, W = image.shape[:2]
    orig_dtype = image.dtype

    # zoom scale
    if zoom_scale is None:
        scale = zoom_scale_from_cat_ratio(
            camera_hfov_deg=camera_hfov_deg,
            cat_per_eye_half_fov_deg=cat_per_eye_half_fov_deg,
            cat_to_human_ratio=cat_to_human_ratio,
        )
    else:
        scale = float(zoom_scale)

    # A) zoomed human view
    zoomed = center_zoom(image, scale=scale)

    # B) cat wide-FOV warp on ORIGINAL frame (works in float [0,1])
    img01 = _to_float01(image)
    cat01 = animal_fov_binocular_warp(
        img01,
        fov_in_deg=camera_hfov_deg,
        per_eye_half_fov_deg=cat_per_eye_half_fov_deg,
        overlap_deg=binocular_overlap_deg,
        out_size=(W, H),
        border_mode=0,        # cv.BORDER_CONSTANT
        border_value=0.0,
    )

    return _from_float01(_to_float01(zoomed), orig_dtype), _from_float01(cat01, orig_dtype)
