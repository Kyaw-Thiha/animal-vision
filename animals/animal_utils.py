import numpy as np
import cv2


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1 + a)) ** 2.4,
    )

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        (1 + a) * (x ** (1 / 2.4)) - a,
    )

def check_input_image(image: np.ndarray) -> bool:
    """
    This function is responsible for checking if the input image is valid

    The function does the following checks:
    - check if it's a np.ndarray instance
    - if it has a dimension of N x M x 3
    - if each values inside are numerical values, instead of from other datatype
    the function returns true iff the above are ALL satisfied
    """

    if not isinstance(image, np.ndarray):
        return False
    if not image.ndim == 3 or not image.shape[2] == 3:
        return False
    if not np.issubdtype(image.dtype, np.number):
        return False

    return True

def get_normalized_image(image: np.ndarray) -> np.ndarray:
    """
    This function returns the normalized version of input image in float32
    """
    image_output = image.astype(np.float32)
    if image_output.max() > 1.0:
        image_output /= 255.0
    image_output = np.clip(image_output, 0.0, 1.0)

    return image_output

def sRGB_to_LMS(image_in_sRGB: np.ndarray) -> np.ndarray:
    """
    The function takes a vector image in sRGB and returns it in LMS
    """
    M_rgb_to_lms = np.array(
        [
            [0.31399022, 0.63951294, 0.04649755],  # L
            [0.15537241, 0.75789446, 0.08670142],  # M
            [0.01775239, 0.10944209, 0.87256922],  # S
        ],
        dtype=np.float32,
    )
    return image_in_sRGB @ M_rgb_to_lms.T

def LMS_to_RGB(image_in_LMS: np.ndarray) -> np.ndarray:
    """
    The function takes a vector image in LMS and returns it in sRGB
    """
    M_LMS_to_sRGB = np.array(
        [
            [ 5.472213,   -4.6419606,   0.16963711],
            [-1.125242,    2.2931712,  -0.16789523],
            [ 0.02980164, -0.19318072,  1.1636479 ]
        ],
    )
    return image_in_LMS @ M_LMS_to_sRGB.T

def merge_L_M(image_in_LMS: np.ndarray, alpha: float) -> np.ndarray:
    """
    **Used for dicromatic mammels like cat, dog**
    The function returns a the image in LMS with L and M merged based on alpha
    """

    LM = alpha * image_in_LMS[:, 0] + (1.0 - alpha) * image_in_LMS[:, 1]
    return np.stack([LM, LM, image_in_LMS[:, 2]], axis=1)

def collapse_LMS_matrix(alpha: float, s_scale: float) -> np.ndarray:
    """
    Build a 3x3 *RGB-linear -> RGB-linear* transform that:
      - merges L & M as: LM = alpha*L + (1-alpha)*M
      - applies S attenuation by s_scale
    Implementation trick: transform the RGB basis through LMS, apply collapse, go back to RGB.
    """

    # 3x3 identity as three basis RGB vectors in linear space
    E = np.eye(3, dtype=np.float32)         # shape (3,3)

    # RGB -> LMS
    LMS = sRGB_to_LMS(E)                    # (3,3), rows are RGB basis expressed in LMS

    # Apply collapse with a simple 3x3 on LMS rows:
    # [LM, LM, s*S] where LM = alpha*L + (1-alpha)*M
    D = np.array(
        [
            [alpha,         1.0 - alpha, 0.0],
            [alpha,         1.0 - alpha, 0.0],
            [0.0,           0.0,         s_scale],
        ],
        dtype=np.float32,
    )
    LMS_collapsed = LMS @ D.T               # still (3,3)

    # Back to RGB
    RGB_out = LMS_to_RGB(LMS_collapsed)     # (3,3)

    # Columns of the resulting 3x3 are exactly the transformed basis vectors.
    # We return a float32 3x3 so you can do: pixels @ T.T
    return RGB_out.astype(np.float32)

def apply_acuity_blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Apply Gaussian blur to simulate reduced visual acuity.

    Parameters
    ----------
    image : np.ndarray
        Input HxWx3 RGB image (float or uint8).
    sigma : float
        Blur strength. Higher sigma = blurrier vision.

    Returns
    -------
    np.ndarray
        Blurred image, same dtype as input.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HxWx3 image")

    dtype = image.dtype
    image_f = image.astype(np.float32, copy=False) if np.issubdtype(dtype, np.integer) else image
    # <- key change: let OpenCV pick kernel size from sigma
    blurred = cv2.GaussianBlur(image_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred.astype(dtype, copy=False)

def apply_anisotropic_acuity_blur_with_streak(image, y_center: float = 0.5,
                                         sigma_streak: float = 0.8,
                                         sigma_far: float = 2.2,
                                         falloff: float = 6.0):
    """
    Preserve acuity near a horizontal 'visual streak' at y_center (0..1).
    Blur increases smoothly away from that band; horizontal pupil → more vertical blur.
    """
    import cv2, numpy as np
    H = image.shape[0]
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    d = np.abs(yy - y_center)                             # distance from streak
    # smoothly vary sigma between streak and far
    sigma_map = sigma_streak + (sigma_far - sigma_streak) * (1.0 - np.exp(-falloff * d**2))
    sigmaY = sigma_map
    sigmaX = np.maximum(0.4, 0.5 * sigma_map)

    # separable blur row-by-row with per-row sigmas
    out = image.astype(np.float32, copy=False)
    tmp = np.empty_like(out)
    for y in range(H):
        tmp[y] = cv2.GaussianBlur(out[y], (0,0), sigmaX=float(sigmaX[y,0]), sigmaY=0.0)

    for y in range(H):
        out[y] = cv2.GaussianBlur(tmp[y], (0,0), sigmaX=1e-16, sigmaY=float(sigmaY[y,0]))
    return out.astype(image.dtype, copy=False)

def apply_chroma_compression(image: np.ndarray, strength: float = 0.4):
    """
    Compresses color saturation toward gray.
    strength=0 → no change
    strength=1 → complete grayscale
    """
    gray = image.mean(axis=2, keepdims=True)
    return gray + (image - gray) * (1 - strength)

def apply_tapetum_bloom(image: np.ndarray, strength: float = 0.12, sigma: float = 3.0) -> np.ndarray:
    """
    Subtle low-light bloom in linear RGB.
    strength: 0..~0.3
    sigma: blur radius for bloom spread
    """
    import cv2, numpy as np
    x = image.astype(np.float32, copy=False)
    x = np.clip(x, 0.0, 1.0)
    # luminance mask to bloom bright areas a bit more
    L = 0.2126 * x[...,0] + 0.7152 * x[...,1] + 0.0722 * x[...,2]
    mask = np.clip((L - 0.4) / 0.6, 0.0, 1.0)  # start blooming above midtones
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=sigma, sigmaY=sigma)[..., None]

    # blurred copy of the image
    blur = cv2.GaussianBlur(x, (0,0), sigmaX=sigma, sigmaY=sigma)

    # screen-like blend, gated by mask
    # screen(a,b) = 1 - (1-a)(1-b)
    screen = 1.0 - (1.0 - x) * (1.0 - blur)
    y = x + strength * mask * (screen - x)
    return np.clip(y, 0.0, 1.0).astype(image.dtype, copy=False)


def apply_rod_vision(
    image: np.ndarray, 
    chroma_scale: float = 0.08,
    luminance_boost: float = 1.4,
    gamma: float = 0.8
) -> np.ndarray:
    """
    Approximate scotopic (rod-dominant) night vision.

    Parameters
    ----------
    image : np.ndarray
        Linear RGB image (0..1 range recommended).
    chroma_scale : float
        Scale factor for color saturation; smaller = more monochrome.
        Rod vision has nearly no color discrimination, so this is usually 0.05–0.15.
    luminance_boost : float
        Brightness multiplier to simulate rods’ higher sensitivity in dim light.
    gamma : float
        Contrast remapping exponent (<1 brightens midtones).

    Returns
    -------
    np.ndarray
        Rod-dominant (night) vision image.
    """
    import cv2, numpy as np

    x = np.clip(image.astype(np.float32), 0.0, 1.0)

    # --- 1) Compute luminance (scotopic weighting) ---
    # Human scotopic luminosity peaks ~507 nm (~green-blue), 
    # close to rod spectral sensitivity.
    L = 0.1 * x[..., 0] + 0.8 * x[..., 1] + 0.1 * x[..., 2]
    L = cv2.GaussianBlur(L, (0, 0), sigmaX=1.2, sigmaY=1.2)

    # --- 2) Desaturate heavily (rods = achromatic) ---
    gray = L[..., None]
    x = gray * (1 - chroma_scale) + x * chroma_scale

    # --- 3) Boost luminance and apply gamma ---
    x = np.clip(x * luminance_boost, 0.0, 1.0)
    x = np.power(x, gamma)

    return x.astype(image.dtype, copy=False)
