# Minimal RGB -> HSI (31 bands) using Colour-Science's Mallett 2019 spectral upsampling (CPU),
# with an optional GPU path (analytic lobes) when device="cuda".
import numpy as np
import cv2 as cv
import colour
import torch


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Vectorised inverse EOTF for sRGB; expects x already in [0,1] if you provide floats."""
    a = 0.055
    thr = 0.04045
    return np.where(x <= thr, x / 12.92, ((x + a) / (1 + a)) ** 2.4).astype(np.float32)


@torch.jit.script
def _srgb_to_linear_t(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    thr = 0.04045
    low = x / 12.92
    high = ((x + a) / (1.0 + a)) ** 2.4
    return torch.where(x <= thr, low, high)


def classic_rgb_to_hsi(
    frame_bgr: np.ndarray,
    *,
    wavelengths: np.ndarray = np.linspace(400.0, 700.0, 31, dtype=np.float32),
    device: str = "cuda",  # "cpu" uses your Colour (Mallett'19) path; "cuda" uses fast GPU analytic path
) -> np.ndarray:
    """
    Convert an OpenCV BGR frame to an HxWxB hyperspectral cube (default B=31, 400..700nm).

    CPU path (default): Colour-Science Mallett2019 (unchanged from your code).
    GPU path (device="cuda"): analytic 3-lobe upsampler on CUDA (no extra conversions added here).
    """
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, "Input must be HxWx3."
    H, W, _ = frame_bgr.shape

    # Ensure uniform grid; build target SpectralShape
    if wavelengths.size < 2:
        raise ValueError("Need at least two wavelengths.")
    step = float(wavelengths[1] - wavelengths[0])
    if not np.allclose(np.diff(wavelengths), step):
        raise ValueError("`wavelengths` must be uniformly spaced.")

    if device.lower() == "cuda" and torch.cuda.is_available():
        # ---------------- GPU analytic path (fast) ----------------
        # Keep your input convention: we don't change channels or scale.
        # If you feed float [0,1], this behaves as intended. If you feed uint8, it will treat 0..255 directly.
        t = torch.as_tensor(frame_bgr, dtype=torch.float32, device="cuda")
        # sRGB -> linear (Torch)
        t = _srgb_to_linear_t(t)
        # Split channels (still in BGR order per your input; we don't swap)
        Bc = t[..., 0]  # B
        G = t[..., 1]  # G
        R = t[..., 2]  # R

        # Wavelength axis on GPU
        wl = torch.as_tensor(wavelengths.astype(np.float32), device="cuda").view(-1, 1, 1)  # Bx1x1

        # Gaussian lobes (nm)
        cR, cG, cB = 610.0, 545.0, 460.0
        sR, sG, sB = 60.0, 60.0, 55.0
        gR = torch.exp(-0.5 * ((wl - cR) / sR) ** 2)  # BxHxW
        gG = torch.exp(-0.5 * ((wl - cG) / sG) ** 2)
        gB = torch.exp(-0.5 * ((wl - cB) / sB) ** 2)

        # Weighted sum by channel intensities
        spec = gR * R.unsqueeze(0) + gG * G.unsqueeze(0) + gB * Bc.unsqueeze(0)  # BxHxW

        # Light per-pixel normalization to keep white ~ flat
        with torch.no_grad():
            denom = (
                torch.exp(-0.5 * ((wl.squeeze() - cR) / sR) ** 2)
                + torch.exp(-0.5 * ((wl.squeeze() - cG) / sG) ** 2)
                + torch.exp(-0.5 * ((wl.squeeze() - cB) ** 2) / (sB**2))
            ).mean()
        spec = spec / (denom + 1e-8)

        spec = spec.clamp_min(0.0).permute(1, 2, 0).contiguous()  # HxWxB
        return spec.detach().cpu().numpy().astype(np.float32)

    # ---------------- CPU Colour-Science path (your original logic) ----------------
    # 1) (As per your request, we do NOT add BGR->RGB or 0..1 scaling here.)
    # 2) sRGB -> linear
    rgb_lin = _srgb_to_linear(frame_bgr)

    # 3) Batch spectral upsampling (Mallett 2019): per-pixel loop
    rgb_vec = rgb_lin.reshape(-1, 3)

    target_shape = colour.SpectralShape(
        start=float(wavelengths[0]),
        end=float(wavelengths[-1]),
        interval=step,
    )

    basis = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019
    fun = colour.recovery.RGB_to_sd_Mallett2019
    N = rgb_vec.shape[0]
    specs = np.empty((N, wavelengths.size), dtype=np.float32)

    chunk = 16384
    for i in range(0, N, chunk):
        block = rgb_vec[i : i + chunk]  # (K,3)
        sds = [fun(rgb1d, basis_functions=basis) for rgb1d in block]  # list of SDs
        arr = np.stack(
            [(sd if sd.shape == target_shape else sd.interpolate(target_shape)).values for sd in sds],
            axis=0,
        ).astype(np.float32)
        specs[i : i + arr.shape[0]] = arr

    hsi = specs.reshape(H, W, wavelengths.size)
    np.maximum(hsi, 0.0, out=hsi)
    return hsi
