from typing import Optional
import numpy as np
# from animals.animal_utils import srgb_to_linear

from animals.animal import Animal


class Cat(Animal):
    def visualize(self, input: np.ndarray) -> Optional[np.ndarray]:
        """
        Simulate a simple cat-vision rendering from an RGB image.

        Steps:
        1) Validate input is an RGB image.
        2) Convert sRGB -> linear RGB.
        3) Map linear RGB to LMS, collapse L & M to a single “LM” channel (dichromacy proxy),
           keep S as-is, then map back LMS -> linear RGB.
        4) Apply a mild box blur to mimic reduced visual acuity.
        5) Convert linear RGB -> sRGB and return in original dtype.
        """
        # ---------- 1) Validate input ----------
        assert isinstance(input, np.ndarray), "Input must be a numpy ndarray."
        assert input.ndim == 3 and input.shape[2] == 3, "Input must be HxWx3 RGB."
        assert np.issubdtype(input.dtype, np.number), "Input array must be numeric."

        # Save dtype to restore later
        orig_dtype = input.dtype

        # Normalize to [0, 1] float32 if needed
        img = input.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        img = np.clip(img, 0.0, 1.0)

        # ---------- helpers ----------
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

        # Stockman & Sharpe–like linear RGB <-> LMS matrices (approximation)
        M_rgb2lms = np.array(
            [
                [0.31399022, 0.63951294, 0.04649755],  # L
                [0.15537241, 0.75789446, 0.08670142],  # M
                [0.01775239, 0.10944209, 0.87256922],  # S
            ],
            dtype=np.float32,
        )
        M_lms2rgb = np.linalg.inv(M_rgb2lms).astype(np.float32)

        # Mild 3x3 box blur to approximate reduced acuity
        def box_blur(img_lin: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
            assert k % 2 == 1, "Kernel size k must be odd."
            pad = k // 2
            kernel_area = float(k * k)
            out = img_lin
            for _ in range(iters):
                # pad H,W dimensions; keep channels unpadded
                padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
                # separable implementation (horizontal then vertical) for efficiency
                # horizontal
                cumsum = np.cumsum(padded, axis=1)
                left = cumsum[:, :-k, :]
                right = cumsum[:, k:, :]
                horiz = (right - left) / k
                # vertical
                cumsum2 = np.cumsum(horiz, axis=0)
                top = cumsum2[:-k, :, :]
                bottom = cumsum2[k:, :, :]
                out = (bottom - top) / k
            return out

        # ---------- 2) sRGB -> linear ----------
        lin = srgb_to_linear(img)

        # ---------- 3) linear RGB -> LMS, collapse L/M (dichromacy proxy), LMS -> linear RGB ----------
        H, W, _ = lin.shape
        lin_reshaped = lin.reshape(-1, 3)
        lms = lin_reshaped @ M_rgb2lms.T  # (N,3)

        # Collapse L and M to a shared channel (simple dichromat proxy)
        LM = 0.5 * (lms[:, 0] + lms[:, 1])
        lms[:, 0] = LM
        lms[:, 1] = LM
        # Keep S as-is

        lin_sim = (lms @ M_lms2rgb.T).reshape(H, W, 3)

        # ---------- 4) light blur ----------
        lin_sim = np.clip(lin_sim, 0.0, 1.0)
        lin_sim = box_blur(lin_sim, k=3, iters=1)

        # ---------- 5) linear -> sRGB and restore dtype ----------
        srgb_sim = np.clip(linear_to_srgb(np.clip(lin_sim, 0.0, 1.0)), 0.0, 1.0)

        if np.issubdtype(orig_dtype, np.integer):
            out = (srgb_sim * 255.0 + 0.5).astype(orig_dtype)
        else:
            out = srgb_sim.astype(orig_dtype)

        return out
