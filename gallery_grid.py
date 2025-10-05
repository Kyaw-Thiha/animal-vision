from __future__ import annotations
from typing import List, Tuple, Optional
import math
import numpy as np
import cv2


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Return HxWx3 uint8 RGB in [0,255]."""
    assert img.ndim == 3 and img.shape[2] == 3, "Expected HxWx3"
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0 + 0.5).astype(np.uint8)
    elif img.dtype != np.uint8:
        # generic cast (assume already 0..255 range)
        img = img.astype(np.uint8)
    return img


def _resize_keep_ar(img: np.ndarray, *, target_h: int) -> np.ndarray:
    """Resize to target_h keeping aspect ratio."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _label_strip(img_rgb: np.ndarray, text: str, *, strip_h: int = 40) -> np.ndarray:
    """Add a bottom label strip and put text centered."""
    h, w = img_rgb.shape[:2]
    strip = np.full((strip_h, w, 3), 0, dtype=np.uint8)  # black strip
    out = np.vstack([img_rgb, strip])

    # text styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    # measure and center
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(6, (w - tw) // 2)
    y = h + strip_h // 2 + th // 2 - 2

    # white text with black outline for legibility
    cv2.putText(out, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def build_labeled_grid(
    tiles: List[Tuple[str, np.ndarray]],
    *,
    tile_height: int = 256,
    pad: int = 8,
    bg: Tuple[int, int, int] = (20, 20, 20),
) -> Optional[np.ndarray]:
    """
    tiles: list of (label, image_RGB)
    Returns grid (RGB uint8) or None if no tiles.
    """
    tiles = [(name, _to_uint8(_resize_keep_ar(img, target_h=tile_height))) for name, img in tiles if img is not None]
    if not tiles:
        return None

    # add label strips
    tiles = [(name, _label_strip(img, name)) for name, img in tiles]
    heights = [t.shape[0] for _, t in tiles]
    widths = [t.shape[1] for _, t in tiles]
    max_w = max(widths)
    max_h = max(heights)

    # pad each tile to same size
    padded: List[np.ndarray] = []
    for _, tile in tiles:
        h, w = tile.shape[:2]
        top = 0
        bottom = max_h - h
        left = 0
        right = max_w - w
        padded.append(cv2.copyMakeBorder(tile, top, bottom, left, right, cv2.BORDER_CONSTANT, value=bg))

    n = len(padded)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # canvas
    cell_h, cell_w = max_h + pad, max_w + pad
    H = rows * cell_h + pad
    W = cols * cell_w + pad
    grid = np.full((H, W, 3), bg, dtype=np.uint8)

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= n:
                break
            y = pad + r * cell_h
            x = pad + c * cell_w
            tile = padded[i]
            th, tw = tile.shape[:2]
            grid[y : y + th, x : x + tw] = tile
            i += 1

    return grid
