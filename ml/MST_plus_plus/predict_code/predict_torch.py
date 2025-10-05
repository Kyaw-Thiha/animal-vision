from typing import List, Tuple, Any, Union, Optional, cast
import numpy as np
import torch
import torch.nn as nn

from ml.MST_plus_plus.predict_code.architecture import model_generator


# ------------------------- small numpy helpers -------------------------


def _to_float01(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.integer):
        x = img.astype(np.float32) / 255.0
    else:
        x = img.astype(np.float32)
        if x.max() > 1.001:
            x = np.clip(x / 255.0, 0.0, 1.0)
    return x


def _pad_to_multiple_reflect(x: np.ndarray, mult: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    H, W = x.shape[:2]
    if mult is None or mult <= 0:
        return x, (0, 0, 0, 0)
    Hn = ((H + mult - 1) // mult) * mult
    Wn = ((W + mult - 1) // mult) * mult
    pad_y, pad_x = Hn - H, Wn - W
    if pad_y == 0 and pad_x == 0:
        return x, (0, 0, 0, 0)
    top, bottom = pad_y // 2, pad_y - (pad_y // 2)
    left, right = pad_x // 2, pad_x - (pad_x // 2)
    x_pad = np.pad(x, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    return x_pad, (top, bottom, left, right)


def _crop_pads(x: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    H, W = x.shape[:2]
    return x[top : H - bottom if bottom else H, left : W - right if right else W, :]


def _infer_grid(n: int) -> Tuple[int, int]:
    r = int(np.floor(np.sqrt(n)))
    for rows in range(r, 0, -1):
        if n % rows == 0:
            return rows, n // rows
    return 1, n


def _merge_tiles_row_major(tiles: List[np.ndarray], grid: Tuple[int, int]) -> np.ndarray:
    rows, cols = grid
    assert len(tiles) == rows * cols, "Tile count must equal rows*cols."
    H, W, C = tiles[0].shape
    for t in tiles:
        assert t.shape == (H, W, C), "All tiles must share identical H, W, and C."
    row_blocks, idx = [], 0
    for _ in range(rows):
        row_tiles = tiles[idx : idx + cols]
        row_blocks.append(np.hstack(row_tiles))
        idx += cols
    return np.vstack(row_blocks)


# ----------------------------- main predictor -----------------------------


def predict_rgb_to_hsi_torch(
    images: Union[np.ndarray, List[np.ndarray]],
    method: str,
    checkpoint: str,
    *,
    stride: int = 16,  # set 0/None to disable; use 8/16 to match net downsampling
    device: Optional[str] = None,  # 'cuda'/'cpu'/None(auto)
    half: bool = False,  # FP16 on CUDA (keep False on CPU)
    strict_load: bool = False,
    grid: Optional[Tuple[int, int]] = None,  # used when a list is passed (row-major)
) -> np.ndarray:
    """
    Run the MST++ PyTorch module directly and return a SINGLE HxWxC float32 array.
      - Single RGB image → returns its HSI.
      - List of RGB tiles → merges their HSIs into one image (row-major order).
    """
    # Device resolve
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if half and dev.type != "cuda":
        print("[TorchPredict][INFO] 'half=True' ignored on CPU.")
        half = False

    # Build model and load weights
    model = model_generator(method, checkpoint)
    model = model.to(dev)
    if half:
        try:
            model.half()
        except Exception:
            print("[TorchPredict][WARN] Model does not support .half(); using FP32.")
            half = False
    model.eval()

    # Normalize input(s)
    if isinstance(images, np.ndarray):
        imgs = [images]
        merging = False
    else:
        imgs = list(images)
        merging = True

    hsi_tiles: List[np.ndarray] = []
    with torch.no_grad():
        for im in imgs:
            x = _to_float01(im)
            pads = (0, 0, 0, 0)
            if stride and stride > 0:
                x, pads = _pad_to_multiple_reflect(x, stride)

            # HWC float32/16 -> NCHW torch
            x_chw = x.transpose(2, 0, 1)  # 3xH'xW'
            t = torch.from_numpy(x_chw).unsqueeze(0).to(dev)  # 1x3xH'xW'
            t = t.half() if half else t.float()

            # forward
            y: torch.Tensor = model(t)  # expect 1xC_hsi x H' x W' or 1xH' x W' x C_hsi
            if y.dim() != 4:
                raise RuntimeError(f"Model output must be 4D; got {tuple(y.shape)}")

            # Try to detect layout: NCHW vs NHWC
            _, d1, d2, d3 = y.shape
            Ht, Wt = t.shape[-2], t.shape[-1]
            if d2 == Ht and d3 == Wt:  # NCHW
                y_nhwc = y.permute(0, 2, 3, 1)  # 1xH'xW'xC
            elif d1 == Ht and d2 == Wt:  # NHWC
                y_nhwc = y  # already 1xH'xW'xC
            else:
                # fallback: assume NCHW
                y_nhwc = y.permute(0, 2, 3, 1)

            hsi = y_nhwc[0].detach().float().cpu().numpy()  # H'xW'xC
            if stride and pads != (0, 0, 0, 0):
                hsi = _crop_pads(hsi, pads)

            hsi_tiles.append(hsi.astype(np.float32, copy=False))

    if not merging:
        return hsi_tiles[0]

    # Merge tiles: require uniform tile size
    Hs = {t.shape[0] for t in hsi_tiles}
    Ws = {t.shape[1] for t in hsi_tiles}
    if len(Hs) != 1 or len(Ws) != 1:
        raise ValueError("All tiles must have the SAME height and width to merge.")
    rows, cols = grid if grid is not None else _infer_grid(len(hsi_tiles))
    return _merge_tiles_row_major(hsi_tiles, (rows, cols))
