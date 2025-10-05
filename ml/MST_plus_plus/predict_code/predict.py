from typing import List, Tuple, Any, Union, Sequence, Optional, cast
import numpy as np
import onnxruntime as ort


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


def predict_rgb_to_hsi(
    images: Union[np.ndarray, List[np.ndarray]],
    onnx_path: str,
    stride: int = 16,  # 0/None disables; use 8/16 to match net downsampling
    device_providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    grid: Optional[Tuple[int, int]] = None,  # only used when a list is passed
) -> np.ndarray:
    """
    Predict HSI with a dynamic-axes ONNX and return a SINGLE HxWxC array.
      - Single RGB image → returns its HSI.
      - List of RGB tiles → merges their HSIs into one image (row-major order).
    """
    # Normalize to a list
    if isinstance(images, np.ndarray):
        imgs = [images]
        merging = False
    else:
        imgs = list(images)
        merging = True

    # Session
    sess = ort.InferenceSession(onnx_path, providers=list(device_providers))
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Probe output layout
    probe_img = _to_float01(imgs[0])
    probe_img, _pads = _pad_to_multiple_reflect(probe_img, stride) if stride and stride > 0 else (probe_img, (0, 0, 0, 0))
    probe_nchw = probe_img.transpose(2, 0, 1)[None, ...]  # 1x3xH'xW'
    out0_any: Any = sess.run([out_name], {in_name: probe_nchw})[0]
    out0 = cast(np.ndarray, out0_any)
    if out0.ndim != 4:
        raise RuntimeError("Expected 4D ONNX output (N,C,H,W) or (N,H,W,C).")
    _, d1, d2, d3 = out0.shape
    Ht, Wt = probe_nchw.shape[-2:]
    if d2 == Ht and d3 == Wt:
        out_is_nchw, C_hsi = True, d1
    elif d1 == Ht and d2 == Wt:
        out_is_nchw, C_hsi = False, d3
    else:
        out_is_nchw, C_hsi = True, d1  # fallback

    # Run per image/tile
    hsi_tiles: List[np.ndarray] = []
    for im in imgs:
        x = _to_float01(im)
        pads = (0, 0, 0, 0)
        if stride and stride > 0:
            x, pads = _pad_to_multiple_reflect(x, stride)
        x_nchw = x.transpose(2, 0, 1)[None, ...]
        pred_any: Any = sess.run([out_name], {in_name: x_nchw})[0]
        pred = cast(np.ndarray, pred_any)
        if out_is_nchw:
            pred = np.transpose(pred, (0, 2, 3, 1))  # 1xH'xW'xC
        hsi = pred[0]
        if stride and pads != (0, 0, 0, 0):
            hsi = _crop_pads(hsi, pads)
        hsi_tiles.append(hsi.astype(np.float32, copy=False))

    # If list was passed, merge tiles; else return the single result
    if not merging:
        return hsi_tiles[0]

    # Merge: require uniform tile size
    Hs = {t.shape[0] for t in hsi_tiles}
    Ws = {t.shape[1] for t in hsi_tiles}
    if len(Hs) != 1 or len(Ws) != 1:
        raise ValueError("All tiles must have the SAME height and width to merge.")
    rows, cols = grid if grid is not None else _infer_grid(len(hsi_tiles))
    return _merge_tiles_row_major(hsi_tiles, (rows, cols))
