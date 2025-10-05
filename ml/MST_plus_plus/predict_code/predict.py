from typing import Tuple, List, Any, cast
import numpy as np
import onnxruntime as ort


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, h, dtype=np.float32))
    wx = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, w, dtype=np.float32))
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _reflect_pad_to_cover(H: int, W: int, tile: int, overlap: int) -> Tuple[int, int]:
    assert 0 <= overlap < tile
    stride = tile - overlap

    def cover(L: int) -> int:
        if L <= tile:
            return tile
        n = int(np.ceil((L - tile) / stride)) + 1
        return stride * n + overlap

    return cover(H), cover(W)


def _extract_tiles(
    img: np.ndarray, tile: int, overlap: int
) -> Tuple[List[Tuple[int, int, np.ndarray]], np.ndarray, int, int, Tuple[int, int, int, int]]:
    """
    Reflect-pad the image so a grid of tiles (tile×tile) fully covers it.

    Returns
    -------
    tiles : list of (y, x, tile_view) coords in the padded image
    img_pad : padded image
    Hp, Wp : padded height/width
    pads : (top_pad, bottom_pad, left_pad, right_pad)
    """
    H, W = img.shape[:2]
    stride = tile - overlap
    Hp, Wp = _reflect_pad_to_cover(H, W, tile, overlap)
    pad_y = Hp - H
    pad_x = Wp - W

    # Split padding per side without going negative.
    top_pad = int(min(pad_y, overlap // 2))
    bottom_pad = int(pad_y - top_pad)
    left_pad = int(min(pad_x, overlap // 2))
    right_pad = int(pad_x - left_pad)

    assert top_pad >= 0 and bottom_pad >= 0 and left_pad >= 0 and right_pad >= 0

    img_pad = np.pad(
        img,
        ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
        mode="reflect",
    )

    tiles: List[Tuple[int, int, np.ndarray]] = []
    for y in range(0, Hp - tile + 1, stride):
        for x in range(0, Wp - tile + 1, stride):
            tiles.append((y, x, img_pad[y : y + tile, x : x + tile, :]))

    return tiles, img_pad, Hp, Wp, (top_pad, bottom_pad, left_pad, right_pad)


def predict_rgb_to_hsi(
    image: np.ndarray,
    onnx_path: str,
    tile: int = 256,
    overlap: int = 64,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Run a 256x256-only ONNX model over an arbitrary-size RGB image by tiling + blending.
    Handles models with fixed batch dimension (e.g., N=4) by padding short batches.

    Args
    ----
    image: HxWx3 (uint8 in [0,255] or float in [0,1]), sRGB-encoded.
    onnx_path: path to the fixed-size ONNX model (expects NCHW 1x3x256x256 typically).
    tile: tile size (must be what the model expects; 256 here).
    overlap: pixels of overlap between tiles to suppress seams (e.g., 32~64).
    batch_size: how many tiles to run per session.run call.

    Returns
    -------
    HxWxC_hsi float32 HSI cube (same H, W as input; bands from the model).
    """
    assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3."

    # Normalize to float32 [0,1]
    if np.issubdtype(image.dtype, np.integer):
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
        if img.max() > 1.001:
            img = np.clip(img / 255.0, 0.0, 1.0)

    # Tiles + exact pads
    tiles, img_pad, Hp, Wp, pads = _extract_tiles(img, tile=tile, overlap=overlap)
    top_pad, bottom_pad, left_pad, right_pad = pads

    # Edge-case: extremely small input → single tile
    if len(tiles) == 0:
        Hp = max(tile, img.shape[0])
        Wp = max(tile, img.shape[1])
        pad_y = Hp - img.shape[0]
        pad_x = Wp - img.shape[1]
        img_pad = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
        tiles = [(0, 0, img_pad[:tile, :tile, :])]
        top_pad = left_pad = 0
        bottom_pad = pad_y
        right_pad = pad_x

    hann = _hann2d(tile, tile).astype(np.float32)[..., None]  # (tile, tile, 1)

    # Session + IO meta
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    output_name = sess.get_outputs()[0].name

    # Static batch detection (e.g., 4) or dynamic (None)
    dim0 = input_meta.shape[0]
    static_batch_k: int | None = dim0 if isinstance(dim0, int) else None

    # Probe with N==k if k is static; else N==1 is fine
    y0, x0, t0 = tiles[0]
    one_tile = t0.transpose(2, 0, 1)[None, ...]  # 1x3xHxW (NCHW typical)
    probe = np.repeat(one_tile, static_batch_k, axis=0) if static_batch_k and static_batch_k > 1 else one_tile
    out0_any: Any = sess.run([output_name], {input_name: probe})[0]
    out0 = cast(np.ndarray, out0_any)
    if out0.ndim != 4:
        raise RuntimeError("Unexpected ONNX output rank; expected 4D (N,C,H,W or N,H,W,C).")
    _, d1, d2, d3 = out0.shape
    if d2 == tile and d3 == tile:  # NCHW
        out_is_nchw = True
        C_hsi = d1
    elif d1 == tile and d2 == tile:  # NHWC
        out_is_nchw = False
        C_hsi = d3
    else:
        raise RuntimeError("Cannot infer output layout; dims do not match tile size.")

    # Accumulators
    H, W = image.shape[:2]
    out_accum = np.zeros((Hp, Wp, C_hsi), dtype=np.float32)
    w_accum = np.zeros((Hp, Wp, C_hsi), dtype=np.float32)

    # If static batch k, align chunk size to k
    if static_batch_k and static_batch_k > 1:
        batch_size = static_batch_k

    def run_batch(batch_tiles: List[Tuple[int, int, np.ndarray]]) -> None:
        if not batch_tiles:
            return

        batch = np.stack([t[2].transpose(2, 0, 1) for t in batch_tiles], axis=0)  # Nx3xHxW
        Ncur = batch.shape[0]

        if static_batch_k and static_batch_k > 1:
            k = static_batch_k
            if Ncur < k:
                pad = np.repeat(batch[-1:], k - Ncur, axis=0)  # repeat last tile
                feed = np.concatenate([batch, pad], axis=0)  # kx3xHxW
            else:
                feed = batch[:k]
            preds_any: Any = sess.run([output_name], {input_name: feed})[0]
            preds = cast(np.ndarray, preds_any)
            if out_is_nchw:
                preds = np.transpose(preds, (0, 2, 3, 1))  # kxHxWxC
            preds_use = preds[: len(batch_tiles)]  # discard padded extras
        else:
            preds_any: Any = sess.run([output_name], {input_name: batch})[0]
            preds = cast(np.ndarray, preds_any)
            if out_is_nchw:
                preds = np.transpose(preds, (0, 2, 3, 1))  # NxHxWxC
            preds_use = preds

        for (y, x, _), pred in zip(batch_tiles, preds_use):
            w = hann if C_hsi == 1 else np.repeat(hann, C_hsi, axis=2)
            out_accum[y : y + tile, x : x + tile, :] += pred * w
            w_accum[y : y + tile, x : x + tile, :] += w

    # Iterate in mini-batches
    cursor = 0
    Ntiles = len(tiles)
    while cursor < Ntiles:
        run_batch(tiles[cursor : cursor + batch_size])
        cursor += batch_size

    # Normalize + crop using the EXACT pads we added
    stitched = out_accum / np.maximum(w_accum, 1e-8)
    stitched = stitched[top_pad : top_pad + H, left_pad : left_pad + W, :]

    return stitched.astype(np.float32)
