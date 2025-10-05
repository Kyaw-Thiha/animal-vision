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


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, h, dtype=np.float32))
    wx = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, w, dtype=np.float32))
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _tile_coords(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Yield tiles as (y0, y1, x0, x1) covering the image with given overlap."""
    stride = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if not ys or ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    if not xs or xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    coords = []
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            # ensure exact tile size by extending backwards when at border
            if (y1 - y0) < tile and H >= tile:
                y0 = y1 - tile
            if (x1 - x0) < tile and W >= tile:
                x0 = x1 - tile
            coords.append((y0, y0 + tile if H >= tile else y1, x0, x0 + tile if W >= tile else x1))
    return coords


def _try_full_frame_once(
    model: nn.Module,
    x_hw3: np.ndarray,
    *,
    device: torch.device,
    use_autocast: bool,
) -> Optional[np.ndarray]:
    """Try single-shot inference. Return HxWxC or None on OOM."""
    x_chw = x_hw3.transpose(2, 0, 1)[None, ...]  # 1x3xH×W
    t = torch.from_numpy(x_chw).to(device).float()
    try:
        with torch.inference_mode():
            ctx = torch.cuda.amp.autocast(enabled=(use_autocast and device.type == "cuda"), dtype=torch.float16)
            with ctx:
                y = model(t)  # 1xC×H×W or 1xH×W×C
        if y.dim() != 4:
            raise RuntimeError(f"Model output must be 4D; got {tuple(y.shape)}")
        _, d1, d2, d3 = y.shape
        Ht, Wt = t.shape[-2], t.shape[-1]
        if d2 == Ht and d3 == Wt:
            y = y.permute(0, 2, 3, 1)  # 1xH×W×C
        # else assume already NHWC
        hsi = y[0].detach().float().cpu().numpy()
        return hsi
    except RuntimeError as e:
        # Only treat CUDA OOM as a signal to tile
        msg = str(e).lower()
        if "out of memory" in msg or "cuda oom" in msg:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return None
        raise


def _run_tile(
    model: nn.Module,
    tile_hw3: np.ndarray,
    *,
    device: torch.device,
    use_autocast: bool,
) -> np.ndarray:
    """Run model on a single tile (H×W×3) and return H×W×C on CPU."""
    x_chw = tile_hw3.transpose(2, 0, 1)[None, ...]
    t = torch.from_numpy(x_chw).to(device).float()
    with torch.inference_mode():
        ctx = torch.cuda.amp.autocast(enabled=(use_autocast and device.type == "cuda"), dtype=torch.float16)
        with ctx:
            y = model(t)
        if y.dim() != 4:
            raise RuntimeError(f"Model output must be 4D; got {tuple(y.shape)}")
        _, d1, d2, d3 = y.shape
        Ht, Wt = t.shape[-2], t.shape[-1]
        if d2 == Ht and d3 == Wt:
            y = y.permute(0, 2, 3, 1)
        hsi = y[0].detach().float().cpu().numpy()
    return hsi


def _predict_one_image_with_auto_tiling(
    model: nn.Module,
    img_hw3: np.ndarray,
    *,
    device: torch.device,
    stride_multiple: int,
    prefer_autocast_fp16: bool,
    start_tile: int = 1024,
    min_tile: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """
    Try full frame; if OOM → try tiles (1024→768→512→384→256). Blend with Hann.
    """

    # optional pad to stride multiple (downsampling friendliness)
    def pad_to_mult(x: np.ndarray, mult: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        H, W = x.shape[:2]
        if mult is None or mult <= 0:
            return x, (0, 0, 0, 0)
        Hn = ((H + mult - 1) // mult) * mult
        Wn = ((W + mult - 1) // mult) * mult
        py, px = Hn - H, Wn - W
        if py == 0 and px == 0:
            return x, (0, 0, 0, 0)
        top, bot = py // 2, py - py // 2
        left, right = px // 2, px - px // 2
        x_pad = np.pad(x, ((top, bot), (left, right), (0, 0)), mode="reflect")
        return x_pad, (top, bot, left, right)

    def crop_pads(x: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
        top, bot, left, right = pads
        H, W = x.shape[:2]
        return x[top : H - bot if bot else H, left : W - right if right else W, :]

    x = _to_float01(img_hw3)
    x, pads = pad_to_mult(x, stride_multiple)

    # 1) Try full frame once
    hsi = _try_full_frame_once(model, x, device=device, use_autocast=prefer_autocast_fp16)
    if hsi is not None:
        out = crop_pads(hsi, pads) if pads != (0, 0, 0, 0) else hsi
        return out.astype(np.float32, copy=False)

    # 2) Auto-tiling attempts
    H, W = x.shape[:2]
    candidate_tiles = [start_tile, 768, 512, 384, 256]
    candidate_tiles = [t for t in candidate_tiles if t >= min_tile]
    hann_cache = {}

    for tile in candidate_tiles:
        try:
            coords = _tile_coords(H, W, tile, overlap)
            # feather window (cache)
            if tile not in hann_cache:
                hw = _hann2d(tile, tile)[..., None]
                hann_cache[tile] = hw
            hann = hann_cache[tile]

            # allocate CPU accumulators
            # run one tile to discover C
            y0, y1, x0, x1 = coords[0]
            pred0 = _run_tile(model, x[y0:y1, x0:x1, :], device=device, use_autocast=prefer_autocast_fp16)
            C = pred0.shape[2]
            out_accum = np.zeros((H, W, C), dtype=np.float32)
            w_accum = np.zeros((H, W, C), dtype=np.float32)

            # place first
            w = hann if C == 1 else np.repeat(hann, C, axis=2)
            out_accum[y0:y1, x0:x1, :] += pred0 * w
            w_accum[y0:y1, x0:x1, :] += w

            # remaining tiles
            for yy0, yy1, xx0, xx1 in coords[1:]:
                pred = _run_tile(model, x[yy0:yy1, xx0:xx1, :], device=device, use_autocast=prefer_autocast_fp16)
                out_accum[yy0:yy1, xx0:xx1, :] += pred * w
                w_accum[yy0:yy1, xx0:xx1, :] += w

            stitched = out_accum / np.maximum(w_accum, 1e-8)
            out = crop_pads(stitched, pads) if pads != (0, 0, 0, 0) else stitched
            return out.astype(np.float32, copy=False)

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda":
                torch.cuda.empty_cache()
                continue  # try smaller tile
            raise  # other errors bubble up

    raise RuntimeError(
        "CUDA OOM even with smallest tile. Try: smaller image, reduce stride_multiple, set half=True (autocast), or run on CPU."
    )


def predict_rgb_to_hsi_torch(
    images: Union[np.ndarray, List[np.ndarray]],
    method: str,
    checkpoint: str,
    *,
    stride: int = 16,  # pad H/W to multiple of this before inference (for downsampling nets)
    device: Optional[str] = None,  # 'cuda'/'cpu'/None(auto)
    half: bool = True,  # use autocast FP16 on CUDA by default (safe & VRAM friendly)
    strict_load: bool = False,
    grid: Optional[Tuple[int, int]] = None,  # used when a list is passed (row-major)
    start_tile: int = 1024,
    min_tile: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """
    Auto memory-aware predictor:
      - Try full-frame once; on OOM auto-tiles with overlap+Hann.
      - Always returns a SINGLE HxWxC float32 array.
    """
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Model
    torch.backends.cudnn.benchmark = True  # speed for convs
    model = model_generator(method, checkpoint).to(dev).eval()

    # Inputs → list
    if isinstance(images, np.ndarray):
        imgs = [images]
        merging = False
    else:
        imgs = list(images)
        merging = True

    # Run each image with auto-tiling
    results: List[np.ndarray] = []
    for im in imgs:
        x = _to_float01(im)
        out = _predict_one_image_with_auto_tiling(
            model,
            x,
            device=dev,
            stride_multiple=stride,
            prefer_autocast_fp16=half and (dev.type == "cuda"),
            start_tile=start_tile,
            min_tile=min_tile,
            overlap=overlap,
        )
        results.append(out)

    if not merging:
        return results[0]

    # Merge tiles back if a list was passed
    Hs = {t.shape[0] for t in results}
    Ws = {t.shape[1] for t in results}
    if len(Hs) != 1 or len(Ws) != 1:
        raise ValueError("All tiles must have the SAME height and width to merge.")
    rows, cols = grid if grid is not None else _infer_grid(len(results))
    return _merge_tiles_row_major(results, (rows, cols))
