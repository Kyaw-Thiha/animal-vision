#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Tuple, Optional, List, Any, cast

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    raise ImportError(
        "onnxruntime is required. Install with `pip install onnxruntime` (or `onnxruntime-gpu` on CUDA machines)."
    ) from e

# ---------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------
_SESSION_CACHE: dict[str, ort.InferenceSession] = {}


def _select_providers() -> List[str]:
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in providers:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in providers:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_session(onnx_path: str) -> ort.InferenceSession:
    key = os.path.abspath(onnx_path)
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=_select_providers())

    # Basic I/O sanity
    assert len(session.get_inputs()) >= 1, "Model has no inputs."
    assert len(session.get_outputs()) >= 1, "Model has no outputs."

    _SESSION_CACHE[key] = session
    return session


def _infer_io_names(session: ort.InferenceSession) -> Tuple[str, str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        raise RuntimeError("Model must have at least one input and one output.")
    return inputs[0].name, outputs[0].name


# ---------------------------------------------------------------------
# Pre/Post utilities + strong validations
# ---------------------------------------------------------------------
def _validate_input_image(img: Any) -> np.ndarray:
    """Ensure img is HxWx3 ndarray with dtype uint8 or float and return it."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"image must be a numpy.ndarray, got {type(img).__name__}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {img.shape}")
    if img.dtype not in (np.uint8, np.float16, np.float32, np.float64):
        raise TypeError(f"Unsupported dtype {img.dtype}. Use uint8 or float (float16/32/64).")
    return img


def _to_float_chw(img: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Convert HWC RGB image to float32 NCHW in [0,1], remember original dtype.
    Asserts the final shape.
    """
    img = _validate_input_image(img)
    orig_dtype = str(img.dtype)

    x = img.astype(np.float32, copy=False)
    if img.dtype == np.uint8:
        x = x / 255.0

    # Assert range if float input (be generous but warn)
    if img.dtype != np.uint8:
        # Not hard failing here, but we’ll clamp later anyway.
        pass

    # HWC -> NCHW, add batch
    x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,H,W)

    # Strong shape assertions
    assert x.ndim == 4, f"Preprocessed input must be 4D, got {x.ndim}D"
    assert x.shape[0] == 1, f"Batch must be 1, got {x.shape[0]}"
    assert x.shape[1] == 3, f"Channels must be 3 (RGB), got {x.shape[1]}"
    return x, orig_dtype


def _to_hwc(out: Any, reference_hw: Tuple[int, int]) -> np.ndarray:
    """
    Convert model output to HWC float32.
    Accepts (1,C,H,W) or (1,H,W,C) or already (H,W,C). Will crop to reference_hw if larger.
    """
    out_arr = np.asarray(out)  # handles OrtValue, lists, etc.
    if out_arr.ndim == 4:
        if out_arr.shape[0] != 1:
            raise ValueError(f"Output batch must be 1, got {out_arr.shape[0]}")
        # Try NCHW
        if out_arr.shape[1] in (1, 2, 3, 4):
            out_arr = np.transpose(out_arr[0], (1, 2, 0))  # (H,W,C)
        # Else NHWC
        elif out_arr.shape[-1] in (1, 2, 3, 4):
            out_arr = out_arr[0]  # (H,W,C)
        else:
            raise ValueError(f"Unsupported 4D output shape {out_arr.shape}; expected channels in dim 1 or -1.")
    elif out_arr.ndim == 3:
        # Assume already HWC
        pass
    else:
        raise ValueError(f"Unsupported output rank {out_arr.ndim}; expected 3 or 4 dims.")

    H, W = reference_hw
    if out_arr.shape[0] < H or out_arr.shape[1] < W:
        raise ValueError(f"Output spatial size {out_arr.shape[:2]} is smaller than input {(H, W)}.")
    if out_arr.shape[0] != H or out_arr.shape[1] != W:
        out_arr = out_arr[:H, :W, :]

    # Channel sanity
    C = out_arr.shape[2]
    if C not in (1, 2, 3, 4):
        raise ValueError(f"Unexpected channel count {C}; expected 1..4.")

    return out_arr.astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def predict_rgb_to_hsi(image: np.ndarray, onnx_path: str) -> np.ndarray:
    """
    Run an ONNX model that maps an RGB image to an HSI image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image as HxWx3. dtype can be uint8 (0..255) or float (expected in [0,1]).
    onnx_path : str
        Path to the ONNX model file.

    Returns
    -------
    np.ndarray
        HSI image as HxWxC float32 (usually C=3), same height/width as input.
    """
    # Preprocess + assertions
    tensor, _ = _to_float_chw(image)  # (1,3,H,W)
    H, W = image.shape[:2]
    assert tensor.shape == (1, 3, H, W), f"Unexpected tensor shape {tensor.shape}"

    # Session + IO names
    session = _get_session(onnx_path)
    input_name, output_name = _infer_io_names(session)

    # Optional: basic input meta check (if present)
    inp_meta = session.get_inputs()[0]
    if inp_meta.shape is not None and isinstance(inp_meta.shape, list):
        # Skip dynamic dims (None/str), but verify channel order if static
        if len(inp_meta.shape) == 4:
            # Heuristics only; many models are NCHW
            pass

    # Run
    # outputs: List[Any] = session.run([output_name], {input_name: tensor})
    # if not outputs:
    #     raise RuntimeError("ONNX Runtime returned no outputs.")
    # out_any: Any = outputs[0]
    out_any: Any = session.run([output_name], {input_name: tensor})[0]

    # Postprocess → HWC float32 + assertions
    hsi = _to_hwc(out_any, (H, W))

    # Numeric sanity
    if not np.all(np.isfinite(hsi)):
        raise ValueError("HSI output contains non-finite values (NaN/Inf).")

    # Range policy: clamp to [0,1] (adjust if your model uses different ranges)
    hsi = np.clip(hsi, 0.0, 1.0)

    # Final shape/dtype assertions
    assert hsi.ndim == 3 and hsi.shape[0] == H and hsi.shape[1] == W, (
        f"Final HSI shape must be (H,W,C) with H={H},W={W}, got {hsi.shape}"
    )
    assert hsi.dtype == np.float32, f"HSI dtype must be float32, got {hsi.dtype}"

    return hsi
