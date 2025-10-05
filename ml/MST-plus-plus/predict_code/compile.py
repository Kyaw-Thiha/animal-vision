#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn

from architecture import model_generator

# ---------------------------------------------------------------------
# A CLI based script to compile checkpoint to ONNX.
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export a .pth checkpoint to ONNX")
    # Required by your spec:
    parser.add_argument("--method", type=str, default="mst_plus_plus")
    parser.add_argument("--pretrained_model_path", type=str, default="./model_zoo/mst_plus_plus.pth")

    # Helpful extras:
    parser.add_argument("--output", type=str, default=None, help="Output .onnx path (defaults to ./export/<method>.onnx)")
    parser.add_argument("--height", type=int, default=256, help="Dummy input height")
    parser.add_argument("--width", type=int, default=256, help="Dummy input width")
    parser.add_argument("--channels", type=int, default=3, help="Input channels (RGB=3)")
    parser.add_argument("--batch", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic axes for batch/height/width")
    parser.add_argument("--half", action="store_true", help="Export with FP16 (if supported)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to instantiate and trace on")
    parser.add_argument("--strict", action="store_true", help="Use strict=True when loading state_dict")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=False)
    return parser.parse_args()


def find_state_dict(ckpt: Any) -> Dict[str, torch.Tensor] | None:
    """
    Accepts a checkpoint that could be:
      - a raw state_dict (param_name -> tensor)
      - {'state_dict': ...}
      - {'model': ...}
      - or anything similar.
    Returns a state_dict or None if not found.
    """
    if isinstance(ckpt, dict):
        # raw state_dict?
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        # common wrappers:
        for key in ("state_dict", "model", "net", "params"):
            if key in ckpt and isinstance(ckpt[key], dict):
                inner = ckpt[key]
                if all(isinstance(v, torch.Tensor) for v in inner.values()):
                    return inner
    return None


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strips a leading 'module.' (from DataParallel) if present.
    """
    needs_strip = any(k.startswith("module.") for k in sd.keys())
    if not needs_strip:
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def load_weights_if_needed(model: nn.Module, ckpt_path: str, device: torch.device, strict: bool = False) -> None:
    """
    Loads weights into model from ckpt_path if the model does not already contain them.
    If model_generator already loaded weights, this is effectively a no-op if keys match.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found at: {ckpt_path}. Proceeding with current model weights.")
        return

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = find_state_dict(ckpt)
    if sd is None:
        print(
            "[WARN] Could not find a state_dict in checkpoint. If weights are loaded inside model_generator, you can ignore this."
        )
        return

    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if isinstance(missing, list) and isinstance(unexpected, list):
        if missing:
            print(f"[INFO] Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[INFO] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")


def make_dummy_input(batch: int, channels: int, height: int, width: int, device: torch.device, half: bool) -> torch.Tensor:
    x = torch.randn(batch, channels, height, width, device=device)
    return x.half() if half else x.float()


def resolve_output_path(method: str, output_arg: str | None) -> str:
    """
    Decide where to save the ONNX model.
    Defaults to ./export/<method>.onnx if not explicitly set.
    """
    if output_arg is not None:
        return output_arg
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in method)
    export_dir = os.path.join(".", "export")
    os.makedirs(export_dir, exist_ok=True)
    return os.path.join(export_dir, f"{safe}.onnx")


def to_device(model: nn.Module, device: torch.device, half: bool) -> nn.Module:
    model = model.to(device)
    if half:
        try:
            model = model.half()
        except Exception:
            print("[WARN] Model does not support .half(); exporting in FP32 instead.")
    model.eval()
    return model


def export_onnx(model: nn.Module, dummy: torch.Tensor, out_path: str, opset: int, dynamic: bool) -> None:
    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    print(f"[INFO] Exporting ONNX → {out_path} (opset={opset}, dynamic={bool(dynamic_axes)})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    print(f"[OK] ONNX saved: {out_path}")


def main():
    opt = parse_args()

    device = torch.device(opt.device if (opt.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if opt.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA not available, falling back to CPU.")

    # Build model (your project must provide this)
    print(f"[INFO] Building model with method='{opt.method}'")
    model = model_generator(opt.method, opt.pretrained_model_path)
    model = to_device(model, device, opt.half)

    load_weights_if_needed(model, opt.pretrained_model_path, device, strict=opt.strict)
    dummy = make_dummy_input(opt.batch, opt.channels, opt.height, opt.width, device, opt.half)

    # Forward once to validate
    try:
        with torch.no_grad():
            _ = model(dummy)
    except Exception as e:
        print(
            "[ERROR] A forward pass failed. Ensure your model's forward(input) signature "
            f"matches the dummy tensor shape BxCxHxW. Details: {e}"
        )
        sys.exit(1)

    # Export
    out_path = resolve_output_path(opt.method, opt.output)
    export_onnx(model, dummy, out_path, opset=opt.opset, dynamic=opt.dynamic)


if __name__ == "__main__":
    main()
