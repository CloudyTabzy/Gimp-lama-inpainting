"""Command-line inference worker used by the GIMP plug-in.

Single mode:
- ``inpaint`` (default, only mode): LaMa inpainting. Takes an image
  PNG and a grayscale mask PNG, writes an inpainted image PNG.

The worker runs in a regular Python environment with Pillow, NumPy, and
ONNX Runtime. GIMP communicates with it through image and result PNGs.

Boundary markers are emitted on stderr with ``flush=True`` so that the
parent GIMP process can parse phase transitions if it wants to. The
format is a single line per marker, prefixed with ``[LAMA_MARKER]``.
The parent does not require these markers; the plug-in functions
whether or not they are present.
"""

from __future__ import annotations

import argparse
import os
import sys


# A simple phase marker that the GIMP parent MAY parse. The format is
# stable but the parent must remain tolerant of missing markers.
_MARKER_PREFIX = "[LAMA_MARKER]"


def _marker(stage: str, detail: str = "") -> None:
    """Emit a parseable phase marker on stderr with flush=True."""
    message = f"{_MARKER_PREFIX} {stage}"
    if detail:
        message = f"{message} {detail}"
    print(message, file=sys.stderr, flush=True)


def _require_file(path: str, label: str) -> str:
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise ValueError(f"{label} not found: {resolved}")
    return resolved


def _output_path(path: str) -> str:
    resolved = os.path.abspath(path)
    parent = os.path.dirname(resolved)
    if not os.path.isdir(parent):
        raise ValueError(f"output directory not found: {parent}")
    if os.path.isdir(resolved):
        raise ValueError(f"output path is a directory: {resolved}")
    return resolved


def _run_inpaint(args: argparse.Namespace) -> None:
    _marker("phase", "validate")
    try:
        import numpy as np
        from PIL import Image
        from lama_inpaint import LamaInpainter
    except ImportError as exc:
        name = exc.name or str(exc)
        raise RuntimeError(f"missing worker dependency: {name}") from None

    image_path = _require_file(args.image, "image")
    mask_path = _require_file(args.mask, "mask")
    model_path = _require_file(args.model, "model")
    output_path = _output_path(args.output)

    with Image.open(image_path) as source:
        source.load()
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
        image_size = source.size

    with Image.open(mask_path) as source_mask:
        source_mask.load()
        mask_size = source_mask.size
        mask_u8 = np.asarray(source_mask.convert("L"), dtype=np.uint8)

    if image_size != mask_size:
        raise ValueError(
            f"image/mask dimensions differ: "
            f"{image_size[0]}x{image_size[1]} vs {mask_size[0]}x{mask_size[1]}"
        )
    if rgba.shape[0] == 0 or rgba.shape[1] == 0:
        raise ValueError("image dimensions must be nonzero")

    image_rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    # Keep the mask SOFT (0..1). The model gets a binarized copy inside
    # inpaint(), and the soft values drive the final composite so
    # antialiased selection edges blend seamlessly.
    mask = mask_u8.astype(np.float32) / 255.0

    _marker("phase", "inference_start")
    result_rgb = LamaInpainter(model_path).inpaint(image_rgb, mask)
    _marker("phase", "inference_done")
    result_u8 = np.rint(np.clip(result_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    output_rgba = np.empty_like(rgba)
    output_rgba[:, :, :3] = result_u8
    output_rgba[:, :, 3] = rgba[:, :, 3]
    Image.fromarray(output_rgba).save(output_path, format="PNG")
    _marker("phase", "result_written")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LaMa inpainting on PNG files",
    )
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument(
        "--mask", required=True, help="grayscale mask path",
    )
    parser.add_argument("--output", required=True, help="result PNG path")
    parser.add_argument("--model", required=True, help="ONNX model path")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        _run_inpaint(args)
    except Exception as exc:
        message = " ".join(str(exc).split()) or exc.__class__.__name__
        if len(message) > 500:
            message = message[:497] + "..."
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
