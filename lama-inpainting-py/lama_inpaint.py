"""LaMa inpainting pipeline.

Drop-in module that takes an RGB float image and a binary mask,
returns the inpainted image with changes confined to the masked region.

Algorithm
---------
1. Find the selection's bounding box from the mask.
2. Pad it with a configurable amount of context (default 1 × max dim).
3. Reflect-pad the image if the ROI extends past its bounds.
4. Resize the ROI to 512 × 512 (the ONNX model's input size).
5. Run a single ONNX inference.
6. Resize the output back to the ROI's original size.
7. Compose: keep the model's output only inside the mask; original pixels
   elsewhere.
8. Paste back into the input image at the ROI position.

The two crucial properties:
- One inference per call regardless of image size.
- Composition preserves the original image outside the mask, so the
  model's "context re-render" is discarded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# The ONNX model was exported at this fixed spatial size.
MODEL_INPUT_SIZE = 512


def _resize_hwc(arr: np.ndarray, h: int, w: int, linear: bool) -> np.ndarray:
    """Resize an HWC array to (h, w). Uses cv2 if available; numpy fallback."""
    if cv2 is not None:
        inter = cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST
        return cv2.resize(arr, (w, h), interpolation=inter)
    # numpy fallback: nearest-neighbor only (bilinear weights not implemented).
    return _resize_numpy(arr, h, w)


def _resize_numpy(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pure-numpy nearest-neighbor resize. Faster than cv2 but no bilinear."""
    src_h, src_w = arr.shape[:2]
    if src_h == h and src_w == w:
        return arr.copy()
    if arr.ndim == 3:
        out = np.zeros((h, w, arr.shape[2]), dtype=arr.dtype)
    else:
        out = np.zeros((h, w), dtype=arr.dtype)

    # Nearest-neighbor sampling (always integer coordinates)
    ys = np.round(np.linspace(0, src_h - 1, h)).astype(np.intp)
    xs = np.round(np.linspace(0, src_w - 1, w)).astype(np.intp)
    out = arr[ys[:, None], xs[None, :]]
    return out


def _crop_with_reflect_pad(
    image: np.ndarray,
    mask: np.ndarray,
    roi_x1: int,
    roi_y1: int,
    roi_w: int,
    roi_h: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Crop ROI from image, edge-replicate padding if ROI extends past bounds.

    When the ROI extends past the image boundary, missing pixels are
    filled with the nearest-edge pixel value (``mode="edge"``). This
    gives LaMa's FFC local branches realistic context instead of the
    mirrored artifacts that ``mode="reflect"`` would produce.

    Returns (img_roi, mask_roi, paste_box) where paste_box is the
    (y1, y2, x1, x2) of the valid (non-padded) region in the
    original image coordinates.
    """
    H, W = image.shape[:2]

    pad_left = max(0, -roi_x1)
    pad_top = max(0, -roi_y1)
    pad_right = max(0, (roi_x1 + roi_w) - W)
    pad_bottom = max(0, (roi_y1 + roi_h) - H)

    img_padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    mask_padded = np.pad(
        mask,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge",
    )

    new_roi_x1 = roi_x1 + pad_left
    new_roi_y1 = roi_y1 + pad_top

    img_roi = img_padded[
        new_roi_y1 : new_roi_y1 + roi_h,
        new_roi_x1 : new_roi_x1 + roi_w,
        :,
    ]
    mask_roi = mask_padded[
        new_roi_y1 : new_roi_y1 + roi_h,
        new_roi_x1 : new_roi_x1 + roi_w,
    ]

    # The "real" (non-reflect-padded) region in original image coords
    y1 = max(0, roi_y1)
    y2 = min(H, roi_y1 + roi_h)
    x1 = max(0, roi_x1)
    x2 = min(W, roi_x1 + roi_w)

    return img_roi, mask_roi, (y1, y2, x1, x2)


class LamaInpainter:
    """Single-image inpainter using ONNX Runtime.

    Parameters
    ----------
    model_path : str | Path
        Path to the ONNX model file (.onnx).
    providers : list[str] | None
        ONNX Runtime execution providers. Defaults to preferring CUDA,
        falling back to CPU.
    """

    def __init__(self, model_path, providers=None):
        self.model_path = str(model_path)
        if providers is None:
            providers = ["CPUExecutionProvider"]
            # Only try CUDA if cuDNN/CUDA shared libraries are present.
            # ONNX Runtime advertises CUDAExecutionProvider regardless of
            # whether the underlying cuDNN/CUDA is installed, so we probe
            # by trying to create a session and silently falling back.
            if os.environ.get("LAMA_FORCE_CPU"):
                pass
            else:
                try:
                    test = ort.InferenceSession(
                        self.model_path,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    if "CUDAExecutionProvider" in test.get_providers():
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                except Exception:
                    pass
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self._input_names = [inp.name for inp in self.session.get_inputs()]
        self._output_name = self.session.get_outputs()[0].name

    @property
    def active_provider(self) -> str:
        return self.session.get_providers()[0]

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context_factor: float = 1.0,
    ) -> np.ndarray:
        """Inpaint the masked region of an image.

        Parameters
        ----------
        image : (H, W, 3) float32
            RGB image, values in [0, 1].
        mask : (H, W) or (H, W, 1) float32
            Binary mask, values in {0, 1}. Pixels with value > 0.5 are
            considered "to inpaint".
        context_factor : float
            How much surrounding context to give the model, as a multiplier
            of max(selection_w, selection_h). Default 1.0.

        Returns
        -------
        result : (H, W, 3) float32
            Same shape as ``image``, with the masked region replaced by
            the model's inpainting. Pixels outside the mask are unchanged.
        """
        # Normalize inputs
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] >= 3:
            image = image[:, :, :3]
        else:
            raise ValueError(f"image must have at least 3 channels, got {image.shape}")
        image = np.clip(image, 0.0, 1.0)

        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Match reference LaMa: any nonzero pixel is masked (predict.py: `(mask > 0) * 1`)
        # GIMP's antialiased selection edges produce values 0.0–0.5 which must be masked.
        mask = (mask > 0).astype(np.float32)

        H, W = image.shape[:2]
        if mask.shape != (H, W):
            raise ValueError(
                f"mask shape {mask.shape} doesn't match image {(H, W)}"
            )

        # Empty mask → no-op
        if mask.sum() == 0:
            return image.copy()

        # 1. Selection bbox
        ys, xs = np.where(mask > 0.5)
        sel_x1, sel_y1 = int(xs.min()), int(ys.min())
        sel_x2, sel_y2 = int(xs.max()) + 1, int(ys.max()) + 1
        sel_w, sel_h = sel_x2 - sel_x1, sel_y2 - sel_y1

        # 2. Padded ROI bbox
        ctx = max(128, int(round(max(sel_w, sel_h) * context_factor)))
        roi_x1 = sel_x1 - ctx
        roi_y1 = sel_y1 - ctx
        roi_w = sel_w + 2 * ctx
        roi_h = sel_h + 2 * ctx

        # 3. Crop with reflect-pad if ROI extends past image
        img_roi, mask_roi, paste_box = _crop_with_reflect_pad(
            image, mask, roi_x1, roi_y1, roi_w, roi_h
        )

        # 4. Pad ROI to multiple of 16 (the FFC spectral layers need /8
        #    with an even bottleneck for the onesided spectral inverse).
        #    We scale down only if the ROI exceeds MAX_SIDE to bound memory.
        MAX_SIDE = 1024
        scale = min(1.0, MAX_SIDE / max(roi_h, roi_w))
        sh, sw = max(1, int(round(roi_h * scale))), max(1, int(round(roi_w * scale)))
        img_s = _resize_hwc(img_roi, sh, sw, linear=True) if scale < 1.0 else img_roi
        mask_s = _resize_hwc(mask_roi, sh, sw, linear=False) if scale < 1.0 else mask_roi
        # Guard: keep the /16-padded side >= 32 so the /8 bottleneck stays
        # >= 4 px (the spectral inverse needs width > its onesided half).
        ph16 = max(32, (sh + 15) // 16 * 16)
        pw16 = max(32, (sw + 15) // 16 * 16)
        pt = (ph16 - sh) // 2
        pl = (pw16 - sw) // 2
        img_pad = np.pad(
            img_s, ((pt, ph16 - sh - pt), (pl, pw16 - sw - pl), (0, 0)), mode="reflect"
        )
        mask_pad = np.pad(
            mask_s, ((pt, ph16 - sh - pt), (pl, pw16 - sw - pl)), mode="reflect"
        )

        # 5. Inference
        img_chw = img_pad.transpose(2, 0, 1)[None, ...]  # (1, 3, ph16, pw16)
        mask_chw = mask_pad[None, None, :, :]  # (1, 1, ph16, pw16)
        out = self.session.run(
            [self._output_name],
            {self._input_names[0]: img_chw, self._input_names[1]: mask_chw},
        )[0]
        out_padded = out[0].transpose(1, 2, 0) / 255.0  # (ph16, pw16, 3) in [0, 1]
        out_padded = np.clip(out_padded, 0.0, 1.0)

        # 6. Crop pad border and resize back to ROI size if we scaled down
        out_cropped = out_padded[pt : pt + sh, pl : pl + sw, :]
        out_roi = _resize_hwc(out_cropped, roi_h, roi_w, linear=True) if scale < 1.0 else out_cropped
        out_roi = np.clip(out_roi, 0.0, 1.0)

        # 7. Masked composition: model's output only inside the mask
        result_roi = np.where(
            mask_roi[:, :, None] > 0.5,
            out_roi,
            img_roi,
        )

        # 8. Paste back into the original image (skip the reflect-pad parts)
        result = image.copy()
        y1, y2, x1, x2 = paste_box
        real_y1 = max(0, -roi_y1)
        real_x1 = max(0, -roi_x1)
        real_y2 = real_y1 + (y2 - y1)
        real_x2 = real_x1 + (x2 - x1)
        result[y1:y2, x1:x2, :] = result_roi[real_y1:real_y2, real_x1:real_x2, :]

        return result


def find_model(default_paths=None, package_dir=None) -> str:
    """Locate the ONNX model file. Search a small list of well-known paths.

    Parameters
    ----------
    default_paths : list[str] | None
        Extra paths to search first.
    package_dir : str | None
        Directory of the GIMP extension (where the plugin source lives).
        When provided, it's searched first along with ``models/`` beneath it.

    Raises FileNotFoundError if no model is found.
    """
    candidates = []
    if default_paths:
        candidates.extend(default_paths)

    # Environment variable override
    env_path = os.environ.get("LAMA_MODEL_PATH")
    if env_path:
        candidates.append(env_path)

    # Package directory (the GIMP extension's own folder)
    if package_dir:
        candidates.extend(
            [
                os.path.join(package_dir, "lama_fp32.onnx"),
                os.path.join(package_dir, "models", "lama_fp32.onnx"),
            ]
        )

    # Common install locations
    candidates.extend(
        [
            "/usr/share/gimp-lama/lama_fp32.onnx",
            "/usr/local/share/gimp-lama/lama_fp32.onnx",
            os.path.expanduser("~/.local/share/gimp-lama/lama_fp32.onnx"),
            os.path.expanduser(
                "~/.local/share/gimp/3.2/extensions/org.gimp.extension.lama-inpainting/lama_fp32.onnx"
            ),
            os.path.expanduser(
                "~/.local/share/gimp/3.0/extensions/org.gimp.extension.lama-inpainting/lama_fp32.onnx"
            ),
            # Windows MSYS2 install
            os.path.expanduser(
                "~/AppData/Roaming/GIMP/3.2/extensions/org.gimp.extension.lama-inpainting/lama_fp32.onnx"
            ),
        ]
    )
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find lama_fp32.onnx. Set LAMA_MODEL_PATH or place "
        "the model in one of the standard install locations:\n  "
        + "\n  ".join(candidates)
    )
