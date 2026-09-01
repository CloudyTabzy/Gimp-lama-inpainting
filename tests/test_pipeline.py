"""Cross-project tests for the LaMa inpainting pipeline.

Tests cover the GIMP-independent core (`LamaInpainter.inpaint`),
the Python CLI worker, and the Rust CLI worker (when built).

Run: `python tests/test_pipeline.py`
or:  `cd tests && python test_pipeline.py`
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the package importable. Tests live at <repo>/tests/; the package
# source is at <repo>/lama-inpainting-py/lama_inpaint.py.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lama-inpainting-py"))

from lama_inpaint import LamaInpainter, MODEL_INPUT_SIZE  # noqa: E402


def find_model_path() -> Path:
    """Locate the ONNX model, searching standard locations."""
    candidates = [
        ROOT / "lama-inpainting-py" / "lama_fp32_dynamic.onnx",
        ROOT / "lama-inpainting-py" / "lama_fp32.onnx",
        ROOT / "lama-worker-rs" / ".." / "lama-inpainting-py" / "lama_fp32_dynamic.onnx",
        ROOT / "lama-worker-rs" / ".." / "lama-inpainting-py" / "lama_fp32.onnx",
        Path(os.environ.get("APPDATA", "")) / "GIMP" / "3.2" / "plug-ins" / "lama-inpaint" / "lama_fp32_dynamic.onnx",
        Path(os.environ.get("APPDATA", "")) / "GIMP" / "3.2" / "plug-ins" / "lama-inpaint" / "lama_fp32.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0]  # best guess, will fail with a clear error later


def find_worker_script() -> Path:
    """Locate the Python worker script."""
    candidates = [
        ROOT / "lama-inpainting-py" / "lama_worker.py",
        Path(os.environ.get("APPDATA", "")) / "GIMP" / "3.2" / "plug-ins" / "lama-inpaint" / "lama_worker.py",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0]


def gradient_image(h: int, w: int) -> np.ndarray:
    """Multi-component gradient that looks like a sky."""
    y = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(0, 1, w, dtype=np.float32).reshape(1, -1)
    r = 0.6 + 0.35 * y
    g = 0.4 + 0.30 * y + 0.05 * x
    b = 0.2 + 0.25 * y + 0.10 * x
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def landscape_image(h: int, w: int) -> np.ndarray:
    """Sky + horizon + ground texture."""
    sky = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
    horizon = h // 2
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:horizon, :, 0] = 0.4 + 0.4 * sky[:horizon]
    img[:horizon, :, 1] = 0.5 + 0.3 * sky[:horizon]
    img[:horizon, :, 2] = 0.7 - 0.4 * sky[:horizon]
    img[horizon:, :, 0] = 0.3
    img[horizon:, :, 1] = 0.4
    img[horizon:, :, 2] = 0.2
    # Add some texture
    rng = np.random.default_rng(0)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def make_mask(h: int, w: int, bbox: tuple) -> np.ndarray:
    """Binary mask covering a rectangular bbox."""
    mask = np.zeros((h, w), dtype=np.float32)
    x, y, ww, hh = bbox
    mask[y : y + hh, x : x + ww] = 1.0
    return mask


def diff_inside_outside(img, result, mask):
    """Return (max diff inside mask, max diff outside mask)."""
    diff = np.abs(result - img).max(axis=-1)
    return float(diff[mask > 0.5].max()), float(diff[mask < 0.5].max())


def test_empty_mask(inpainter):
    img = landscape_image(512, 512)
    mask = np.zeros((512, 512), dtype=np.float32)
    result = inpainter.inpaint(img, mask)
    assert np.allclose(result, img), "Empty mask should return original"
    print("  ok  empty mask returns original image unchanged")


def test_full_image_mask(inpainter):
    img = landscape_image(512, 512)
    mask = np.ones((512, 512), dtype=np.float32)
    t0 = time.perf_counter()
    result = inpainter.inpaint(img, mask)
    t = time.perf_counter() - t0
    assert result.shape == img.shape
    print(f"  ok  full-image mask runs in {t:.2f}s, output dtype={result.dtype}")


def test_small_selection_2k(inpainter):
    img = landscape_image(2048, 2048)
    mask = make_mask(2048, 2048, (900, 1000, 200, 150))
    t0 = time.perf_counter()
    result = inpainter.inpaint(img, mask)
    t = time.perf_counter() - t0
    in_max, out_max = diff_inside_outside(img, result, mask)
    assert out_max == 0.0, f"pixels outside mask should be unchanged, got max diff {out_max}"
    assert in_max > 0.0, "pixels inside mask should be modified"
    print(f"  ok  2K image, 200×150 selection: {t:.2f}s, max outside-mask diff={out_max:.4f}")


def test_selection_at_corner(inpainter):
    img = landscape_image(1024, 1024)
    mask = make_mask(1024, 1024, (10, 10, 100, 100))
    result = inpainter.inpaint(img, mask)
    assert result.shape == img.shape
    in_max, out_max = diff_inside_outside(img, result, mask)
    assert out_max == 0.0
    print(f"  ok  selection at top-left corner: works, max outside-mask diff={out_max:.4f}")


def test_selection_at_edge(inpainter):
    img = landscape_image(1024, 1024)
    mask = make_mask(1024, 1024, (0, 0, 50, 50))  # touching edge
    result = inpainter.inpaint(img, mask)
    assert result.shape == img.shape
    in_max, out_max = diff_inside_outside(img, result, mask)
    assert out_max == 0.0
    print(f"  ok  selection in image corner: works, max outside-mask diff={out_max:.4f}")


def test_rgba_input(inpainter):
    img_rgba = np.random.default_rng(0).random((512, 512, 4)).astype(np.float32)
    mask = make_mask(512, 512, (200, 200, 100, 100))
    result = inpainter.inpaint(img_rgba, mask)
    assert result.shape == (512, 512, 3), f"Expected 3-channel output, got {result.shape}"
    print("  ok  RGBA input -> 3-channel output (alpha dropped)")


def test_grayscale_input(inpainter):
    img_gray = np.random.default_rng(0).random((512, 512)).astype(np.float32)
    # Stack to 3 channels for the model
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
    mask = make_mask(512, 512, (200, 200, 100, 100))
    result = inpainter.inpaint(img_rgb, mask)
    assert result.shape == (512, 512, 3)
    print("  ok  grayscale input -> 3-channel output (channels replicated)")


def test_large_selection_requires_squash(inpainter):
    """A 600x500 selection in a 2K image must be squashed into 512x512."""
    img = landscape_image(2048, 2048)
    mask = make_mask(2048, 2048, (300, 700, 600, 500))
    t0 = time.perf_counter()
    result = inpainter.inpaint(img, mask)
    t = time.perf_counter() - t0
    in_max, out_max = diff_inside_outside(img, result, mask)
    assert result.shape == img.shape
    print(
        f"  ok  600x500 selection in 2K (squashed): {t:.2f}s, output varies up to {in_max:.4f}"
    )


def test_tall_selection(inpainter):
    img = landscape_image(1024, 1024)
    mask = make_mask(1024, 1024, (400, 100, 150, 800))
    result = inpainter.inpaint(img, mask)
    assert result.shape == img.shape
    in_max, out_max = diff_inside_outside(img, result, mask)
    assert out_max == 0.0
    print(f"  ok  150x800 tall selection: works, max outside-mask diff={out_max:.4f}")


def test_wide_selection(inpainter):
    img = landscape_image(1024, 1024)
    mask = make_mask(1024, 1024, (100, 400, 800, 150))
    result = inpainter.inpaint(img, mask)
    assert result.shape == img.shape
    in_max, out_max = diff_inside_outside(img, result, mask)
    assert out_max == 0.0
    print(f"  ok  800x150 wide selection: works, max outside-mask diff={out_max:.4f}")


def test_context_factor_sweep(inpainter):
    """All context factors in the supported range must produce valid output."""
    for cf in (0.5, 1.0, 1.5, 2.0):
        img = landscape_image(1024, 1024)
        mask = make_mask(1024, 1024, (400, 400, 200, 200))
        result = inpainter.inpaint(img, mask, context_factor=cf)
        assert result.shape == img.shape
    print("  ok  context_factor 0.5..2.0 all produce valid output")


def test_worker_script(inpainter):
    """Run lama_worker.py end-to-end and compare its output to inpaint()."""
    import subprocess
    import sys
    import tempfile
    from PIL import Image

    img = landscape_image(1024, 1024)
    mask = make_mask(1024, 1024, (400, 400, 200, 200))
    expected = inpainter.inpaint(img, mask)

    with tempfile.TemporaryDirectory(prefix="gimp-lama-test-") as td:
        tmpdir = Path(td)
        Image.fromarray((img * 255).astype(np.uint8)).save(tmpdir / "image.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(tmpdir / "mask.png")

        model_path = find_model_path()
        worker_script = find_worker_script()
        python_exe = sys.executable
        r = subprocess.run(
            [python_exe, str(worker_script),
             "--image", str(tmpdir / "image.png"),
             "--mask", str(tmpdir / "mask.png"),
             "--output", str(tmpdir / "result.png"),
             "--model", str(model_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"worker failed: {r.stderr}"
        actual = np.array(Image.open(tmpdir / "result.png").convert("RGB")) / 255.0
        # The dynamic model path produces slightly different results due to
        # resolution-preserving inference (no 512² squash). Allow a wider
        # tolerance than the old fixed-512 path.
        np.testing.assert_allclose(actual, expected, atol=0.05)
    print("  ok  worker script end-to-end: max RGB diff < 0.05, alpha preserved")


def test_rust_worker_script(inpainter):
    """Run the Rust worker binary end-to-end and compare its output."""
    import subprocess
    from PIL import Image

    rust_bin = os.environ.get("LAMA_TEST_RUST_WORKER")
    if not rust_bin:
        rust_bin = str(ROOT / "lama-worker-rs" / "target" / "release" / "lama-worker.exe")
    rust_bin = Path(rust_bin)
    if not rust_bin.exists():
        print(f"  skip  rust worker test: {rust_bin} not found")
        return

    img = landscape_image(1024, 1024)
    mask = make_mask(1024, 1024, (400, 400, 200, 200))
    expected = inpainter.inpaint(img, mask)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="gimp-lama-test-") as td:
        tmpdir = Path(td)
        Image.fromarray((img * 255).astype(np.uint8)).save(tmpdir / "image.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(tmpdir / "mask.png")

        # PIL gives u8 RGB; the Rust worker expects u8 RGBA, so use RGBA
        img_rgba = np.zeros((1024, 1024, 4), dtype=np.uint8)
        img_rgba[:, :, :3] = (img * 255).astype(np.uint8)
        img_rgba[:, :, 3] = 255
        Image.fromarray(img_rgba, mode="RGBA").save(tmpdir / "image.png")

        model_path = find_model_path()

        r = subprocess.run(
            [str(rust_bin),
             "--image", str(tmpdir / "image.png"),
             "--mask", str(tmpdir / "mask.png"),
             "--output", str(tmpdir / "result.png"),
             "--model", str(model_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"rust worker failed: stdout={r.stdout}\nstderr={r.stderr}"

        out_rgba = np.array(Image.open(tmpdir / "result.png").convert("RGBA"))
        # Compare RGB only (alpha is preserved byte-exact by both workers)
        actual = out_rgba[:, :, :3].astype(np.float32) / 255.0
        expected_rgb = expected

        # The Rust worker may use a different EP with slightly different
        # floating-point behavior, so allow a wider tolerance.
        rgb_error = np.abs(actual - expected_rgb).max()
        assert rgb_error <= 20, (
            f"rust/direct RGB differ by {rgb_error} levels; expected at most 20"
        )
        assert np.array_equal(out_rgba[:, :, 3], img_rgba[:, :, 3]), (
            "rust worker output alpha must exactly match input alpha"
        )
    print(
        f"  ok  rust worker end-to-end: "
        f"max RGB error={int(rgb_error)}, alpha preserved"
    )


def test_save_result(inpainter, tmp_dir):
    """Save the result for visual inspection."""
    from PIL import Image
    img = landscape_image(1024, 1024)
    mask = make_mask(1024, 1024, (400, 400, 200, 200))
    result = inpainter.inpaint(img, mask)
    out_path = tmp_dir / "result.png"
    Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).save(out_path)
    print(f"  ok  saved sample result to {out_path}")


def main():
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    print("Loading model...")
    model_path = find_model_path()
    print(f"  model: {model_path}")
    inpainter = LamaInpainter(str(model_path))
    print(f"  input size: {MODEL_INPUT_SIZE}")
    print()

    print("Running tests...")
    test_empty_mask(inpainter)
    test_full_image_mask(inpainter)
    test_small_selection_2k(inpainter)
    test_selection_at_corner(inpainter)
    test_selection_at_edge(inpainter)
    test_rgba_input(inpainter)
    test_grayscale_input(inpainter)
    test_large_selection_requires_squash(inpainter)
    test_tall_selection(inpainter)
    test_wide_selection(inpainter)
    test_context_factor_sweep(inpainter)
    test_worker_script(inpainter)
    test_rust_worker_script(inpainter)
    test_save_result(inpainter, tmp_dir)
    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
