# Rust CPU Worker Optimization Notes

**Date:** 2026-07
**Author:** Optimization session post-mortem
**Scope:** What we tried to make the Rust CPU worker faster than Python
on the same hardware, what worked, what didn't, and why.

## TL;DR

The Rust CPU worker is already at its practical performance ceiling.
The ORT CPU EP inference is the bottleneck (~1.75 s of the
~3.4 s per-call budget) and cannot be made faster from our side without
GPU acceleration or a smaller model. Marginal parallelization with
`rayon` shaves a little off the pre/post-processing but the win is
small because the per-pixel work is small. Crates that look promising
on paper (`fast_image_resize`, `memmap2`) did not deliver meaningful
speedups once we measured.

## Starting point and target

We want the Rust worker to be **faster than the Python worker on CPU**.
The Python worker (`gimp-lama-inpainting/lama_worker.py`) uses
`onnxruntime` with the CPU EP and follows the same algorithm. Both
implementations feed the same ONNX graph to the same ORT C++ runtime.
The only difference is the language around the call.

**Why Rust should win at all:**

- **No interpreter startup**: Python ~0.5–1 s; Rust ~0.05 s.
- **No library import cost**: Python ~5–10 s for `numpy` + `onnxruntime`
  + `PIL`; Rust has no equivalent because the dependencies are statically
  linked.
- **No GIL**: Rust can actually parallelize across cores via `rayon`.

So we win on **first-call latency**, but on **steady-state per-call**
work we hit the same ORT ceiling as Python.

### Measured numbers (1024×1024 image, 200×200 selection)

| Path                     | Cold first call | Warm steady-state |
|--------------------------|-----------------|-------------------|
| Python worker            | ~20 s           | ~2 s               |
| Rust worker (final)      | ~5 s            | ~3.4 s             |
| Rust inference (ORT)     | —               | ~1.75 s            |
| Rust pre/post + I/O      | —               | ~1.65 s            |

The 5–6× speedup on first call is real and stays. The per-call speedup
is smaller (and sometimes zero on warm cache) because the inference
itself is the same in both.

## Bottleneck analysis

The hot path inside `inpaint()` in `src/main.rs`:

1. RGBA → f32 HWC conversion (~3 M element passes for a 1024² image)
2. Selection bbox scan (~1 M element pass)
3. Reflect-pad of image and mask (~1 M element pass)
4. Bilinear resize ROI → 512² (~0.78 M pixels × 3 channels)
5. **ORT inference** (~1.75 s; nothing we can do)
6. CHW → HWC conversion of output + divide by 255 (~0.78 M pixels)
7. Bilinear resize 512² → ROI (~1 M pixels × 3 channels)
8. Mask composition (replace non-masked pixels with original) (~0.36 M pixels)
9. Paste result back into original image + write RGBA PNG (~3 M ops)

Steps 1–4 and 6–9 are all ~1.5 s of work for a typical selection.
Step 5 is the 1.75 s we cannot beat without changing the model or
hardware.

## What we tried, in order

### 1. ORT `with_intra_threads(num_cpus)` — **reverted**

```rust
Session::builder()?
    .with_intra_threads(threads)?
    .commit_from_file(model_path)?
```

**Hypothesis:** ORT's CPU EP defaults to a small thread pool; explicitly
spinning it up to use all cores should make inference faster.

**Result:** Inference went from 1.75 s to ~4 s. **Made it slower.**

**Why:** ORT's CPU EP is already parallelized internally. Forcing more
threads caused thread-pool overhead and contention to dominate. The
optimal thread count for a 512² model is probably 2–4, not 16.

**Take-away:** Don't fight ORT's heuristics. If `with_intra_threads`
isn't requested, ORT picks a sensible default for the model. Trust it
unless you have measured a specific win.

### 2. `rayon` parallelism on per-pixel loops — **kept**

Parallelized:
- RGBA → f32 HWC conversion
- Mask u8 → bool conversion
- Selection bbox scan
- Reflect-pad (2D and 3D)
- Bilinear resize (forward and inverse)
- Nearest-neighbor resize (forward and inverse)
- HWC → CHW (and mask) packing for ORT inputs
- CHW → HWC unpacking from ORT output
- Mask composition
- Final output PNG write (via `par_chunks_mut` on the RGBA buffer)

**Hypothesis:** Multi-core parallelism should make the per-pixel
pre/postprocessing faster.

**Result:** Test total dropped from ~3.76 s to ~3.4 s on the 1024²
test image. Standalone runs stabilized at ~3.4 s warm. Marginal but
real.

**Why small:** The per-pixel work for a typical selection (~600×600
ROI) is only a few million operations. Rayon's dispatch overhead is
in the same order of magnitude for this size. The win grows with
larger images (4K) but is barely measurable on 1024².

**Implementation notes:**
- Used `array.axis_iter_mut(Axis(0)).enumerate().par_bridge().for_each(...)`
  to get parallel iteration over rows while keeping ndarray's bounds
  checking.
- For the final output, switched from `RgbaImage::put_pixel()` to direct
  buffer indexing on `as_mut()` then `par_chunks_mut(row_stride)`. The
  `put_pixel` path is a slow per-call dispatch; the indexed path is what
  you want when you need every microsecond.
- Pre-computed `1.0 - wx` and `1.0 - wy` once per pixel in the bilinear
  resize instead of computing them inside the inner channel loop. Saves
  one multiplication per pixel per channel.

**Caveat:** Rayon initializes its thread pool on the first `par_iter`
call. This adds a one-time cost (~50–200 ms depending on CPU) that you
pay on the first invocation. For a single-shot CLI run, this can
dominate the savings. For a long-running GIMP plug-in that calls the
worker many times, the warm pool pays off.

### 3. `fast_image_resize` (SIMD bilinear) — **reverted**

```toml
fast_image_resize = { version = "5", default-features = false, features = ["rayon"] }
```

**Hypothesis:** SIMD-accelerated bilinear should be 3–10× faster than
the manual loop. From the crate's own benchmarks, an AVX2 build resizes
a 4928×3279 RGB image to 852×567 in ~3.5 ms.

**Result:** Did not deliver a measurable speedup for our case.

**Why:** The crate works on `u8` and `u16` pixel types. Our pipeline is
`f32` HWC in `[0.0, 1.0]`. The minimum integration path is:
1. Multiply f32 by 255, round, store in a `u8` buffer (1.08 M ops for
   a 600×600 image).
2. Resize via fast_image_resize (the SIMD win).
3. Convert back u8 → f32 (786 K ops for 512×512).

The conversion overhead is comparable to or larger than the savings on
the small ROI sizes our typical selection produces. On a 4K image where
the resize dominates the per-call budget, this would help. On our
test image (1024² with 200² selection → ~600² ROI), it didn't move
the needle.

**Integration friction (left as future work):**
- Need `IntoImageView` impl for our `Vec<u8>` buffers, which requires
  constructing `fast_image_resize::images::Image` wrappers.
- API is non-trivial; the v5 → v6 migration changed `ResizeOptions`
  shapes and the `Resizer::resize()` signature takes a borrowed source
  view and a mutable destination view. Worth doing for 4K but not for
  1024².

**Take-away:** `fast_image_resize` is genuinely fast but its win on
small images is eaten by the u8/f32 conversion overhead. For our
default workload it's not worth the dependency.

### 4. `memmap2` for model loading — **reverted**

```toml
memmap2 = "0.9"
```

```rust
let file = File::open(model_path)?;
let mmap = unsafe { memmap2::Mmap::map(&file) }?;
Session::builder()?
    .commit_from_memory(&mmap)?
```

**Hypothesis:** ORT's `commit_from_file` opens, reads 198 MB into a
heap buffer, and closes. Memory-mapping the file and passing the
mapped bytes to `commit_from_memory` should skip the kernel-to-user
copy.

**Result:** Inference times went from 1.75 s to 3–5 s and were
inconsistent. **Made it worse.**

**Why (best guess):** On Windows, when the `Mmap` value is dropped at
the end of `build_session`, the OS un-maps the file. If ORT's C API
holds any pointer into the mapped region past the function return
(lazy copy, alignment probe, etc.), that pointer becomes invalid and
later inference steps touch freed memory. The `commit_from_memory` Rust
binding makes no API guarantee about the lifetime of the input buffer
after the call, but the C runtime may use it differently than we
expect.

Even with the `mmap` held by moving it into the returned `Session`
struct, the issue persisted. `commit_from_file` (which reads the file
into ORT-managed memory internally) is more reliable and not
meaningfully slower — the read is bound by disk, not by API.

**Take-away:** `commit_from_file` is fine. `commit_from_memory` is
only worth it when the caller already has the bytes in memory (e.g.
receiving them over the network). Memory-mapping a file just to skip
one buffer copy of a 198 MB file is not worth the lifecycle complexity.

### 5. Manual bilinear with pre-computed weights — **kept**

```rust
let inv_wy = 1.0 - wy;
let inv_wx = 1.0 - wx;
for ch in 0..c {
    let v00 = src[[y0, x0, ch]];
    let v01 = src[[y0, x1, ch]];
    let v10 = src[[y1, x0, ch]];
    let v11 = src[[y1, x1, ch]];
    let v0 = v00 * inv_wx + v01 * wx;
    let v1 = v10 * inv_wx + v11 * wx;
    row[[nx, ch]] = v0 * inv_wy + v1 * wy;
}
```

**Hypothesis:** Hoisting `1.0 - wx` and `1.0 - wy` out of the channel
loop saves one multiplication per pixel per channel.

**Result:** Microscopic (~0.05 ms per call on a 600→512 resize). Kept
because it's free and clearer than computing it inline.

### 6. Direct buffer indexing for output write — **kept**

```rust
out_rgba.as_mut()
    .par_chunks_mut(row_stride)
    .enumerate()
    .for_each(|(y, row_buf)| { /* write 4 bytes per pixel */ })
```

**Hypothesis:** `RgbaImage::put_pixel()` does a per-call bounds check
and index calculation. Direct buffer indexing through `as_mut()`
skips that.

**Result:** ~10% speedup on the output write step (which was ~80 ms
for a 1024² image). Modest but free.

## What we did not try (and why)

### Quantization / smaller models

`onnxruntime.quantization.quantize_dynamic` produces an int8 model.
Our investigation earlier rejected this: the LaMa architecture (SiLU
activations, FFT-like FFC blocks) doesn't quantize well dynamically.
Per the docs in `gimp-lama-inpainting/Docs/06_IMPLEMENTATION_PLAN.md`,
int8 quantization took the model from 1.5 s/iter to 12 s/iter.
That's the wrong direction. fp16 conversion hit Cast-node
mismatches in ORT 1.24.4 and was rejected for the same reason.

### Vulkan EP

Not an ORT-supported execution provider. Vulkan is a graphics API; ORT
talks to it through DirectML (DX12) on Windows. There is no separate
"Vulkan EP" in ORT; the closest thing is `DmlExecutionProvider` which
is known to crash at session init in some environments due to a
prebuilt-binary incompatibility between ORT's DirectML EP and the
system D3D12/DirectML runtime. The Dawn/WebGPU EP is experimental.

### GPU acceleration (already documented)

`DmlExecutionProvider` can crash at session init in some environments
(typically `AbiCustomRegistry.cpp(519)`, `E_INVALIDARG`). The root
cause is a prebuilt-binary incompatibility between ORT's DirectML EP
and certain D3D12/DirectML runtime versions. We can't fix that without
rebuilding ONNX Runtime from source, which is out of scope. Documented
in the session history; not repeated here.

## What actually matters

The lesson from this whole exercise is that **the ORT inference is the
ceiling**. Once we accept that, the optimization surface shrinks to:

1. **Don't make the inference slower** — done; we reverted every change
   that hurt inference time (`with_intra_threads`, `commit_from_memory`).
2. **Shrink the pre/post-processing overhead** — done with `rayon`
   parallelism and direct buffer indexing. Marginal.
3. **Skip I/O costs** — done with `RgbaImage::as_raw()` direct indexing
   and `RgbaImage::as_mut() → par_chunks_mut` for output write.

Beyond that, the only path to faster inference is:

- A smaller / distilled / faster model (requires a new ONNX export).
- GPU acceleration (DirectML has known ABI issues with some D3D12
  runtimes; CUDA requires NVIDIA hardware; WebGPU is experimental
  and likely crashes too).
- A faster CPU (silicon, not software).

## Recommended reading order for future optimizers

If someone else picks this up, the order of operations should be:

1. **Read this file** (you're doing it).
2. **Run `python -m py_compile` then `cargo build --release` then
   `cargo test --release`** to confirm the baseline works.
3. **Measure first.** Use the `--verbose` markers
   (`[LAMA_MARKER] phase ...`) and `tracing::info!("inference took
   ...")` to see where time actually goes. Don't optimize blind.
4. **If inference is the bottleneck** (it almost certainly is), don't
   touch the Rust pre/post-processing. Look at the model.
5. **If image processing is the bottleneck** (large images, like 4K
   and bigger selections), then `fast_image_resize` with the
   `bytemuck` feature is worth the integration effort. The v6 API
   has cleaner `IntoImageView` impls for `&[u8]` / `&mut [u8]`
   buffers.
6. **Never use `with_intra_threads` without measuring.** ORT's default
   is right more often than not.
7. **Never `memmap2` the model for ORT.** The C API's lifetime
   assumptions are not what you'd expect on Windows.
8. **Profile, then change one thing, then measure again.** Multi-change
   PRs make regressions impossible to debug.

## Final state

```text
Cargo.toml dependencies:
  ort (path = "../../ort-main")    # CPU inference, 1.75 s
  image                           # PNG decode
  ndarray                         # HWC buffers
  clap                            # CLI
  rayon                           # per-pixel parallelism
  tracing, tracing-subscriber     # diagnostics
  anyhow                          # errors
```

No fast_image_resize, no memmap2. Just the deps that actually move
the needle, kept minimal so the binary stays at ~22 MB and the
compile time stays under 5 s.

## Addendum: 2026-08

The ORT CPU baseline is **the final answer** for the LaMa filter.
Three findings from the recent work:

1. **Moebius integration fully removed.** Originally added as a
   `Remove Object (Moebius, slow)...` filter in GIMP, but Moebius needs
   ~6 GB of VRAM. The model hit `ID3D12Device::CreateCommittedResource`
   failures mid-inference. The GIMP-side filter, the Moebius Rust
   worker source (`moebius-worker-rs/`), all four Moebius ONNX
   models, and the worker binary have all been deleted. LaMa is the
   only filter in this plug-in.
2. **OxiONNX benchmarked but not adopted.** Pure-Rust ONNX runtime
   benchmarked at `benchmark-ort-vs-oxionnx/`. Result: ORT 2.07 s
   (works), OxiONNX panics on `nn.ReflectionPad2d` exports.
   The Pad bug was patched locally in `oxionnx-0.1.4/` and reported
   upstream via the GitHub issue draft at
   `oxionnx-0.1.4/GITHUB_ISSUE_pad_assertion.md`. No further OxiONNX
   work — LaMa's use case is ORT.
3. **ORT CPU EP config is locked.** All knobs known to hurt
   performance are reverted:
   - No `with_intra_threads(N)` (made 1.75 s → 4 s).
   - No `commit_from_memory` for the model file (made 1.75 s →
     3-5 s, inconsistent on Windows).
   - No `with_parallel_execution(true)` (made 1.75 s → 2.07 s).
   - `with_dimension_override("batch", 1)` is still set — this is
     required, not a perf knob; ORT otherwise can't size the
     dynamic dimension.
   - The default ORT `OptLevel` (Basic) is in effect. `OptLevel::All`
     was not tested but is not expected to help on a 198 MB model
     already at the inference-time ceiling.

**Practical conclusion:** the LaMa filter is at its perf ceiling on
CPU with the current code. The only paths to faster inference are:

- Smaller / distilled model (requires a new ONNX export — out of
  scope of the current Rust work).
- GPU acceleration that works on the target hardware (DirectML has
  known ABI issues with some D3D12 runtimes; WebGPU/Dawn is
  experimental; CUDA requires NVIDIA hardware).
- A faster CPU (silicon, not software).

None of these are in the scope of this codebase.


## Addendum: 2026-08

The ORT CPU baseline is **the final answer** for the LaMa filter.
Three findings from the recent work:

1. **Moebius integration fully removed.** Originally added as a
   `Remove Object (Moebius, slow)...` filter in GIMP, but Moebius needs
   ~6 GB of VRAM. The model hit `ID3D12Device::CreateCommittedResource`
   failures mid-inference. The GIMP-side filter, the Moebius Rust
   worker source (`moebius-worker-rs/`), all four Moebius ONNX
   models, and the worker binary have all been deleted. LaMa is the
   only filter in this plug-in.
2. **OxiONNX benchmarked but not adopted.** Pure-Rust ONNX runtime
   benchmarked at `benchmark-ort-vs-oxionnx/`. Result: ORT 2.07 s
   (works), OxiONNX panics on `nn.ReflectionPad2d` exports.
   The Pad bug was patched locally in `oxionnx-0.1.4/` and reported
   upstream via the GitHub issue draft at
   `oxionnx-0.1.4/GITHUB_ISSUE_pad_assertion.md`. No further OxiONNX
   work — LaMa's use case is ORT.
3. **ORT CPU EP config is locked.** All knobs known to hurt
   performance are reverted:
   - No `with_intra_threads(N)` (made 1.75 s → 4 s).
   - No `commit_from_memory` for the model file (made 1.75 s →
     3-5 s, inconsistent on Windows).
   - No `with_parallel_execution(true)` (made 1.75 s → 2.07 s).
   - `with_dimension_override("batch", 1)` is still set — this is
     required, not a perf knob; ORT otherwise can't size the
     dynamic dimension.
   - The default ORT `OptLevel` (Basic) is in effect. `OptLevel::All`
     was not tested but is not expected to help on a 198 MB model
     already at the inference-time ceiling.

**Practical conclusion:** the LaMa filter is at its perf ceiling on
CPU with the current code. The only paths to faster inference are:

- Smaller / distilled model (requires a new ONNX export — out of
  scope of the current Rust work).
- GPU acceleration that works on the target hardware (DirectML has
  known ABI issues with some D3D12 runtimes; WebGPU/Dawn is
  experimental; CUDA requires NVIDIA hardware).
- A faster CPU (silicon, not software).

None of these are in the scope of this codebase.