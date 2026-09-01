# LaMa Inpainting Plug-in — Lessons & Future Directions

> Condensed from the development session. Keeps what is durable and
> useful; drops the implementation plan, the chronological attempts log,
> and anything superseded by the final `.interp` shebang solution.

**Date:** 2026-07
**Status:** Reference notes; not a build plan

---

## 1. The wall: MINGW Clang vs MSVC wheels

GIMP 3.2 ships a MINGW Clang–built Python 3.14 with `gi` and GEGL,
but **every** `cp314` wheel on PyPI (`numpy`, `onnxruntime`,
`opencv-python`, `imath`) is MSVC-built. The two ABIs do not
match, and `numpy`'s import code refuses to load MSVC `.pyd` files
in a MINGW Python with a hard error — not a warning.

Three ways past it:

1. **MSYS2 + build MINGW numpy/onnxruntime from source** — 1+ GB
   toolchain, hours of build time. Out of scope.
2. **Portable MINGW Python + MINGW-built wheel bundle** — same
   problem at a different scale.
3. **Sidecar architecture** — keep the GIMP side pure-stdlib + `gi`
   + GEGL; do the ML work in a separate configured system Python
   process where MSVC wheels load normally. **Chosen.**

The sidecar is the only realistic path. The whole GIMP side stays
clean (`import gi`, `Gimp`, `GimpUi`, `Gegl`, `GLib` only); the
worker process gets `numpy`, `onnxruntime`, `Pillow`. They talk via
temp PNGs.

---

## 2. The sidecar pattern

```
GIMP process  (MINGW Python 3.14, gi + GEGL only)
  └─ #!lama-gimp-python  (shebang → user-level .interp mapping)
       └─ lama-inpaint.py
            ├─ exports drawable + selection to temp PNGs via GEGL
            ├─ spawns worker subprocess (system Python or Rust)
            └─ loads result PNG into drawable's shadow buffer,
               merge_shadow(True)
                    │
                    ▼
           worker process  (MSVC Python 3.10+ or Rust)
               onnxruntime / ort CPU provider
               🌟 model pipeline:
                  reflect-pad ROI → 512² resize → inference →
                  masked composition → RGBA output
                  (input alpha bytes preserved)
```

**Invariants that must hold for every future filter:**

- The GIMP-side plug-in imports **only** `gi`, `Gimp`, `GimpUi`, `Gegl`,
  `GLib`. No `numpy`, no `PIL`, no `onnxruntime`, no `cv2`. This is the
  entire point of the architecture.
- Worker communication is file-based (temp PNGs). No shared
  Python interpreter, no shared C ABI, no sockets.
- One inference per call. The core reflect-pads the selection ROI,
  crops+resizes to model input, infers, resizes back, and does masked
  composition. Pixels outside the mask are byte-identical to input.
- No GIMP installation is modified. Per-user interpreter alias
  (`.interp` files in `%APPDATA%\GIMP\3.2\interpreters\`) + shebang on
  the plug-in's first line is the canonical handoff, not patching
  `pygimp.interp`, not modifying `PATH`, not re-exec'ing across CRTs.

---

## 3. Getting GIMP to actually use its own Python

On stock Windows GIMP 3.2, third-party `.py` plug-ins can initially
be launched by the first system `python.exe`/`pythonw.exe` on
`PATH` (typically system Python 3.13, which has no `gi`). We saw this
break empirically before we fixed it.

The fix is **not** to patch `<GIMP install>\lib\gimp\3.0\interpreters\
pygimp.interp` — that's a system file and modifying it breaks the
next GIMP update. The fix is a per-user interpreter alias:

1. The plug-in's first line is `#!lama-gimp-python` (a project-specific
   alias, never `python` or `python3`).
2. The installer writes two `.interp` files into
   `%APPDATA%\GIMP\3.2\interpreters\`:
   - `lama-gimp-python.interp` → maps to `bin\python.exe` (console)
   - `lama-gimp-python_win.interp` → maps to `bin\pythonw.exe` (GUI)
3. GIMP's shebang resolution runs before `.py` extension resolution
   (`gimpinterpreterdb.c:887-899`), so this mapping is scoped to
   this single plug-in. Every other `.py` plug-in still uses the
   system Python.
4. Use **forward slashes** in `.interp` file paths. GIMP's parser
   requires them on Windows.
5. Removal is two file deletions. No GIMP install touched.

**Important:** the alias must be a unique name, not `python` or
`python3`. Using a common name would globally override `.py`
resolution for all plug-ins.

---

## 4. Image-scoped procedure gotcha (don't waste a day on this)

`Gimp.ImageProcedure` with `<Image>/Filters/...` menu path, an
image-type filter like `"RGB*, GRAY*"`, and `DRAWABLE` sensitivity
mask is **image-scoped**. GIMP is allowed — and does, by default —
to **omit the menu entry when no image/drawable is open**.

Symptoms when you forget: the plug-in is registered (visible in
`gimp --verbose` as `Querying plug-in: ...`), the PDB shows the
right `MENU_PATHS` and `SENSITIVITY`, but the filter is missing from
the menu.

Test rule: always test menu presence **with an image loaded**.
Headless `gimp --verbose` with no image will not show the entry —
this is correct GIMP behavior, not a bug.

Other things that hide an image-scoped menu entry:
- No image open
- No drawable/layer active
- The image type is not in `set_image_types(...)` (e.g. indexed color)

---

## 5. `subprocess.Popen` pipe leak locks the worker EXE on Windows

**Symptom:** after running the plug-in once, you cannot replace
`lama_worker_rust.exe` (or `lama_worker.py`) in the plug-in folder
with "the folder or a file in it is open in another program". Task
Manager shows the EXE still running with non-zero CPU even after GIMP
is closed. Stays locked until PC restart.

**Cause:** the inference function used `subprocess.Popen` with
`stdout=PIPE, stderr=PIPE` and read lines in a loop. When the
inference done marker was detected, an early `return` exited the
loop without closing the pipes or waiting for the child. Windows
holds a file lock on the EXE as long as any handle is open, and
those pipe handles are owned by the GIMP process (which may run
for hours).

**Fix:** use `try/finally` to unconditionally close all handles and
`process.wait()` to reap the child:

```python
try:
    for line in iter(process.stdout.readline, ""):
        if line.startswith("RESULT:"):
            result_data = json.loads(line[7:])
            break  # break, not return
finally:
    process.stdout.close()
    process.stderr.close()
    process.wait()  # releases the file lock immediately
```

If you don't need streaming, just use `subprocess.run()` — it
always waits and closes. Only use `Popen` when you actually need
incremental output.

---

## 6. Execution provider cascade (what to try, in order)

For a CPU-only Rust worker:

| Provider | Tries? | Notes |
|---|---|---|
| `CPUExecutionProvider` | Yes, default | Always works. ~1.75 s inference for 512². |
| `CUDAExecutionProvider` | Optional | Needs matching CUDA 12 + cuDNN 9 on PATH. NVIDIA only. |
| `DmlExecutionProvider` | Known issues | ORT prebuilt DirectML binary can crash at session init with `AbiCustomRegistry.cpp(519)`, `E_INVALIDARG` in some environments due to a prebuilt-binary incompatibility with certain D3D12 runtime versions. |
| `WebGPU / Dawn` | Experimental | ORT upstream marks it experimental; can't be combined with other GPU EPs in the prebuilt binary. |

**Three things that hurt ORT CPU inference specifically:**

1. **`with_intra_threads(N)`** — counterintuitively *slows down*
   inference (1.75 s → 4 s). ORT's CPU EP is already internally
   parallelized. Forcing more threads creates pool overhead and
   contention. **Don't override ORT's default thread count unless
   you've measured a specific win.**
2. **`with_memory_pattern(false)`** — required for DirectML,
   harmless for CPU. Keep it.
3. **`with_parallel_execution(false)`** — required for DirectML,
   harmless for CPU. Keep it.

**What actually helps for ORT CPU on this model:** (this model = LaMa FFC)

- `with_dimension_override("batch", 1)` — required because the
  LaMa model has a dynamic batch dimension. The `FFC` block's
  grouped convolutions produce batch-dependent 3D tensors that
  DirectML can't pre-compile kernels for. Fixing batch=1 is always
  correct (one image per call).
- Pre-computed resize weights (`1.0 - wx`, `1.0 - wy` hoisted
  out of the channel loop).
- `rayon` parallelism for per-pixel pre/postprocessing
  (`axis_iter_mut(Axis(0)).par_bridge().for_each(...)`). Marginal
  gain for typical selections (~600² ROI) but a real win on 4K.

---

## 7. Performance — the actual ceiling

The model runs on ORT CPU EP. Both workers (Python and Rust) feed
the same C++ backend, so the inference time is identical
(~1.75 s). The difference is in everything *around* the inference:

| Path | First call | Warm steady-state |
|---|---|---|
| Python worker | ~20 s | ~2 s |
| Rust worker | ~5 s | ~3.4 s |

That's a **~4× speedup on first call** due to no interpreter /
library import overhead. Steady-state is similar.

**Per-call breakdown (Rust worker, typical 1024² / 200² selection):**
- ORT inference: ~1.75 s
- Pre/postprocessing + I/O: ~1.65 s
- Total: ~3.4 s

The ORT inference is the ceiling. The Rust worker wins on startup
overhead, not on inference time. Further speedup requires:

- A smaller / distilled / faster model
- GPU acceleration that works on the target hardware (DirectML has
  known ABI issues with some D3D12 runtimes; CUDA requires NVIDIA;
  WebGPU is experimental)
- Faster silicon

**Crates that look promising but didn't help:**

| Crate | Tried | Verdict |
|---|---|---|
| `fast_image_resize` (SIMD) | Yes | U8/f32 conversion overhead ate the SIMD speedup for our image sizes. Would help on 4K but not on 1024². |
| `memmap2` (model mmap) | Yes | Made things worse — 1.75 s → 3-5 s. ORT's commit_from_memory doesn't guarantee lifetime semantics on Windows. Stick with `commit_from_file`. |
| `with_intra_threads(N)` | Yes | Made things worse (see §6). |

`rayon` is the only parallelization crate that actually helped.

---

## 8. Cross-CRT re-exec — DON'T do this

We tried a stdlib-only re-exec bootstrap that detected when the
plug-in was launched with the wrong Python and relaunched under
GIMP's Python. It "worked" in the sense that `import gi` succeeded
in the child. But:

1. **The re-exec loses GIMP's wire-protocol descriptor table.** On
   Windows, FD numbers are owned by the CRT instance. MSVC ↔ MINGW
   re-exec means the child gets an FD that points to a different
   descriptor table, and the wire protocol immediately returns EOF.
2. **Importing `gi` in the child is not proof the wire protocol
   survived.** The wire protocol is the problem, not the import.
3. **The reliable symptom is `LibGimpBase-WARNING: gimp_wire_read():
   unexpected EOF` with no Python traceback.** End-to-end test
   (real `gimp --verbose`) is the only honest check.

Don't try to re-exec across CRTs. Use the per-user `.interp` alias
instead — it's the supported way.

---

## 9. The seven big mistakes we made (don't repeat)

1. **Patching `pygimp.interp`** to force GIMP to use its bundled
   Python. The user pushed back on this and they were right — there
   is a supported way (`*.interp` files in the user interpreter
   dir) that doesn't touch GIMP's install.
2. **Bundling wheels in `vendor/`** with `sys.path` prepending. The
   C-ABI mismatch is at the binary level; no amount of file
   shuffling makes MSVC `.pyd` files loadable in a MINGW Python.
3. **Trusting `pip install` to fix GIMP-internal dependencies**
   (e.g. PyGObject). GIMP's runtime has no dev headers; the
   supported path is to use GIMP's bundled Python, not to rebuild
   it.
4. **Re-exec'ing across CRTs** to "get into" GIMP's Python. The
   wire-protocol descriptor table does not survive.
5. **Treating "import gi succeeded" as proof the bootstrap
   worked.** It isn't — the wire protocol is the real test, and
   the symptom is `gimp_wire_read(): unexpected EOF` with no
   Python traceback.
6. **Assuming the model is "fast enough on CPU"** without measuring.
   The model is the ceiling. Both workers hit the same ORT
   inference time; the Rust worker only wins on startup.
7. **Trusting `cargo install` to silently work** in install scripts.
   The Rust build can take minutes; install scripts should not
   silently fail and leave the user without a working plug-in.
   The installer should print clear status (built / already
   present / source not found / cargo missing / build failed)
   and never block on a Rust build.

---

## 10. Where to go from here

### Short term (low risk, high value)

- **Distill or quantize the model** — int8 was rejected earlier
  (~12 s/iter) but a properly calibrated static quantization might
  work better. fp16 was rejected (Cast-node mismatches in ORT 1.24.4)
  but a newer ORT might handle it. These are the only paths to
  faster CPU inference without a different model.
- **Fix Linux `install.sh`** — it's still the legacy vendoring
  installer. Migrate to the same `.interp`-free, per-user Python
  approach as `install.bat`.
- **Test the menu visibility properly** — add a CI step that
  launches GIMP headless with a test image and asserts the menu
  entry resolves. The image-scoped procedure gotcha is real and
  easy to miss in manual testing.

### Medium term (moderate risk, big upside)

- **GGUF support** — the original goal. Same sidecar pattern, new
  worker binary, new file-extension registration. The architecture
  doesn't need to change. Just add a new Gimp.Procedure that
  dispatches to a GGUF-capable worker.
- **Persistent worker** — instead of spawning a new worker per
  call (with model reload each time), keep one worker running and
  communicate via stdin/stdout JSON. The model loading (~1-2 s)
  disappears from the per-call budget. Uses the same temp PNG
  handoff or upgrades to length-prefixed binary IPC.
- **Better provider probe** — try the DirectML build with a fresh
  ORT release, in case the `AbiCustomRegistry` crash was a
  prebuilt-binary bug that's since been fixed.

### Long term (architectural changes, only if needed)

- **Direct GIMP filter C ABI** — write a C shared library that
  GIMP loads natively, and call ORT from C. Sidesteps the Python
  ABI wall entirely. Cost: a real C build pipeline, lost
  cross-platform Python tooling, more brittle deploy.
- **GPU plug-in** — if the user can ever get DirectML or CUDA
  working on their actual machine, the same sidecar with
  `--features directml` already exists in the Rust worker and
  just needs to be enabled.
- **Multiple-model plug-in** — register one entry per
  Gimp.Procedure (e.g. "LaMa Standard", "LaMa Fast", "LaMa
  High-Res"), each dispatching to a different ONNX model. Same
  architecture; just more plug-in registrations.

---

## 11. Quick reference — the right shape for a new filter

If you're adding a new ONNX-backed filter to this project (or a
similar one), the shape is:

1. `filter-foo-py/foo-foo.py` — GIMP-side plug-in, shebang on
   first line, only `gi`/`Gimp`/`GimpUi`/`Gegl`/`GLib` imports.
2. `filter-foo-py/foo_worker.py` — system-Python CLI sidecar,
   `numpy` + `onnxruntime` + `PIL`, no GIMP, no `gi`. Takes
   PNGs in, returns PNGs out.
3. `filter-foo-py/foo_inpaint.py` (or equivalent) — the
   GIMP-independent inference core. Testable without GIMP.
4. `filter-foo-py/install.bat` — detects GIMP Python, writes
   `.interp` files into the user interpreter dir, copies plug-in
   files to `%APPDATA%\GIMP\3.2\plug-ins\foo\`. Never touches
   `<GIMP install>`.
5. `tests/test_pipeline.py` — covers the core pipeline plus both
   workers. `max RGB error=0` is the contract; anything else
   means something changed.
6. Optional: `filter-foo-rs/` — Rust sidecar for faster
   startup. CPU-only by default; GPU providers are opt-in
   Cargo features.

Don't add a Rust sidecar unless you need the first-call
speedup. The Python worker is always the default and is fully
functional.

---

## 12. Upscaler pass — removed

The original plug-in included an experimental post-inpaint
detail-enhancement pass using Real-ESRGAN and DAT. Three approaches
were tried (isolated patch + padding, full-canvas downscale→upscale→
crop, and detail-transfer with feather). All three produced a visible
"patch" boundary in real-world images, so the entire feature was
removed from this plug-in.

The relevant learnings and the correct approach (uniform detail
transfer without selection boundary) are documented in:

`C:\Dev\GIMP_Native_Plugin\GOAL-Detail-Enhance-Plugin.md`

That file is the spec for a future standalone "Detail Enhance..."
filter plug-in. The whole `RealEsrganUpscaler` class, the DAT model,
and the `--mode upscale` worker dispatch have been removed from this
project's source tree.

(This section is kept as historical documentation. The Detail Enhance
filter is **not part of this project** — it belongs to a separate
project that was never started. The path above is external to the
repo and may not exist on disk.)

---

## 13. GIMP 3.x Python plug-in registration pitfalls (2026-08)

When adding the manga model support, we hit several GIMP 3.x API
issues that caused the plug-in to silently fail to register. All of
these are now documented in the workspace-wide forensics guide, but
the LaMa-specific lessons are recorded here.

### 13a. Missing `gi.require_version()` calls

**Symptom:** `PyGIWarning` at import time, plug-in not discovered.

**Fix:** Add version requirements *before* importing from
`gi.repository`:

```python
import gi
gi.require_version('Gegl', '0.4')
gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')

from gi.repository import Gegl, Gimp, GimpUi, GLib, GObject
```

The warnings alone don't block discovery, but they indicate the
import order is wrong and can cause version mismatches at runtime.

### 13b. `add_choice_argument` signature

**Symptom:** `Plug-in failed to create procedure` in gimp-console
verbose output. Plug-in appears in pluginrc but menu item does
nothing when clicked.

**Wrong:**
```python
procedure.add_choice_argument(
    "model", "Mo_del",
    choice,           # missing description
    "lama",
    Gimp.PARAM_FLAGEMPLARY,  # wrong flag constant
)
```

**Correct:**
```python
procedure.add_choice_argument(
    "model",                          # name
    "Mo_del",                         # label
    "Inpainting model to use",        # description (REQUIRED)
    choice,                           # Gimp.Choice object
    "lama",                           # default value
    GObject.ParamFlags.READWRITE,     # flags (not Gimp.PARAM_FLAG_*)
)
```

Key differences:
- **Description parameter is required** — omitting it shifts all
  subsequent args, causing a silent type mismatch.
- **Flags use `GObject.ParamFlags.READWRITE`**, not
  `Gimp.PARAM_FLAGEMPLARY`. The `Gimp.PARAM_FLAG_*` constants
  don't exist in GIMP 3.x Python bindings.

### 13c. `config.get_choice()` does not exist

**Symptom:** Runtime error:
`'GimpProcedureConfigRun-plug-in-lama-inpaint' object has no attribute 'get_choice'`

**Wrong:** `config.get_choice("model")`

**Correct:** `config.get_property("model")`

GIMP 3.x config objects use GObject's `get_property()` for all
parameter types, including choices. There is no `get_choice()` method.

### 13d. Interactive mode requires `GimpUi.ProcedureDialog`

**Symptom:** Menu item appears but nothing happens when clicked.

**Cause:** The `run()` method was not showing a dialog for
`Gimp.RunMode.INTERACTIVE`. GIMP 3.x does not auto-generate dialogs
for plug-in parameters.

**Fix:**
```python
def run(self, procedure, run_mode, image, drawables, config, run_data):
    if run_mode == Gimp.RunMode.INTERACTIVE:
        dialog = GimpUi.ProcedureDialog.new(procedure, config)
        dialog.fill(["model"])  # list of parameter names to show
        if not dialog.run():
            dialog.destroy()
            return procedure.new_return_values(
                Gimp.PDBStatusType.CANCEL, GLib.Error())
        dialog.destroy()

    model_choice = config.get_property("model")
    return self._run_lama(procedure, run_mode, image, drawables, model_choice)
```

### 13e. Debugging checklist

When a plug-in doesn't appear or doesn't respond:

1. **Delete `pluginrc`** and restart GIMP to force rescan.
2. **Run `gimp-console --verbose`** and grep for the plug-in name.
   Look for `Querying plug-in` and `failed to create procedure`.
3. **Check `py_compile`** — syntax errors block discovery.
4. **Check imports** — `gi.require_version()` must precede
   `from gi.repository import ...`.
5. **Check `add_*_argument` signatures** — wrong arg count/type
   causes silent failure.
6. **Check `Gimp.main()`** — must use `sys.argv`, not hardcoded args.
7. **Check `run()` signature** — must be
   `(self, procedure, run_mode, image, drawables, config, data)`.

---

## 15. Porting LaMa safetensors to candle (2026-08)

The manga model ships as a PyTorch state dict (`.safetensors`), not
ONNX. Converting to ONNX is impossible with current tooling: PyTorch's
ONNX exporter has no symbolic for ANY fft op (`fft_rfftn`, `fft_fft`,
`complex`) even though ONNX has had DFT since opset 17 — see
pytorch/pytorch#112382. DFT-matrix replacement in the exported graph is
O(N²) and hung tracing. So: native inference via `candle-core/-nn 0.11`
in the Rust worker, dispatched on file extension.

### 15a. Verified layer-index map (big-lama, n_ds=3, n_blocks=18)

Read this off the safetensors header, not from assumptions. The
Sequential indices are:

```
0   ReflectionPad2d(3)          no params
1   FFC_BN_ACT(4→64, k7)        bn_l only          (ratio 0/0)
2-4 FFC_BN_ACT downsample ×3    bn_l; #4 also bn_g (last one gout=.75)
5-22 FFCResnetBlock ×18         52 tensors each    (.75/.75)
23  ConcatTupleLayer            NO PARAMS ← never request weights here
24/25  ConvT(512→256)+BN       ConvT HAS bias     (+26 ReLU)
27/28  ConvT(256→128)+BN                          (+29 ReLU)
30/31  ConvT(128→64)+BN                           (+32 ReLU)
33  ReflectionPad2d(3)          no params
34  Conv2d(64→3, k7)            HAS bias           (FFC convs do NOT)
35  Sigmoid                     no params
```

Off-by-one here cost three debug cycles: upsample base is
`2+n_ds+n_bl+1`, final conv is that plus `n_ds*3+1`.

### 15b. FFC convs are bias-free

Every conv inside FFC/SpectralTransform/FourierUnit is `bias=False`
(`conv2d_no_bias`). Only the two bookend layers (#24-style ConvT and
#34 final conv) carry biases. Requesting a missing bias tensor aborts
the load.

### 15c. Spectral channel packing order

PyTorch packs FFT real/imag as **per-channel interleaved**:
`(re0, im0, re1, im1, …)` — stack(dim=inner)·reshape, NOT
cat([all_re, all_im]). The trained 1×1 `conv_layer` reads that exact
order. Mirror it on unpack (`view(b,c,2,h,m)` then split).

### 15d. irfftn discards Im(DC) and Im(Nyquist)

`torch.fft.irfftn` treats the spectrum as one-sided of a REAL signal:
bins 0 and N/2 contribute their **real part only**. A full-spectrum
IDFT reconstruction must zero those imaginary parts before mirroring,
or errors leak through 36 BN+ReLU stages.

### 15e. candle API traps (each cost a build cycle)

- `matmul` is strict `(M,K)@(K,N)`; no batched-2D broadcasting. Flatten
  leading dims yourself; output width = `rhs.dims()[0]` when rhs kept raw.
- `transpose(a,b)` swaps ONE pair only. Contracting axis 1 of (B,C,H,W)
  needs the double-swap route B,C,H,W → B,W,C,H → flatten → mm → invert.
  A single swap silently contracts the wrong axis whenever H==W.
- `narrow` returns strided views; `flip` (and any index-select) demands
  contiguous input. Insert `.contiguous()` between narrow→flip chains.
- Zero-channel tensors are fragile — thread the global branch as
  `Option<Tensor>` from layer 1 until #4 actually produces global channels.

### 15f. THE input contract: generators eat `img * (1 - mask)`

The single most expensive bug of the port (cost a full "why is it
just pasting white" round-trip):

`saicinpainting/training/trainers/default.py` feeds the generator
`masked_img = img * (1 - mask)` — **hole pixels are zeroed before
inference**. A raw state-dict generator does NOT zero internally.
The shipped `lama_fp32.onnx` works without this step only because
its exporter baked the multiply into the graph. When we fed the raw
image to the candle port, the model received an out-of-distribution
input and produced flat white / saturated garbage — and worse, our
first "ground truth" torch script replicated the same omission, so
the wrongness matched itself and looked like a passing comparison.

Rules:
- Any new backend must apply `img * (1-mask)` before concat with the
  mask channel (`CandleInpainter::inpaint` does this; grep for
  `one_minus`).
- Ground-truth harnesses must replicate the FULL preprocessing
  contract, not just architecture + weights.
- Tell-tale symptom of a missing preprocess step: output looks like
  a constant fill while the reference "agrees" — both paths are
  jointly wrong. Vary the INPUT REGIME (constant vs textured vs
  partial masks) when validating; agreements that survive regime
  changes are real, agreements under one regime may be shared bugs.

### 15g. Verification methodology (do this FIRST next time)

The loop that finally worked — run it before touching GIMP:

1. Dump the safetensors header and classify every top-level index;
   reconcile against the architecture source line-by-line.
2. Standalone smoke test: synthetic PNG pair straight into
   `lama-worker.exe --model <safetensors>`. No GIMP in the loop.
3. Ground truth: run the ORIGINAL PyTorch generator (lama-main modules,
   shim `kornia`/`pytorch_lightning` if imports fail) on IDENTICAL input
   (full-image mask ⇒ identical preprocessing) and diff pixels — with
   the full §15f input contract applied on BOTH sides.
4. Acceptance seen: manga content → mean 0.28–9.9, p99 = 0 in the
   saturated regime; bounded ≤85/255 once inputs are in-distribution
   (float-path divergence through 18 BN blocks — expected across FFT
   backends, no structural artifacts). Residual sub-1% bands where the
   reference itself emits [0,255,0]-style garbage are reference chaos.
5. Behavioral gates that actually matter to users: outside-mask pixels
   byte-identical; fill textured (std >> 0), not constant; line work
   continues through filled rows; output deterministic run-to-run.

### 15g. Reflect padding is part of the contract (real bug found via stage bisect)

FFC builds every k3/k7 conv as
`nn.Conv2d(..., padding=N, padding_mode='reflect')`. candle's Conv2d
only does zero padding. Zero-padded convs matched torch to ~4 decimal
places on means while extremes diverged — and the error propagated
inward through the FFT global path as scattered pixels (the "glitchy"
report). Fix: manual `reflect_pad2d` pre-pad + conv with `padding: 0`
(`RefPadConv`). After the fix, layers 1–4 match torch **exactly**
including per-stage maxima.

Diagnostic that found it: instrument BOTH backends to print per-layer
mean/std/min/max (`LAMA_DEBUG_STAGES=1` env in the worker; forward
hooks in torch) and walk the table until the first divergence.

### 15h. This checkpoint is chaotically unstable — cross-backend pixel equality is impossible

After the reflect fix, residual divergence enters at the first
FourierUnit and grows. Evidence it is NOT a port bug:

- f64 FFT bases produced bit-identical outputs to f32 → not precision.
- **torch-f32 vs torch-f64 on identical input diverge comparably**
  (seq5_g max 23.5 vs 36.2; final range [0.70,0.88] vs [0.17,1.0]).
- Internal activations reach thousands regardless of input regime.
- torch emits `[0,255,0]`-style saturated garbage on some OOD inputs;
  our port stays smoother there.

Conclusion: hot BN channels make the generator a chaotic map; any
second implementation (different BLAS/FFT reduction order) rides a
different trajectory of equal validity. Practical acceptance gates are
behavioral, not bitwise: outside-mask bytes exact, fill textured,
line-work continues, output deterministic, bounded mean|d| vs
reference (≤~10/255 observed). For photographic content expect the
Manga model to look speckly by nature — that is the checkpoint, not
the port; General LaMa remains the right tool for photos.

### 15i. Resolution-preserving inference for the candle path

IOPaint (upstream of the anime-manga checkpoint) pads to modulo-8 at
original resolution; it never squashes to 512². Forcing manga ROI
through a 512² resize round-trip aliases screentone/lines into moiré
that reads as "scattered pixels". The candle branch now:

- keeps ROI resolution, reflect-padding to the next multiple of 8
  (`MAX_SIDE 1024` cap with proportional downscale above that);
- builds DFT bases lazily per runtime size (`FourierUnit::bases_for`),
  since FFC is size-agnostic given /8 dims.

Validation: fill Laplacian hits are 99.6% connected curve structure
(7 isolated dots / 1907) — crisp reconstructed linework, not noise.
The ONNX branch still resizes (fixed-size export).

---

## 17. Native-resolution ONNX + soft-mask compositing (2026-09)

The 512²-squash ONNX path produced visibly blurry, warped inpaints.
Three root causes, all fixed:

1. **Fixed 512×512 export.** The shipped `lama_fp32.onnx` had static
   H/W. Any ROI was force-squashed to a 512² square: blur from the
   downscale→upscale round-trip, warp from anisotropic stretch of
   non-square ROIs. The FFC generator is fully convolutional, so the
   fix was to patch the ONNX input dims to dynamic (`height`/`width`
   dim_params) — no re-export from a checkpoint needed. The dynamic
   file is now the canonical `lama_fp32.onnx` (the fixed-512 export
   is gone). Constraint discovered empirically: **spatial dims must be
   multiples of 16**, not 8 — the /8 bottleneck must stay *even* for
   the onesided spectral inverse (520 fails, 512 works; 264 fails,
   256 works). We pad to mod-16, minimum 32.

2. **Resize alignment bug.** `resize_bilinear_hwc` used
   `src = dst * (src/dst)` (align_corners=True-style) while claiming
   OpenCV parity. OpenCV/PyTorch `align_corners=False` sample at
   `(dst + 0.5) * scale - 0.5`. The half-pixel shift was applied
   twice (down then up) → systematic warp on lines/edges. Both
   bilinear and nearest resize now use half-pixel-center mapping.

3. **Hard binary composite on antialiased masks.** GIMP selections
   are soft (0–255). The Python worker binarized at `>127` *before*
   `inpaint()`, destroying edge softness; both workers then did a
   hard binary paste → visible seam/color cut. Now: the model input
   mask is binarized at `> 0` (reference behavior), but compositing
   blends with the **soft** mask (`a*out + (1-a)*orig`). Pixels at
   a=0 stay bit-exact original; antialiased edges blend gradually.

**Full-image path.** Reference LaMa runs the *whole* frame through
the model (the FFC global branch wants full context for consistent
color/shading). Images ≤ 4 MP (ORT) / ≤ 1 MP (candle, O(N²) DFT
matmul) now skip the ROI crop entirely: pad full frame to mod-16,
one inference at native res, soft composite. A 1131×1600 photo takes
this path with zero resize. Above the budget the ROI path remains,
with the cap raised to 2048 px and edge-replicate crop padding in
both workers (Rust was reflect, Python was edge — now both edge;
mirror padding made the model copy symmetric structures into fills).

Timing note: quality-first means the 2K-image test case went from
~2 s (512² squash) to ~26 s (3 MP native-res ROI). Expected and
accepted — accuracy over speed.

---

## 18. See also

- `AGENTS.md` at the workspace root — project conventions and
  rules.
- `README.md` — user-facing install and usage docs.
- `lama-worker-rs/OPTIMIZATION.md` — Rust worker post-mortem.
- `lama-worker-rs/README.md` — Rust worker build, env vars, EP
  features.
- `C:\Dev\GIMP_Plugin\Documentation\GIMP-plugin-common-pitfalls.md`
  — broader GIMP development traps.
- `C:\Dev\GIMP_Plugin\Documentation\GIMP-Plugin-Connectivity-Forensics-Guide.md`
  — systematic debugging workflow for plug-in discovery issues.
