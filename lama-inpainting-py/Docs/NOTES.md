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

---

## 13. See also

- `AGENTS.md` at the workspace root — project conventions and
  rules.
- `README.md` — user-facing install and usage docs.
- `lama-worker-rs/OPTIMIZATION.md` — Rust worker post-mortem.
- `lama-worker-rs/README.md` — Rust worker build, env vars, EP
  features.
- GIMP 3.x plug-in pitfalls (workspace root) — broader GIMP
  development traps; this doc is LaMa-specific.
