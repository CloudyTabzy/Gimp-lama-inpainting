# GIMP LaMa Inpainting

This is a GIMP 3.2+ filter plug-in for LaMa inpainting, plus an optional
Rust sidecar that replaces the Python inference worker for faster
startup.

This file is for AI agents and human contributors who need to get up
to speed on the project quickly. Read this before making any non-trivial
change.

---

## What this project is

A GIMP filter that runs the LaMa ONNX inpainting model on the active
selection. The model is LaMa (Large Mask inpainting), a 198 MB ONNX
file with dynamic height/width — it accepts any mod-16 spatial size
and is run at native resolution (whole image ≤ 4 MP, ROI above that).
The GIMP plug-in is image-scoped: make a selection, run the
filter, get the result inside the selection bounds.

**Quality bar (v1.1+):** The pipeline is tuned for maximum visual
fidelity. Images ≤ 4 MP run at native resolution (no ROI crop, no
resize — exactly like reference LaMa). The model input mask is
binarized at `> 0`, but compositing blends with the soft mask so
antialiased edges transition seamlessly. The tradeoff is speed:
expect ~25–30 s for a 1.8 MP photo on CPU instead of ~2 s. This is
considered the right call for a tool you run once per edit; if you
need speed over quality, reduce the pixel budget or lower the
MAX_SIDE cap in the worker code.

The project is laid out as two sub-projects in one repo:

```
Gimp-lama-inpainting/
├── lama-inpainting-py/      # the GIMP plug-in (Python, GEGL glue)
├── lama-worker-rs/          # optional Rust sidecar (CPU ORT)
└── tests/                   # cross-project Python tests
```

The Rust worker is **opt-in** and **CPU-only by default**; the Python
worker remains the default. The Python side of the plug-in only
imports `gi`, `Gimp`, `GimpUi`, `Gegl`, and `GLib` — no `numpy`, no
`onnxruntime`, no `PIL`. ML inference is always out of process in a
configured worker.

---

## Build, test, install

```bash
# Syntax check
python -m py_compile lama-inpainting-py/lama-inpaint.py lama-inpainting-py/lama_worker.py

# Cross-project tests
python tests/test_pipeline.py

# Rust worker (optional, opt-in)
cd lama-worker-rs
cargo test --release

# Install (Windows)
cd lama-inpainting-py
./install.bat
```

The installer writes per-user `.interp` files into
`%APPDATA%\GIMP\3.2\interpreters\` and copies the plug-in files into
`%APPDATA%\GIMP\3.2\plug-ins\lama-inpaint\`. It also attempts to build
the Rust worker via `cargo install` and copy it next to the Python
worker; if `cargo` isn't available the Python worker is the default.

---

## Architecture

```
GIMP process  (MINGW Python 3.14, gi + GEGL only)
  └─ #!lama-gimp-python  (shebang → user-level .interp mapping)
       └─ lama-inpaint.py
            ├─ exports drawable + selection mask to temp PNGs via GEGL
            ├─ spawns worker subprocess (Python or Rust)
            └─ loads result PNG into drawable shadow buffer, merge_shadow
                    │
                    ▼
           worker process  (Python 3.10+ OR Rust)
               onnxruntime / ort CPU provider
               🌟 model pipeline (dynamic-H/W ONNX):
                  ≤4 MP: pad full frame to mod-16 → single inference
                         at native resolution (reference behavior)
                  >4 MP: bbox → context pad → edge-pad crop →
                         pad to mod-16 (downscale only above 2048 px)
                  → soft-mask composite (a*out + (1-a)*orig) →
                  RGBA output (input alpha bytes preserved)
```

---

## Where to read for context

Before making any meaningful change, read in this order:

1. **This file** (you're here). Project overview, rules, where to look.
2. **`lama-inpainting-py/Docs/NOTES.md`** — consolidated durable
   lessons (MINGW/MSVC ABI wall, sidecar architecture, `.interp`
   shebang handoff, image-scoped procedure gotcha, subprocess pipe
   leak on Windows, ORT config, the seven big mistakes, "shape for
   a new filter" template). Essential reading before touching the
   GIMP integration. The earlier `00_SUMMARY.md` / `04_LESSONS.md`
   / `06_IMPLEMENTATION_PLAN.md` / `03_ATTEMPTS.md` were rolled
   into this single file; if you can't find one of them, it was
   consolidated here, not deleted.
3. **`lama-worker-rs/OPTIMIZATION.md`** — what was tried, what worked,
   what didn't, in the Rust sidecar.
4. **`README.md`** — user-facing install/usage docs.

For the bigger cross-project context (external reference checkouts,
pitfalls docs that don't belong in the repo, session history), look
at the parent workspace directory.

---

## Rules for AI agents and contributors

1. **Never delete content from `Docs/NOTES.md`** — it is a
   chronological log of what was tried (including failed
   approaches) and the seven big mistakes. The value is in not
   repeating them. Adding new sections is fine; rewriting or
   removing existing ones is not.
2. **Never re-exec across CRTs.** GIMP's wire protocol (CRT-level file
   descriptors) does not survive an MSVC↔MINGW handoff. The
   user-level `.interp` alias is the supported way to start a plug-in
   in a different Python.
3. **Never `import numpy`, `onnxruntime`, `PIL`, or `cv2` in the
   GIMP-side plug-in.** That is the entire point of the sidecar
   architecture. The plug-in runs in GIMP's MINGW Clang Python 3.14,
   which cannot load MSVC-built wheels.
4. **Never patch GIMP's installation files** (under `<GIMP install>\lib\
   gimp\3.0\interpreters\`). Per-user `.interp` files in
   `%APPDATA%\GIMP\3.2\interpreters\` are the supported way.
5. **Never commit the model into a sub-path that's wrong** — the ONNX
   file goes in `lama-inpainting-py/lama_fp32.onnx` (next to the
   Python source). The installer copies it to the plug-in directory.
6. **Never commit `__pycache__/` or `target/`.** The `.gitignore`
   covers them. `Cargo.lock` IS committed — this is a binary
   crate, we want reproducible builds.
7. **Run the test suite before committing.** The Python
   `tests/test_pipeline.py` covers the core pipeline plus both
   workers. The Rust `cargo test --release` covers the Rust worker's
   end-to-end behavior. If you change the core pipeline, run both.
8. **If you change the inference algorithm, you change a contract.**
   The Python core, the Rust worker, and the ONNX model all
   implement the same algorithm (≤4 MP: pad full frame to mod-16 →
   single inference at native resolution; >4 MP: bbox → context pad →
   edge-pad crop → pad to mod-16 → inference → soft-mask composite
   `a*out + (1-a)*orig`). The model input mask is binarized (`> 0`);
   the composite uses the soft mask. Any of them changing means the
   test suite's "max RGB error=0" assertion might fail and need
   updating.
9. **Document the output convention of any model.** The shipped LaMa
   model outputs 0–255 (divided by 255 after inference). A model swap
   must preserve or adjust this convention; `lama_inpaint.py` and
   `lama-worker-rs/src/main.rs` both divide by 255 after inference.
10. **At session end, review what stale/abandoned artifacts remain**
    and flag them. The plug-in's `install.sh` is known stale (Linux
    not migrated to the sidecar layout) and `vendor/` is from a failed
    bundling approach — neither is in the repo.
11. **If you don't know whether something is the right approach,**
    check `lama-inpainting-py/Docs/NOTES.md` first. Most mistakes
    we made were already made and documented there.
12. **GIMP 3.x Python plug-in API is strict about signatures.** When
    adding parameters or changing procedure registration, always:
    - Use `gi.require_version()` before `from gi.repository import ...`
    - Include description in `add_*_argument()` calls
    - Use `GObject.ParamFlags.READWRITE`, not `Gimp.PARAM_FLAG_*`
    - Use `config.get_property("name")`, not `config.get_choice("name")`
    - Show `GimpUi.ProcedureDialog` for `INTERACTIVE` run mode
    - Debug with `gimp-console --verbose` and check for
      `failed to create procedure` errors
    See `Docs/NOTES.md` §13 and
    `C:\Dev\GIMP_Plugin\Documentation\GIMP-plugin-common-pitfalls.md`
    for the full list.

---

## Code style and conventions

### Python (lama-inpainting-py/, tests/)

- The plug-in's first line is `#!lama-gimp-python`. The installer
  writes two `.interp` files mapping this alias to GIMP's bundled
  Python 3.14 (`bin\python.exe` for console, `bin\pythonw.exe` for
  GUI).
- `import gi` is the first import. Then `gi.require_version(...)`
  for each GIMP/GimpUi/Gegl version, then `from gi.repository import ...`.
- The plug-in class is `LamaInpaint(Gimp.PlugIn)`. Subclass
  `Gimp.PlugIn`, not `Gimp.ImageProcedure` (the latter is for
  procedures created from existing PDB procedures).
- All progress calls go through a `_safe_progress` wrapper that
  swallows exceptions, because GIMP can be in non-interactive
  contexts where progress callbacks are no-ops.
- All subprocess calls use `subprocess.Popen` (not `run`) with a
  daemon reader thread, so the GIMP UI stays responsive while the
  worker runs. See `_run_worker_with_progress` in
  `lama-inpaint.py`.

### Rust (lama-worker-rs/)

- Edition 2024, MSRV 1.94. Bump the `rust-version` field in
  `Cargo.toml` in lock-step with this line when raising the floor;
  the pinned `ort-2.0.0-rc.12` sets its own MSRV at 1.88, so any
  value ≥ 1.88 is acceptable on our side.
- CLI via `clap` derive; markers via `eprintln!` with explicit
  `flush()`.
- The session is built with `with_dimension_override("batch", 1)`
  because the LaMa model has a dynamic batch dimension. This is
  always correct (one image per call) and required for DirectML.
  Do **not** add `with_intra_threads(N)` — see OPTIMIZATION.md for
  the 1.75 s → 4 s regression. ORT picks its own internal thread
  count and the default is already optimal for the 198 MB LaMa
  graph on CPU.
- **Two separate parallelism layers, kept distinct:**
  - **Rayon** (12 call sites) parallelises the *pre/post-processing*
    outside the ORT inference call: RGBA → f32 conversion,
    reflect-pad, bilinear and nearest-neighbour resize, mask
    composition, output write. Use
    `array.axis_iter_mut(Axis(0)).enumerate().par_bridge().for_each(...)`
    for `ndarray` rows and `image.as_mut().par_chunks_mut(row_stride)`
    for the final RGBA write. `rayon` owns the thread pool for
    this layer.
  - **ORT** parallelises the *inference* (`session.run(...)`).
    This is opaque to us; it is not a Rayon thread pool and
    must not be confused with one.
  The two layers do not overlap. Removing Rayon would not
  hand off to ORT; it would simply leave the pre/post loops
  single-threaded.
- Per-row resize weights (`inv_wx`, `inv_wy`) are precomputed once
  per pixel, not per channel.
- 1.94+ lints worth being aware of (none currently fire in this
  crate, but the new warns may in future): `unused_visibilities`
  on `const _` declarations, and the closure-capture change that
  can promote non-move closures to move-closure of pattern
  bindings. Keep clippy clean.

### Documentation

- `Docs/` is for *human* readers, not for the code to generate. Keep
  the markdown current when you change behavior, but don't add
  generated content or move large code blocks into prose.
- `OPTIMIZATION.md` in the Rust worker is the place to record failed
  optimization attempts, not commit messages. Future optimizers
  read it first.

---

## Layout (what's where)

```
Gimp-lama-inpainting/
├── AGENTS.md                 ← this file
├── README.md                 ← user-facing install/usage
├── .gitignore
│
├── lama-inpainting-py/       ← the GIMP plug-in (Python, GEGL glue)
│   ├── lama-inpaint.py       ← GIMP-side entry, registered as `plug-in-lama-inpaint`
│   ├── lama_worker.py        ← Python CLI sidecar (default worker)
│   ├── lama_inpaint.py       ← GIMP-independent inference core (LamaInpainter class)
│   ├── install.bat           ← Windows per-user installer
│   ├── install.sh            ← [STALE] Linux installer, not migrated
│   ├── build.py              ← [STALE] legacy bundler, not used
│   ├── lama_fp32.onnx        ← the model (198 MB, shipped in repo)
│   ├── lama_config.json      ← worker Python path (written by install.bat)
│   ├── gimp-verbose.bat      ← launches GIMP with console messages visible
│   ├── vendor/               ← [STALE] extracted numpy/onnxruntime wheels, not used
│   └── Docs/                 ← architecture docs, lessons, plan
│
├── lama-worker-rs/           ← optional Rust sidecar (CPU ORT)
│   ├── Cargo.toml            ← CPU-only default; cuda/directml are opt-in features
│   ├── src/main.rs           ← the binary
│   ├── tests/integration.rs  ← spawns the binary, asserts alpha + mask behavior
│   ├── scripts/run-tests.ps1 ← `cargo test --release` wrapper
│   ├── OPTIMIZATION.md       ← what we tried, what worked, what didn't
│   └── README.md             ← user-facing docs for the Rust worker
│
└── tests/                    ← cross-project Python tests
    └── test_pipeline.py       ← core + Python worker + Rust worker end-to-end
```

External checkouts at the workspace root (`ort-main/`, `gimp-GIMP_3_2_4/`,
`lama-main/`, `pygobject-master/`) are **not** in this repo. They are
read-only references for development. `lama-worker-rs/Cargo.toml`
references `../../ort-main` as a path dependency.
