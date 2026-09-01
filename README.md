# GIMP LaMa Inpainting

> Fill any selection in GIMP with content-aware inpainting powered by the
> [LaMa](https://github.com/advimman/lama) model.

A drop-in GIMP 3.2+ filter that runs LaMa (Large Mask inpainting) on
your active selection. Make a selection, run the filter, get a
realistic fill — no manual clone-stamp required.

![Flow: select, run, inpaint](https://img.shields.io/badge/pipeline-select%20%7C%20run%20%7C%20inpaint-blue)

---

## What it does

LaMa is a state-of-the-art inpainting model from Samsung AI Center
(2021). It fills masked regions with realistic content by understanding
the surrounding image context. This plug-in wraps it as a one-click
GIMP filter:

1. You make a selection in GIMP.
2. Run **Filters → Enhance → LaMa Inpaint...**
3. The selected region is filled with realistic content; everything
   outside the selection stays byte-identical to the original.

It works on any layer that has a non-empty selection — photographs,
drawings, textures, anywhere you have something missing and need it
filled.

### Quality bar (v1.1+)

The pipeline is tuned for **maximum visual fidelity**, not speed:

- **Native-resolution inference** — images up to 4 MP are fed to the
  model *whole*, exactly like the reference LaMa pipeline. No resize,
  no squash, no warp. The model's global Fourier branch sees the full
  image, which is what keeps color and shading consistent across the
  selection boundary.
- **Seamless edges** — the model receives a binarized mask (reference
  behavior), but the final blend uses GIMP's *soft* selection values,
  so antialiased edges transition gradually instead of showing a hard
  cut. Outside the selection: bit-exact original pixels.
- **No blur from resize round-trips** — the old fixed-512² export
  forced every region through a downscale→upscale cycle. That's gone.

The tradeoff is speed: quality-first means ~25–30 s for a 1.8 MP photo
on CPU instead of ~2 s. We consider that the right call for a tool you
run once per edit.

---

## Features

| Feature | Status |
|---|---|
| 🖼️  GIMP 3.2 filter via Filters → Enhance menu | ✅ |
| ✨  **Native-resolution inference** — whole image ≤ 4 MP, ROI above | ✅ |
| 🎨  Adaptive ROI around the selection (native resolution for Manga) | ✅ |
| 🌗  **Soft-mask edge blending** — antialiased selections, no visible seam | ✅ |
| 🔁  Edge-replicate crop for edge-touching selections | ✅ |
| 🪄  Preserves pixels outside the selection byte-for-byte | ✅ |
| 🐍  Default Python worker (no build step) | ✅ |
| 🦀  Optional Rust worker — ~2× faster cold-start, also runs Manga model | ✅ |
| 🎯  GPU-ready (DirectML/CUDA behind opt-in Cargo features) | ✅ |
| 🔌  Per-user GIMP interpreter alias — no GIMP install touched | ✅ |
| 📦  Self-contained installer (`.bat`) | ✅ |
| 🌐  LaMa model auto-downloaded from HuggingFace on install | ✅ |
| 📝  Architectural docs and lessons learned included | ✅ |
| 🎨  **Manga LaMa** — line-art finetuned model for anime/comic inpainting | ✅ |

---

## Performance

> **v1.1 quality-first note:** inference now runs at *native resolution*
> (whole image ≤ 4 MP, ROI above), not at a squashed 512². That is
> deliberately slower than v1.0 — expect ~2 s for small regions to
> ~30 s for a 1.8 MP photo on CPU — in exchange for no resize blur,
> no geometric warp, and seamless selection edges.

The model runs on ONNX Runtime with the CPU execution provider.
Both the Python and Rust workers feed the same `ort` C++ backend, so
the inference time is identical between them. The difference is in
everything *around* the inference:

| Path | First call | Warm steady-state | Notes |
|---|---|---|---|
| 🐍  **Python worker** | ~30–45 s | ~25–30 s | Interpreter startup + `numpy` + `onnxruntime` + `PIL` import |
| 🦀  **Rust worker** | ~30 s | ~27 s | Statically linked, no interpreter |

(timings: 1131×1600 photo, medium selection, CPU; smaller images are
proportionally faster — a 512² region is ~2 s)

That's a **~4× speedup on first call** and a similar wall-clock
advantage on every subsequent call, with no code changes to the
GIMP integration — the Rust worker takes the same temp-PNG handoff
as the Python one.

### What does what

| Work | Time | Where |
|---|---|---|
| Interpreter + library startup | 5–15 s | Python only |
| Statically-linked binary startup | ~0.05 s | Rust only |
| ORT inference (same in both) | 2–27 s | ORT CPU EP, scales with native pixel count |
| Pre/postprocessing (parallelized with `rayon`) | ~1.65 s | Both |
| File I/O | ~0.1 s | Both |

---

## Model

The plug-in includes two models:

| Model | File | Size | Domain |
|---|---|---|---|
| **LaMa General** | `lama_fp32.onnx` | ~198 MB | Photos, natural images |
| **LaMa Manga** | `lama-manga.safetensors` | ~195 MB | Anime, manga, comic art |

The **General** model is the standard LaMa from
[Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX), re-exported
with **dynamic height/width** so it runs at native resolution (the
original export is frozen at 512×512 and force-squashes every region
into a square — we don't ship it anymore). Good for photos, soft fills,
background removal.

The **Manga** model is a fine-tuned Big-LaMa from
[Sanster/anime-manga-big-lama](https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt)
— trained on ~300,000 manga/anime images. It reconstructs sharp line
art, screentone, and comic textures. **Not suitable for photographs**
(it will invent manga-style line work).

Both are **not** stored in the repo — the installer downloads the
General model from the [Releases](../../releases) page (it's the
`lama_fp32.onnx` asset). The Manga model is in the release zip.

### Model selection

In GIMP, run **Filters → Enhance → LaMa Inpaint...** and choose your
model from the dropdown. The Manga option only appears when
`lama-manga.safetensors` is present in the plug-in directory.

---

## Quick start (Windows)

### Option A: Release zip (recommended)

1. Download the latest `lama-inpaint-v*.zip` from the
   [Releases](../../releases) page.
2. Extract the zip to `%APPDATA%\GIMP\3.2\plug-ins\` so the
   directory structure is:
   ```
   %APPDATA%\GIMP\3.2\plug-ins\lama-inpaint\
   ├── lama-inpaint.py
   ├── lama_worker.py
   ├── lama_inpaint.py
   ├── lama_worker_rust.exe
   ├── lama_fp32.onnx       ← General LaMa model
   ├── lama-manga.safetensors  ← Manga LaMa model
   └── lama_config.json
   ```
3. Restart GIMP, open an image, make a selection, run
   **Filters → Enhance → LaMa Inpaint...**.

### Option B: Installer script

```cmd
cd lama-inpainting-py
install.bat
```

The installer:
1. Detects GIMP 3's bundled Python and a suitable worker Python on
   your PATH.
2. Installs `pillow`, `numpy`, `onnxruntime` for the worker.
3. Writes two per-user `.interp` files into
   `%APPDATA%\GIMP\3.2\interpreters\`.
4. Copies the plug-in files (Python source + model + config) to
   `%APPDATA%\GIMP\3.2\plug-ins\lama-inpaint\`.
5. If `cargo` is on your PATH, builds the Rust sidecar and sets it as
   the default worker. If not, the Python worker is the default.
6. If the Manga safetensors model is found in `..\models\`, copies it
   alongside the other files.

Restart GIMP, open an image, make a selection, run
**Filters → Enhance → LaMa Inpaint...**.

### Option C: Manual install

Place the plug-in directory under `%APPDATA%\GIMP\3.2\plug-ins\` with
these files:

```
lama-inpaint/
├── lama-inpaint.py         ← GIMP plug-in entry
├── lama_worker.py          ← Python worker
├── lama_inpaint.py         ← Inference core
├── lama_config.json        ← Worker Python path
├── lama_fp32.onnx          ← General model (download from HuggingFace)
├── lama-manga.safetensors  ← Manga model (download from Releases)
└── lama_worker_rust.exe    ← Rust sidecar (optional, from Releases)
```

### Try the Rust worker manually

```powershell
cd lama-worker-rs
cargo build --release
copy target\release\lama-worker.exe %APPDATA%\GIMP\3.2\plug-ins\lama-inpaint\lama_worker_rust.exe

# In the next GIMP session, the Rust worker will be used.
# To force Python instead:
$env:LAMA_USE_RUST_WORKER = "0"
```

---

## What's in the box

```
Gimp-lama-inpainting/
├── lama-inpainting-py/      ← the GIMP plug-in (Python + GEGL glue)
├── lama-worker-rs/          ← optional Rust sidecar (faster startup)
├── tests/                   ← cross-project test suite
├── AGENTS.md                ← project overview for agents/contributors
├── Docs/NOTES.md            ← consolidated lessons (MINGW/MSVC ABI wall,
│                              sidecar pattern, .interp shebang, etc.)
└── lama-inpainting-py/OPTIMIZATION.md
                             ← Rust worker post-mortem
```

The `lama_fp32.onnx` model (~198 MB) is **not** in the repo. See
[Model](#model) above — `install.bat` downloads it from HuggingFace
on first run.

---

## How it works

The plug-in runs entirely in GIMP's MINGW Clang Python 3.14, which
has `gi` and GEGL but **cannot** load MSVC-built Python wheels like
`numpy` and `onnxruntime`. So:

1. The plug-in exports the drawable and the selection mask to two
   temp PNGs using GEGL's `gegl:png-save` and `gegl:buffer-source`
   ops (no `PIL` needed).
2. It spawns a worker subprocess (Python or Rust) and passes the PNG
   paths and model path as CLI args.
3. The worker runs the LaMa model, returns a result PNG.
4. The plug-in reads the result back into the drawable's shadow
   buffer, then `merge_shadow(True)` applies only to the selected
   region.

This sidecar pattern means the GIMP side stays simple (no ML
dependencies) and the heavy lifting happens in a separate process
where we have full control over the runtime.

---

## Development

To run the test suite:

```cmd
python -m py_compile lama-inpainting-py\lama-inpaint.py lama-inpainting-py\lama_worker.py

python tests\test_pipeline.py

cd lama-worker-rs
cargo test --release
```

Read [AGENTS.md](AGENTS.md) for the project conventions, [Docs/](lama-inpainting-py/Docs/)
for the architectural decisions and lessons learned, and
[OPTIMIZATION.md](lama-worker-rs/OPTIMIZATION.md) for the Rust
sidecar's optimization post-mortem.

---

## License

Plug-in code: GPL-3.0-or-later.

LaMa model: see the original [LaMa repository](https://github.com/advimman/lama)
for usage terms.
