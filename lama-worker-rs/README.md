# lama-worker-rs

LaMa inpainting sidecar for the GIMP Native plug-in (Rust CPU).
Faster than the Python worker due to lower startup overhead; no GPU
dependencies. GPU providers (CUDA, DirectML) are opt-in Cargo features.

CLI mirrors the Python `lama_worker.py` one-for-one so the GIMP
plug-in can swap the implementation behind a flag.

## Status

Opt-in Rust worker for the GIMP plug-in at
`gimp-lama-inpainting/`. The Python worker (`lama_worker.py`) remains
the default; the Rust worker activates when `LAMA_USE_RUST_WORKER=1`
or `"worker_kind": "rust"` is set in `lama_config.json`.

## Layout

```text
lama-worker-rs/
|-- Cargo.toml              # path = "../../ort-main" + CPU-only default; cuda/directml opt-in
|-- src/main.rs             # the binary
|-- tests/integration.rs    # spawns the binary, asserts alpha + mask behavior
|-- scripts/run-tests.ps1   # `cargo test --release` wrapper
`-- README.md
```

## CLI

Same arguments as the Python worker:

```text
lama-worker --image INPUT.png --mask MASK.png --output RESULT.png --model MODEL.onnx
```

Emits four boundary markers on stderr with `flush` between them:

```text
[LAMA_MARKER] phase validate
[LAMA_MARKER] phase provider
[LAMA_MARKER] phase inference_start
[LAMA_MARKER] phase inference_done
[LAMA_MARKER] phase result_written
```

Other stderr output is `tracing`-driven (`RUST_LOG` env var, default
`info`). The plug-in does not require these markers and works whether
or not they are present.

### Environment variables

| Variable | Effect |
| --- | --- |
| `LAMA_FORCE_CPU=1` | Skip the DirectML/CUDA probe, stay on CPU. |
| `LAMA_FORCE_PROVIDER=cuda` | Only try `CUDAExecutionProvider` (then CPU). |
| `LAMA_FORCE_PROVIDER=directml` | Only try `DmlExecutionProvider` (then CPU). |
| `LAMA_FORCE_PROVIDER=cpu` | Same as `LAMA_FORCE_CPU=1`. |

Default cascade: **DirectML → CPU** (when DirectML crashes or is
unsupported, falls back silently). CUDA is opt-in (see features below).

## Building

Default CPU-only build:

```powershell
cd C:\Dev\GIMP_Native_Plugin\Gimp-lama-inpainting\lama-worker-rs
cargo build --release
```

The binary lives at `target\release\lama-worker.exe`.

No GPU runtime dependencies. The `onnxruntime` shared library is pulled
automatically by `download-binaries` and copied next to the binary by
`copy-dylibs`.

### GPU features (opt-in)

| Feature | Command | GPU |
|---|---|---|
| `cuda` | `cargo build --release --features cuda` | NVIDIA (needs CUDA 12 + cuDNN 9 on PATH) |
| `directml` | `cargo build --release --features directml` | Any DX12 GPU |

When a GPU feature is enabled, the corresponding execution provider is
registered. If GPU session init fails, it falls back to CPU.

### Session options

The session is built with `with_dimension_override("batch", 1)` to fix
the model's dynamic batch dimension (always correct: one image per
inference call).

## Testing

```powershell
cd C:\Dev\GIMP_Native_Plugin\Gimp-lama-inpainting\lama-worker-rs
cargo test --release
```

Or: `.\scripts\run-tests.ps1`

The integration test:
1. Generates a 512x512 RGBA PNG with a nontrivial alpha pattern.
2. Writes a rectangular mask PNG.
3. Spawns the compiled binary.
4. Asserts byte-for-byte alpha preservation, inside-mask pixel changes,
   and outside-mask byte match.

Model search order: `LAMA_TEST_MODEL` env var → `../../models/lama_fp32.onnx`
→ `../../../models/lama_fp32.onnx` → `C:\Dev\GIMP_Native_Plugin\models\lama_fp32.onnx`.

## Hidden-console support

The worker is a normal Rust binary that does not call `AllocConsole` or
attach to its parent's console. The GIMP side hides the worker console
in its `subprocess.Popen` call (`CREATE_NO_WINDOW` +
`STARTF_USESHOWWINDOW` / `SW_HIDE`). No extra code needed on the Rust
side.

## See also

- `OPTIMIZATION.md` in this directory — what we tried, what worked,
  what didn't, and why.
- `Docs/06_IMPLEMENTATION_PLAN.md` in the parent project — the
  end-to-end design rationale.
