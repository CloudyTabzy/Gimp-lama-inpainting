# TODO

Pending work tracked across the project. Format: `- [ ] <item>`
plus the file/place where it belongs.

## ZITS-PlusPlus / ZITS inpainting (deferred, not pursuing)

- [x] ZITS++ and ZITS inpainting repos deleted from
      `C:\Dev\GIMP_Native_Plugin\` (2026-07-28). Reasoning recorded in
      `GOAL-Detail-Enhance-Plugin.md` and session log. Do not re-fetch.

## Detail-enhance filter (separate, deferred)

- [ ] Implement `Filters > Enhance > Detail Enhance...` as a GIMP filter
      separate from inpainting. Uses DAT x4 ONNX model (already
      exported at `DAT-main/`). Full spec in
      `C:\Dev\GIMP_Native_Plugin\GOAL-Detail-Enhance-Plugin.md`. Adds
      detail uniformly to the entire image, no selection boundary. ~30
      lines of Python + a small Rust worker.

## ORT dependency pinning

- [x] Pinned to `ort-2.0.0-rc.12` (NOT `ort-main`) — `ort-main` ships
      ORT 1.28 prebuilts that need newer MSVC STL than is available
      (`__std_max_element_8i` etc.). 2.0.0-rc.12 / ORT 1.24.2 builds
      clean. Reason recorded in `lama-worker-rs/Cargo.toml`.

- [ ] When a newer ORT version becomes available, retest. The blocker
      is whether the prebuilt ORT matches our MSVC toolchain. Try
      `ort-2.0.0-rc.13`, `ort-2.0.0-rc.14`, etc. as they're released.

## Test infrastructure

- [ ] **CI / smoke test.** None of the tests run automatically.
      Could add a `scripts/smoke-test.sh` that runs the Rust worker's
      `--help`, plus `python tests/test_pipeline.py` if a model is
      present. Low priority.

## Stretch goals (deferred)

- [ ] **MAT (Mask-Aware Transformer) inpainter** — single-ONNX
      inpainting model with structural reasoning built in. Would be a
      "fast high-quality" option. Worth investigating if LaMa quality
      isn't enough.

- [ ] **Plug-in icon / better menu organization.** Currently
      `Filters > Enhance > LaMa Inpaint...`. Once we add another
      inpainter or the detail-enhance filter, consider a submenu or
      labels.


## Cleanup (2026-08)

- [x] Moebius implementation fully removed from the GIMP plug-in:
      GIMP-side filter code, `moebius-worker-rs/` Rust source, all
      four Moebius ONNX models, the worker binary, the GIMP
      install-dir copies, and the references in `install.bat` and
      `lama-inpaint.py`. Project is now LaMa-only.
- [x] OxiONNX benchmark files cleaned; only `ort_only.rs` kept for LaMa CPU perf experiments
- [x] OxiONNX Pad op bug patched locally in `oxionnx-0.1.4/oxionnx-ops/src/{shape/sequence.rs,registry/conv_ops/pad.rs}`
- [x] GitHub issue drafts ready at `oxionnx-0.1.4/GITHUB_ISSUE_pad_assertion.md` and `oxionnx-0.1.4/PAD_FIX_FINDINGS.md`
- [x] `lama-worker-rs/OPTIMIZATION.md` updated with 2026-08 addendum (Moebius removed, OxiONNX benchmark done, ORT config locked at CPU baseline)
- [x] LaMa worker confirmed on ORT CPU baseline — no `with_intra_threads`, no `with_parallel_execution`, no `OptLevel::All`