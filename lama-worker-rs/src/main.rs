//! LaMa inpainting sidecar worker (Rust CPU).
//!
//! CLI mirrors the Python `lama_worker.py` one-for-one so the GIMP
//! Native plug-in can swap the implementation behind a flag without
//! changing its own invocation. CPU-only build by default; GPU
//! providers (CUDA, DirectML) are opt-in via Cargo features.
//!
//! Boundary markers `[LAMA_MARKER] phase <name>` are emitted on stderr
//! with `flush` between validate / inference_start / inference_done /
//! result_written so the parent GIMP process can parse phase
//! transitions if it wants to. Public output is otherwise minimal and
//! stable.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use ndarray::{s, Array2, Array3, Array4, Axis};
use ort::{
    inputs,
    session::Session,
    value::TensorRef,
};
use rayon::prelude::*;

mod candle_infer;

/// Fixed LaMa ONNX input spatial size. The model was exported at 512x512.
const MODEL_INPUT: usize = 512;
/// Mask threshold in 0..=255. Pixels strictly above become "inpaint".
const MASK_THRESHOLD: u8 = 127;

#[derive(Parser, Debug)]
#[command(
    name = "lama-worker",
    version,
    about = "LaMa inpainting sidecar worker (Rust CPU)"
)]
struct Args {
    /// Path to the input PNG.
    #[arg(long)]
    image: PathBuf,
    /// Grayscale mask PNG path.
    #[arg(long)]
    mask: PathBuf,
    /// Path to the result PNG that the worker writes (RGBA).
    #[arg(long)]
    output: PathBuf,
    /// Path to the ONNX model file.
    #[arg(long)]
    model: PathBuf,
}

fn main() {
    if let Err(err) = run() {
        // Reduce the error chain to a single concise line. The Python
        // worker does the same: no traceback on stderr, just a stable
        // `ERROR: <message>` line and a non-zero exit.
        let mut message = format!("{:#}", err);
        message = message.split_whitespace().collect::<Vec<_>>().join(" ");
        if message.is_empty() {
            message = err.to_string();
        }
        if message.len() > 500 {
            message.truncate(497);
            message.push_str("...");
        }
        eprintln!("ERROR: {}", message);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    // Tracing is for worker startup diagnostics. We route tracing to
    // stderr and use the `RUST_LOG` env var for the filter; default is
    // `info`. The stable public output of the worker is the
    // `[LAMA_MARKER]` lines and the `ERROR:` line on failure.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    let args = Args::parse();

    marker("validate");

    run_inpaint(&args)
}

fn run_inpaint(args: &Args) -> Result<()> {
    let image_path = require_file(&args.image, "image")?;
    let mask_path = require_file(&args.mask, "mask")?;
    let model_path = require_file(&args.model, "model")?;
    let output_path = output_path(&args.output)?;

    // Load the input as RGBA so we can preserve the exact 8-bit alpha
    // bytes. Pillow's `Image.open(...).convert("RGBA")` is the Python
    // equivalent.
    let img_rgba_u8 = image::open(&image_path)
        .with_context(|| format!("failed to open image: {}", image_path.display()))?
        .to_rgba8();
    let (width, height) = (img_rgba_u8.width() as usize, img_rgba_u8.height() as usize);
    if width == 0 || height == 0 {
        bail!("image dimensions must be nonzero");
    }

    // Load the mask as a single-channel u8 grayscale. Anything strictly
    // above 127 becomes the inpaint region (matches the Python worker).
    let mask_gray_u8 = image::open(&mask_path)
        .with_context(|| format!("failed to open mask: {}", mask_path.display()))?
        .to_luma8();
    let (mask_w, mask_h) = (mask_gray_u8.width() as usize, mask_gray_u8.height() as usize);
    if mask_w != width || mask_h != height {
        bail!(
            "image/mask dimensions differ: {}x{} vs {}x{}",
            width,
            height,
            mask_w,
            mask_h
        );
    }

    // Build the f32 HWC image and the bool HW mask. The model itself
    // only consumes RGB; alpha is preserved separately. The conversion
    // is parallelized across rows with rayon since it's purely
    // independent per-pixel work.
    let image_hwc: Array3<f32> = {
        let mut arr = Array3::<f32>::zeros((height, width, 3));
        arr.axis_iter_mut(Axis(0))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row)| {
                let y_u32 = y as u32;
                for x in 0..width {
                    let p = img_rgba_u8[(x as u32, y_u32)];
                    row[[x, 0]] = p[0] as f32 / 255.0;
                    row[[x, 1]] = p[1] as f32 / 255.0;
                    row[[x, 2]] = p[2] as f32 / 255.0;
                }
            });
        arr
    };
    let mask: Array2<bool> = {
        let mut arr = Array2::<bool>::from_elem((height, width), false);
        arr.axis_iter_mut(Axis(0))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row)| {
                let y_u32 = y as u32;
                for x in 0..width {
                    row[[x]] = mask_gray_u8[(x as u32, y_u32)].0[0] > MASK_THRESHOLD;
                }
            });
        arr
    };

    marker("inference_start");
    let inference_start = Instant::now();

    let use_candle = model_path
        .extension()
        .map_or(false, |e| e == "safetensors");

    // ONNX path: fixed 512×512 export → must resize.
    let (img_512, mask_512, roi_info) = preprocess(&image_hwc, &mask)?;

    let result_hwc = if use_candle {
        marker("provider");
        tracing::info!("provider: candle-cpu (safetensors)");
        let device = candle_core::Device::Cpu;
        let model = candle_infer::CandleInpainter::from_safetensors(&model_path, &device)
            .context("failed to load safetensors model")?;
        // Candle path is resolution-preserving: the FFC generator accepts
        // any H,W divisible by 8. Resizing manga screentone/lines to
        // 512² and back produces moiré/scatter, so we pad the ROI to
        // /8 instead (scaling down only when it exceeds a sane cap).
        const MAX_SIDE: usize = 1024;
        let (img_roi, mask_roi) = (roi_info.img_roi.clone(), roi_info.mask_roi.clone());
        let (rh, rw) = (img_roi.dim().0, img_roi.dim().1);
        let scale = if rh.max(rw) > MAX_SIDE {
            MAX_SIDE as f32 / rh.max(rw) as f32
        } else {
            1.0
        };
        let (sh, sw) = (
            ((rh as f32 * scale).round() as usize).max(1),
            ((rw as f32 * scale).round() as usize).max(1),
        );
        let img_s = if scale < 1.0 { resize_bilinear_hwc(&img_roi, sh, sw) } else { img_roi };
        let mask_s = if scale < 1.0 { resize_nearest_hw(&mask_roi, sh, sw) } else { mask_roi };
        // Guard: keep the /8-padded side >= 32 so the /8 bottleneck stays
        // >= 4 px (the spectral inverse needs width > its onesided half).
        let ph8 = (((sh + 7) / 8 * 8).max(32));
        let pw8 = (((sw + 7) / 8 * 8).max(32));
        let pt = (ph8 - sh) / 2;
        let pl = (pw8 - sw) / 2;
        let img_pad = reflect_pad_3d(&img_s, pt, pl, ph8 - sh - pt, pw8 - sw - pl);
        let mask_pad = reflect_pad_2d(&mask_s, pt, pl, ph8 - sh - pt, pw8 - sw - pl);
        let img_chw = hwc_to_chw_tensor(&img_pad, &device)?;
        let mask_chw = mask_to_chw_tensor(&mask_pad, &device)?;
        let out = model
            .inpaint(&img_chw, &mask_chw)
            .context("candle inference failed")?;
        let mut out_hwc = chw_tensor_to_hwc(&out)?;
        // Crop the pad border and undo the optional downscale so the
        // result aligns with postprocess()'s ROI expectations.
        out_hwc = out_hwc
            .slice(ndarray::s![pt..pt + sh, pl..pl + sw, ..])
            .to_owned();
        let out_roi_size = if scale < 1.0 {
            resize_bilinear_hwc(&out_hwc, rh, rw)
        } else {
            out_hwc
        };
        postprocess(&out_roi_size, &image_hwc, &roi_info)?
    } else {
        let (mut session, provider_name) = build_session(&model_path)
            .with_context(|| {
                format!("failed to build ORT session for {}", model_path.display())
            })?;
        marker("provider");
        tracing::info!("provider: {}", provider_name);
        run_ort_inpaint(&mut session, &image_hwc, &mask, &img_512, &mask_512, &roi_info)?
    };
    let inference_secs = inference_start.elapsed().as_secs_f32();
    marker("inference_done");
    tracing::info!("inference took {:.3}s", inference_secs);

    // Compose the result with the unchanged 8-bit alpha and write the
    // RGBA PNG. We round to the nearest 8-bit value (matches the
    // Python `np.rint` path). Parallelized across rows via direct
    // buffer indexing (image::RgbaImage stores RGBA as a flat u8
    // array, 4 bytes per pixel).
    let mut out_rgba = image::RgbaImage::new(width as u32, height as u32);
    {
        let img_raw = img_rgba_u8.as_raw();
        let row_stride = width * 4;
        out_rgba
            .as_mut()
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_buf)| {
                let in_row = y * row_stride;
                for x in 0..width {
                    let r = quantize(result_hwc[[y, x, 0]]);
                    let g = quantize(result_hwc[[y, x, 1]]);
                    let b = quantize(result_hwc[[y, x, 2]]);
                    let a = img_raw[in_row + x * 4 + 3];
                    let p = x * 4;
                    row_buf[p] = r;
                    row_buf[p + 1] = g;
                    row_buf[p + 2] = b;
                    row_buf[p + 3] = a;
                }
            });
    }
    out_rgba
        .save(&output_path)
        .with_context(|| format!("failed to write result: {}", output_path.display()))?;

    marker("result_written");
    Ok(())
}

fn quantize(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Convert HWC f32 array to CHW candle Tensor (1,C,H,W).
fn hwc_to_chw_tensor(hwc: &Array3<f32>, dev: &candle_core::Device) -> Result<candle_core::Tensor> {
    let (h, w, c) = hwc.dim();
    let mut data = Vec::with_capacity(c * h * w);
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                data.push(hwc[[y, x, ch]]);
            }
        }
    }
    candle_core::Tensor::from_vec(data, (1, c, h, w), dev).map_err(|e| anyhow!("{}", e))
}

/// Convert HW bool mask to (1,1,H,W) f32 candle Tensor.
fn mask_to_chw_tensor(mask: &Array2<bool>, dev: &candle_core::Device) -> Result<candle_core::Tensor> {
    let (h, w) = mask.dim();
    let data: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| if mask[[y, x]] { 1.0f32 } else { 0.0 }))
        .collect();
    candle_core::Tensor::from_vec(data, (1, 1, h, w), dev).map_err(|e| anyhow!("{}", e))
}

/// Convert candle (1,3,H,W) Tensor in [0,255] to HWC f32 array in [0,1].
fn chw_tensor_to_hwc(t: &candle_core::Tensor) -> Result<Array3<f32>> {
    let (b, c, h, w) = t.dims4()?;
    if b != 1 || c != 3 {
        bail!("unexpected tensor shape {:?}", t.shape());
    }
    let data = t
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| anyhow!("{}", e))?
        .flatten_all()
        .map_err(|e| anyhow!("{}", e))?
        .to_vec1::<f32>()
        .map_err(|e| anyhow!("{}", e))?;
    let inv = 1.0f32 / 255.0;
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let v = data[ch * h * w + y * w + x] * inv;
                out[[y, x, ch]] = v.clamp(0.0, 1.0);
            }
        }
    }
    Ok(out)
}

fn marker(stage: &str) {
    eprintln!("[LAMA_MARKER] phase {}", stage);
    let _ = std::io::stderr().flush();
}

fn require_file(path: &Path, label: &str) -> Result<PathBuf> {
    let resolved = std::fs::canonicalize(path)
        .with_context(|| format!("{} path not found: {}", label, path.display()))?;
    if !resolved.is_file() {
        bail!("{} not a file: {}", label, resolved.display());
    }
    Ok(resolved)
}

fn output_path(path: &Path) -> Result<PathBuf> {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to read current directory")?
            .join(path)
    };
    if let Some(parent) = resolved.parent() {
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            bail!("output directory not found: {}", parent.display());
        }
    }
    if resolved.is_dir() {
        bail!("output path is a directory: {}", resolved.display());
    }
    Ok(resolved)
}

fn build_session(model_path: &Path) -> Result<(Session, &'static str)> {
    let session = Session::builder()
        .map_err(|e| anyhow!("failed to create ORT session builder: {}", e))?
        .with_dimension_override("batch", 1)
        .map_err(|e| anyhow!("failed to override batch dimension: {}", e))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("ORT session build failed: {}", e))?;
    // ort 2.0.0-rc.12 has no Session::providers() method; the build-time
    // cargo feature is the only honest signal we have about which EPs
    // are statically linked into the binary. Active providers are
    // determined by ORT at runtime and printed as warnings during init.
    #[cfg(feature = "webgpu")]
    let provider_name: &'static str = "WebGPU";
    #[cfg(not(feature = "webgpu"))]
    let provider_name: &'static str = "CPU";
    Ok((session, provider_name))
}

/// ROI context shared between preprocessing and postprocessing.
struct RoiInfo {
    roi_x1: i64,
    roi_y1: i64,
    roi_w: usize,
    roi_h: usize,
    paste_box: (usize, usize, usize, usize),
    img_roi: Array3<f32>,
    mask_roi: Array2<bool>,
}

/// Preprocess: bbox → context pad → reflect-pad crop → resize to 512×512.
fn preprocess(
    image: &Array3<f32>,
    mask: &Array2<bool>,
) -> Result<(Array3<f32>, Array2<bool>, RoiInfo)> {
    let (height, width) = (image.dim().0, image.dim().1);
    if mask.iter().all(|v| !*v) {
        bail!("empty mask");
    }
    let (mut sel_x1, mut sel_y1) = (width, height);
    let (mut sel_x2, mut sel_y2): (usize, usize) = (0, 0);
    for y in 0..height {
        for x in 0..width {
            if mask[[y, x]] {
                sel_x1 = sel_x1.min(x);
                sel_y1 = sel_y1.min(y);
                sel_x2 = sel_x2.max(x);
                sel_y2 = sel_y2.max(y);
            }
        }
    }
    let sel_w = sel_x2 - sel_x1 + 1;
    let sel_h = sel_y2 - sel_y1 + 1;
    let ctx = std::cmp::max(1, (std::cmp::max(sel_w, sel_h) as f32).round() as usize);
    let roi_x1 = sel_x1 as i64 - ctx as i64;
    let roi_y1 = sel_y1 as i64 - ctx as i64;
    let roi_w = sel_w + 2 * ctx;
    let roi_h = sel_h + 2 * ctx;
    let (img_roi, mask_roi, paste_box) =
        crop_with_reflect_pad(image, mask, roi_x1, roi_y1, roi_w, roi_h);
    let img_512 = resize_bilinear_hwc(&img_roi, MODEL_INPUT, MODEL_INPUT);
    let mask_512 = resize_nearest_hw(&mask_roi, MODEL_INPUT, MODEL_INPUT);
    Ok((
        img_512,
        mask_512,
        RoiInfo {
            roi_x1,
            roi_y1,
            roi_w,
            roi_h,
            paste_box,
            img_roi,
            mask_roi,
        },
    ))
}

/// Postprocess: resize 512×512 output back → mask-compose → paste into original.
fn postprocess(
    out_hwc_512: &Array3<f32>,
    image: &Array3<f32>,
    roi: &RoiInfo,
) -> Result<Array3<f32>> {
    let out_roi = resize_bilinear_hwc(out_hwc_512, roi.roi_h, roi.roi_w);
    let mut composed_roi = out_roi.clone();
    composed_roi
        .axis_iter_mut(Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(y, mut row)| {
            for x in 0..roi.roi_w {
                if !roi.mask_roi[[y, x]] {
                    for ch in 0..3 {
                        row[[x, ch]] = roi.img_roi[[y, x, ch]];
                    }
                }
            }
        });
    let (y1, y2, x1, x2) = roi.paste_box;
    let real_y1 = std::cmp::max(0, -roi.roi_y1) as usize;
    let real_x1 = std::cmp::max(0, -roi.roi_x1) as usize;
    let real_y2 = real_y1 + (y2 - y1);
    let real_x2 = real_x1 + (x2 - x1);
    let mut result = image.clone();
    let mut sub = result.slice_mut(s![y1..y2, x1..x2, ..]).to_owned();
    sub.assign(&composed_roi.slice(s![real_y1..real_y2, real_x1..real_x2, ..]));
    result
        .slice_mut(s![y1..y2, x1..x2, ..])
        .assign(&sub);
    Ok(result)
}

/// ORT inference path: convert to CHW, run session, convert back.
fn run_ort_inpaint(
    session: &mut Session,
    image: &Array3<f32>,
    mask: &Array2<bool>,
    img_512: &Array3<f32>,
    mask_512: &Array2<bool>,
    roi: &RoiInfo,
) -> Result<Array3<f32>> {
    let input_names: Vec<String> = session
        .inputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    if input_names.len() != 2 {
        bail!("expected exactly 2 ORT inputs, got {}", input_names.len());
    }
    let output_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .ok_or_else(|| anyhow!("ORT session has no outputs"))?;

    let img_chw: Array4<f32> = {
        let mut arr = Array4::<f32>::zeros((1, 3, MODEL_INPUT, MODEL_INPUT));
        arr.axis_iter_mut(Axis(2))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row)| {
                for x in 0..MODEL_INPUT {
                    for ch in 0..3 {
                        row[[0, ch, x]] = img_512[[y, x, ch]];
                    }
                }
            });
        arr
    };
    let mask_chw: Array4<f32> = {
        let mut arr = Array4::<f32>::zeros((1, 1, MODEL_INPUT, MODEL_INPUT));
        arr.axis_iter_mut(Axis(2))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row)| {
                for x in 0..MODEL_INPUT {
                    row[[0, 0, x]] = if mask_512[[y, x]] { 1.0_f32 } else { 0.0_f32 };
                }
            });
        arr
    };

    let outputs = session
        .run(inputs![
            input_names[0].as_str() => TensorRef::from_array_view(img_chw.view())?,
            input_names[1].as_str() => TensorRef::from_array_view(mask_chw.view())?,
        ])
        .map_err(|e| anyhow!("ORT run failed: {}", e))?;

    let out_view = outputs
        .get(output_name.as_str())
        .ok_or_else(|| anyhow!("ORT output `{}` missing", output_name))?
        .try_extract_array::<f32>()
        .map_err(|e| anyhow!("ORT output extraction failed: {}", e))?;
    let out_4d = out_view.to_owned();
    drop(outputs);

    let out_hwc_512: Array3<f32> = {
        let shape = out_4d.shape().to_vec();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
            bail!("unexpected ORT output shape {:?}", shape);
        }
        let (h, w) = (shape[2], shape[3]);
        let inv_255 = 1.0_f32 / 255.0;
        let mut hwc = Array3::<f32>::zeros((h, w, 3));
        hwc.axis_iter_mut(Axis(0))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    for ch in 0..3 {
                        let v = out_4d[[0, ch, y, x]] * inv_255;
                        row[[x, ch]] = v.clamp(0.0, 1.0);
                    }
                }
            });
        hwc
    };

    postprocess(&out_hwc_512, image, roi)
}

/// Core inpainting routine. Mirrors the Python `LamaInpainter.inpaint`
/// step-for-step: bbox -> context pad -> reflect-pad crop -> 512x512
/// resize -> single inference -> resize back -> mask composition -> paste.
fn inpaint(
    session: &mut Session,
    image: &Array3<f32>,
    mask: &Array2<bool>,
) -> Result<Array3<f32>> {
    let (height, width) = (image.dim().0, image.dim().1);

    // 1. Selection bbox.
    if mask.iter().all(|v| !*v) {
        return Ok(image.clone());
    }
    let mut sel_x1 = width;
    let mut sel_y1 = height;
    let mut sel_x2: usize = 0;
    let mut sel_y2: usize = 0;
    for y in 0..height {
        for x in 0..width {
            if mask[[y, x]] {
                if x < sel_x1 {
                    sel_x1 = x;
                }
                if y < sel_y1 {
                    sel_y1 = y;
                }
                if x >= sel_x2 {
                    sel_x2 = x;
                }
                if y >= sel_y2 {
                    sel_y2 = y;
                }
            }
        }
    }
    let sel_w = sel_x2 - sel_x1 + 1;
    let sel_h = sel_y2 - sel_y1 + 1;

    // 2. Padded ROI bbox (one context-size on each side).
    let ctx = std::cmp::max(1, (std::cmp::max(sel_w, sel_h) as f32).round() as usize);
    let roi_x1 = sel_x1 as i64 - ctx as i64;
    let roi_y1 = sel_y1 as i64 - ctx as i64;
    let roi_w = sel_w + 2 * ctx;
    let roi_h = sel_h + 2 * ctx;

    // 3. Crop with reflect-pad if ROI extends past image bounds.
    let (img_roi, mask_roi, paste_box) =
        crop_with_reflect_pad(image, mask, roi_x1, roi_y1, roi_w, roi_h);

    // 4. Resize to 512x512. The source mask is already a 0/1 bool;
    //    nearest-neighbor resize of a bool keeps it binary, so we do
    //    not re-threshold here (the Python re-threshold is a no-op for
    //    binary masks and the input is already binary).
    let img_512 = resize_bilinear_hwc(&img_roi, MODEL_INPUT, MODEL_INPUT);
    let mask_512 = resize_nearest_hw(&mask_roi, MODEL_INPUT, MODEL_INPUT);

    // 5. Inference. The model has two inputs in the order declared by
    // `session.inputs()`: image then mask (1x3x512x512 and 1x1x512x512).
    let input_names: Vec<String> = session
        .inputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    if input_names.len() != 2 {
        bail!(
            "expected exactly 2 ORT inputs, got {}",
            input_names.len()
        );
    }
    let output_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .ok_or_else(|| anyhow!("ORT session has no outputs"))?;

    let img_chw: Array4<f32> = {
        let mut arr = Array4::<f32>::zeros((1, 3, MODEL_INPUT, MODEL_INPUT));
        arr.axis_iter_mut(Axis(2))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row)| {
                for x in 0..MODEL_INPUT {
                    for ch in 0..3 {
                        row[[0, ch, x]] = img_512[[y, x, ch]];
                    }
                }
            });
        arr
    };
    let mask_chw: Array4<f32> = {
        let mut arr = Array4::<f32>::zeros((1, 1, MODEL_INPUT, MODEL_INPUT));
        arr.axis_iter_mut(Axis(2))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row)| {
                for x in 0..MODEL_INPUT {
                    row[[0, 0, x]] = if mask_512[[y, x]] { 1.0_f32 } else { 0.0_f32 };
                }
            });
        arr
    };

    let outputs = session
        .run(inputs![
            input_names[0].as_str() => TensorRef::from_array_view(img_chw.view())?,
            input_names[1].as_str() => TensorRef::from_array_view(mask_chw.view())?,
        ])
        .map_err(|e| anyhow!("ORT run failed: {}", e))?;

    let out_view = outputs
        .get(output_name.as_str())
        .ok_or_else(|| anyhow!("ORT output `{}` missing", output_name))?
        .try_extract_array::<f32>()
        .map_err(|e| anyhow!("ORT output extraction failed: {}", e))?;
    // out_view shape: (1, 3, 512, 512)
    let out_4d = out_view.to_owned();
    drop(outputs); // release borrow before allocating

    // 6. Convert CHW -> HWC, divide by 255 (model outputs in 0..=255),
    //    clip to [0, 1]. Per-row parallel.
    let out_hwc_512: Array3<f32> = {
        let shape = out_4d.shape().to_vec();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
            bail!(
                "unexpected ORT output shape {:?}; expected (1, 3, H, W)",
                shape
            );
        }
        let (h, w) = (shape[2], shape[3]);
        let inv_255 = 1.0_f32 / 255.0;
        let mut hwc = Array3::<f32>::zeros((h, w, 3));
        hwc.axis_iter_mut(Axis(0))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    for ch in 0..3 {
                        let v = out_4d[[0, ch, y, x]] * inv_255;
                        row[[x, ch]] = if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v };
                    }
                }
            });
        hwc
    };

    // 7. Resize output back to ROI size, then mask-compose: keep the
    //    original image outside the mask, the model output inside.
    let out_roi = resize_bilinear_hwc(&out_hwc_512, roi_h, roi_w);
    let mut composed_roi = out_roi.clone();
    composed_roi
        .axis_iter_mut(Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(y, mut row)| {
            for x in 0..roi_w {
                if !mask_roi[[y, x]] {
                    for ch in 0..3 {
                        row[[x, ch]] = img_roi[[y, x, ch]];
                    }
                }
            }
        });

    // 8. Paste back into the original image, skipping the reflect-pad
    //    region. `paste_box` is the (y1, y2, x1, x2) of the valid
    //    (non-pad) region in the original image coordinates.
    let (y1, y2, x1, x2) = paste_box;
    let real_y1 = (std::cmp::max(0, -roi_y1)) as usize;
    let real_x1 = (std::cmp::max(0, -roi_x1)) as usize;
    let real_y2 = real_y1 + (y2 - y1);
    let real_x2 = real_x1 + (x2 - x1);
    let mut result = image.clone();
    let mut sub = result
        .slice_mut(s![y1..y2, x1..x2, ..])
        .to_owned();
    sub.assign(&composed_roi.slice(s![real_y1..real_y2, real_x1..real_x2, ..]));
    result
        .slice_mut(s![y1..y2, x1..x2, ..])
        .assign(&sub);
    Ok(result)
}

fn crop_with_reflect_pad(
    image: &Array3<f32>,
    mask: &Array2<bool>,
    roi_x1: i64,
    roi_y1: i64,
    roi_w: usize,
    roi_h: usize,
) -> (Array3<f32>, Array2<bool>, (usize, usize, usize, usize)) {
    let (h, w) = (image.dim().0 as i64, image.dim().1 as i64);
    let pad_left = std::cmp::max(0, -roi_x1) as usize;
    let pad_top = std::cmp::max(0, -roi_y1) as usize;
    let pad_right = std::cmp::max(0, (roi_x1 + roi_w as i64) - w) as usize;
    let pad_bottom = std::cmp::max(0, (roi_y1 + roi_h as i64) - h) as usize;

    let img_padded = reflect_pad_3d(image, pad_top, pad_left, pad_bottom, pad_right);
    let mask_padded = reflect_pad_2d(mask, pad_top, pad_left, pad_bottom, pad_right);

    let new_roi_x1 = (roi_x1 + pad_left as i64) as usize;
    let new_roi_y1 = (roi_y1 + pad_top as i64) as usize;

    let img_roi = img_padded
        .slice(s![
            new_roi_y1..new_roi_y1 + roi_h,
            new_roi_x1..new_roi_x1 + roi_w,
            ..
        ])
        .to_owned();
    let mask_roi = mask_padded
        .slice(s![
            new_roi_y1..new_roi_y1 + roi_h,
            new_roi_x1..new_roi_x1 + roi_w
        ])
        .to_owned();

    let y1 = std::cmp::max(0, roi_y1) as usize;
    let y2 = std::cmp::min(h, roi_y1 + roi_h as i64) as usize;
    let x1 = std::cmp::max(0, roi_x1) as usize;
    let x2 = std::cmp::min(w, roi_x1 + roi_w as i64) as usize;

    (img_roi, mask_roi, (y1, y2, x1, x2))
}

fn reflect_index(i: i64, size: i64) -> i64 {
    if size <= 1 {
        return 0;
    }
    let period = 2 * (size - 1);
    let m = ((i % period) + period) % period;
    if m < size {
        m
    } else {
        period - m
    }
}

fn reflect_pad_2d(
    arr: &Array2<bool>,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> Array2<bool> {
    let (h, w) = (arr.dim().0 as i64, arr.dim().1 as i64);
    let new_h = (h as usize) + pad_top + pad_bottom;
    let new_w = (w as usize) + pad_left + pad_right;
    let mut out = Array2::<bool>::from_elem((new_h, new_w), false);
    out.axis_iter_mut(Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(ny, mut out_row)| {
            let oy = reflect_index(ny as i64 - pad_top as i64, h) as usize;
            for nx in 0..new_w {
                let ox = reflect_index(nx as i64 - pad_left as i64, w) as usize;
                out_row[[nx]] = arr[[oy, ox]];
            }
        });
    out
}

fn reflect_pad_3d(
    arr: &Array3<f32>,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> Array3<f32> {
    let (h, w, c) = (arr.dim().0 as i64, arr.dim().1 as i64, arr.dim().2);
    let new_h = (h as usize) + pad_top + pad_bottom;
    let new_w = (w as usize) + pad_left + pad_right;
    let mut out = Array3::<f32>::zeros((new_h, new_w, c));
    out.axis_iter_mut(Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(ny, mut out_row)| {
            let oy = reflect_index(ny as i64 - pad_top as i64, h) as usize;
            for nx in 0..new_w {
                let ox = reflect_index(nx as i64 - pad_left as i64, w) as usize;
                for ch in 0..c {
                    out_row[[nx, ch]] = arr[[oy, ox, ch]];
                }
            }
        });
    out
}

/// Bilinear resize of an HWC f32 array to (new_h, new_w). The
/// algorithm is the same as OpenCV's `INTER_LINEAR` (no half-pixel
/// adjustment): for each output pixel (j, i), the source coordinates
/// are `fx = j * (src_w / dst_w)` and `fy = i * (src_h / dst_h)`, with
/// the standard four-tap blend. Weights are precomputed once per row
/// and column to keep the inner loop branch-free, and the per-row
/// loop body is parallelized across cores via rayon.
fn resize_bilinear_hwc(src: &Array3<f32>, new_h: usize, new_w: usize) -> Array3<f32> {
    let (h, w, c) = (src.dim().0, src.dim().1, src.dim().2);
    if h == new_h && w == new_w {
        return src.to_owned();
    }
    let mut out = Array3::<f32>::zeros((new_h, new_w, c));
    let sy = h as f32 / new_h as f32;
    let sx = w as f32 / new_w as f32;
    out.axis_iter_mut(Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(ny, mut row)| {
            let fy = ny as f32 * sy;
            let y0 = (fy as i64).clamp(0, (h as i64) - 1) as usize;
            let y1 = (y0 + 1).min(h - 1);
            let wy = (fy - y0 as f32).clamp(0.0, 1.0);
            let inv_wy = 1.0 - wy;
            for nx in 0..new_w {
                let fx = nx as f32 * sx;
                let x0 = (fx as i64).clamp(0, (w as i64) - 1) as usize;
                let x1 = (x0 + 1).min(w - 1);
                let wx = (fx - x0 as f32).clamp(0.0, 1.0);
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
            }
        });
    out
}

/// Nearest-neighbor resize of an HW array of bools. Parallelized
/// across rows with rayon.
fn resize_nearest_hw(src: &Array2<bool>, new_h: usize, new_w: usize) -> Array2<bool> {
    let (h, w) = (src.dim().0, src.dim().1);
    if h == new_h && w == new_w {
        return src.to_owned();
    }
    let mut out = Array2::<bool>::from_elem((new_h, new_w), false);
    let sy = h as f32 / new_h as f32;
    let sx = w as f32 / new_w as f32;
    out.axis_iter_mut(Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(ny, mut row)| {
            let oy = ((ny as f32 * sy) as usize).min(h - 1);
            for nx in 0..new_w {
                let ox = ((nx as f32 * sx) as usize).min(w - 1);
                row[[nx]] = src[[oy, ox]];
            }
        });
    out
}
