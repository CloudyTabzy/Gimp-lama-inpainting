//! End-to-end integration test for the LaMa worker binary.
//!
//! Spawns the compiled binary with a synthetic 512x512 RGBA image and
//! a rectangular mask, then asserts that the result preserves the
//! alpha bytes byte-for-byte, leaves pixels outside the mask
//! unchanged, and modifies pixels inside the mask.

use std::path::PathBuf;
use std::process::Command;

use image::{ImageBuffer, Rgba, RgbaImage};

fn test_model_path() -> PathBuf {
    if let Ok(p) = std::env::var("LAMA_TEST_MODEL") {
        return PathBuf::from(p);
    }
    // Walk up from CARGO_MANIFEST_DIR to find a model.
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let candidates = [
        // The lama_fp32.onnx may be checked in alongside the Python project.
        format!("{}/../lama-inpainting-py/lama_fp32.onnx", manifest),
        // Or in a top-level models/ directory outside this repo.
        format!("{}/../../models/lama_fp32.onnx", manifest),
        format!("{}/../../../models/lama_fp32.onnx", manifest),
        "C:/Dev/GIMP_Native_Plugin/models/lama_fp32.onnx".to_string(),
        // Also accept the model bundled with the installed GIMP plug-in.
        format!("{}/../../../Gimp-lama-inpainting/lama-inpainting-py/lama_fp32.onnx", manifest),
    ];
    for c in &candidates {
        let p = std::path::Path::new(c);
        if p.exists() {
            return p.to_path_buf();
        }
    }
    panic!(
        "lama_fp32.onnx not found. Set LAMA_TEST_MODEL or place the model at \
         one of: {:?}",
        candidates
    );
}

fn make_test_image(width: u32, height: u32) -> RgbaImage {
    let mut img: RgbaImage = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let p = ((x as f32 / width as f32) * 255.0) as u8;
            let a = if (x + y) % 2 == 0 { 200 } else { 255 };
            img.put_pixel(x, y, Rgba([p, 128, 64, a]));
        }
    }
    img
}

fn make_test_mask(width: u32, height: u32, bbox: (u32, u32, u32, u32)) -> RgbaImage {
    let mut mask: RgbaImage = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let in_bbox =
                x >= bbox.0 && x < bbox.0 + bbox.2 && y >= bbox.1 && y < bbox.1 + bbox.3;
            let v = if in_bbox { 255 } else { 0 };
            mask.put_pixel(x, y, Rgba([v, v, v, 255]));
        }
    }
    mask
}

#[test]
fn worker_preserves_alpha_and_changes_mask_only() {
    let model = test_model_path();
    let tmp = std::env::temp_dir().join("lama-worker-integration-test");
    let _ = std::fs::create_dir_all(&tmp);
    let image_path = tmp.join("image.png");
    let mask_path = tmp.join("mask.png");
    let output_path = tmp.join("output.png");

    let width = 512u32;
    let height = 512u32;
    let img = make_test_image(width, height);
    img.save(&image_path).expect("save image");

    let bbox = (200u32, 200u32, 100u32, 100u32);
    let mask = make_test_mask(width, height, bbox);
    mask.save(&mask_path).expect("save mask");

    let bin = env!("CARGO_BIN_EXE_lama-worker");
    let output = Command::new(bin)
        .arg("--image")
        .arg(&image_path)
        .arg("--mask")
        .arg(&mask_path)
        .arg("--output")
        .arg(&output_path)
        .arg("--model")
        .arg(&model)
        .output()
        .expect("spawn worker");

    assert!(
        output.status.success(),
        "lama-worker exited with non-zero status {:?}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let in_img = image::open(&image_path).unwrap().to_rgba8();
    let out_img = image::open(&output_path)
        .unwrap()
        .to_rgba8();

    // Alpha bytes must match exactly.
    assert_eq!(
        in_img.as_raw().len(),
        out_img.as_raw().len(),
        "input and output have different sizes"
    );
    let in_raw: &[u8] = in_img.as_raw();
    let out_raw: &[u8] = out_img.as_raw();
    let mut mismatches = 0;
    for (i, (a, b)) in in_raw
        .chunks_exact(4)
        .zip(out_raw.chunks_exact(4))
        .enumerate()
    {
        if a[3] != b[3] {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("alpha mismatch at pixel {}: in={} out={}", i, a[3], b[3]);
            }
        }
    }
    assert_eq!(mismatches, 0, "alpha bytes must be preserved exactly");

    // Inside mask: RGB should change.
    let mut inside_changed = 0;
    let mut inside_total = 0;
    for y in 0..height {
        for x in 0..width {
            if x >= bbox.0 && x < bbox.0 + bbox.2 && y >= bbox.1 && y < bbox.1 + bbox.3 {
                let p_in = in_img.get_pixel(x, y);
                let p_out = out_img.get_pixel(x, y);
                if p_in.0[..3] != p_out.0[..3] {
                    inside_changed += 1;
                }
                inside_total += 1;
            }
        }
    }
    assert!(
        inside_changed > inside_total / 4,
        "expected at least 25% of masked pixels to change; got {}/{}",
        inside_changed,
        inside_total
    );
}
