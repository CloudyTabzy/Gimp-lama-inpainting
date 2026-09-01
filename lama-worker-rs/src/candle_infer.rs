//! Candle-based LaMa FFC ResNet generator for `.safetensors` checkpoints.
//!
//! Faithful port of `saicinpainting.training.modules.ffc.FFCResNetGenerator`
//! (big-lama config). The Fourier units use real-valued DFT matrix
//! multiplication because candle has no FFT kernels.
//!
//! Layer-index map for n_downsampling=3, n_blocks=18 (verified against the
//! shipped safetensors header — do not "clean these up" without re-checking):
//!
//! ```text
//!  0      ReflectionPad2d(3)            no params
//!  1      FFC_BN_ACT(4->64,   k7, 0/0)     bn_l only
//!  2..3   FFC_BN_ACT downsample x2       bn_l only
//!  4      FFC_BN_ACT downsample (gout=.75) bn_l + bn_g
//!  5..22  FFCResnetBlock x18 (.75/.75)   52 tensors each
//!  23     ConcatTupleLayer               NO PARAMS
//!  24/25  ConvT(512->256) + BN           (+26 ReLU, no params)
//!  27/28  ConvT(256->128) + BN           (+29 ReLU)
//!  30/31  ConvT(128->64)  + BN           (+32 ReLU)
//!  33     ReflectionPad2d(3)             no params
//!  34     Conv(64->3, k7)  WITH bias
//!  35     Sigmoid                        no params
//! ```

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{BatchNorm, Conv2d, ConvTranspose2d, Module, ModuleT, VarBuilder};

use crate::MODEL_INPUT;

// ── DFT bases ──────────────────────────────────────────────────────────
//
// One {c, s} pair per transform length N, ortho-normalized:
//   c[k,n] = cos(2πkn/N)/√N      (even → identical for ±exponent)
//   s[k,n] = sin(+2πkn/N)/√N     (stored with POSITIVE exponent)
//
// Formulas (derived once, used everywhere):
//   fwd real x        : P = x@cᵀ, Q = x@sᵀ          spectrum Re=P, Im=−Q
//   fwd complex (R,I) : Re'=R@cᵀ+I@sᵀ   Im'=I@cᵀ−R@sᵀ
//   inv complex (R,I) : Re'=R@cᵀ−I@sᵀ   Im'=I@cᵀ+R@sᵀ
struct Basis {
    c: Tensor, // (N, N) or (m, N) onesided
    s: Tensor,
}

impl Basis {
    fn full(n: usize, dev: &Device) -> Result<Self> {
        Self::build(n, n, dev)
    }
    /// One-sided: m = N/2+1 output bins.
    fn onesided(n: usize, dev: &Device) -> Result<Self> {
        Self::build(n / 2 + 1, n, dev)
    }
    fn build(rows: usize, n: usize, dev: &Device) -> Result<Self> {
        // Bases in f32. An f64 variant was tested and produced identical
        // outputs — cross-backend divergence vs torch is chaotic
        // amplification inside the checkpoint (see NOTES §15h), not
        // transform precision.
        let scale = 1.0 / (n as f64).sqrt();
        let mut c = Vec::with_capacity(rows * n);
        let mut s = Vec::with_capacity(rows * n);
        for k in 0..rows {
            for j in 0..n {
                let a = 2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
                c.push((a.cos() * scale) as f32);
                s.push((a.sin() * scale) as f32);
            }
        }
        Ok(Self {
            c: Tensor::from_vec(c, (rows, n), dev)?,
            s: Tensor::from_vec(s, (rows, n), dev)?,
        })
    }
}

/// y[..., j] = Σ_k x[..., k] · m[j, k]  (contract LAST axis; m is the RAW (J,K) basis).
fn mm_last(x: &Tensor, m: &Tensor) -> Result<Tensor> {
    let d = x.dims();
    let k = d[d.len() - 1];
    debug_assert_eq!(k, m.dims()[1], "mm_last contraction dim");
    let rows: usize = d[..d.len() - 1].iter().product();
    let mut shape: Vec<usize> = d.to_vec();
    shape[d.len() - 1] = m.dims()[0];
    let mt = m.t().map_err(|e| anyhow::anyhow!("{}", e))?; // (K,J)
    Ok(x.reshape((rows, k))?.matmul(&mt)?.reshape(shape)?)
}

/// Transform along the H axis of (B,C,H,W): contract axis 1 against an
/// (H,N) pre-transposed matrix. candle's transpose only swaps pairs, so we
/// route B,C,H,W → B,W,C,H → flatten(B·W·C, H) → matmul → B,W,C,N → B,N,C,W → B,C,N,W.
fn contract_h(x: &Tensor, mt: &Tensor) -> Result<Tensor> {
    let (b, _c, _h, w) = x.dims4()?;
    let n = mt.dims()[1];
    let t1 = x.transpose(1, 3)?; // (B,W,C,H)
    let y = t1.reshape((b * w * _c, _h))?.matmul(mt)?; // (B·W·C, N)
    let t2 = y.reshape((b, w, _c, n))?; // (B,W,C,N)
    let t3 = t2.transpose(1, 3)?; // (B,N,C,W)
    Ok(t3.transpose(1, 2)?) // (B,C,N,W)
}

/// Forward rfftn on real input. Returns (Re, Im), each (B,C,H,W/2+1).
fn rfft2(x: &Tensor, bh: &Basis, bw: &Basis) -> Result<(Tensor, Tensor)> {
    // NOTE: f64 here was tested and gave bit-identical results to f32 —
    // the divergence vs torch is chaotic-amplification inside the
    // checkpoint, not transform precision. Keep f32 for speed.
    let ct = bh.c.t().map_err(|e| anyhow::anyhow!("{}", e))?;
    let st = bh.s.t().map_err(|e| anyhow::anyhow!("{}", e))?;
    // H pass on real x: spectrum Re=P, Im=−Q with P=x@cᵀ, Q=x@sᵀ.
    let p = contract_h(x, &ct)?;
    let q = contract_h(x, &st)?;
    // W pass (onesided), complex input (P, −Q):
    //   Re' = P@cwᵀ + Q@swᵀ ,  Im' = −(Q@cwᵀ + P@swᵀ)
    let pcw = mm_last(&p, &bw.c)?;
    let psw = mm_last(&p, &bw.s)?;
    let qcw = mm_last(&q, &bw.c)?;
    let qsw = mm_last(&q, &bw.s)?;
    let re = (&pcw + &qsw)?; // (b,c,h,m)
    let im = (&qcw + &psw)?.neg()?; // (b,c,h,m)
    Ok((re, im))
}

/// Pack (Re,Im) into PyTorch's interleaved layout: (B, 2C, H, M) ordered
/// (re0, im0, re1, im1, ...). Matches torch stack(real,imag,-1)-permute-view.
fn pack(re: &Tensor, im: &Tensor) -> Result<Tensor> {
    // (b,c,h,m) x2 -> stack at dim 2 -> (b,c,2,h,m) -> view (b,c*2,h,m)
    Tensor::stack(&[re.clone(), im.clone()], 2)?
        .reshape((re.dims4()?.0, re.dims4()?.1 * 2, re.dims4()?.2, re.dims4()?.3))
        .map_err(Into::into)
}

/// Unpack (B,2C,H,M) interleaved back into (Re, Im).
fn unpack(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, c2, h, m) = x.dims4()?;
    let c = c2 / 2;
    let t = x.reshape((b, c, 2, h, m))?;
    let re = t.narrow(2, 0, 1)?.reshape((b, c, h, m))?.contiguous()?;
    let im = t.narrow(2, 1, 1)?.reshape((b, c, h, m))?.contiguous()?;
    Ok((re, im))
}

/// Inverse DFT along the H axis of a (B,C,H,W) tensor (complex input,
/// real part returned): Re' = R@chᵀ − I@shᵀ, contracted over axis 1.
fn idft_h(r: &Tensor, im: &Tensor, basis: &Basis) -> Result<Tensor> {
    let ct = basis.c.t().map_err(|e| anyhow::anyhow!("{}", e))?;
    let st = basis.s.t().map_err(|e| anyhow::anyhow!("{}", e))?;
    let pr = contract_h(r, &ct)?; // (b,c,h,w) — N==h here
    let pi = contract_h(im, &st)?;
    (&pr - &pi).map_err(Into::into)
}

/// Inverse irfftn from one-sided (Re,Im). Returns REAL (B,C,H,W).
fn irfft2(re: &Tensor, im: &Tensor, bw_full: &Basis, bh: &Basis, out_w: usize) -> Result<Tensor> {
    let (_b, _c, _h, m) = re.dims4()?;
    let tail = out_w - m; // = m-1 for even sizes; excludes Nyquist bin
    if tail == 0 {
        bail!("irfft2: degenerate width");
    }
    // f32 throughout (see rfft2 note).
    // torch.fft.irfftn semantics: the imaginary parts of the DC bin and
    // the Nyquist bin are DISCARDED (a real signal has purely-real bins
    // there). Zero them before mirroring so our full-spectrum IDFT
    // matches torch bit-for-bit.
    let im = if m >= 2 {
        let mid = im.narrow(3, 1, m - 2)?;
        let z = Tensor::zeros((mid.dims4()?.0, mid.dims4()?.1, mid.dims4()?.2, 1), im.dtype(), im.device())?;
        Tensor::cat(&[z.clone(), mid, z], 3)?
    } else {
        Tensor::zeros_like(&im)?
    };
    // Hermitian mirror along W: appended bins = conj(F[w-k]) for k>=m,
    // which in tensor terms is conj(flip(F[..., 1 .. 1+tail])).
    let rt = re
        .contiguous()?
        .narrow(3, 1, tail)?
        .contiguous()?
        .flip(&[3])?
        .contiguous()?;
    let it = im
        .contiguous()?
        .narrow(3, 1, tail)?
        .contiguous()?
        .flip(&[3])?
        .neg()?
        .contiguous()?;
    let fr = Tensor::cat(&[re.clone(), rt], 3)?; // (b,c,h,out_w)
    let fi = Tensor::cat(&[im.clone(), it], 3)?;

    // Full IDFT along W (complex in, keep both parts):
    //   Wr = Fr@cwᵀ − Fi@swᵀ ,  Wi = Fi@cwᵀ + Fr@swᵀ
    let frc = mm_last(&fr, &bw_full.c)?;
    let frs = mm_last(&fr, &bw_full.s)?;
    let fic = mm_last(&fi, &bw_full.c)?;
    let fis = mm_last(&fi, &bw_full.s)?;
    let wr = (&frc - &fis)?; // (b,c,h,w)
    let wi = (&fic + &frs)?;

    // Full IDFT along H on that complex field; return its real part.
    idft_h(&wr, &wi, bh)
}

// ── Layers ─────────────────────────────────────────────────────────────

const BN_CFG: candle_nn::BatchNormConfig = candle_nn::BatchNormConfig {
    eps: 1e-5,
    remove_mean: true,
    affine: true,
    momentum: 0.1,
};

struct FourierUnit {
    conv: Conv2d,
    bn: BatchNorm,
    /// DFT bases per runtime spatial size, built lazily. The generator is
    /// fully convolutional, so arbitrary H,W (multiples of 8) only need
    /// per-size matrices.
    bases: std::cell::RefCell<
        std::collections::HashMap<(usize, usize), std::rc::Rc<SizeBases>>,
    >,
}

struct SizeBases {
    bh: Basis,  // full (h,h)      - forward H and inverse H
    bw: Basis,  // onesided (m,w)  - forward W
    bwf: Basis, // full (w,w)      - inverse W
}

impl FourierUnit {
    fn load(vb: VarBuilder, ch: usize) -> Result<Self> {
        // conv_layer: 1x1, no bias (weights confirm).
        let conv =
            candle_nn::conv2d_no_bias(ch * 2, ch * 2, 1, Default::default(), vb.pp("conv_layer"))?;
        let bn = candle_nn::batch_norm(ch * 2, BN_CFG, vb.pp("bn"))?;
        Ok(Self {
            conv,
            bn,
            bases: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    fn bases_for(
        &self,
        h: usize,
        w: usize,
        dev: &candle_core::Device,
    ) -> Result<std::rc::Rc<SizeBases>> {
        if let Some(b) = self.bases.borrow().get(&(h, w)) {
            return Ok(b.clone());
        }
        let b = std::rc::Rc::new(SizeBases {
            bh: Basis::full(h, dev)?,
            bw: Basis::onesided(w, dev)?,
            bwf: Basis::full(w, dev)?,
        });
        self.bases.borrow_mut().insert((h, w), b.clone());
        Ok(b)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;
        let basis = self.bases_for(h, w, x.device())?;
        let (re, im) = rfft2(x, &basis.bh, &basis.bw)?;
        let packed = pack(&re, &im)?;
        let y = self.conv.forward(&packed)?;
        let y = self.bn.forward_t(&y, false)?;
        let y = y.relu()?;
        let (yr, yi) = unpack(&y)?;
        irfft2(&yr, &yi, &basis.bwf, &basis.bh, w)
    }
}
struct SpectralTransform {
    avgpool: bool,
    conv1: Conv2d,
    bn1: BatchNorm,
    fu: FourierUnit,
    conv2: Conv2d,
}

impl SpectralTransform {
    fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        stride: usize,
    ) -> Result<Self> {
        debug_assert!(!vb.contains_tensor("lfu.conv_layer.weight"), "LFU unsupported");
        let half = out_ch / 2;
        let conv1 = candle_nn::conv2d_no_bias(
            in_ch,
            half,
            1,
            Default::default(),
            vb.pp("conv1").pp("0"),
        )?;
        let bn1 = candle_nn::batch_norm(half, BN_CFG, vb.pp("conv1").pp("1"))?;
        let fu = FourierUnit::load(vb.pp("fu"), half)?;
        let conv2 = candle_nn::conv2d_no_bias(half, out_ch, 1, Default::default(), vb.pp("conv2"))?;
        Ok(Self {
            avgpool: stride == 2,
            conv1,
            bn1,
            fu,
            conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.avgpool { x.avg_pool2d(2)? } else { x.clone() };
        let x = self.bn1.forward_t(&self.conv1.forward(&x)?, false)?.relu()?;
        let out = self.fu.forward(&x)?;
        let s = (x + out)?;
        Ok(self.conv2.forward(&s)?)
    }
}

/// Sum of optional contributions; at least one must be present.
fn add_opt(a: Option<Tensor>, b: Option<Tensor>) -> Result<Option<Tensor>> {
    match (a, b) {
        (Some(a), Some(b)) => Ok(Some((&a + &b)?)),
        (a, None) => Ok(a),
        (None, b) => Ok(b),
    }
}

struct FfcBlock {
    l2l: Option<RefPadConv>,
    l2g: Option<RefPadConv>,
    g2l: Option<RefPadConv>,
    g2g: Option<SpectralTransform>,
    has_gout: bool, // ratio_gout != 0
    has_gin: bool,  // ratio_gin != 0
}

/// Conv2d whose implicit padding is REFLECT (PyTorch padding_mode='reflect',
/// which FFC uses everywhere). candle only supports zero padding, so we
/// reflect-pad manually and run the conv with padding=0.
struct RefPadConv {
    pad: usize,
    conv: Conv2d,
}

impl RefPadConv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.pad > 0 { reflect_pad2d(x, self.pad)? } else { x.clone() };
        Ok(self.conv.forward(&x)?)
    }
}

impl FfcBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        ratio_gin: f64,
        ratio_gout: f64,
        stride: usize,
        padding: usize,
        kernel: usize,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        let in_g = (in_ch as f64 * ratio_gin) as usize;
        let in_l = in_ch - in_g;
        let out_g = (out_ch as f64 * ratio_gout) as usize;
        let out_l = out_ch - out_g;
        // Reflect pre-pad replaces the conv's own padding entirely.
        let cfg = candle_nn::Conv2dConfig {
            stride,
            padding: 0,
            ..Default::default()
        };
        let cv = |name: &str, ci: usize, co: usize| -> Result<RefPadConv> {
            Ok(RefPadConv {
                pad: padding,
                conv: candle_nn::conv2d_no_bias(ci, co, kernel, cfg, vb.pp(name))?,
            })
        };
        Ok(Self {
            l2l: (in_l > 0 && out_l > 0).then(|| cv("convl2l", in_l, out_l)).transpose()?,
            l2g: (in_l > 0 && out_g > 0).then(|| cv("convl2g", in_l, out_g)).transpose()?,
            g2l: (in_g > 0 && out_l > 0).then(|| cv("convg2l", in_g, out_l)).transpose()?,
            g2g: (in_g > 0 && out_g > 0)
                .then(|| {
                    SpectralTransform::load(vb.pp("convg2g"), in_g, out_g, stride)
                })
                .transpose()?,
            has_gout: ratio_gout != 0.0,
            has_gin: ratio_gin != 0.0,
        })
    }

    /// xl always present; xg None until some layer emits global channels.
    fn forward(&self, xl: &Tensor, xg: Option<&Tensor>) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // Local output branch.
        let out_l = if !self.has_gout || self.l2l.is_some() || self.g2l.is_some() {
            let a = match &self.l2l {
                Some(c) => Some(c.forward(xl)?),
                None => None,
            };
            let b = match (&self.g2l, xg) {
                (Some(c), Some(g)) => Some(c.forward(g)?),
                _ => None,
            };
            add_opt(a, b)?
        } else {
            None
        };

        // Global output branch.
        let out_g = if self.has_gout {
            let a = match &self.l2g {
                Some(c) => Some(c.forward(xl)?),
                None => None,
            };
            let b = match (&self.g2g, xg) {
                (Some(c), Some(g)) => Some(c.forward(g)?),
                _ => None,
            };
            add_opt(a, b)?
        } else {
            None
        };
        Ok((out_l, out_g))
    }
}

struct FfcBnAct {
    ffc: FfcBlock,
    bn_l: Option<BatchNorm>,
    bn_g: Option<BatchNorm>,
}

impl FfcBnAct {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        ratio_gin: f64,
        ratio_gout: f64,
        stride: usize,
        padding: usize,
        kernel: usize,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        let ffc = FfcBlock::load(
            vb.pp("ffc"), in_ch, out_ch, ratio_gin, ratio_gout, stride, padding, kernel, h, w,
        )?;
        let out_g = (out_ch as f64 * ratio_gout) as usize;
        let out_l = out_ch - out_g;
        Ok(Self {
            bn_l: (ratio_gout != 1.0 && out_l > 0)
                .then(|| {
                    candle_nn::batch_norm(out_l, BN_CFG, vb.pp("bn_l"))
                        .map_err(|e| anyhow::anyhow!("{}", e))
                })
                .transpose()?,
            bn_g: (ratio_gout != 0.0 && out_g > 0)
                .then(|| {
                    candle_nn::batch_norm(out_g, BN_CFG, vb.pp("bn_g"))
                        .map_err(|e| anyhow::anyhow!("{}", e))
                })
                .transpose()?,
            ffc,
        })
    }

    fn forward(
        &self,
        xl: &Tensor,
        xg: Option<&Tensor>,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (ol, og) = self.ffc.forward(xl, xg)?;
        let ol = ol
            .map(|t| -> Result<Tensor> {
                let t = match &self.bn_l {
                    Some(bn) => bn.forward_t(&t, false)?,
                    None => t,
                };
                Ok(t.relu()?)
            })
            .transpose()?;
        let og = og
            .map(|t| -> Result<Tensor> {
                let t = match &self.bn_g {
                    Some(bn) => bn.forward_t(&t, false)?,
                    None => t,
                };
                Ok(t.relu()?)
            })
            .transpose()?;
        Ok((ol, og))
    }
}

struct ResnetBlock {
    conv1: FfcBnAct,
    conv2: FfcBnAct,
}

impl ResnetBlock {
    fn load(
        vb: VarBuilder,
        dim: usize,
        ratio_gin: f64,
        ratio_gout: f64,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        Ok(Self {
            conv1: FfcBnAct::load(vb.pp("conv1"), dim, dim, ratio_gin, ratio_gout, 1, 1, 3, h, w)?,
            conv2: FfcBnAct::load(vb.pp("conv2"), dim, dim, ratio_gin, ratio_gout, 1, 1, 3, h, w)?,
        })
    }

    fn forward(&self, xl: &Tensor, xg: &Tensor) -> Result<(Tensor, Tensor)> {
        let (ol, og) = self.conv1.forward(xl, Some(xg))?;
        let (ol, og) = match (ol, og) {
            (Some(l), Some(g)) => self.conv2.forward(&l, Some(&g))?,
            _ => bail!("resnet block dropped a branch"),
        };
        let (l, g) = match (ol, og) {
            (Some(l), Some(g)) => ((xl + &l)?, (xg + &g)?),
            _ => bail!("resnet block dropped a branch"),
        };
        Ok((l, g))
    }
}

// ── Generator ──────────────────────────────────────────────────────────

pub struct CandleInpainter {
    init: FfcBnAct,
    downs: Vec<FfcBnAct>,
    res: Vec<ResnetBlock>,
    up_conv: Vec<ConvTranspose2d>,
    up_bn: Vec<BatchNorm>,
    final_conv: Conv2d,
}

/// torch.nn.ReflectionPad2d(p) via flip+narrow+cat.
fn reflect_pad2d(x: &Tensor, p: usize) -> Result<Tensor> {
    let (_b, _c, h, w) = x.dims4()?;
    let left = x.flip(&[3])?.contiguous()?.narrow(3, w - p, p)?;
    let right = x.flip(&[3])?.contiguous()?.narrow(3, 0, p)?;
    let row = Tensor::cat(&[left, x.clone(), right], 3)?;
    let top = row.flip(&[2])?.contiguous()?.narrow(2, h - p, p)?;
    let bottom = row.flip(&[2])?.contiguous()?.narrow(2, 0, p)?;
    Tensor::cat(&[top, row, bottom], 2).map_err(Into::into)
}

impl CandleInpainter {
    pub fn from_safetensors(model_path: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)
                .map_err(|e| anyhow::anyhow!("{}", e))?
        };

        const IN_NC: usize = 4;
        const OUT_NC: usize = 3;
        const NGF: usize = 64;
        const N_DS: usize = 3;
        const N_BL: usize = 18;
        const RG: f64 = 0.75;
        const MAX_F: usize = 1024;

        let h0 = MODEL_INPUT;
        let w0 = MODEL_INPUT;
        let hb = h0 >> N_DS; // bottleneck spatial (64)

        // 1: init FFC_BN_ACT, kernel 7 (pad handled by reflect_pad2d).
        let init = FfcBnAct::load(
            vb.pp("model").pp("1"), IN_NC, NGF, 0.0, 0.0, 1, 0, 7, h0 + 6, w0 + 6,
        )?;

        // 2..=4: downsampling FFC_BN_ACTs, kernel 3, stride 2, pad 1.
        // Last one inherits resnet ratio_gout (=RG); others are pure local.
        let mut downs = Vec::with_capacity(N_DS);
        for i in 0..N_DS {
            let mult = 1usize << i;
            let (rgin, rgout) = if i == N_DS - 1 { (0.0, RG) } else { (0.0, 0.0) };
            downs.push(FfcBnAct::load(
                vb.pp("model").pp(&(i + 2).to_string()),
                (NGF * mult).min(MAX_F),
                (NGF * mult * 2).min(MAX_F),
                rgin,
                rgout,
                2,
                1,
                3,
                h0 >> (i + 1),
                w0 >> (i + 1),
            )?);
        }

        // 5..=22: resnet blocks at bottleneck resolution.
        let feats = (NGF << N_DS).min(MAX_F);
        let mut res = Vec::with_capacity(N_BL);
        for i in 0..N_BL {
            res.push(ResnetBlock::load(
                vb.pp("model").pp(&(2 + N_DS + i).to_string()),
                feats,
                RG,
                RG,
                hb,
                hb,
            )?);
        }

        // ConcatTupleLayer sits at index 2+N_DS+N_BL = 23 (NO parameters).
        // Upsample groups start at 24: [ConvT, BN, ReLU] × N_DS.
        let up_base = 2 + N_DS + N_BL + 1;
        let mut up_conv = Vec::with_capacity(N_DS);
        let mut up_bn = Vec::with_capacity(N_DS);
        for i in 0..N_DS {
            let mult = 1usize << (N_DS - i);
            let ic = (NGF * mult).min(MAX_F);
            let oc = (NGF * mult / 2).min(MAX_F);
            let base = up_base + i * 3;
            up_conv.push(candle_nn::conv_transpose2d(
                ic,
                oc,
                3,
                candle_nn::ConvTranspose2dConfig {
                    stride: 2,
                    padding: 1,
                    output_padding: 1,
                    ..Default::default()
                },
                vb.pp("model").pp(&base.to_string()),
            ).map_err(|e| anyhow::anyhow!("{}", e))?);
            up_bn.push(candle_nn::batch_norm(oc, BN_CFG, vb.pp("model").pp(&(base + 1).to_string()))
                .map_err(|e| anyhow::anyhow!("{}", e))?);
        }

        // Final conv at up_base + N_DS*3 + 1 (skip the ReflectionPad index).
        let final_idx = up_base + N_DS * 3 + 1;
        let final_conv = candle_nn::conv2d(
            NGF,
            OUT_NC,
            7,
            Default::default(),
            vb.pp("model").pp(&final_idx.to_string()),
        )
        .map_err(|e| anyhow::anyhow!("{}", e))?;

        Ok(Self { init, downs, res, up_conv, up_bn, final_conv })
    }

    /// image (1,3,512,512) in [0,1], mask (1,1,512,512) in {0,1}
    /// → (1,3,512,512) in [0,255] (worker convention divides by 255 later).
    pub fn inpaint(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let dbg = std::env::var("LAMA_DEBUG_STAGES").is_ok();
        macro_rules! stage {
            ($name:expr, $t:expr) => {
                if dbg {
                    let t: &Tensor = &$t;
                    let v = t.flatten_all().map_err(|e| anyhow::anyhow!("{}", e))?;
                    let n = v.elem_count().max(1) as f64;
                    let mut m = 0f64;
                    let vals = v.to_vec1::<f32>().unwrap_or_default();
                    for &x in &vals {
                        m += x as f64;
                    }
                    m /= n;
                    let mut s = 0f64;
                    for &x in &vals {
                        let dd = x as f64 - m;
                        s += dd * dd;
                    }
                    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
                    for &x in &vals {
                        mn = mn.min(x as f64);
                        mx = mx.max(x as f64);
                    }
                    eprintln!(
                        "STAGE {} shape {:?} mean {:.5} std {:.5} min {:.4} max {:.4}",
                        $name,
                        t.dims(),
                        m,
                        (s / n).sqrt(),
                        mn,
                        mx
                    );
                }
            };
        }
        // Training contract (trainers/default.py): the generator consumes
        // img * (1 - mask) — hole pixels ZEROED — plus the mask channel.
        // Raw state-dict generators do NOT zero internally; the shipped
        // ONNX export baked this multiply into its graph, which is why
        // the ONNX path never needed it explicitly.
        let one_minus = mask.neg()?.affine(1.0, 1.0)?; // 1 - mask
        let masked_img = (image * one_minus.broadcast_as(image.shape())?)?;
        let x = Tensor::cat(&[masked_img, mask.clone()], 1)?;
        stage!("cat_input", x);
        let x = reflect_pad2d(&x, 3)?; // layer 0
        stage!("pad3", x);

        let z = Tensor::zeros_like(&x)?;
        // Init consumes a synthetic empty global branch.
        let (mut ol, mut og) = self.init.forward(&x, None)?;
        if dbg {
            if let Some(ref t) = ol {
                stage!("L1_init_l", t.clone());
            }
        }
        let mut ol = ol.context("init produced no local branch")?;
        drop(z);

        for (i, d) in self.downs.iter().enumerate() {
            let (nl, ng) = d.forward(&ol, og.as_ref())?;
            ol = nl.context("downsample dropped local branch")?;
            og = ng;
            stage!(format!("L{}_down_l", i + 2), ol);
            if let Some(ref g) = og {
                stage!(format!("L{}_down_g", i + 2), g.clone());
            }
        }

        let g = og.context("no global branch before resnet blocks")?;
        let (mut ol, mut og) = (ol, g);
        for (i, r) in self.res.iter().enumerate() {
            let (nl, ng) = r.forward(&ol, &og)?;
            ol = nl;
            og = ng;
            if i < 2 || i == self.res.len() - 1 {
                stage!(format!("res{}_l", 5 + i), ol.clone());
                stage!(format!("res{}_g", 5 + i), og.clone());
            }
        }
        stage!("post_res_l", ol.clone());
        stage!("post_res_g", og.clone());

        // ConcatTupleLayer (index 23): cat along channels.
        let mut x = Tensor::cat(&[ol, og], 1)?;
        stage!("concat", x);

        for i in 0..self.up_conv.len() {
            x = self.up_conv[i].forward(&x)?;
            stage!(format!("up{}_convt", i), x.clone());
            x = self.up_bn[i].forward_t(&x, false)?;
            stage!(format!("up{}_bn", i), x.clone());
            x = x.relu()?;
        }

        let x = reflect_pad2d(&x, 3)?; // layer 33
        let x = self.final_conv.forward(&x)?; // layer 34 (has bias)
        let x = x.neg()?.exp()?; // sigmoid
        let x = (&x + 1.0)?.recip()?;
        (x * 255.0).map_err(Into::into)
    }
}
