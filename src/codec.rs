//! Fish codec (modded DAC) on the GPU: frames of codes <-> 44.1 kHz audio.
//!
//! Decoding is streaming and exact: `DecodeStream` takes frames in order and keeps every
//! causal state between calls (the transformer's ring KV cache, the ConvNeXt depthwise
//! histories and each conv layer's history), so chunked output equals one full pass.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::ffi;
use crate::gpu::{check, launched, Cublas, DevBuf, HostBuf, Stream};

pub const SAMPLE_RATE: u32 = 44100;
/// Audio samples per frame of codes.
pub const FRAME_SAMPLES: usize = 2048;
const D: usize = 1024;
const HEADS: usize = 16;
const WINDOW: i32 = 128;
const FFN: usize = 3072;
const SEM_SIZE: i32 = 4096;
const RES_SIZE: i32 = 1024;
const NCB: usize = 10;
const DEC_IN: usize = 1536;
const RATES: [usize; 4] = [8, 8, 4, 2];
const DILATIONS: [usize; 3] = [1, 3, 9];
/// Frames per front pass, and the ring size of the transformer KV cache (window + chunk).
const FRONT_CHUNK: usize = 128;
const RING: usize = WINDOW as usize + FRONT_CHUNK;

/// A weight: in VRAM, or in mapped page-locked RAM the kernels read over PCIe.
enum T {
    Dev(DevBuf),
    Host(HostBuf),
}

impl T {
    fn ptr<P>(&self) -> *mut P {
        match self {
            T::Dev(b) => b.ptr::<P>(),
            T::Host(b) => b.dev::<P>(),
        }
    }
}

/// Weights only the encoder (reference voices) reads.
fn encoder_only(name: &str) -> bool {
    ["enc.", "down0.", "down1.", "pre.", "vq"].iter().any(|p| name.starts_with(p))
}

/// One layer's input: `ctx` rows of history followed by up to `cap` current rows.
struct Stage {
    buf: DevBuf,
    tmp: DevBuf,
    ctx: usize,
    c: usize,
}

impl Stage {
    fn new(ctx: usize, cap: usize, c: usize, st: &Stream) -> Result<Self> {
        let s = Self { buf: DevBuf::zeros((ctx + cap) * c * 4, st)?, tmp: DevBuf::alloc(ctx.max(1) * c * 4)?, ctx, c };
        Ok(s)
    }
    /// First current row.
    fn cur(&self) -> *mut f32 {
        self.buf.at::<f32>(self.ctx * self.c)
    }
    /// Keep the last ctx rows (history + the n current rows) as the next call's history.
    fn shift(&self, n: usize, st: &Stream) -> Result<()> {
        if self.ctx == 0 || n == 0 {
            return Ok(());
        }
        let bytes = self.ctx * self.c * 4;
        let src = self.buf.at::<f32>(n * self.c) as *const std::ffi::c_void;
        if n >= self.ctx {
            check(unsafe { ffi::dr_d2d(self.buf.ptr(), src, bytes, st.raw) }, "d2d")
        } else {
            // Overlapping ranges: bounce through tmp.
            check(unsafe { ffi::dr_d2d(self.tmp.ptr(), src, bytes, st.raw) }, "d2d")?;
            check(unsafe { ffi::dr_d2d(self.buf.ptr(), self.tmp.ptr::<std::ffi::c_void>(), bytes, st.raw) }, "d2d")
        }
    }
}

/// Zero rows kept before a full-sequence buffer so causal convs can read "before" row 0.
const PRE: usize = 64;

/// A full-sequence activation [rows, c] preceded by PRE zero rows.
struct Seq {
    buf: DevBuf,
    rows: usize,
    c: usize,
}

impl Seq {
    fn new(rows: usize, c: usize, st: &Stream) -> Result<Self> {
        let buf = DevBuf::alloc((PRE + rows) * c * 4)?;
        check(unsafe { ffi::dr_memset(buf.ptr(), 0, PRE * c * 4, st.raw) }, "memset")?;
        Ok(Self { buf, rows, c })
    }
    fn at(&self, row: usize) -> *mut f32 {
        self.buf.at::<f32>((PRE + row) * self.c)
    }
}

/// Scratch for the transformer / ConvNeXt over up to `rows` rows.
struct FrontScratch {
    x16: DevBuf,
    big: DevBuf,
    q: DevBuf,
    y: DevBuf,
    o16: DevBuf,
}

impl FrontScratch {
    fn new(rows: usize) -> Result<Self> {
        Ok(Self {
            x16: DevBuf::alloc(rows * 4 * D * 2)?,
            big: DevBuf::alloc(rows * (2 * FFN).max(4 * D) * 4)?,
            q: DevBuf::alloc(rows * D * 4)?,
            y: DevBuf::alloc(rows * 2 * D * 4)?,
            o16: DevBuf::alloc(rows * FFN * 2)?,
        })
    }
}

fn alloc_caches(layers: usize, ring: usize) -> Result<Vec<(DevBuf, DevBuf)>> {
    (0..layers).map(|_| Ok((DevBuf::alloc(ring * D * 4)?, DevBuf::alloc(ring * D * 4)?))).collect()
}

pub struct Codec {
    t: HashMap<String, T>,
    inv_freq: DevBuf,
    blas: Cublas,
    st: Stream,
}

unsafe impl Send for Codec {}
// Shared with the decode thread; callers never use one Codec from two threads at once.
unsafe impl Sync for Codec {}

fn f(b: &DevBuf) -> *mut f32 {
    b.ptr::<f32>()
}
fn h(b: &DevBuf) -> *mut u16 {
    b.ptr::<u16>()
}

/// Streaming decoder state for one utterance.
pub struct DecodeStream {
    /// Most frames one pass takes: the buffers below are sized for it.
    pass: usize,
    frames_done: usize,
    caches: Vec<(DevBuf, DevBuf)>,
    up: Vec<Stage>,
    stage_in: Stage,
    blocks: Vec<(Stage, Vec<Stage>, usize, usize, usize)>,
    out_stage: Stage,
    scr: FrontScratch,
    x: DevBuf,
    codes: DevBuf,
    col: DevBuf,
    tmp: DevBuf,
    gem: DevBuf,
    audio: DevBuf,
}

unsafe impl Send for DecodeStream {}

impl Codec {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with(path, false)
    }

    /// `encoder_in_ram`: the encoder's weights stay in RAM (it runs once per reference voice), leaving their VRAM to
    /// the decoder and the LM.
    pub fn load_with(path: &Path, encoder_in_ram: bool) -> Result<Self> {
        let st = Stream::new()?;
        let blas = Cublas::new(&st)?;
        let file = std::fs::File::open(path).map_err(|e| anyhow!("open {}: {e}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        let sf = SafeTensors::deserialize(&map)?;
        let mut t = HashMap::new();
        for (name, view) in sf.tensors() {
            if !matches!(view.dtype(), Dtype::F32 | Dtype::F16) {
                bail!("{name}: unexpected dtype {:?}", view.dtype());
            }
            let w = if encoder_in_ram && encoder_only(&name) { T::Host(HostBuf::from_bytes(view.data())?) } else { T::Dev(DevBuf::from_bytes(view.data(), &st)?) };
            t.insert(name.to_string(), w);
        }
        let inv: Vec<f32> = (0..32).map(|i| 1.0f32 / 10000f32.powf((2 * i) as f32 / 64.0)).collect();
        let inv_freq = DevBuf::from_slice(&inv, &st)?;
        st.sync()?;
        Ok(Self { t, inv_freq, blas, st })
    }

    /// Bytes of the encoder's weights and of the rest, from the file header.
    pub fn sizes(path: &Path) -> Result<(u64, u64)> {
        let file = std::fs::File::open(path).map_err(|e| anyhow!("open {}: {e}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        let sf = SafeTensors::deserialize(&map)?;
        let (mut enc, mut rest) = (0u64, 0u64);
        for (name, view) in sf.tensors() {
            *if encoder_only(&name) { &mut enc } else { &mut rest } += view.data().len() as u64;
        }
        Ok((enc, rest))
    }

    fn w(&self, name: &str) -> Result<&T> {
        self.t.get(name).ok_or_else(|| anyhow!("codec tensor {name} missing"))
    }
    fn p32(&self, name: &str) -> Result<*const f32> {
        Ok(self.w(name)?.ptr())
    }
    fn p16(&self, name: &str) -> Result<*const u16> {
        Ok(self.w(name)?.ptr())
    }

    fn s(&self) -> ffi::Stream {
        self.st.raw
    }

    #[allow(clippy::too_many_arguments)]
    fn epilogue(&self, y: *const f32, n: usize, c: usize, bias: *const f32, scale: *const f32, res: *const f32, act: i32, out: *mut f32, out16: *mut u16) -> Result<()> {
        launched(unsafe { ffi::dr_epilogue(y, n, c as i32, bias, scale, res, act, out, out16, self.s()) }, "epilogue")
    }

    /// Causal windowed transformer (fish-speech WindowLimitedTransformer) over rows at
    /// positions t0.., in place, including its final RMSNorm. caches: per layer (k, v) rings.
    #[allow(clippy::too_many_arguments)]
    fn transformer(&self, prefix: &str, layers: usize, x: *mut f32, rows: usize, window: i32, t0: usize, caches: &[(DevBuf, DevBuf)], ring: usize, scr: &FrontScratch) -> Result<()> {
        let s = self.s();
        let null = std::ptr::null::<f32>();
        let nullm = std::ptr::null_mut::<f32>();
        let null16 = std::ptr::null_mut::<u16>();
        for (l, (kc, vc)) in caches.iter().enumerate().take(layers) {
            let p = format!("{prefix}.l{l}.");
            launched(unsafe { ffi::dr_rmsnorm_plain(x, rows as i32, D as i32, self.p32(&(p.clone() + "attn_norm"))?, 1e-5, nullm, h(&scr.x16), s) }, "rms")?;
            self.blas.gemm_f16(h(&scr.x16), self.p16(&(p.clone() + "wqkv"))?, f(&scr.big), rows, 3 * D, D, 0.0)?;
            launched(
                unsafe { ffi::dr_codec_rope(f(&scr.big), rows as i32, HEADS as i32, f(&self.inv_freq), t0 as i32, ring as i32, f(&scr.q), f(kc), f(vc), s) },
                "rope",
            )?;
            launched(unsafe { ffi::dr_codec_attn(f(&scr.q), f(kc), f(vc), rows as i32, HEADS as i32, window, t0 as i32, ring as i32, h(&scr.o16), s) }, "attn")?;
            self.blas.gemm_f16(h(&scr.o16), self.p16(&(p.clone() + "wo"))?, f(&scr.y), rows, D, D, 0.0)?;
            self.epilogue(f(&scr.y), rows * D, D, null, self.p32(&(p.clone() + "attn_scale"))?, x, 0, x, null16)?;
            launched(unsafe { ffi::dr_rmsnorm_plain(x, rows as i32, D as i32, self.p32(&(p.clone() + "ffn_norm"))?, 1e-5, nullm, h(&scr.x16), s) }, "rms")?;
            self.blas.gemm_f16(h(&scr.x16), self.p16(&(p.clone() + "w13"))?, f(&scr.big), rows, 2 * FFN, D, 0.0)?;
            launched(unsafe { ffi::dr_swiglu16(f(&scr.big), h(&scr.o16), rows as i32, FFN as i32, s) }, "swiglu")?;
            self.blas.gemm_f16(h(&scr.o16), self.p16(&(p.clone() + "w2"))?, f(&scr.y), rows, D, FFN, 0.0)?;
            self.epilogue(f(&scr.y), rows * D, D, null, self.p32(&(p + "ffn_scale"))?, x, 0, x, null16)?;
        }
        launched(unsafe { ffi::dr_rmsnorm_plain(x, rows as i32, D as i32, self.p32(&format!("{prefix}.norm"))?, 1e-5, x, null16, s) }, "rms")
    }

    /// ConvNeXt block: out = input + gamma * pw2(gelu(pw1(LN(dwconv(input))))). The 6 rows
    /// before `input` must be readable (history or zeros). out may alias input.
    fn convnext(&self, p: &str, input: *const f32, out: *mut f32, rows: usize, scr: &FrontScratch) -> Result<()> {
        let s = self.s();
        let null = std::ptr::null::<f32>();
        let nullm = std::ptr::null_mut::<f32>();
        let null16 = std::ptr::null_mut::<u16>();
        let w = |n: &str| self.p32(&format!("{p}.{n}"));
        launched(unsafe { ffi::dr_dwconv(input, rows as i32, D as i32, 7, w("dw_w")?, w("dw_b")?, f(&scr.y), s) }, "dwconv")?;
        launched(unsafe { ffi::dr_layernorm(f(&scr.y), rows as i32, D as i32, w("ln_w")?, w("ln_b")?, 1e-6, h(&scr.x16), s) }, "layernorm")?;
        self.blas.gemm_f16(h(&scr.x16), self.p16(&format!("{p}.pw1_w"))?, f(&scr.big), rows, 4 * D, D, 0.0)?;
        self.epilogue(f(&scr.big), rows * 4 * D, 4 * D, w("pw1_b")?, null, null, 2, nullm, h(&scr.x16))?;
        self.blas.gemm_f16(h(&scr.x16), self.p16(&format!("{p}.pw2_w"))?, f(&scr.y), rows, D, 4 * D, 0.0)?;
        self.epilogue(f(&scr.y), rows * D, D, w("pw2_b")?, w("gamma")?, input, 0, out, null16)
    }

    /// Full-sequence causal conv: out = act/bias(conv(snake?(x))) (+ res), in row chunks.
    #[allow(clippy::too_many_arguments)]
    fn conv_seq(&self, x: &Seq, w: &str, b: &str, k: usize, d: usize, stride: usize, alpha: Option<&str>, cout: usize, res: Option<&Seq>, out: &Seq, scratch: &(DevBuf, DevBuf)) -> Result<()> {
        let s = self.s();
        let cin = x.c;
        let rows = x.rows / stride;
        assert_eq!(out.rows, rows);
        assert_eq!(out.c, cout);
        let (col, gem) = scratch;
        let chunk = (col.bytes() / (k * cin * 2)).min(gem.bytes() / (cout * 4)).max(1);
        let alpha = match alpha {
            Some(a) => self.p32(a)?,
            None => std::ptr::null(),
        };
        let mut r0 = 0;
        while r0 < rows {
            let n = (rows - r0).min(chunk);
            launched(unsafe { ffi::dr_im2col(x.at(r0 * stride), n as i32, cin as i32, k as i32, d as i32, stride as i32, alpha, h(col), s) }, "im2col")?;
            self.blas.gemm_f16(h(col), self.p16(w)?, f(gem), n, cout, k * cin, 0.0)?;
            let res_p = res.map(|r| r.at(r0) as *const f32).unwrap_or(std::ptr::null());
            self.epilogue(f(gem), n * cout, cout, self.p32(b)?, std::ptr::null(), res_p, 0, out.at(r0), std::ptr::null_mut())?;
            r0 += n;
        }
        Ok(())
    }

    /// Encode 44.1 kHz mono audio to codes ([frames][10], codebook 0 first).
    pub fn encode(&self, audio: &[f32]) -> Result<Vec<i32>> {
        let st = &self.st;
        let s = self.s();
        let hop = 2048;
        let len = audio.len().div_ceil(hop) * hop;
        if len == 0 {
            bail!("empty reference audio");
        }
        let mut padded = audio.to_vec();
        padded.resize(len, 0.0);
        let scratch = (DevBuf::alloc(96 << 20)?, DevBuf::alloc(96 << 20)?);
        let x = Seq::new(len, 1, st)?;
        x.buf.upload_at(PRE * 4, &padded, st)?;
        let mut a = Seq::new(len, 64, st)?;
        self.conv_seq(&x, "enc.in.w", "enc.in.b", 7, 1, 1, None, 64, None, &a, &scratch)?;
        drop(x);
        let strides = [2usize, 4, 8, 8];
        for (bi, &stride) in strides.iter().enumerate() {
            let c = a.c;
            for (u, &dil) in DILATIONS.iter().enumerate() {
                let p = format!("enc.b{bi}.ru{u}.");
                let y1 = Seq::new(a.rows, c, st)?;
                self.conv_seq(&a, &(p.clone() + "c1.w"), &(p.clone() + "c1.b"), 7, dil, 1, Some(&(p.clone() + "a1")), c, None, &y1, &scratch)?;
                let out = Seq::new(a.rows, c, st)?;
                self.conv_seq(&y1, &(p.clone() + "c2.w"), &(p.clone() + "c2.b"), 1, 1, 1, Some(&(p + "a2")), c, Some(&a), &out, &scratch)?;
                a = out;
            }
            let p = format!("enc.b{bi}.");
            let out = Seq::new(a.rows / stride, 2 * c, st)?;
            self.conv_seq(&a, &(p.clone() + "down.w"), &(p.clone() + "down.b"), 2 * stride, 1, stride, Some(&(p.clone() + "alpha")), 2 * c, None, &out, &scratch)?;
            a = out;
            if self.t.contains_key(&format!("enc.b{bi}.tr.norm")) {
                let scr = FrontScratch::new(a.rows)?;
                let caches = alloc_caches(4, a.rows)?;
                self.transformer(&format!("enc.b{bi}.tr"), 4, a.at(0), a.rows, 512, 0, &caches, a.rows, &scr)?;
            }
        }
        let out = Seq::new(a.rows, D, st)?;
        self.conv_seq(&a, "enc.out.w", "enc.out.b", 3, 1, 1, Some("enc.out.alpha"), D, None, &out, &scratch)?;
        a = out;
        for i in 0..2 {
            let out = Seq::new(a.rows / 2, D, st)?;
            self.conv_seq(&a, &format!("down{i}.w"), &format!("down{i}.b"), 2, 1, 2, None, D, None, &out, &scratch)?;
            a = out;
            let scr = FrontScratch::new(a.rows)?;
            self.convnext(&format!("down{i}.cnx"), a.at(0), a.at(0), a.rows, &scr)?;
        }
        let scr = FrontScratch::new(a.rows)?;
        let caches = alloc_caches(8, a.rows)?;
        self.transformer("pre", 8, a.at(0), a.rows, WINDOW, 0, &caches, a.rows, &scr)?;
        let frames = a.rows;
        let codes = DevBuf::alloc(frames * NCB * 4)?;
        for i in 0..NCB {
            let (n, row0) = if i == 0 { (SEM_SIZE as usize, 0) } else { (RES_SIZE as usize, SEM_SIZE as usize + (i - 1) * RES_SIZE as usize) };
            let table = unsafe { self.p32("rvq.table")?.add(row0 * D) };
            launched(
                unsafe {
                    ffi::dr_vq_encode(
                        a.at(0), frames as i32, D as i32, self.p32(&format!("vq{i}.in_w"))?, self.p32(&format!("vq{i}.in_b"))?,
                        self.p32(&format!("vq{i}.codebook_n"))?, n as i32, table, codes.at::<i32>(i), NCB as i32, s,
                    )
                },
                "vq_encode",
            )?;
        }
        codes.download::<i32>(frames * NCB, st)
    }

    /// A fresh streaming decoder for passes of up to FRONT_CHUNK frames.
    pub fn stream(&self) -> Result<DecodeStream> {
        self.stream_for(FRONT_CHUNK)
    }

    /// A fresh streaming decoder whose buffers hold passes of up to `pass` frames: its memory grows with
    /// `pass`, about 1.6 GiB at FRONT_CHUNK, so a stream fed a few frames at a time takes a small one.
    pub fn stream_for(&self, pass: usize) -> Result<DecodeStream> {
        let pass = pass.clamp(1, FRONT_CHUNK);
        let chunk = 4 * pass;
        let st = &self.st;
        let mut blocks = Vec::new();
        let mut c_in = DEC_IN;
        let mut cap = chunk;
        for &r in &RATES {
            let c_out = c_in / 2;
            cap *= r;
            let tconv_in = Stage::new(1, cap / r, c_in, st)?;
            let rus: Vec<Stage> = DILATIONS.iter().map(|&d| Stage::new(6 * d, cap, c_out, st)).collect::<Result<_>>()?;
            blocks.push((tconv_in, rus, c_in, c_out, r));
            c_in = c_out;
        }
        let ds = DecodeStream {
            pass,
            frames_done: 0,
            caches: alloc_caches(8, RING)?,
            up: vec![Stage::new(6, 2 * pass, D, st)?, Stage::new(6, 4 * pass, D, st)?],
            stage_in: Stage::new(6, chunk, D, st)?,
            out_stage: Stage::new(6, cap, c_in, st)?,
            blocks,
            scr: FrontScratch::new(4 * pass)?,
            x: DevBuf::alloc(4 * pass * D * 4)?,
            codes: DevBuf::alloc(pass * NCB * 4)?,
            col: DevBuf::alloc((cap * 7 * 96).max(chunk * 8 * 2 * DEC_IN).max(chunk * 7 * D) * 2)?,
            tmp: DevBuf::alloc(cap * 96 * 4 + chunk * 8 * DEC_IN * 4)?,
            gem: DevBuf::alloc(cap * 96 * 4 + chunk * 8 * DEC_IN * 4)?,
            audio: DevBuf::alloc(cap * 4)?,
        };
        st.sync()?;
        Ok(ds)
    }

    /// Feed the next frames ([n][10] codes) and get their audio (n * FRAME_SAMPLES samples).
    pub fn decode_chunk(&self, ds: &mut DecodeStream, codes: &[i32]) -> Result<Vec<f32>> {
        let mut audio = Vec::with_capacity(codes.len() / NCB * FRAME_SAMPLES);
        for part in codes.chunks(ds.pass * NCB) {
            self.front(ds, part)?;
            let rows = part.len() / NCB * 4;
            let mut r0 = 0;
            while r0 < rows {
                let n = (rows - r0).min(4 * ds.pass);
                audio.extend(self.convs(ds, unsafe { ds.x.ptr::<f32>().add(r0 * D) }, n)?);
                r0 += n;
            }
        }
        Ok(audio)
    }

    /// RVQ sum + transformer + 2 x (transposed conv, ConvNeXt): n frames -> 4n latent rows in ds.x.
    fn front(&self, ds: &mut DecodeStream, codes: &[i32]) -> Result<()> {
        let st = &self.st;
        let s = self.s();
        let null = std::ptr::null::<f32>();
        let null16 = std::ptr::null_mut::<u16>();
        let n = codes.len() / NCB;
        ds.codes.upload_at(0, codes, st)?;
        launched(unsafe { ffi::dr_rvq_sum(f(&ds.codes) as *const i32, n as i32, NCB as i32, self.p32("rvq.table")?, f(&ds.x), D as i32, SEM_SIZE, RES_SIZE, s) }, "rvq")?;
        self.transformer("post", 8, f(&ds.x), n, WINDOW, ds.frames_done, &ds.caches, RING, &ds.scr)?;
        let mut rows = n;
        for u in 0..2 {
            let p = format!("up{u}.");
            launched(unsafe { ffi::dr_tconv_gather(f(&ds.x), rows as i32, D as i32, 0, null, h(&ds.scr.x16), s) }, "gather")?;
            self.blas.gemm_f16(h(&ds.scr.x16), self.p16(&(p.clone() + "w"))?, f(&ds.scr.y), rows, 2 * D, D, 0.0)?;
            rows *= 2;
            let stage = &ds.up[u];
            self.epilogue(f(&ds.scr.y), rows * D, D, self.p32(&(p.clone() + "b"))?, null, null, 0, stage.cur(), null16)?;
            self.convnext(&(p + "cnx"), stage.cur(), f(&ds.x), rows, &ds.scr)?;
            stage.shift(rows, st)?;
        }
        ds.frames_done += n;
        Ok(())
    }

    /// The decoder convolution stack over n latent rows (86 Hz) -> n * 512 samples.
    fn convs(&self, ds: &mut DecodeStream, latent: *const f32, n: usize) -> Result<Vec<f32>> {
        let st = &self.st;
        let s = self.s();
        let null = std::ptr::null::<f32>();
        let null16 = std::ptr::null_mut::<u16>();
        check(unsafe { ffi::dr_d2d(ds.stage_in.cur() as *mut std::ffi::c_void, latent as *const std::ffi::c_void, n * D * 4, s) }, "d2d")?;
        launched(unsafe { ffi::dr_im2col(ds.stage_in.cur(), n as i32, D as i32, 7, 1, 1, null, h(&ds.col), s) }, "im2col")?;
        self.blas.gemm_f16(h(&ds.col), self.p16("dec.in.w")?, f(&ds.gem), n, DEC_IN, 7 * D, 0.0)?;
        self.epilogue(f(&ds.gem), n * DEC_IN, DEC_IN, self.p32("dec.in.b")?, null, null, 0, ds.blocks[0].0.cur(), null16)?;
        ds.stage_in.shift(n, st)?;
        let mut m = n;
        for b in 0..ds.blocks.len() {
            let (tci, rus, ci, co, r) = &ds.blocks[b];
            let (ci, co, r) = (*ci, *co, *r);
            let p = format!("dec.b{b}.");
            // snake + transposed conv (k = 2r): m rows -> m*r rows
            launched(unsafe { ffi::dr_tconv_gather(tci.cur(), m as i32, ci as i32, 1, self.p32(&(p.clone() + "alpha"))?, h(&ds.col), s) }, "gather")?;
            self.blas.gemm_f16(h(&ds.col), self.p16(&(p.clone() + "up.w"))?, f(&ds.gem), m, r * co, 2 * ci, 0.0)?;
            tci.shift(m, st)?;
            m *= r;
            self.epilogue(f(&ds.gem), m * co, co, self.p32(&(p.clone() + "up.b"))?, null, null, 0, rus[0].cur(), null16)?;
            for u in 0..3 {
                let x_in = &rus[u];
                let q = format!("{p}ru{u}.");
                let dil = DILATIONS[u];
                launched(unsafe { ffi::dr_im2col(x_in.cur(), m as i32, co as i32, 7, dil as i32, 1, self.p32(&(q.clone() + "a1"))?, h(&ds.col), s) }, "im2col")?;
                self.blas.gemm_f16(h(&ds.col), self.p16(&(q.clone() + "c1.w"))?, f(&ds.gem), m, co, 7 * co, 0.0)?;
                self.epilogue(f(&ds.gem), m * co, co, self.p32(&(q.clone() + "c1.b"))?, null, null, 0, f(&ds.tmp), null16)?;
                launched(unsafe { ffi::dr_tconv_gather(f(&ds.tmp), m as i32, co as i32, 0, self.p32(&(q.clone() + "a2"))?, h(&ds.col), s) }, "gather")?;
                self.blas.gemm_f16(h(&ds.col), self.p16(&(q.clone() + "c2.w"))?, f(&ds.gem), m, co, co, 0.0)?;
                let dst = if u < 2 { rus[u + 1].cur() } else if b + 1 < ds.blocks.len() { ds.blocks[b + 1].0.cur() } else { ds.out_stage.cur() };
                self.epilogue(f(&ds.gem), m * co, co, self.p32(&(q + "c2.b"))?, null, x_in.cur(), 0, dst, null16)?;
                x_in.shift(m, st)?;
            }
        }
        launched(
            unsafe {
                ffi::dr_conv_out(ds.out_stage.cur(), m as i32, ds.out_stage.c as i32, 7, self.p32("dec.out.alpha")?, self.p32("dec.out.w")?, self.p32("dec.out.b")?, f(&ds.audio), s)
            },
            "conv_out",
        )?;
        ds.out_stage.shift(m, st)?;
        ds.audio.download::<f32>(m, st)
    }

    /// Decode a whole utterance ([frames][10] codes) to samples in [-1, 1].
    pub fn decode(&self, codes: &[i32]) -> Result<Vec<f32>> {
        let mut ds = self.stream()?;
        self.decode_chunk(&mut ds, codes)
    }
}

pub use crate::audio_io::{write_wav, resample, read_wav, read_wav_bytes};
