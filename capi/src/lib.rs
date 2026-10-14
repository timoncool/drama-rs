//! C ABI over drama-rs for programs that load the engine at run time (`drama_engine.h`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use drama::codec;
use drama::engine::{Cancelled, SampleParams};
use drama::prompt::Reference;
use drama::tts::{SpeakOpts, Tts};

pub const DRAMA_OK: i32 = 0;
pub const DRAMA_ERROR: i32 = 1;
pub const DRAMA_CANCELLED: i32 = 2;

#[repr(C)]
pub struct DramaParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u64,
    pub max_new: u32,
}

/// A reference clip is encoded again only when its file or transcript changes.
#[derive(PartialEq, Eq, Hash)]
struct RefKey {
    path: PathBuf,
    transcript: String,
    bytes: u64,
    modified: Option<SystemTime>,
}

pub struct DramaEngine {
    tts: Mutex<Tts>,
    stop: Arc<AtomicBool>,
    refs: Mutex<HashMap<RefKey, Reference>>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: &str) {
    let text = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = text);
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| payload.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
}

unsafe fn text_arg(ptr: *const c_char, what: &str) -> Result<String, String> {
    if ptr.is_null() {
        return Err(format!("{what} is null"));
    }
    CStr::from_ptr(ptr).to_str().map(str::to_string).map_err(|_| format!("{what} is not UTF-8"))
}

#[no_mangle]
pub extern "C" fn drama_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn drama_sample_rate() -> u32 {
    codec::SAMPLE_RATE
}

#[no_mangle]
pub extern "C" fn drama_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// # Safety
/// `model_dir` is a NUL-terminated UTF-8 path.
#[no_mangle]
pub unsafe extern "C" fn drama_load(model_dir: *const c_char, max_len: u32, prefill_chunk: u32) -> *mut DramaEngine {
    let loaded = catch_unwind(AssertUnwindSafe(|| -> Result<DramaEngine, String> {
        let dir = text_arg(model_dir, "model_dir")?;
        let tts = Tts::load(Path::new(&dir), max_len.max(256) as usize, prefill_chunk.max(16) as usize).map_err(|e| format!("{e:#}"))?;
        let stop = tts.engine.stop.clone();
        Ok(DramaEngine { tts: Mutex::new(tts), stop, refs: Mutex::new(HashMap::new()) })
    }));
    match loaded {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)),
        Ok(Err(e)) => {
            set_error(&e);
            std::ptr::null_mut()
        }
        Err(p) => {
            set_error(&panic_text(p.as_ref()));
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `engine` comes from `drama_load` and is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn drama_free(engine: *mut DramaEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// # Safety
/// `engine` is live; may be called from any thread.
#[no_mangle]
pub unsafe extern "C" fn drama_cancel(engine: *mut DramaEngine) {
    if let Some(e) = engine.as_ref() {
        e.stop.store(true, Ordering::Relaxed);
    }
}

fn reference(engine: &DramaEngine, tts: &Tts, wav: &str, transcript: &str) -> Result<Reference, String> {
    let path = PathBuf::from(wav);
    let meta = std::fs::metadata(&path).map_err(|e| format!("{wav}: {e}"))?;
    let key = RefKey { path: path.clone(), transcript: transcript.to_string(), bytes: meta.len(), modified: meta.modified().ok() };
    let mut refs = engine.refs.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(r) = refs.get(&key) {
        return Ok(r.clone());
    }
    let r = tts.reference(&path, transcript).map_err(|e| format!("{wav}: {e:#}"))?;
    refs.insert(key, r.clone());
    Ok(r)
}

/// # Safety
/// `engine` is live, the strings are NUL-terminated UTF-8, `params`, `out_audio` and `out_len` are valid.
#[no_mangle]
pub unsafe extern "C" fn drama_speak(
    engine: *mut DramaEngine, text: *const c_char, ref_wav: *const c_char, ref_text: *const c_char, params: *const DramaParams,
    out_audio: *mut *mut f32, out_len: *mut usize,
) -> i32 {
    speak(engine, text, ref_wav, ref_text, params, |_| {}, out_audio, out_len)
}

/// Called with each decoded piece of audio while `drama_speak_stream` runs, on the engine's codec thread.
pub type DramaChunkFn = Option<unsafe extern "C" fn(audio: *const f32, len: usize, user: *mut c_void)>;

/// # Safety
/// As `drama_speak`; `on_chunk` (may be null) is safe to call from another thread with `user`.
#[no_mangle]
pub unsafe extern "C" fn drama_speak_stream(
    engine: *mut DramaEngine, text: *const c_char, ref_wav: *const c_char, ref_text: *const c_char, params: *const DramaParams,
    on_chunk: DramaChunkFn, user: *mut c_void, out_audio: *mut *mut f32, out_len: *mut usize,
) -> i32 {
    let user = UserPtr(user);
    speak(engine, text, ref_wav, ref_text, params, move |a: &[f32]| {
        if let Some(f) = on_chunk {
            f(a.as_ptr(), a.len(), user.get());
        }
    }, out_audio, out_len)
}

#[derive(Clone, Copy)]
struct UserPtr(*mut c_void);
unsafe impl Send for UserPtr {}

impl UserPtr {
    fn get(self) -> *mut c_void {
        self.0
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn speak(
    engine: *mut DramaEngine, text: *const c_char, ref_wav: *const c_char, ref_text: *const c_char, params: *const DramaParams,
    mut on_chunk: impl FnMut(&[f32]) + Send, out_audio: *mut *mut f32, out_len: *mut usize,
) -> i32 {
    let Some(engine) = engine.as_ref() else {
        set_error("engine is null");
        return DRAMA_ERROR;
    };
    if params.is_null() || out_audio.is_null() || out_len.is_null() {
        set_error("params or output is null");
        return DRAMA_ERROR;
    }
    let p = &*params;
    let spoken = catch_unwind(AssertUnwindSafe(|| -> Result<Vec<f32>, (i32, String)> {
        let fail = |e: String| (DRAMA_ERROR, e);
        let text = text_arg(text, "text").map_err(fail)?;
        let wav = text_arg(ref_wav, "ref_wav").map_err(fail)?;
        let transcript = text_arg(ref_text, "ref_text").map_err(fail)?;
        let mut tts = engine.tts.lock().unwrap_or_else(|p| p.into_inner());
        engine.stop.store(false, Ordering::Relaxed);
        let voice = reference(engine, &tts, &wav, &transcript).map_err(fail)?;
        let opts = SpeakOpts {
            params: SampleParams { temperature: p.temperature, top_k: p.top_k, top_p: p.top_p, pad: 0, seed: p.seed },
            max_new: p.max_new.max(1) as usize,
            first_chunk: 16,
            chunk: 16,
            split: true,
            keep_stalled: false,
        };
        let result = tts.speak(&text, &[voice], &opts, |a| {
            on_chunk(a);
            Ok(())
        });
        engine.stop.store(false, Ordering::Relaxed);
        match result {
            Ok(s) => Ok(s.audio),
            Err(e) if e.downcast_ref::<Cancelled>().is_some() => Err((DRAMA_CANCELLED, e.to_string())),
            Err(e) => Err((DRAMA_ERROR, format!("{e:#}"))),
        }
    }));
    match spoken {
        Ok(Ok(audio)) => {
            let boxed = audio.into_boxed_slice();
            *out_len = boxed.len();
            *out_audio = Box::into_raw(boxed) as *mut f32;
            DRAMA_OK
        }
        Ok(Err((code, e))) => {
            set_error(&e);
            code
        }
        Err(p) => {
            set_error(&panic_text(p.as_ref()));
            DRAMA_ERROR
        }
    }
}

/// # Safety
/// `audio` and `len` are exactly what one `drama_speak` returned.
#[no_mangle]
pub unsafe extern "C" fn drama_free_audio(audio: *mut f32, len: usize) {
    if !audio.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(audio, len)));
    }
}
