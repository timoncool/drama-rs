//! HTTP server: `POST /v1/speak` renders text to WAV (whole, or streamed while it is generated).
//!
//! The GPU engine is single-tenant, so one worker thread owns the [`Tts`] and requests queue for it; the ntex
//! handlers only parse requests and forward audio. Endpoints:
//!
//! - `POST /v1/speak`: JSON `{"text": ..., "voice": "name", "stream": false, "temperature": 1.0, "top_k": 30,
//!   "top_p": 0.9, "seed": 0, "split": true, "max_new": 2048}` (only `text` is required), or the text itself as a
//!   plain body. For dialogue, `"voices": ["alice", "bob"]` clones alice as `<|speaker:0|>` and bob as
//!   `<|speaker:1|>` (or `{"2": "alice"}` to choose the numbers); speaker tags and inline tags like `[laughs]` in
//!   `text` work as on the command line. Returns `audio/wav`, 44.1 kHz mono 16-bit. With `"stream": true` (or `?stream=1`) the WAV is sent
//!   as it is generated, with an open-ended header that players such as `pw-play`, `ffplay` and `mpv` accept.
//! - `POST /v1/voices?name=NAME&text=TRANSCRIPT`: body is a WAV clip; registers a voice to clone. Without `text`
//!   the clip is transcribed first when `fluid-speech` is installed.
//! - `GET /v1/voices`: registered voice names. `GET /health`: `ok`.
//! - `POST /v1/transcribe`: body is an audio clip; returns `{"text": ...}`. Runs the `fluid-speech` binary (found on
//!   `PATH`, or named by `$FLUID_SPEECH`) on the CPU (`$DRAMA_TRANSCRIBE_DEVICE` to change), because the GPU is
//!   taken by the TTS model. `501` when it is not installed. `GET /v1/info` says whether it is.
//! - `GET /`: a web UI for all of the above (built from `ui/` and embedded). `GET /openapi.json`: the API as
//!   OpenAPI 3.1, and `GET /docs` renders it (Scalar, loaded from a CDN).

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::mpsc as std_mpsc;
use std::task::{Context, Poll};

use anyhow::{anyhow, Result};
use ntex::http::StatusCode;
use ntex::util::{Bytes, Stream};
use ntex::web::{self, App, HttpRequest, HttpResponse, HttpServer};
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};

use crate::codec::SAMPLE_RATE;
use crate::engine::SampleParams;
use crate::prompt::Reference;
use crate::tts::{SpeakOpts, Tts};

/// Generation defaults for requests that leave a field out.
#[derive(Clone, Copy)]
pub struct Defaults {
    pub params: SampleParams,
    pub max_new: usize,
    pub first_chunk: usize,
    pub chunk: usize,
    pub split: bool,
}

#[derive(Deserialize, Default)]
struct SpeakRequest {
    text: String,
    voice: Option<String>,
    /// Several voices for multi-speaker text: `["alice", "bob"]` makes alice speaker 0 and bob speaker 1, or
    /// `{"0": "alice", "3": "bob"}` picks the speaker numbers
    voices: Option<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    split: Option<bool>,
    max_new: Option<usize>,
}

#[derive(Deserialize)]
struct SpeakQuery {
    stream: Option<String>,
    voice: Option<String>,
}

#[derive(Deserialize)]
struct VoiceQuery {
    name: String,
    text: Option<String>,
}

const FLUID_SPEECH_URL: &str = "https://fluid-speech-rs.apiplant.com";

/// The `fluid-speech` binary: `$FLUID_SPEECH` if set, else the first one on `PATH`. Looked up on every call, so
/// installing it while the server runs is enough.
fn fluid_speech() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("FLUID_SPEECH").filter(|p| !p.is_empty()) {
        return Some(p.into());
    }
    let exe = if cfg!(windows) { "fluid-speech.exe" } else { "fluid-speech" };
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(exe)).find(|p| p.is_file())
}

fn audio_extension(b: &[u8]) -> &'static str {
    match b {
        [b'R', b'I', b'F', b'F', ..] => "wav",
        [b'f', b'L', b'a', b'C', ..] => "flac",
        [b'O', b'g', b'g', b'S', ..] => "ogg",
        [b'I', b'D', b'3', ..] | [0xff, 0xfb | 0xf3 | 0xf2, ..] => "mp3",
        [_, _, _, _, b'f', b't', b'y', b'p', ..] => "m4a",
        _ => "wav",
    }
}

/// Transcribes an audio clip with `fluid-speech`. Blocking: it writes a temp file, runs the binary and reads its stdout.
fn transcribe_blocking(bin: &std::path::Path, audio: &[u8]) -> Result<String> {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("drama-transcribe-{}-{n}.{}", std::process::id(), audio_extension(audio)));
    std::fs::write(&path, audio)?;
    let device = std::env::var("DRAMA_TRANSCRIBE_DEVICE").unwrap_or_else(|_| "cpu".into());
    let out = std::process::Command::new(bin).args(["--device", &device]).arg(&path).output();
    let _ = std::fs::remove_file(&path);
    let out = out.map_err(|e| anyhow!("could not run {}: {e}", bin.display()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("fluid-speech failed: {}", err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("no output"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).split_whitespace().collect::<Vec<_>>().join(" "))
}

/// [`transcribe_blocking`] on its own thread; `None` when `fluid-speech` is not installed.
async fn transcribe(audio: Bytes) -> Option<Result<String>> {
    let bin = fluid_speech()?;
    let (tx, rx) = oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(transcribe_blocking(&bin, &audio));
    });
    Some(rx.await.unwrap_or_else(|_| Err(anyhow!("transcription thread died"))))
}

fn transcription_disabled() -> HttpResponse {
    error(StatusCode::NOT_IMPLEMENTED, format!("transcription is disabled: install fluid-speech from {FLUID_SPEECH_URL} and put it on PATH"))
}

type Chunk = Result<Bytes, std::io::Error>;

enum Job {
    Speak { req: SpeakRequest, out: mpsc::UnboundedSender<Chunk>, started: oneshot::Sender<Result<()>> },
    AddVoice { name: String, text: String, wav: Bytes, done: oneshot::Sender<Result<()>> },
    Voices { done: oneshot::Sender<Vec<String>> },
}

fn to_pcm16(audio: &[f32]) -> Bytes {
    let mut b = Vec::with_capacity(audio.len() * 2);
    for &s in audio {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    Bytes::from(b)
}

/// A 44-byte WAV header for 16-bit mono at [`SAMPLE_RATE`]; `data_len` `None` = unknown (streaming).
fn wav_header(data_len: Option<u32>) -> Bytes {
    let data = data_len.unwrap_or(u32::MAX - 36);
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36u32.wrapping_add(data)).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    h.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    Bytes::from(h)
}

/// `transcript` with its leading speaker tag set to `speaker`, so the reference clones that speaker.
fn retag(transcript: &str, speaker: u32) -> String {
    let t = transcript.trim_start();
    let rest = match crate::prompt::leading_speaker(t) {
        Some(_) => t.find("|>").map(|i| &t[i + 2..]).unwrap_or(t).trim_start(),
        None => t,
    };
    format!("<|speaker:{speaker}|> {rest}")
}

/// The references a request asks for: `voice` (as registered), and/or `voices` (list or speaker map).
fn resolve_voices(req: &SpeakRequest, voices: &HashMap<String, Reference>) -> Result<Vec<Reference>> {
    let get = |name: &str| voices.get(name).ok_or_else(|| anyhow!("unknown voice {name:?} (POST /v1/voices to add it)"));
    let mut refs = Vec::new();
    if let Some(v) = &req.voice {
        refs.push(get(v)?.clone());
    }
    let pairs: Vec<(u32, String)> = match &req.voices {
        None => vec![],
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .enumerate()
            .map(|(i, v)| v.as_str().map(|n| (i as u32, n.to_string())).ok_or_else(|| anyhow!("voices: names must be strings")))
            .collect::<Result<_>>()?,
        Some(serde_json::Value::Object(m)) => m
            .iter()
            .map(|(k, v)| {
                let n: u32 = k.parse().map_err(|_| anyhow!("voices: keys are speaker numbers, got {k:?}"))?;
                let name = v.as_str().ok_or_else(|| anyhow!("voices: names must be strings"))?;
                Ok((n, name.to_string()))
            })
            .collect::<Result<_>>()?,
        Some(_) => anyhow::bail!("voices: a list of names or {{\"speaker\": name}} map"),
    };
    for (speaker, name) in pairs {
        if speaker > 15 {
            anyhow::bail!("voices: speaker {speaker} out of range 0..15");
        }
        let mut r = get(&name)?.clone();
        r.transcript = retag(&r.transcript, speaker);
        refs.push(r);
    }
    Ok(refs)
}

/// The engine thread: runs jobs one at a time.
fn worker(mut tts: Tts, mut voices: HashMap<String, Reference>, defaults: Defaults, jobs: std_mpsc::Receiver<Job>) {
    for job in jobs {
        match job {
            Job::Voices { done } => {
                let mut names: Vec<String> = voices.keys().cloned().collect();
                names.sort();
                let _ = done.send(names);
            }
            Job::AddVoice { name, text, wav, done } => {
                let r = crate::codec::read_wav_bytes(&wav)
                    .and_then(|(audio, sr)| tts.reference_audio(&audio, sr, &text))
                    .map(|reference| {
                        eprintln!("voice {name:?} registered");
                        voices.insert(name, reference);
                    });
                let _ = done.send(r);
            }
            Job::Speak { req, out, started } => {
                let refs = match resolve_voices(&req, &voices) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = started.send(Err(e));
                        continue;
                    }
                };
                let d = defaults;
                let opts = SpeakOpts {
                    params: SampleParams {
                        temperature: req.temperature.unwrap_or(d.params.temperature),
                        top_k: req.top_k.unwrap_or(d.params.top_k),
                        top_p: req.top_p.unwrap_or(d.params.top_p),
                        seed: req.seed.unwrap_or(d.params.seed),
                        ..d.params
                    },
                    max_new: req.max_new.unwrap_or(d.max_new),
                    first_chunk: d.first_chunk,
                    chunk: d.chunk,
                    split: req.split.unwrap_or(d.split),
                    keep_stalled: true,
                };
                let _ = started.send(Ok(()));
                let sink = |audio: &[f32]| -> Result<()> {
                    out.send(Ok(to_pcm16(audio))).map_err(|_| anyhow!("client went away"))
                };
                match tts.speak(&req.text, &refs, &opts, sink) {
                    Ok(s) => eprintln!(
                        "spoke {:.2}s of audio in {:.2}s ({} chars)",
                        s.audio.len() as f64 / SAMPLE_RATE as f64,
                        s.elapsed.as_secs_f64(),
                        req.text.chars().count()
                    ),
                    Err(e) => {
                        eprintln!("speak failed: {e:#}");
                        let _ = out.send(Err(std::io::Error::other(e.to_string())));
                    }
                }
            }
        }
    }
}

/// Body stream over the worker's audio chunks, optionally preceded by the WAV header.
struct AudioStream {
    header: Option<Bytes>,
    rx: mpsc::UnboundedReceiver<Chunk>,
}

impl Stream for AudioStream {
    type Item = Chunk;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Chunk>> {
        if let Some(h) = self.header.take() {
            return Poll::Ready(Some(Ok(h)));
        }
        self.rx.poll_recv(cx)
    }
}

/// The web UI's files (`ui/dist`) as `(path, bytes)`, embedded by build.rs; empty when it was not built.
static UI: &[(&str, &[u8])] = include!(concat!(env!("OUT_DIR"), "/ui.rs"));

const DOCS_HTML: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>drama API</title>
  </head>
  <body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@1"></script>
  </body>
</html>
"#;

fn openapi() -> String {
    include_str!("openapi.json").replacen("\"VERSION\"", &format!("{:?}", env!("CARGO_PKG_VERSION")), 1)
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("json") => "application/json",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

async fn ui(req: HttpRequest) -> HttpResponse {
    let path = req.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match UI.iter().find(|(p, _)| *p == path) {
        Some((p, bytes)) => {
            let cache = if p.starts_with("assets/") { "public, max-age=31536000, immutable" } else { "no-cache" };
            HttpResponse::Ok().content_type(content_type(p)).header("cache-control", cache).body(Bytes::from_static(bytes))
        }
        None if UI.is_empty() && path == "index.html" => error(
            StatusCode::NOT_FOUND,
            "this drama was built without its web UI: run `pnpm install && pnpm build` in ui/, then rebuild",
        ),
        None => error(StatusCode::NOT_FOUND, "not found"),
    }
}

fn error(status: StatusCode, msg: impl std::fmt::Display) -> HttpResponse {
    HttpResponse::builder(status).content_type("text/plain; charset=utf-8").body(format!("{msg}\n"))
}

async fn speak(jobs: std_mpsc::Sender<Job>, req: HttpRequest, body: Bytes) -> HttpResponse {
    let query = web::types::Query::<SpeakQuery>::from_query(req.query_string()).ok();
    let is_json = req.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains("json"))
        || body.first() == Some(&b'{');
    let mut sreq = if is_json {
        match serde_json::from_slice::<SpeakRequest>(&body) {
            Ok(r) => r,
            Err(e) => return error(StatusCode::BAD_REQUEST, format!("bad JSON: {e}")),
        }
    } else {
        SpeakRequest { text: String::from_utf8_lossy(&body).into_owned(), ..Default::default() }
    };
    if let Some(q) = query {
        let q = q.into_inner();
        if let Some(s) = q.stream {
            sreq.stream = !matches!(s.as_str(), "0" | "false" | "no");
        }
        if q.voice.is_some() {
            sreq.voice = q.voice;
        }
    }
    if sreq.text.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "no text");
    }
    let stream = sreq.stream;
    let (out, mut rx) = mpsc::unbounded_channel();
    let (started, started_rx) = oneshot::channel();
    if jobs.send(Job::Speak { req: sreq, out, started }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match started_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, e),
        Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
    if stream {
        return HttpResponse::Ok()
            .content_type("audio/wav")
            .streaming(AudioStream { header: Some(wav_header(None)), rx });
    }
    let mut pcm = Vec::new();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            Ok(b) => pcm.extend_from_slice(&b),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
    let mut wav = wav_header(Some(pcm.len() as u32)).to_vec();
    wav.extend_from_slice(&pcm);
    HttpResponse::Ok().content_type("audio/wav").body(wav)
}

async fn add_voice(jobs: std_mpsc::Sender<Job>, req: HttpRequest, body: Bytes) -> HttpResponse {
    let q = match web::types::Query::<VoiceQuery>::from_query(req.query_string()) {
        Ok(q) => q.into_inner(),
        Err(_) => return error(StatusCode::BAD_REQUEST, "need ?name=NAME&text=TRANSCRIPT and a WAV body"),
    };
    let text = match q.text.filter(|t| !t.trim().is_empty()) {
        Some(t) => t,
        None => match transcribe(body.clone()).await {
            None => return error(StatusCode::BAD_REQUEST, format!("no transcript given, and transcription is disabled: pass ?text= or install fluid-speech from {FLUID_SPEECH_URL}")),
            Some(Err(e)) => return error(StatusCode::BAD_REQUEST, format!("{e:#}")),
            Some(Ok(t)) if t.is_empty() => return error(StatusCode::BAD_REQUEST, "no speech found in the clip"),
            Some(Ok(t)) => t,
        },
    };
    let (done, rx) = oneshot::channel();
    if jobs.send(Job::AddVoice { name: q.name, text, wav: body, done }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match rx.await {
        Ok(Ok(())) => HttpResponse::Ok().body("ok\n"),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

async fn transcribe_route(body: Bytes) -> HttpResponse {
    if body.is_empty() {
        return error(StatusCode::BAD_REQUEST, "the body must be an audio clip");
    }
    match transcribe(body).await {
        None => transcription_disabled(),
        Some(Ok(text)) => HttpResponse::Ok().json(&serde_json::json!({ "text": text })),
        Some(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn info() -> HttpResponse {
    let bin = fluid_speech();
    HttpResponse::Ok().json(&serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "sample_rate": SAMPLE_RATE,
        "transcription": {
            "enabled": bin.is_some(),
            "binary": bin.map(|p| p.display().to_string()),
            "install_url": FLUID_SPEECH_URL,
        },
    }))
}

async fn list_voices(jobs: std_mpsc::Sender<Job>) -> HttpResponse {
    let (done, rx) = oneshot::channel();
    if jobs.send(Job::Voices { done }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match rx.await {
        Ok(names) => HttpResponse::Ok().json(&names),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

/// Load the engine with `load` on a dedicated thread (CUDA state stays on the thread that created it), then serve
/// it on `addr` until the process is stopped (Ctrl-C).
pub fn run<F>(load: F, defaults: Defaults, addr: &str) -> Result<()>
where
    F: FnOnce() -> Result<(Tts, HashMap<String, Reference>)> + Send + 'static,
{
    let (jobs, rx) = std_mpsc::channel::<Job>();
    let (ready_tx, ready) = std_mpsc::channel::<Result<()>>();
    std::thread::Builder::new().name("drama-engine".into()).spawn(move || match load() {
        Ok((tts, voices)) => {
            let _ = ready_tx.send(Ok(()));
            worker(tts, voices, defaults, rx)
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e));
        }
    })?;
    ready.recv().map_err(|_| anyhow!("engine thread died while loading"))??;
    serve(jobs, addr.to_string())?;
    Ok(())
}

#[ntex::main(name = "drama", signals = true)]
async fn serve(jobs: std_mpsc::Sender<Job>, addr: String) -> std::io::Result<()> {
    eprintln!("listening on http://{addr}  (web UI at /, API docs at /docs)");
    HttpServer::new(async move |_| {
        let (j1, j2, j3) = (jobs.clone(), jobs.clone(), jobs.clone());
        // voice clips are uploaded whole: allow a few minutes of 48 kHz stereo 16-bit WAV
        let cfg = web::WebAppConfig::new().set_state(web::types::PayloadConfig::new(64 << 20));
        App::new()
            .with_config(cfg)
            .service(web::resource("/health").to(async || "ok\n"))
            .service(web::resource("/openapi.json").to(async || HttpResponse::Ok().content_type("application/json").body(openapi())))
            .service(web::resource("/docs").to(async || HttpResponse::Ok().content_type("text/html; charset=utf-8").body(DOCS_HTML)))
            .service(web::resource("/v1/speak").route(web::post().to(move |req: HttpRequest, body: Bytes| speak(j1.clone(), req, body))))
            .service(
                web::resource("/v1/voices")
                    .route(web::post().to(move |req: HttpRequest, body: Bytes| add_voice(j2.clone(), req, body)))
                    .route(web::get().to(move || list_voices(j3.clone()))),
            )
            .service(web::resource("/v1/transcribe").route(web::post().to(transcribe_route)))
            .service(web::resource("/v1/info").route(web::get().to(info)))
            .default_service(web::get().to(ui))
    })
    .bind(addr.as_str(), ntex::SharedCfg::new("drama"))?
    .workers(1)
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffs_audio_extensions() {
        assert_eq!(audio_extension(b"RIFF....WAVE"), "wav");
        assert_eq!(audio_extension(b"fLaC...."), "flac");
        assert_eq!(audio_extension(b"ID3\x04"), "mp3");
        assert_eq!(audio_extension(b"\0\0\0\x20ftypM4A "), "m4a");
    }

    /// Runs the real `fluid-speech` when it is installed; skipped otherwise.
    #[test]
    fn transcribes_a_clip_when_fluid_speech_is_installed() {
        let Some(bin) = fluid_speech() else { return };
        let sr = 16000u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + sr * 2).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        for v in [16u32, 0x0001_0001, sr, sr * 2, 0x0010_0002] {
            wav.extend_from_slice(&v.to_le_bytes());
        }
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(sr * 2).to_le_bytes());
        wav.extend(std::iter::repeat(0u8).take(sr as usize * 2));
        // silence must come back as an empty transcript, not an error
        assert_eq!(transcribe_blocking(&bin, &wav).unwrap(), "");
    }
}
