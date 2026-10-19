import { For, Show, createEffect, createMemo, createSignal, onSettled, untrack } from "solid-js";
import { info as fetchInfo, listVoices, speak, type Info, type SpeakRequest } from "./api";
import { StreamPlayer, fmtTime, peaksOf, wavBlob } from "./audio";
import { VoiceDialog } from "./VoiceDialog";
import {
  btn, btnGhost, btnIcon, btnLg, btnPrimary, btnSm, btnSmIcon, btnStop, chipBtn, dot, editorBack, editorInput, errorBox, input, kbd,
  panel, panelHead, panelTitle, sectionLabel, select, take, tokSpk, tokTag,
} from "./styles";
import { ClipPlayer, Icon, Logo, Slider, Toggle, Waveform, speakerColor, voiceColor } from "./ui";

interface Take {
  id: number;
  text: string;
  voice: string;
  url: string;
  peaks: Float32Array;
  seconds: number;
  took: number;
  partial: boolean;
  autoplay: boolean;
}

const TAGS = ["[laughs]", "[whispers]", "[sighs]", "[pause]", "[excited]", "[sad]"];
const DEFAULT_TEXT = "Hello! This is drama, speaking from your own GPU. [laughs] Pretty fast, right?";
const STORE = "drama.ui.v2";
let nextId = 0;

interface Saved {
  text: string;
  voice: string;
  cast: Record<number, string>;
  temperature: number;
  topK: number;
  topP: number;
  seed: string;
  stream: boolean;
  split: boolean;
}

function loadSaved(): Partial<Saved> {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "{}");
  } catch {
    return {};
  }
}

const SPEAKER_RE = /<\|speaker:(\d+)\|>/g;
const speakersIn = (t: string) => [...new Set([...t.matchAll(SPEAKER_RE)].map((m) => +m[1]))].sort((a, b) => a - b);

/** The script as text and highlighted tokens (speaker tags, [inline tags]) for the layer behind the editor. */
function tokens(t: string): { t: string; kind?: "spk" | "tag"; n?: number }[] {
  const out: { t: string; kind?: "spk" | "tag"; n?: number }[] = [];
  let at = 0;
  for (const m of t.matchAll(/<\|speaker:(\d+)\|>|\[[^\[\]\n]{1,40}\]/g)) {
    if (m.index > at) out.push({ t: t.slice(at, m.index) });
    out.push(m[1] !== undefined ? { t: m[0], kind: "spk", n: +m[1] } : { t: m[0], kind: "tag" });
    at = m.index + m[0].length;
  }
  out.push({ t: t.slice(at) + "\n" });
  return out;
}

export function App() {
  const saved = loadSaved();
  const [voices, setVoices] = createSignal<string[]>([]);
  const [info, setInfo] = createSignal<Info | null>(null);
  const [online, setOnline] = createSignal(true);
  const [text, setText] = createSignal(saved.text ?? DEFAULT_TEXT);
  const [voice, setVoice] = createSignal(saved.voice ?? "");
  const [cast, setCast] = createSignal<Record<number, string>>(saved.cast ?? {});
  const [stream, setStream] = createSignal(saved.stream ?? true);
  const [split, setSplit] = createSignal(saved.split ?? true);
  const [temperature, setTemperature] = createSignal(saved.temperature ?? 1.0);
  const [topK, setTopK] = createSignal(saved.topK ?? 30);
  const [topP, setTopP] = createSignal(saved.topP ?? 0.9);
  const [seed, setSeed] = createSignal(saved.seed ?? "");
  const [busy, setBusy] = createSignal<"" | "waiting" | "buffering" | "streaming" | "rendering">("");
  const [elapsed, setElapsed] = createSignal(0);
  const [error, setError] = createSignal("");
  const [takes, setTakes] = createSignal<Take[]>([]);
  const [live, setLive] = createSignal<Float32Array>(new Float32Array());
  const [liveSecs, setLiveSecs] = createSignal(0);
  const [dialog, setDialog] = createSignal<{ file: File | null } | null>(null);
  const [dragOver, setDragOver] = createSignal(false);

  let abort: AbortController | null = null;
  let player: StreamPlayer | null = null;
  let ctx: AudioContext | null = null;
  let ta!: HTMLTextAreaElement;
  let back!: HTMLDivElement;
  let touched = false;

  const speakers = createMemo(() => speakersIn(text()));
  const chars = createMemo(() => text().replace(/<\|speaker:\d+\|>/g, "").length);
  const estSeconds = () => chars() / 14;
  const segments = createMemo(() => tokens(text()));

  const refresh = () =>
    listVoices()
      .then(setVoices)
      .catch((e) => setError(`Could not list voices: ${e.message}`));
  const refreshInfo = () =>
    fetchInfo()
      .then(setInfo)
      .catch(() => {});

  createEffect(
    () => JSON.stringify({ text: text(), voice: voice(), cast: cast(), temperature: temperature(), topK: topK(), topP: topP(), seed: seed(), stream: stream(), split: split() }),
    (json) => {
      try {
        localStorage.setItem(STORE, json);
      } catch {}
    },
  );

  onSettled(() => {
    refresh();
    refreshInfo();
    const ping = () =>
      fetch("/health")
        .then((r) => setOnline(r.ok))
        .catch(() => setOnline(false));
    const timer = setInterval(ping, 8000);
    const key = (e: KeyboardEvent) => {
      if (dialog()) return;
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!busy()) generate();
      } else if (e.key === "Escape" && busy()) {
        stopAll();
      }
    };
    window.addEventListener("keydown", key);
    return () => {
      clearInterval(timer);
      window.removeEventListener("keydown", key);
    };
  });

  function request(): SpeakRequest {
    const req: SpeakRequest = { text: text(), stream: stream(), split: split(), temperature: temperature(), top_k: topK(), top_p: topP() };
    if (seed().trim() !== "") req.seed = Number(seed());
    const tags = speakers();
    if (tags.length) {
      const map: Record<string, string> = {};
      for (const n of tags) if (cast()[n]) map[String(n)] = cast()[n];
      if (Object.keys(map).length) req.voices = map;
    } else if (voice()) req.voice = voice();
    return req;
  }

  const voiceLabel = (req: SpeakRequest) =>
    req.voices && !Array.isArray(req.voices) ? [...new Set(Object.values(req.voices))].join(" + ") : (req.voice ?? "Default voice");

  function stopAll() {
    abort?.abort();
    player?.stop();
  }

  async function generate() {
    if (busy()) return stopAll();
    if (!text().trim()) return;
    setError("");
    setBusy(stream() ? "waiting" : "rendering");
    setLive(new Float32Array());
    setLiveSecs(0);
    setElapsed(0);
    abort = new AbortController();
    const req = request();
    const t0 = performance.now();
    const timer = setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    const bins: number[] = [];
    let carry = new Int16Array();
    const BIN = 882; // 20 ms
    let pcm: Uint8Array | null = null;
    let partial = false;
    try {
      const res = await speak(req, abort.signal);
      if (req.stream) {
        ctx ??= new AudioContext({ sampleRate: 44100 });
        await ctx.resume();
        player = new StreamPlayer(ctx, (chunk) => {
          if (busy() === "waiting") setBusy("buffering");
          const all = new Int16Array(carry.length + chunk.length);
          all.set(carry);
          all.set(chunk, carry.length);
          let i = 0;
          for (; i + BIN <= all.length; i += BIN) {
            let p = 0;
            for (let j = i; j < i + BIN; j++) p = Math.max(p, Math.abs(all[j]));
            bins.push(p / 32768);
          }
          carry = all.slice(i);
          setLive(Float32Array.from(bins));
          setLiveSecs(player!.samples / 44100);
        }, Math.max(2, chars() / 14), () => setBusy("streaming"));
        try {
          pcm = await player.play(res);
        } catch (e: any) {
          if (e.name !== "AbortError" || !player.samples) throw e;
          partial = true;
          pcm = new Uint8Array(player.chunks.reduce((a, c) => a + c.length, 0));
          let o = 0;
          for (const c of player.chunks) {
            pcm.set(c, o);
            o += c.length;
          }
        }
      } else {
        pcm = new Uint8Array(await res.arrayBuffer()).subarray(44);
      }
      if (pcm.length) {
        const blob = wavBlob(pcm);
        setTakes((t) => [
          {
            id: nextId++,
            text: req.text,
            voice: voiceLabel(req),
            url: URL.createObjectURL(blob),
            peaks: peaksOf(pcm!),
            seconds: pcm!.length / 2 / 44100,
            took: (performance.now() - t0) / 1000,
            partial,
            autoplay: !req.stream,
          },
          ...t,
        ]);
      }
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      clearInterval(timer);
      setBusy("");
      abort = null;
      player = null;
    }
  }

  function removeTake(id: number) {
    const t = takes().find((x) => x.id === id);
    if (t) URL.revokeObjectURL(t.url);
    setTakes((all) => all.filter((x) => x.id !== id));
  }

  /** Insert `s` where the cursor is (or at the end if the editor was never used). */
  function insert(s: string, own = false) {
    const t = untrack(text);
    let a = touched ? ta.selectionStart : t.length;
    let b = touched ? ta.selectionEnd : t.length;
    let ins = s;
    if (own && a > 0 && t[a - 1] !== "\n") ins = "\n" + ins;
    const next = t.slice(0, a) + ins + t.slice(b);
    setText(next);
    const caret = a + ins.length;
    setTimeout(() => {
      ta.focus();
      ta.setSelectionRange(caret, caret);
      touched = true;
    }, 0);
  }

  const nextSpeaker = () => {
    const s = speakers();
    return s.length ? Math.min(15, s[s.length - 1] + 1) : 0;
  };

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer?.files[0];
    if (f) setDialog({ file: f });
  }

  const voiceSelect = (value: () => string, onChange: (v: string) => void, none: string) => (
    <div class="relative w-full">
      <select class={select} value={value()} onChange={(e) => onChange(e.currentTarget.value)}>
        <option value="">{none}</option>
        <For each={voices()}>{(v) => <option value={v}>{v}</option>}</For>
      </select>
      <Icon name="chevron" size={13} class="pointer-events-none absolute top-2.5 right-2.5 text-ink-400" />
    </div>
  );

  const status = () => {
    if (!online()) return { cls: "bg-rec shadow-[0_0_8px_var(--color-rec)]", label: "Server offline" };
    switch (busy()) {
      case "waiting":
        return { cls: "bg-warn animate-pulse", label: "Generating…" };
      case "rendering":
        return { cls: "bg-warn animate-pulse", label: "Rendering…" };
      case "buffering":
        return { cls: "bg-warn animate-pulse", label: "Buffering…" };
      case "streaming":
        return { cls: "bg-accent animate-pulse", label: "Streaming" };
      default:
        return { cls: "bg-ok shadow-[0_0_8px_var(--color-ok)]", label: "Engine ready" };
    }
  };

  return (
    <div class="flex min-h-full flex-col lg:h-full">
      {/* ---- top bar ---- */}
      <header class="flex h-12 flex-none items-center gap-4 border-b border-ink-750 bg-ink-900 px-4">
        <div class="flex items-center gap-2.5">
          <Logo size={24} />
          <span class="text-[15px] font-semibold tracking-tight text-ink-100">drama</span>
          <Show when={info()}>
            <span class="rounded bg-ink-800 px-1.5 py-px font-mono text-[10.5px] text-ink-400">v{info()!.version}</span>
          </Show>
        </div>
        <div class="flex-1" />
        <div class="flex items-center gap-2">
          <span class="inline-flex h-6 items-center gap-1.5 rounded-full border border-ink-700 bg-ink-850 px-2.5 text-xs text-ink-300">
            <span class={`${dot} ${status().cls}`} /> {status().label}
          </span>
          <Show when={info()}>
            <Show
              when={info()!.transcription.enabled}
              fallback={
                <a
                  class={chipBtn}
                  href={info()!.transcription.install_url}
                  target="_blank"
                  rel="noreferrer"
                  title="Transcription is disabled: install fluid-speech to transcribe voice clips automatically"
                >
                  <span class={`${dot} bg-ink-500`} /> Transcription off <Icon name="external" size={11} />
                </a>
              }
            >
              <span class="inline-flex h-6 items-center gap-1.5 rounded-full border border-ink-700 bg-ink-850 px-2.5 text-xs text-ink-300" title={info()!.transcription.binary ?? ""}>
                <span class={`${dot} bg-ok`} /> Transcription on
              </span>
            </Show>
          </Show>
          <a class={`${btnGhost} !h-7 !text-xs`} href="/docs" target="_blank">
            <Icon name="book" size={13} /> API
          </a>
        </div>
      </header>

      {/* ---- workspace ---- */}
      <div class="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[264px_minmax(0,1fr)_296px] lg:overflow-hidden">
        {/* voices */}
        <section
          class={`${panel} order-2 lg:order-1 ${dragOver() ? "border-accent bg-accent/5" : ""}`}
          onDragOver={(e) => {
            if (e.dataTransfer?.types.includes("Files")) {
              e.preventDefault();
              setDragOver(true);
            }
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <div class={panelHead}>
            <span class={panelTitle}>Voices</span>
            <span class="font-mono text-[11px] text-ink-500">{voices().length}</span>
            <div class="flex-1" />
            <button class={`${btnPrimary} !h-[26px] !rounded-[7px] !px-2.5 !text-xs`} onClick={() => setDialog({ file: null })}>
              <Icon name="plus" size={13} /> New
            </button>
          </div>
          <div class="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto p-2">
            <button class={voiceRow(voice() === "")} aria-pressed={voice() === "" ? "true" : "false"} onClick={() => setVoice("")}>
              <span class="grid size-8 flex-none place-items-center rounded-[9px] border border-dashed border-ink-600 text-ink-400">
                <Icon name="waves" />
              </span>
              <span class="min-w-0">
                <span class="block truncate text-ink-100">Default voice</span>
                <span class="block text-xs text-ink-400">No cloning</span>
              </span>
            </button>
            <For each={voices()}>
              {(v) => (
                <button class={voiceRow(voice() === v)} aria-pressed={voice() === v ? "true" : "false"} onClick={() => setVoice(v)}>
                  <span class="grid size-8 flex-none place-items-center rounded-[9px] text-[13px] font-bold text-ink-950" style={{ background: voiceColor(v) }}>
                    {v.slice(0, 1).toUpperCase()}
                  </span>
                  <span class="min-w-0 flex-1">
                    <span class="block truncate text-ink-100">{v}</span>
                    <span class="block text-xs text-ink-400">Cloned voice</span>
                  </span>
                  <Show when={voice() === v}>
                    <Icon name="check" size={14} class="text-accent" />
                  </Show>
                </button>
              )}
            </For>
            <Show when={!voices().length}>
              <div class="m-1 mt-3 rounded-xl border border-dashed border-ink-700 p-4 text-center text-ink-400">
                <Icon name="mic" size={20} class="mx-auto mb-2" />
                <p class="leading-snug">Clone a voice from a short clip. Drop an audio file here, or press <b class="text-ink-200">New</b>.</p>
              </div>
            </Show>
          </div>
        </section>

        {/* script + takes */}
        <div class="order-1 flex min-h-0 flex-col gap-3 lg:order-2">
          <section class={`${panel} min-h-[320px] flex-[1.15]`}>
            <div class={panelHead}>
              <span class={panelTitle}>Script</span>
              <div class="flex-1" />
              <div class="flex flex-wrap items-center gap-1.5">
                <For each={[0, 1, 2, 3]}>
                  {(n) => (
                    <button class={`${chipBtn} !px-2`} style={{ color: speakerColor(n) }} onMouseDown={(e) => e.preventDefault()} onClick={() => insert(`<|speaker:${n}|> `, true)} title={`Insert speaker ${n}`}>
                      S{n}
                    </button>
                  )}
                </For>
                <Show when={nextSpeaker() > 3}>
                  <button class={`${chipBtn} !px-2`} style={{ color: speakerColor(nextSpeaker()) }} onMouseDown={(e) => e.preventDefault()} onClick={() => insert(`<|speaker:${nextSpeaker()}|> `, true)}>
                    S{nextSpeaker()}
                  </button>
                </Show>
              </div>
            </div>
            <div class="flex flex-wrap items-center gap-1.5 border-b border-ink-750 px-3.5 py-2">
              <span class="mr-1 text-[11px] uppercase tracking-wider text-ink-500">Tags</span>
              <For each={TAGS}>
                {(t) => (
                  <button class={`${chipBtn} !text-warn/90`} onMouseDown={(e) => e.preventDefault()} onClick={() => insert(`${t} `)}>
                    {t}
                  </button>
                )}
              </For>
            </div>
            <div class="relative min-h-[160px] flex-1">
              <div class={editorBack} ref={back} aria-hidden="true">
                <For each={segments()}>
                  {(s) =>
                    s.kind === "spk" ? (
                      <span class={tokSpk} style={{ color: speakerColor(s.n!) }}>{s.t}</span>
                    ) : s.kind === "tag" ? (
                      <span class={tokTag}>{s.t}</span>
                    ) : (
                      s.t
                    )
                  }
                </For>
              </div>
              <textarea
                ref={ta}
                class={editorInput}
                value={text()}
                spellcheck={false}
                onInput={(e) => {
                  touched = true;
                  setText(e.currentTarget.value);
                }}
                onFocus={() => (touched = true)}
                onScroll={() => (back.scrollTop = ta.scrollTop)}
                placeholder="Type what should be said. Use [laughs] style tags, or the S0 / S1 buttons for a dialogue between several voices."
              />
            </div>
            <Show when={error()}>
              <div class={`${errorBox} mx-3 mb-2`}>
                <span class="flex-1 whitespace-pre-wrap">{error()}</span>
                <button class="cursor-pointer text-ink-400 hover:text-ink-100" onClick={() => setError("")} aria-label="Dismiss">
                  <Icon name="x" size={13} />
                </button>
              </div>
            </Show>
            <div class="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-ink-750 px-3.5 py-2.5">
              <span class="font-mono text-xs text-ink-400">
                {chars()} chars · ~{fmtTime(estSeconds())}
                <Show when={speakers().length}> · {speakers().length} speaker{speakers().length > 1 ? "s" : ""}</Show>
              </span>
              <div class="flex-1" />
              <span class="hidden text-xs text-ink-500 sm:inline">
                <span class={kbd}>Ctrl</span> <span class={kbd}>↵</span> generate · <span class={kbd}>Esc</span> stop
              </span>
              <Show
                when={busy()}
                fallback={
                  <button class={`${btnPrimary} ${btnLg}`} onClick={generate} disabled={!text().trim() || !online()}>
                    <Icon name="bolt" size={15} /> Generate
                  </button>
                }
              >
                <button class={`${btnStop} ${btnLg}`} onClick={stopAll}>
                  <Icon name="stop" size={13} /> Stop · {elapsed().toFixed(1)}s
                </button>
              </Show>
            </div>
          </section>

          <section class={`${panel} min-h-[200px] flex-1`}>
            <div class={panelHead}>
              <span class={panelTitle}>Takes</span>
              <span class="font-mono text-[11px] text-ink-500">{takes().length}</span>
              <div class="flex-1" />
              <Show when={takes().length}>
                <button
                  class={`${btnGhost} ${btnSm}`}
                  onClick={() => {
                    takes().forEach((t) => URL.revokeObjectURL(t.url));
                    setTakes([]);
                  }}
                >
                  Clear all
                </button>
              </Show>
            </div>
            <div class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-3">
              <Show when={busy()}>
                <div class={`${take} border-accent/50 !bg-accent/10`}>
                  <div class="mb-2 flex items-center gap-2 text-xs text-ink-300">
                    <span class={`${dot} animate-pulse bg-rec`} />
                    <span class="font-medium text-ink-100">{busy() === "streaming" ? "Streaming" : busy() === "buffering" ? "Buffering so playback won’t stall" : busy() === "rendering" ? "Rendering" : "Starting"}</span>
                    <span class="font-mono text-ink-400">{liveSecs().toFixed(1)}s audio · {elapsed().toFixed(1)}s</span>
                  </div>
                  <Show
                    when={live().length}
                    fallback={<div class="h-11 animate-shimmer rounded-lg bg-[linear-gradient(90deg,transparent,rgba(154,134,255,0.2),transparent)] bg-[length:200%_100%]" />}
                  >
                    <Waveform peaks={live()} tail />
                  </Show>
                </div>
              </Show>
              <For each={takes()}>
                {(t) => (
                  <article class={take}>
                    <div class="mb-2 flex items-start gap-3">
                      <p class="line-clamp-2 min-w-0 flex-1 whitespace-pre-wrap break-words text-ink-100">{t.text}</p>
                      <div class="flex flex-none gap-1">
                        <button class={`${btnGhost} ${btnSmIcon}`} title="Load text into the script" onClick={() => setText(t.text)}>
                          <Icon name="copy" size={13} />
                        </button>
                        <a class={`${btnGhost} ${btnSmIcon}`} title="Download WAV" href={t.url} download={`drama-${t.id + 1}.wav`}>
                          <Icon name="download" size={13} />
                        </a>
                        <button class={`${btnGhost} ${btnSmIcon}`} title="Delete take" onClick={() => removeTake(t.id)}>
                          <Icon name="trash" size={13} />
                        </button>
                      </div>
                    </div>
                    <ClipPlayer src={t.url} peaks={t.peaks} seconds={t.seconds} autoplay={t.autoplay} />
                    <div class="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] text-ink-400">
                      <span class="font-sans text-ink-300">{t.voice}</span>
                      <span>{t.seconds.toFixed(1)}s audio</span>
                      <span>{t.took.toFixed(1)}s</span>
                      <span class="text-ok">{(t.seconds / t.took).toFixed(1)}× realtime</span>
                      <Show when={t.partial}>
                        <span class="text-warn">stopped early</span>
                      </Show>
                    </div>
                  </article>
                )}
              </For>
              <Show when={!takes().length && !busy()}>
                <div class="grid flex-1 place-items-center py-6 text-center text-ink-500">
                  <div>
                    <Icon name="waves" size={26} class="mx-auto mb-2" />
                    <p>Generated audio shows up here.</p>
                  </div>
                </div>
              </Show>
            </div>
          </section>
        </div>

        {/* inspector */}
        <aside class={`${panel} order-3 overflow-y-auto`}>
          <div class={panelHead}>
            <span class={panelTitle}>Inspector</span>
          </div>
          <div class="flex flex-col gap-5 p-3.5">
            <div>
              <div class={sectionLabel}>{speakers().length ? "Cast" : "Voice"}</div>
              <Show
                when={speakers().length}
                fallback={
                  <>
                    {voiceSelect(voice, setVoice, "Default voice (no clone)")}
                    <p class="mt-2 text-xs leading-snug text-ink-500">Add speaker lines (S0, S1…) to the script to cast several voices.</p>
                  </>
                }
              >
                <div class="flex flex-col gap-2">
                  <For each={speakers()}>
                    {(n) => (
                      <div class="flex items-center gap-2">
                        <span class="w-9 flex-none rounded-md py-1 text-center font-mono text-xs font-semibold" style={{ color: speakerColor(n), background: "color-mix(in srgb, currentColor 14%, transparent)" }}>
                          S{n}
                        </span>
                        {voiceSelect(
                          () => cast()[n] ?? "",
                          (v) => setCast((c) => ({ ...c, [n]: v })),
                          "Default",
                        )}
                      </div>
                    )}
                  </For>
                </div>
              </Show>
            </div>

            <div>
              <div class={sectionLabel}>Sampling</div>
              <Slider label="Temperature" value={temperature()} min={0.1} max={1.5} step={0.05} onChange={setTemperature} />
              <Slider label="Top-p" value={topP()} min={0.1} max={1} step={0.01} onChange={setTopP} />
              <label class="mt-1.5 flex items-center justify-between gap-3 py-1.5">
                <span class="text-ink-300">Top-k</span>
                <input class={`${input} !w-20 text-right font-mono`} type="number" min="1" max="1000" value={topK()} onInput={(e) => setTopK(+e.currentTarget.value || 30)} />
              </label>
              <label class="flex items-center justify-between gap-3 py-1.5">
                <span class="text-ink-300">Seed</span>
                <span class="flex items-center gap-1.5">
                  <input class={`${input} !w-24 text-right font-mono`} type="number" min="0" placeholder="random" value={seed()} onInput={(e) => setSeed(e.currentTarget.value)} />
                  <button class={`${btn} ${btnIcon}`} title="Pick a random seed" onClick={() => setSeed(String(Math.floor(Math.random() * 1e6)))}>
                    <Icon name="dice" />
                  </button>
                </span>
              </label>
              <button
                class="mt-1 cursor-pointer text-xs text-ink-500 hover:text-ink-200"
                onClick={() => {
                  setTemperature(1);
                  setTopP(0.9);
                  setTopK(30);
                  setSeed("");
                }}
              >
                Reset to defaults
              </button>
            </div>

            <div>
              <div class={sectionLabel}>Output</div>
              <Toggle label="Play while generating" hint="Stream audio as soon as it is ready." checked={stream()} onChange={setStream} />
              <Toggle label="Sentence by sentence" hint="Steadier voice and no stalls on long text." checked={split()} onChange={setSplit} />
            </div>
          </div>
        </aside>
      </div>

      <Show when={dialog()}>
        {(d) => (
          <VoiceDialog
            info={info()}
            file={d().file}
            onClose={() => setDialog(null)}
            onSaved={(name) => {
              setDialog(null);
              setVoice(name);
              refresh();
            }}
          />
        )}
      </Show>
    </div>
  );
}

const voiceRow = (on: boolean) =>
  `flex w-full cursor-pointer items-center gap-2.5 rounded-[10px] border px-2.5 py-2 text-left transition-colors ${on ? "border-accent/45 bg-accent/12" : "border-transparent hover:bg-ink-800"}`;
