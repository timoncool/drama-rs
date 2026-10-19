import { Show, createSignal, onSettled } from "solid-js";
import { addVoice, transcribe, type Info } from "./api";
import { toWav } from "./audio";
import { btn, btnGhost, btnPrimary, btnStop, errorBox, input, modal, notice, scrim, sectionLabel, textarea } from "./styles";
import { ClipPlayer, Icon } from "./ui";

interface Loaded {
  wav: Blob;
  url: string;
  seconds: number;
  peaks: Float32Array;
}

/** Modal to register a voice: pick, drop or record a clip, get its transcript (automatically when the server can), name it. */
export function VoiceDialog(props: { info: Info | null; file: File | null; onClose: () => void; onSaved: (name: string) => void }) {
  const [name, setName] = createSignal("");
  const [clip, setClip] = createSignal<Loaded | null>(null);
  const [text, setText] = createSignal("");
  const [busy, setBusy] = createSignal<"" | "decoding" | "transcribing" | "saving">("");
  const [error, setError] = createSignal("");
  const [over, setOver] = createSignal(false);
  const [recording, setRecording] = createSignal(false);
  const [recSecs, setRecSecs] = createSignal(0);
  let recorder: MediaRecorder | null = null;
  let recTimer = 0;
  let fileInput!: HTMLInputElement;

  const canTranscribe = () => !!props.info?.transcription.enabled;

  async function runTranscribe(wav: Blob) {
    setBusy("transcribing");
    try {
      setText(await transcribe(wav));
    } catch (e: any) {
      setError(`Transcription failed: ${e.message}`);
    } finally {
      setBusy("");
    }
  }

  async function load(file: Blob, suggested?: string) {
    setError("");
    setBusy("decoding");
    try {
      const { wav, seconds, peaks } = await toWav(file);
      const old = clip();
      if (old) URL.revokeObjectURL(old.url);
      setClip({ wav, seconds, peaks, url: URL.createObjectURL(wav) });
      if (suggested && !name().trim()) setName(suggested);
      if (seconds > 40) setError("That clip is long; 5–30 seconds works best for cloning.");
      setBusy("");
      if (canTranscribe()) await runTranscribe(wav);
    } catch (e: any) {
      setBusy("");
      setError(`Could not read that audio: ${e.message ?? e}`);
    }
  }

  const fromFile = (f: File) => load(f, f.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim().slice(0, 40));

  async function toggleRecord() {
    if (recording()) {
      recorder?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const parts: Blob[] = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => parts.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        clearInterval(recTimer);
        setRecording(false);
        load(new Blob(parts, { type: recorder?.mimeType }));
      };
      recorder.start();
      setRecSecs(0);
      recTimer = window.setInterval(() => setRecSecs((s) => s + 1), 1000);
      setRecording(true);
    } catch (e: any) {
      setError(`Microphone unavailable: ${e.message ?? e} (browsers only allow it on localhost or HTTPS)`);
    }
  }

  async function save() {
    const c = clip();
    if (!c) return;
    setBusy("saving");
    setError("");
    try {
      await addVoice(name().trim(), c.wav, text().trim());
      props.onSaved(name().trim());
    } catch (e: any) {
      setError(e.message);
      setBusy("");
    }
  }

  onSettled(() => {
    if (props.file) fromFile(props.file);
    const key = (e: KeyboardEvent) => e.key === "Escape" && !busy() && props.onClose();
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("keydown", key);
      clearInterval(recTimer);
      if (recorder?.state === "recording") recorder.stop();
      const c = clip();
      if (c) URL.revokeObjectURL(c.url);
    };
  });

  const ready = () => !!clip() && !!name().trim() && !!text().trim() && !busy();

  return (
    <div class={scrim} onPointerDown={(e) => e.target === e.currentTarget && !busy() && props.onClose()}>
      <div class={modal} role="dialog" aria-label="New voice">
        <div class="flex items-center justify-between border-b border-ink-750 px-5 py-4">
          <div>
            <h2 class="text-base font-semibold text-ink-100">New voice</h2>
            <p class="mt-0.5 text-ink-400">A clean 5–30 second clip of one person speaking.</p>
          </div>
          <button class={`${btnGhost} !w-[30px] !px-0`} onClick={props.onClose} aria-label="Close">
            <Icon name="x" />
          </button>
        </div>

        <div class="flex flex-col gap-4 p-5">
          <Show
            when={clip()}
            fallback={
              <div
                class={`flex flex-col items-center gap-3 rounded-xl border-[1.5px] border-dashed px-4 py-7 text-center transition-colors ${over() ? "border-accent bg-accent/10" : "border-ink-600 bg-ink-850"}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setOver(true);
                }}
                onDragLeave={() => setOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setOver(false);
                  const f = e.dataTransfer?.files[0];
                  if (f) fromFile(f);
                }}
              >
                <Icon name="upload" size={22} class="text-ink-400" />
                <p class="text-ink-300">Drop an audio file here (wav, mp3, m4a, ogg, flac…)</p>
                <div class="flex gap-2">
                  <button class={btn} onClick={() => fileInput.click()} disabled={!!busy()}>
                    <Icon name="upload" /> Choose file
                  </button>
                  <button class={recording() ? btnStop : btn} onClick={toggleRecord} disabled={!!busy()}>
                    <Show when={recording()} fallback={<Icon name="mic" />}>
                      <span class="size-2 animate-pulse rounded-full bg-rec" />
                    </Show>
                    {recording() ? `Stop · ${recSecs()}s` : "Record"}
                  </button>
                </div>
                <Show when={busy() === "decoding"}>
                  <p class="text-ink-400">Reading audio…</p>
                </Show>
                <input ref={fileInput} type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac" class="hidden" onChange={(e) => e.currentTarget.files?.[0] && fromFile(e.currentTarget.files[0])} />
              </div>
            }
          >
            {(c) => (
              <div class="rounded-xl border border-ink-700 bg-ink-850 p-3">
                <ClipPlayer src={c().url} peaks={c().peaks} seconds={c().seconds} />
                <div class="mt-2 flex justify-end">
                  <button
                    class={`${btnGhost} !h-[26px] !px-2 !text-xs`}
                    onClick={() => {
                      setClip(null);
                      setText("");
                    }}
                    disabled={!!busy()}
                  >
                    Replace clip
                  </button>
                </div>
              </div>
            )}
          </Show>

          <label class="block">
            <span class={sectionLabel}>Name</span>
            <input class={input} value={name()} onInput={(e) => setName(e.currentTarget.value)} placeholder="alice" maxlength={40} />
          </label>

          <div>
            <span class={sectionLabel}>
              Transcript
              <Show when={canTranscribe() && clip()}>
                <button class="cursor-pointer text-[11px] normal-case tracking-normal text-accent hover:underline disabled:opacity-50" onClick={() => runTranscribe(clip()!.wav)} disabled={!!busy()}>
                  <Icon name="wand" size={11} /> transcribe again
                </button>
              </Show>
            </span>
            <textarea
              class={textarea}
              rows={3}
              value={text()}
              onInput={(e) => setText(e.currentTarget.value)}
              placeholder={busy() === "transcribing" ? "Transcribing…" : "Exactly what the clip says"}
              disabled={busy() === "transcribing"}
            />
            <Show when={busy() === "transcribing"}>
              <p class="mt-1.5 text-ink-400">Transcribing with fluid-speech (the first run may download its model)…</p>
            </Show>
          </div>

          <Show when={props.info && !canTranscribe()}>
            <div class={notice}>
              <Icon name="wand" size={16} class="mt-0.5 flex-none text-warn" />
              <p>
                <b class="text-ink-100">Transcription is disabled.</b> Install <b>fluid-speech</b> from{" "}
                <a href={props.info!.transcription.install_url} target="_blank" rel="noreferrer">
                  {props.info!.transcription.install_url.replace("https://", "")}
                </a>{" "}
                and put it on your PATH to get transcripts automatically. Until then, type the transcript by hand.
              </p>
            </div>
          </Show>

          <Show when={error()}>
            <div class={errorBox}>{error()}</div>
          </Show>
        </div>

        <div class="flex items-center justify-end gap-2 border-t border-ink-750 px-5 py-3.5">
          <button class={btnGhost} onClick={props.onClose} disabled={busy() === "saving"}>
            Cancel
          </button>
          <button class={btnPrimary} onClick={save} disabled={!ready()}>
            {busy() === "saving" ? "Encoding voice…" : "Save voice"}
          </button>
        </div>
      </div>
    </div>
  );
}
