/* The HTTP API of `drama serve`; /openapi.json describes the same endpoints. */

export const SAMPLE_RATE = 44100;

export interface SpeakRequest {
  text: string;
  voice?: string;
  voices?: string[] | Record<string, string>;
  stream?: boolean;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  seed?: number;
  split?: boolean;
  max_new?: number;
}

async function check(r: Response): Promise<Response> {
  if (!r.ok) throw new Error((await r.text()).trim() || `${r.status} ${r.statusText}`);
  return r;
}

export interface Info {
  version: string;
  sample_rate: number;
  transcription: { enabled: boolean; binary: string | null; install_url: string };
}

export async function info(): Promise<Info> {
  return (await check(await fetch("/v1/info"))).json();
}

export async function listVoices(): Promise<string[]> {
  return (await check(await fetch("/v1/voices"))).json();
}

/** Registers a voice; without `text` the server transcribes the clip itself (when it can). */
export async function addVoice(name: string, wav: Blob, text?: string): Promise<void> {
  const q = new URLSearchParams({ name });
  if (text) q.set("text", text);
  await check(await fetch(`/v1/voices?${q}`, { method: "POST", body: wav, headers: { "content-type": "audio/wav" } }));
}

/** Transcribes a clip with the server's `fluid-speech`; rejects with a message when it is not installed. */
export async function transcribe(audio: Blob, signal?: AbortSignal): Promise<string> {
  const r = await check(await fetch("/v1/transcribe", { method: "POST", body: audio, signal }));
  return (await r.json()).text;
}

/** POST /v1/speak; the response body is a WAV (open-ended when `stream` is set). */
export async function speak(req: SpeakRequest, signal?: AbortSignal): Promise<Response> {
  return check(
    await fetch("/v1/speak", {
      method: "POST",
      body: JSON.stringify(req),
      headers: { "content-type": "application/json" },
      signal,
    }),
  );
}
