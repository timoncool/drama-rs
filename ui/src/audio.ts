/* Browser-side audio helpers: WAV encoding, decoding any clip to mono WAV, waveform peaks, streamed playback. */
import { SAMPLE_RATE } from "./api";

/** 16-bit mono PCM at `rate` as a complete WAV file. */
export function wavBlob(pcm: Uint8Array, rate = SAMPLE_RATE): Blob {
  const h = new DataView(new ArrayBuffer(44));
  const str = (o: number, s: string) => [...s].forEach((c, i) => h.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  h.setUint32(4, 36 + pcm.length, true);
  str(8, "WAVEfmt ");
  h.setUint32(16, 16, true);
  h.setUint16(20, 1, true);
  h.setUint16(22, 1, true);
  h.setUint32(24, rate, true);
  h.setUint32(28, rate * 2, true);
  h.setUint16(32, 2, true);
  h.setUint16(34, 16, true);
  str(36, "data");
  h.setUint32(40, pcm.length, true);
  return new Blob([h.buffer, pcm as BlobPart], { type: "audio/wav" });
}

/** Any browser-decodable clip (wav, mp3, ogg, m4a, webm…) as a mono 16-bit WAV at 44.1 kHz, plus its length. */
export async function toWav(file: Blob): Promise<{ wav: Blob; seconds: number; peaks: Float32Array }> {
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  try {
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    const n = buf.length;
    const mono = new Float32Array(n);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const ch = buf.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += ch[i] / buf.numberOfChannels;
    }
    const pcm = new Int16Array(n);
    for (let i = 0; i < n; i++) pcm[i] = Math.round(Math.max(-1, Math.min(1, mono[i])) * 32767);
    const bytes = new Uint8Array(pcm.buffer);
    return { wav: wavBlob(bytes, buf.sampleRate), seconds: n / buf.sampleRate, peaks: peaksOf(bytes) };
  } finally {
    ctx.close();
  }
}

/** Min/max-free loudness envelope of 16-bit PCM, `bins` values in 0..1. */
export function peaksOf(pcm: Uint8Array, bins = 600): Float32Array {
  const n = pcm.length >> 1;
  const view = new DataView(pcm.buffer, pcm.byteOffset, n * 2);
  const out = new Float32Array(bins);
  const per = Math.max(1, n / bins);
  for (let b = 0; b < bins; b++) {
    let peak = 0;
    const end = Math.min(n, Math.floor((b + 1) * per));
    for (let i = Math.floor(b * per); i < end; i++) peak = Math.max(peak, Math.abs(view.getInt16(i * 2, true)));
    out[b] = peak / 32768;
  }
  return out;
}

export function fmtTime(s: number): string {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

/** Streams a /v1/speak response through Web Audio as it arrives, and can be stopped. */
export class StreamPlayer {
  private sources = new Set<AudioBufferSourceNode>();
  private at = 0;
  private stopped = false;
  readonly chunks: Uint8Array[] = [];
  samples = 0;
  /** Chunks held back until enough audio is buffered to play through without stalling. */
  private pending: Int16Array[] = [];
  private playing = false;
  private t0 = performance.now();
  /** Seconds of audio buffered before playback starts, once known (for display). */
  startedAfter = 0;

  /**
   * `estSeconds` is a guess of the whole clip's length. If the server generates slower than realtime, playback
   * waits until the audio produced so far covers the shortfall (`est * (1 - rate)`), so it never runs dry.
   */
  constructor(
    private ctx: AudioContext,
    private onChunk: (pcm: Int16Array) => void,
    private estSeconds = 0,
    private onStart: () => void = () => {},
  ) {}

  /** Audio seconds that must be buffered before starting, given the generation speed measured so far. */
  private needed(): number {
    const produced = this.samples / SAMPLE_RATE;
    const rate = produced / Math.max(0.05, (performance.now() - this.t0) / 1000);
    const lead = 0.4;
    if (rate >= 1.15) return lead;
    return Math.min(this.estSeconds, lead + this.estSeconds * (1 - Math.min(rate, 1)) * 1.1);
  }

  get buffering(): boolean {
    return !this.playing && !this.stopped;
  }

  private schedule(pcm: Int16Array) {
    const n = pcm.length;
    const audio = this.ctx.createBuffer(1, n, SAMPLE_RATE);
    const ch = audio.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = pcm[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = audio;
    src.connect(this.ctx.destination);
    src.onended = () => this.sources.delete(src);
    this.at = Math.max(this.at, this.ctx.currentTime + 0.05);
    src.start(this.at);
    this.sources.add(src);
    this.at += audio.duration;
  }

  private feed(pcm: Int16Array, last = false) {
    if (this.stopped) return;
    if (pcm.length) this.pending.push(pcm);
    if (!this.playing && (last || this.samples / SAMPLE_RATE >= this.needed())) {
      this.playing = true;
      this.startedAfter = (performance.now() - this.t0) / 1000;
      this.onStart();
    }
    if (this.playing) {
      for (const p of this.pending) this.schedule(p);
      this.pending = [];
    }
  }

  stop() {
    this.stopped = true;
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {}
    }
    this.sources.clear();
  }

  /** Reads the response to its end; resolves to the whole clip's PCM. */
  async play(res: Response): Promise<Uint8Array> {
    const reader = res.body!.getReader();
    this.t0 = performance.now();
    let header = 44;
    let carry: Uint8Array | null = null;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      let buf = value;
      if (header > 0) {
        const skip = Math.min(header, buf.length);
        header -= skip;
        buf = buf.subarray(skip);
      }
      if (carry) {
        const joined = new Uint8Array(carry.length + buf.length);
        joined.set(carry);
        joined.set(buf, carry.length);
        buf = joined;
        carry = null;
      }
      if (buf.length % 2) {
        carry = buf.slice(buf.length - 1);
        buf = buf.subarray(0, buf.length - 1);
      }
      if (!buf.length) continue;
      const own = buf.slice();
      this.chunks.push(own);
      const view = new DataView(own.buffer);
      const n = own.length / 2;
      const pcm = new Int16Array(n);
      for (let i = 0; i < n; i++) pcm[i] = view.getInt16(i * 2, true);
      this.samples += n;
      this.onChunk(pcm);
      this.feed(pcm);
    }
    this.feed(new Int16Array(), true);
    const pcm = new Uint8Array(this.chunks.reduce((a, c) => a + c.length, 0));
    let o = 0;
    for (const c of this.chunks) {
      pcm.set(c, o);
      o += c.length;
    }
    return pcm;
  }

  /** Seconds of audio still queued to play. */
  get remaining(): number {
    return Math.max(0, this.at - this.ctx.currentTime);
  }
}
