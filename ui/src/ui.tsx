/* Small shared building blocks: icons, a toggle, a labelled field, the waveform and the clip player. */
import { Show, createEffect, createSignal, onSettled } from "solid-js";
import { fmtTime } from "./audio";
import { btn, slider, switchCls } from "./styles";

const PATHS: Record<string, string> = {
  play: "M7 4.5v15l12-7.5z",
  pause: "M7 4.5h3.5v15H7zM13.5 4.5H17v15h-3.5z",
  stop: "M6 6h12v12H6z",
  plus: "M12 5v14M5 12h14",
  download: "M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19h14",
  trash: "M5 7h14M10 7V5h4v2m-6 0l.7 12h6.6L16 7",
  mic: "M12 4a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V7a3 3 0 0 0-3-3zM6 11.5a6 6 0 0 0 12 0M12 17.5V21",
  upload: "M12 16V5m0 0L7.5 9.5M12 5l4.5 4.5M5 19h14",
  dice: "M6 4h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM8.5 8.5h.01M15.5 8.5h.01M12 12h.01M8.5 15.5h.01M15.5 15.5h.01",
  x: "M6 6l12 12M18 6L6 18",
  check: "M5 12.5l4.5 4.5L19 7.5",
  bolt: "M13 3L5 14h6l-1 7 8-11h-6z",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  wand: "M15 4l1 2.5L18.5 7.5 16 8.5 15 11l-1-2.5-2.5-1L14 6.5zM5 19l9-9m-9 9l2-2",
  refresh: "M20 11a8 8 0 0 0-14.5-4M4 4v4h4M4 13a8 8 0 0 0 14.5 4M20 20v-4h-4",
  doc: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  waves: "M3 12h.01M7 8v8M11 4v16M15 7v10M19 10v4",
  chevron: "M6 9l6 6 6-6",
  copy: "M9 9h10v11H9zM5 15V4h10",
  book: "M5 4h10a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3zM5 17a3 3 0 0 1 3-3h10",
};

export function Icon(props: { name: keyof typeof PATHS | string; size?: number; class?: string }) {
  return (
    <svg
      class={props.class}
      width={props.size ?? 15}
      height={props.size ?? 15}
      viewBox="0 0 24 24"
      fill={props.name === "play" || props.name === "pause" || props.name === "stop" ? "currentColor" : "none"}
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[props.name] ?? ""} />
    </svg>
  );
}

export function Logo(props: { size?: number }) {
  const s = props.size ?? 22;
  return (
    <svg width={s} height={s} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#1b1e2a" />
      <path d="M7 13v6M12 9v14M17 5v22M22 9v14M27 13v6" stroke="#a897ff" stroke-width="2.2" fill="none" stroke-linecap="round" />
    </svg>
  );
}

export function Toggle(props: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <div class="flex items-center justify-between gap-3 py-1.5">
      <span class="min-w-0">
        <span class="block text-ink-100">{props.label}</span>
        <Show when={props.hint}>
          <span class="block text-[12px] leading-snug text-ink-400">{props.hint}</span>
        </Show>
      </span>
      <button class={switchCls} role="switch" aria-checked={props.checked ? "true" : "false"} aria-label={props.label} onClick={() => props.onChange(!props.checked)} />
    </div>
  );
}

export function Slider(props: { label: string; value: number; min: number; max: number; step: number; digits?: number; onChange: (v: number) => void }) {
  const fill = () => `${((props.value - props.min) / (props.max - props.min)) * 100}%`;
  return (
    <label class="block py-1.5">
      <span class="mb-0.5 flex justify-between text-ink-300">
        {props.label}
        <span class="font-mono text-[12px] text-ink-100">{props.value.toFixed(props.digits ?? 2)}</span>
      </span>
      <input
        class={slider}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        style={{ "--fill": fill() }}
        onInput={(e) => props.onChange(+e.currentTarget.value)}
      />
    </label>
  );
}

export function speakerColor(n: number): string {
  return `var(--color-spk${n % 8})`;
}

export function voiceColor(name: string): string {
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return `hsl(${h % 360} 62% 68%)`;
}

/** Bar waveform of `peaks` (0..1). `tail` shows the newest bars only (a live recording); `progress` tints the played part. */
export function Waveform(props: {
  peaks: Float32Array;
  progress?: number;
  tail?: boolean;
  color?: string;
  onSeek?: (frac: number) => void;
}) {
  let box!: HTMLDivElement;
  let canvas!: HTMLCanvasElement;
  const [width, setWidth] = createSignal(0);

  onSettled(() => {
    const ro = new ResizeObserver(() => setWidth(box.clientWidth));
    ro.observe(box);
    setWidth(box.clientWidth);
    return () => ro.disconnect();
  });

  createEffect(
    () => ({ peaks: props.peaks, progress: props.progress ?? 0, w: width(), tail: props.tail, color: props.color }),
    ({ peaks, progress, w, tail, color }) => {
      const h = box.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(w * dpr));
      canvas.height = Math.max(1, Math.floor(h * dpr));
      const g = canvas.getContext("2d")!;
      g.scale(dpr, dpr);
      g.clearRect(0, 0, w, h);
      const css = getComputedStyle(box);
      const base = css.getPropertyValue("--color-ink-500") || "#565e76";
      const lit = color || css.getPropertyValue("--color-accent") || "#9a86ff";
      const barW = 3;
      const gap = 2;
      const bars = Math.max(1, Math.floor(w / (barW + gap)));
      const n = peaks.length;
      const mid = h / 2;
      for (let b = 0; b < bars; b++) {
        let v = 0;
        if (tail) {
          const i = n - bars + b;
          v = i >= 0 && i < n ? peaks[i] : 0;
        } else if (n) {
          const from = Math.floor((b * n) / bars);
          const to = Math.max(from + 1, Math.floor(((b + 1) * n) / bars));
          for (let i = from; i < to && i < n; i++) v = Math.max(v, peaks[i]);
        }
        const bh = Math.max(2, Math.pow(v, 0.7) * (h - 8));
        const x = b * (barW + gap);
        g.fillStyle = (b + 0.5) / bars <= progress ? lit : base;
        g.globalAlpha = (b + 0.5) / bars <= progress ? 1 : 0.55;
        g.beginPath();
        g.roundRect(x, mid - bh / 2, barW, bh, 1.5);
        g.fill();
      }
      g.globalAlpha = 1;
    },
  );

  return (
    <div
      class="relative h-11 w-full cursor-pointer overflow-hidden rounded-lg bg-ink-900"
      ref={box}
      onPointerDown={(e) => {
        const r = box.getBoundingClientRect();
        props.onSeek?.(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
      }}
    >
      <canvas class="absolute inset-0 size-full" ref={canvas} />
    </div>
  );
}

let current: HTMLAudioElement | null = null;

/** Play/pause button, waveform with a playhead (click to seek) and the time, for one clip. */
export function ClipPlayer(props: { src: string; peaks: Float32Array; seconds: number; autoplay?: boolean; color?: string }) {
  const [playing, setPlaying] = createSignal(false);
  const [pos, setPos] = createSignal(0);
  let audio: HTMLAudioElement | null = null;
  let raf = 0;

  const get = () => {
    if (audio && audio.src.endsWith(props.src.replace(/^.*\//, ""))) return audio;
    audio?.pause();
    const a = new Audio(props.src);
    a.onended = () => {
      setPlaying(false);
      setPos(0);
    };
    a.onpause = () => setPlaying(false);
    a.onplay = () => {
      setPlaying(true);
      const tick = () => {
        if (!audio || audio.paused) return;
        setPos(audio.currentTime);
        raf = requestAnimationFrame(tick);
      };
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    return (audio = a);
  };

  const toggle = () => {
    const a = get();
    if (!a.paused) return a.pause();
    if (current && current !== a) current.pause();
    current = a;
    a.play().catch(() => {});
  };

  onSettled(() => {
    if (props.autoplay) toggle();
    return () => {
      cancelAnimationFrame(raf);
      audio?.pause();
    };
  });

  return (
    <div class="flex items-center gap-2.5">
      <button class={`${btn} !size-9 !rounded-full !px-0`} onClick={toggle} aria-label={playing() ? "Pause" : "Play"}>
        <Icon name={playing() ? "pause" : "play"} size={14} />
      </button>
      <div class="min-w-0 flex-1">
        <Waveform
          peaks={props.peaks}
          progress={props.seconds ? pos() / props.seconds : 0}
          color={props.color}
          onSeek={(f) => {
            const a = get();
            a.currentTime = f * props.seconds;
            setPos(a.currentTime);
            if (a.paused) toggle();
          }}
        />
      </div>
      <span class="w-[100px] flex-none whitespace-nowrap text-right font-mono text-[12px] text-ink-300">
        {fmtTime(pos())} <span class="text-ink-500">/ {fmtTime(props.seconds)}</span>
      </span>
    </div>
  );
}
