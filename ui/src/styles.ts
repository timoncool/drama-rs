/* Tailwind class strings shared by the components. */
const focus = "focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent/70";
const btnBase = `inline-flex cursor-pointer items-center justify-center gap-1.5 whitespace-nowrap font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${focus}`;

export const btn = `${btnBase} h-[30px] rounded-lg border border-ink-600 bg-ink-800 px-[11px] text-ink-100 enabled:hover:border-ink-500 enabled:hover:bg-ink-750`;
export const btnPrimary = `${btnBase} h-[30px] rounded-lg border border-transparent bg-accent-strong px-[11px] font-semibold text-white enabled:hover:bg-accent`;
export const btnStop = `${btnBase} h-[30px] rounded-lg border border-rec/60 bg-rec/20 px-[11px] text-[#ffd3d9] enabled:hover:border-rec enabled:hover:bg-rec/30`;
export const btnGhost = `${btnBase} h-[30px] rounded-lg border border-transparent bg-transparent px-[11px] text-ink-300 enabled:hover:bg-ink-800 enabled:hover:text-ink-100`;
export const btnLg = "!h-10 !rounded-[10px] !px-[18px] !text-sm";
export const btnSm = "!h-[26px] !rounded-[7px] !px-2 !text-xs";
export const btnIcon = "!w-[30px] !px-0";
export const btnSmIcon = "!h-[26px] !w-[26px] !rounded-[7px] !px-0";

export const input = `h-8 w-full rounded-lg border border-ink-700 bg-ink-850 px-2.5 text-ink-100 transition-colors placeholder:text-ink-500 hover:border-ink-600 focus:border-accent focus:outline-none`;
export const textarea = `w-full resize-y rounded-lg border border-ink-700 bg-ink-850 px-2.5 py-2 leading-normal text-ink-100 transition-colors placeholder:text-ink-500 hover:border-ink-600 focus:border-accent focus:outline-none`;
export const select = `${input} cursor-pointer appearance-none pr-7`;

export const panel = "flex min-h-0 flex-col rounded-xl border border-ink-700 bg-ink-900";
export const panelHead = "flex h-[42px] flex-none items-center gap-2 border-b border-ink-750 px-3.5";
export const panelTitle = "text-[11px] font-semibold uppercase tracking-[0.09em] text-ink-400";
export const sectionLabel = "mb-2 flex items-baseline justify-between text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-400";
export const kbd = "rounded border border-ink-600 bg-ink-850 px-[5px] py-px font-mono text-[10.5px] text-ink-300";
export const chip = "inline-flex h-6 items-center gap-[5px] rounded-full border border-ink-700 bg-ink-850 px-[9px] text-xs text-ink-300";
export const chipBtn = `${chip} cursor-pointer transition-colors hover:border-ink-500 hover:text-ink-100 ${focus}`;
export const dot = "size-[7px] flex-none rounded-full";
export const notice = "flex gap-2.5 rounded-[10px] border border-warn/35 bg-warn/10 px-3 py-2.5 leading-normal text-ink-200 [&_a]:text-warn [&_a]:underline [&_a]:underline-offset-2";
export const errorBox = "flex gap-2.5 rounded-[10px] border border-rec/45 bg-rec/10 px-3 py-2.5 leading-normal text-ink-200";
export const take = "rounded-[10px] border border-ink-750 bg-ink-850 px-3 py-[11px]";
export const modal = "max-h-[calc(100vh-2rem)] w-full max-w-[560px] overflow-auto rounded-2xl border border-ink-600 bg-ink-900 shadow-[0_30px_80px_rgba(0,0,0,0.6)]";
export const scrim = "fixed inset-0 z-50 grid place-items-center bg-ink-950/75 p-4 backdrop-blur-md";

export const slider = [
  "h-[18px] w-full cursor-pointer appearance-none bg-transparent",
  "[&::-webkit-slider-runnable-track]:h-1 [&::-webkit-slider-runnable-track]:rounded-sm",
  "[&::-webkit-slider-runnable-track]:bg-[linear-gradient(to_right,var(--color-accent)_var(--fill),var(--color-ink-700)_var(--fill))]",
  "[&::-webkit-slider-thumb]:-mt-[5px] [&::-webkit-slider-thumb]:size-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-ink-100 [&::-webkit-slider-thumb]:shadow-[0_0_0_3px_rgba(154,134,255,0.3)]",
  "[&::-moz-range-track]:h-1 [&::-moz-range-track]:rounded-sm [&::-moz-range-track]:bg-ink-700",
  "[&::-moz-range-progress]:h-1 [&::-moz-range-progress]:rounded-sm [&::-moz-range-progress]:bg-accent",
  "[&::-moz-range-thumb]:size-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-ink-100",
].join(" ");

export const switchCls =
  "relative h-[17px] w-[30px] flex-none cursor-pointer rounded-[9px] border-0 bg-ink-600 p-0 transition-colors aria-checked:bg-accent-strong after:absolute after:top-0.5 after:left-0.5 after:size-[13px] after:rounded-full after:bg-white after:transition-transform after:content-[''] aria-checked:after:translate-x-[13px]";

const layer = "absolute inset-0 m-0 whitespace-pre-wrap border-0 p-[16px_18px] font-mono text-sm leading-[1.8] [overflow-wrap:anywhere] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden";
export const editorBack = `${layer} pointer-events-none overflow-hidden text-ink-100`;
export const editorInput = `${layer} resize-none overflow-y-auto bg-transparent text-transparent caret-ink-100 outline-none placeholder:text-ink-500`;
export const tokSpk = "rounded bg-current/15 shadow-[0_0_0_1px_color-mix(in_srgb,currentColor_30%,transparent)]";
export const tokTag = "rounded bg-warn/15 text-warn";
