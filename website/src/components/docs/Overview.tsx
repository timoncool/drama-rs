import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        drama-rs runs Fish Audio's{" "}
        <a
          href="https://huggingface.co/fishaudio/drama-3"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          Drama 3
        </a>{" "}
        text-to-speech model on a single 24 GB GPU, in Rust and hand-written CUDA, with no PyTorch at run time. It ships
        as a library crate and the <IC>drama</IC> command.
      </Lead>

      <Section>
        <H2>The model</H2>
        <UL>
          <LI>
            <strong class="font-medium text-ink">Backbone</strong>: Qwen3.5-MoE, 35B total / 3B active, 40 layers of
            gated-DeltaNet linear attention with full attention every 4th layer.
          </LI>
          <LI>
            <strong class="font-medium text-ink">Codebook tower</strong>: 13 transformer layers that predict codebooks 1
            to 9 of every frame.
          </LI>
          <LI>
            <strong class="font-medium text-ink">Codec</strong>: the Fish modded-DAC encoder (voice cloning) and decoder,
            streaming and exact.
          </LI>
        </UL>
        <P>
          Routed experts are int4 (asymmetric, group 64, MSE clip search), the other linears int8, and the lm_head keeps
          only the 4,097 sampleable rows. Weights and codec take about 21.4 GiB of a 24 GB card.
        </P>
        <CopyBlock command={`hf download apiplant/drama-q4 --local-dir drama-q4`} />
        <P>
          To rebuild the int4 weights from the original FP8 checkpoint, see <IC>tools/convert.py</IC> in the
          repository.
        </P>
      </Section>

      <Section>
        <H2>Speed (RTX 4090)</H2>
        <FlagTable
          rows={[
            { flag: "decode", meaning: "~5.2 ms per frame, about 190 frames/s, 8.9x real time (21.5 frames = 1 s of audio)" },
            { flag: "prefill", meaning: "~140 ms for a 300-row prompt (text + 10 s reference voice)" },
            { flag: "time to first audio", meaning: "50 to 120 ms after the prompt is ready (streaming)" },
            { flag: "codec decode", meaning: "~200x real time" },
          ]}
        />
        <P>
          Each decode step is one CUDA-graph replay: lm_head, sampling with repetition-aware sampling, the 13 tower
          layers with 9 codebook samplers, and the backbone step for the new frame. Weight-streaming GEMVs prefetch the
          next kernel's weights into L2 to hide launch gaps.
        </P>
      </Section>

      <Section>
        <H2>Requirements</H2>
        <UL>
          <LI>CUDA: an NVIDIA GPU with 24 GB of memory, Ampere or newer (A10, 3090, 4090, L40, A100, H100, 5090).</LI>
          <LI>CUDA: Linux x86_64 with a driver for CUDA 12 (the release builds) or your own toolkit (source builds).</LI>
          <LI>
            Apple Silicon (MLX): macOS 15 or newer and 32 GB or more of unified memory. Experts run as int4 (
            <IC>--experts auto</IC> selects <IC>q4</IC>); the FP8 expert placement options are CUDA-only. Throughput has not been measured
            on a Mac yet.
          </LI>
          <LI>
            <IC>pw-play</IC> (PipeWire) on Linux, or <IC>ffplay</IC> (FFmpeg) on macOS, for <IC>--play</IC>.
          </LI>
        </UL>
      </Section>
    </DocsLayout>
  );
}
