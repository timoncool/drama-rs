import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>drama-rs</IC> crate (library name <IC>drama</IC>) is the engine behind the <IC>drama</IC> command:
        model and codec loading, prompt building, the graph-captured decode loop and the streaming codec.
      </Lead>

      <Section>
        <H2>Add the dependency</H2>
        <P>
          The command-line binary (clap) is behind the default <IC>cli</IC> feature, which also enables <IC>download</IC> (ureq; find or fetch the model) and <IC>serve</IC> (the ntex HTTP server, <IC>drama::serve</IC>). Without them the crate depends on
          anyhow, half, hound, memmap2, safetensors, serde and tokenizers. The build script compiles the CUDA kernels
          with <IC>nvcc</IC> (<IC>CUDA_PATH</IC>, default <IC>/opt/cuda</IC>) for <IC>DRAMA_CUDA_ARCH</IC> (default{" "}
          <IC>sm_89</IC>; a comma-separated list builds several).
        </P>
        <CopyBlock command={`cargo add drama-rs --no-default-features`} />
      </Section>

      <Section>
        <H2>Speak with a cloned voice</H2>
        <Pre caption="src/main.rs" lang="rust">{`use drama::engine::SampleParams;
use drama::tts::{SpeakOpts, Tts};

let mut tts = Tts::load("drama-q4".as_ref(), 4096, 512)?;   // context length, prefill chunk
let voice = tts.reference("voice.wav".as_ref(), "Exactly what is said in voice.wav.")?;
let opts = SpeakOpts { params: SampleParams::default(), max_new: 2048, first_chunk: 4, chunk: 16, split: true };

// the closure receives 44.1 kHz audio as it is generated
let spoken = tts.speak("Hello there.", &[voice], &opts, |chunk| Ok(()))?;
drama::codec::write_wav("hello.wav".as_ref(), &spoken.audio)?;`}</Pre>
        <P>
          <IC>Spoken</IC> also carries the generated frames, the number of sentences and the time to first audio. Pass
          no references to let the model pick a voice: with <IC>split</IC> on, the first sentence becomes the reference
          for the rest so the voice stays the same.
        </P>
      </Section>
    </DocsLayout>
  );
}
