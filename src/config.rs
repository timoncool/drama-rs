//! Checkpoint configuration shared by the CUDA and MLX engines.
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeParameters,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub dim: usize,
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub num_codebooks: usize,
    pub vocab_size: usize,
    pub shared_enrichment_layers: usize,
    pub rope_base: f64,
    pub rotary_percent: f64,
    pub vq_backbone_codebooks: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub audio_decoder_config: DecoderConfig,
    pub semantic_start_token_id: i32,
    pub semantic_end_token_id: i32,
    pub eos_token_id: i32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Meta {
    pub(crate) format: String,
    pub(crate) q4_group: usize,
    pub(crate) q8_group: usize,
    pub(crate) config: Config,
}

impl Config {
    pub fn tower_layers(&self) -> usize {
        self.audio_decoder_config.shared_enrichment_layers + self.audio_decoder_config.num_codebooks
            - 1
    }
    pub fn frame_width(&self) -> usize {
        1 + self.audio_decoder_config.num_codebooks
    }
    pub fn n_candidates(&self) -> usize {
        self.audio_decoder_config.vocab_size + 1
    }
    pub fn is_linear(&self, layer: usize) -> bool {
        self.text_config.layer_types[layer] == "linear_attention"
    }
}

/// Which routed-expert weights to run.
#[derive(Debug, Clone, Default)]
pub enum ExpertMode {
    /// The converted int4 experts, resident on the GPU.
    #[default]
    Q4,
    /// The original block-FP8 experts from this checkpoint directory (fishaudio/drama-3). The first `vram_layers`
    /// layers' experts are uploaded to the GPU, the rest stay in pinned host RAM and are read by the kernels over
    /// PCIe. Everything else stays as converted.
    Fp8 { dir: PathBuf, vram_layers: usize },
    /// The converted int4 experts in page-locked host RAM, for a card the model does not fit: decoding keeps the
    /// experts in use in a VRAM cache that takes what `vram_limit` bytes leave (0: the free VRAM less a margin). The
    /// output is the same as `Q4`; a larger cache only makes it faster.
    Q4Ram { vram_limit: usize },
}
