//! OpenVINO Whisper engine configuration.

use serde::{Deserialize, Serialize};

use super::super::{default_on_demand_loading, default_true};

/// OpenVINO Whisper speech-to-text configuration (Intel NPU/CPU/GPU via OpenVINO GenAI).
/// Requires: cargo build --features openvino-whisper
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenVinoConfig {
    /// Model name or path to a directory containing OpenVINO IR model files.
    /// Quantized names include "base.en-int8", "small.en-fp16", and "tiny-int4".
    /// Short names such as "base.en" use `quantized` to select int8 or fp16.
    pub model: String,

    /// OpenVINO inference device: "NPU", "CPU", "GPU", or "AUTO".
    #[serde(default = "default_openvino_device")]
    pub device: String,

    /// Prefer int8 quantized model variants.
    #[serde(default = "default_true")]
    pub quantized: bool,

    /// Number of CPU threads (ignored for NPU/GPU).
    #[serde(default)]
    pub threads: Option<usize>,

    /// Whisper language code.
    #[serde(default = "default_openvino_language")]
    pub language: String,

    /// Translate non-English speech to English.
    #[serde(default)]
    pub translate: bool,

    /// Load the model when recording starts instead of at daemon startup.
    #[serde(default = "default_on_demand_loading")]
    pub on_demand_loading: bool,

    /// OpenVINO installation directory containing the GenAI shared library.
    /// Also settable through VOXTYPE_OPENVINO_DIR.
    #[serde(default)]
    pub openvino_dir: Option<String>,
}

fn default_openvino_device() -> String {
    "NPU".to_string()
}

fn default_openvino_language() -> String {
    "en".to_string()
}

impl Default for OpenVinoConfig {
    fn default() -> Self {
        Self {
            model: "base.en".to_string(),
            device: default_openvino_device(),
            quantized: true,
            threads: None,
            language: default_openvino_language(),
            translate: false,
            on_demand_loading: false,
            openvino_dir: None,
        }
    }
}
