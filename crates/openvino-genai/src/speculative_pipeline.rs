//! Speculative-decoding (draft-model) LLM pipeline.
//!
//! The public OpenVINO GenAI C API does not expose draft-model speculative decoding, so this
//! module is backed by a small C++ shim in `openvino-genai-sys` that wraps the C++
//! `ov::genai::LLMPipeline(main, device, draft_model(...))` constructor and the
//! `SDPerModelsPerfMetrics` payload of `DecodedResults::extended_perf_metrics`.
//!
//! Because the shim owns its own opaque types end-to-end (and the public C-API does not ship
//! getters for arbitrary [`GenerationConfig`] fields), this module also exposes a parallel
//! [`SpeculativeGenerationConfig`] with the small set of fields the C++ pipeline needs.
//!
//! # Example
//!
//! ```no_run
//! use openvino_genai::{SpeculativeGenerationConfig, SpeculativeLlmPipeline};
//!
//! let mut pipe = SpeculativeLlmPipeline::builder("path/to/main", "CPU")
//!     .draft("path/to/draft", "CPU")
//!     .build()
//!     .expect("to create the pipeline");
//!
//! let mut cfg = SpeculativeGenerationConfig::new().unwrap();
//! cfg.set_max_new_tokens(100).unwrap();
//! cfg.set_num_assistant_tokens(4).unwrap();
//!
//! let results = pipe.generate("Hello", Some(&cfg), None).unwrap();
//! println!("{}", results.get_string().unwrap());
//!
//! if let Some(metrics) = results.get_sd_perf_metrics().unwrap() {
//!     println!("accepted draft tokens: {}", metrics.num_accepted_tokens().unwrap());
//! }
//! ```
//!
//! [`GenerationConfig`]: crate::GenerationConfig

use crate::error::LoadingError;
use crate::{cstr, drop_using_function, try_unsafe, util::Result, SetupError, Streamer};
use openvino_genai_sys as sys;
use std::ffi::CString;
use std::os::raw::c_char;

/// Generation parameters for [`SpeculativeLlmPipeline`].
///
/// Mirrors a subset of the regular [`crate::GenerationConfig`] surface: only the fields the
/// shim's C++ pipeline needs are exposed. If you need to set a parameter that isn't here, open
/// an issue or add the corresponding setter to the shim.
pub struct SpeculativeGenerationConfig {
    ptr: *mut sys::ov_genai_sd_generation_config,
}
drop_using_function!(
    SpeculativeGenerationConfig,
    sys::ov_genai_sd_generation_config_free
);

unsafe impl Send for SpeculativeGenerationConfig {}

impl SpeculativeGenerationConfig {
    /// Create a default config (equivalent to a default-constructed
    /// `ov::genai::GenerationConfig`).
    pub fn new() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(sys::ov_genai_sd_generation_config_create(
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Self { ptr })
    }

    /// Set the maximum number of new tokens to generate.
    pub fn set_max_new_tokens(&mut self, value: usize) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_max_new_tokens(
            self.ptr, value
        ))
    }

    /// Set the maximum total length (prompt + generated tokens).
    pub fn set_max_length(&mut self, value: usize) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_max_length(
            self.ptr, value
        ))
    }

    /// Set the sampling temperature.
    pub fn set_temperature(&mut self, value: f32) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_temperature(
            self.ptr, value
        ))
    }

    /// Set the top-p (nucleus sampling) threshold.
    pub fn set_top_p(&mut self, value: f32) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_top_p(self.ptr, value))
    }

    /// Set the top-k filtering value.
    pub fn set_top_k(&mut self, value: usize) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_top_k(self.ptr, value))
    }

    /// Enable multinomial random sampling.
    pub fn set_do_sample(&mut self, value: bool) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_do_sample(
            self.ptr,
            i32::from(value)
        ))
    }

    /// Set the number of beams for beam search. `1` disables beam search.
    pub fn set_num_beams(&mut self, value: usize) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_num_beams(
            self.ptr, value
        ))
    }

    /// Set the repetition penalty. `1.0` means no penalty.
    pub fn set_repetition_penalty(&mut self, value: f32) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_repetition_penalty(
            self.ptr, value
        ))
    }

    /// Set the presence penalty.
    pub fn set_presence_penalty(&mut self, value: f32) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_presence_penalty(
            self.ptr, value
        ))
    }

    /// Set the frequency penalty.
    pub fn set_frequency_penalty(&mut self, value: f32) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_frequency_penalty(
            self.ptr, value
        ))
    }

    /// Set the random number generator seed.
    pub fn set_rng_seed(&mut self, value: usize) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_rng_seed(
            self.ptr, value
        ))
    }

    /// Set the number of candidate tokens the draft model produces per iteration. With the
    /// stateful backend this is the initial value and the runtime adjusts it based on recent
    /// acceptance rate; with the continuous-batching backend it is used as-is. Defaults to `5`.
    pub fn set_num_assistant_tokens(&mut self, value: usize) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_generation_config_set_num_assistant_tokens(
            self.ptr, value
        ))
    }

    /// Set the minimum probability a draft candidate must clear to be validated by the main
    /// model (continuous-batching backend only). Mutually exclusive with
    /// [`set_num_assistant_tokens`](Self::set_num_assistant_tokens) as the strategy selector.
    pub fn set_assistant_confidence_threshold(&mut self, value: f32) -> Result<()> {
        try_unsafe!(
            sys::ov_genai_sd_generation_config_set_assistant_confidence_threshold(self.ptr, value)
        )
    }

    fn as_ptr(&self) -> *const sys::ov_genai_sd_generation_config {
        self.ptr
    }
}

/// A draft-model speculative-decoding LLM pipeline.
///
/// Construct one with [`SpeculativeLlmPipeline::builder`]. The pipeline runs a "main" model
/// against a smaller "draft" model: the draft proposes candidate tokens which the main model
/// validates in a single forward pass, accepting a prefix when its distribution agrees.
pub struct SpeculativeLlmPipeline {
    ptr: *mut sys::ov_genai_sd_llm_pipeline,
}
drop_using_function!(SpeculativeLlmPipeline, sys::ov_genai_sd_llm_pipeline_free);

unsafe impl Send for SpeculativeLlmPipeline {}

/// Builder for [`SpeculativeLlmPipeline`].
pub struct SpeculativeLlmPipelineBuilder<'a> {
    main_path: &'a str,
    main_device: &'a str,
    draft_path: Option<&'a str>,
    draft_device: &'a str,
    main_properties: Vec<(&'a str, &'a str)>,
    draft_properties: Vec<(&'a str, &'a str)>,
}

impl<'a> SpeculativeLlmPipelineBuilder<'a> {
    /// Set the draft model directory and device.
    ///
    /// This must be called before [`build`](Self::build) — the pipeline is speculative by
    /// definition, so a draft model is mandatory.
    pub fn draft(mut self, path: &'a str, device: &'a str) -> Self {
        self.draft_path = Some(path);
        self.draft_device = device;
        self
    }

    /// Append a property for the main model. Property keys and values are passed through to
    /// OpenVINO as strings.
    pub fn main_property(mut self, key: &'a str, value: &'a str) -> Self {
        self.main_properties.push((key, value));
        self
    }

    /// Append a property for the draft model.
    pub fn draft_property(mut self, key: &'a str, value: &'a str) -> Self {
        self.draft_properties.push((key, value));
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> std::result::Result<SpeculativeLlmPipeline, SetupError> {
        sys::library::load().map_err(LoadingError::SystemFailure)?;
        let draft_path = self
            .draft_path
            .ok_or(SetupError::Loading(LoadingError::CannotFindLibraryPath))?;

        let main_path = cstr!(self.main_path);
        let main_device = cstr!(self.main_device);
        let draft_path_c = cstr!(draft_path);
        let draft_device = cstr!(self.draft_device);

        let main_kv = flatten_properties(&self.main_properties);
        let draft_kv = flatten_properties(&self.draft_properties);
        let main_ptrs = pointer_slice(&main_kv);
        let draft_ptrs = pointer_slice(&draft_kv);

        let mut ptr = std::ptr::null_mut();
        try_unsafe!(sys::ov_genai_sd_llm_pipeline_create(
            main_path.as_ptr(),
            main_device.as_ptr(),
            self.main_properties.len(),
            main_ptrs.as_ptr(),
            draft_path_c.as_ptr(),
            draft_device.as_ptr(),
            self.draft_properties.len(),
            draft_ptrs.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(SpeculativeLlmPipeline { ptr })
    }
}

impl SpeculativeLlmPipeline {
    /// Start building a speculative pipeline with the main model path and device. Call
    /// [`SpeculativeLlmPipelineBuilder::draft`] to specify the draft model before
    /// [`build`](SpeculativeLlmPipelineBuilder::build).
    pub fn builder<'a>(
        main_path: &'a str,
        main_device: &'a str,
    ) -> SpeculativeLlmPipelineBuilder<'a> {
        SpeculativeLlmPipelineBuilder {
            main_path,
            main_device,
            draft_path: None,
            draft_device: "",
            main_properties: Vec::new(),
            draft_properties: Vec::new(),
        }
    }

    /// Generate text from a prompt.
    ///
    /// Optionally pass a [`SpeculativeGenerationConfig`] and/or a [`Streamer`] callback.
    pub fn generate(
        &mut self,
        prompt: &str,
        config: Option<&SpeculativeGenerationConfig>,
        streamer: Option<&Streamer>,
    ) -> Result<SpeculativeDecodedResults> {
        let prompt = cstr!(prompt);
        let config_ptr = config.map_or(std::ptr::null(), SpeculativeGenerationConfig::as_ptr);
        let streamer_raw = streamer.map(Streamer::as_raw);
        let streamer_ptr = streamer_raw
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref);
        let mut results_ptr = std::ptr::null_mut();
        try_unsafe!(sys::ov_genai_sd_llm_pipeline_generate(
            self.ptr,
            prompt.as_ptr(),
            config_ptr,
            streamer_ptr,
            std::ptr::addr_of_mut!(results_ptr)
        ))?;
        Ok(SpeculativeDecodedResults { ptr: results_ptr })
    }

    /// Begin a chat session that retains KV cache between calls.
    pub fn start_chat(&mut self) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_llm_pipeline_start_chat(self.ptr))
    }

    /// End a chat session and clear the KV cache.
    pub fn finish_chat(&mut self) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_llm_pipeline_finish_chat(self.ptr))
    }

    /// Set the default generation config for subsequent [`generate`](Self::generate) calls.
    pub fn set_generation_config(&mut self, config: &SpeculativeGenerationConfig) -> Result<()> {
        try_unsafe!(sys::ov_genai_sd_llm_pipeline_set_generation_config(
            self.ptr,
            config.as_ptr()
        ))
    }
}

/// Results from a speculative-decoding generate call.
pub struct SpeculativeDecodedResults {
    ptr: *mut sys::ov_genai_sd_decoded_results,
}
drop_using_function!(
    SpeculativeDecodedResults,
    sys::ov_genai_sd_decoded_results_free
);

impl SpeculativeDecodedResults {
    /// Get the generated text as a single string. Multi-sequence outputs are concatenated as
    /// they are by `ov::genai::DecodedResults::operator std::string()`.
    pub fn get_string(&self) -> Result<String> {
        unsafe {
            crate::util::get_c_string_two_call(|buf, size| {
                sys::ov_genai_sd_decoded_results_get_string(self.ptr, buf, size)
            })
        }
    }

    /// Extract the speculative-decoding performance metrics, if present.
    ///
    /// Returns `Ok(None)` only if the underlying `extended_perf_metrics` was not an
    /// `SDPerModelsPerfMetrics`. For pipelines created through [`SpeculativeLlmPipeline`] this
    /// is always `Some`.
    pub fn get_sd_perf_metrics(&self) -> Result<Option<SdPerfMetrics>> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(sys::ov_genai_sd_decoded_results_get_sd_perf_metrics(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(SdPerfMetrics { ptr }))
    }
}

/// Speculative-decoding performance metrics for both the main and draft model, plus the
/// number of tokens accepted from the draft.
pub struct SdPerfMetrics {
    ptr: *mut sys::ov_genai_sd_perf_metrics,
}
drop_using_function!(SdPerfMetrics, sys::ov_genai_sd_perf_metrics_free);

impl SdPerfMetrics {
    /// Number of draft-proposed tokens accepted by the main model across the call. Total
    /// generated tokens equals this plus one main-model token per validation step.
    pub fn num_accepted_tokens(&self) -> Result<usize> {
        let mut out: usize = 0;
        try_unsafe!(sys::ov_genai_sd_perf_metrics_get_num_accepted_tokens(
            self.ptr, &mut out
        ))?;
        Ok(out)
    }

    /// Metrics for the main model.
    pub fn main_model_metrics(&self) -> SdModelMetrics<'_> {
        SdModelMetrics {
            parent: self,
            side: 0,
        }
    }

    /// Metrics for the draft model.
    pub fn draft_model_metrics(&self) -> SdModelMetrics<'_> {
        SdModelMetrics {
            parent: self,
            side: 1,
        }
    }
}

/// Borrowed view onto one side (main or draft) of the [`SdPerfMetrics`].
pub struct SdModelMetrics<'a> {
    parent: &'a SdPerfMetrics,
    side: i32,
}

impl SdModelMetrics<'_> {
    /// Mean and standard deviation of Time-to-First-Token (ms).
    pub fn ttft(&self) -> Result<(f32, f32)> {
        self.mean_std(sys::ov_genai_sd_perf_metrics_get_ttft)
    }

    /// Mean and standard deviation of Time-to-Second-Token (ms).
    pub fn ttst(&self) -> Result<(f32, f32)> {
        self.mean_std(sys::ov_genai_sd_perf_metrics_get_ttst)
    }

    /// Mean and standard deviation of Time-per-Output-Token (ms/token).
    pub fn tpot(&self) -> Result<(f32, f32)> {
        self.mean_std(sys::ov_genai_sd_perf_metrics_get_tpot)
    }

    /// Mean and standard deviation of inference latency from the third token onward (ms).
    pub fn latency(&self) -> Result<(f32, f32)> {
        self.mean_std(sys::ov_genai_sd_perf_metrics_get_latency)
    }

    /// Mean and standard deviation of total generate-call duration (ms).
    pub fn generate_duration(&self) -> Result<(f32, f32)> {
        self.mean_std(sys::ov_genai_sd_perf_metrics_get_generate_duration)
    }

    /// Total number of tokens generated by this side.
    pub fn num_generated_tokens(&self) -> Result<usize> {
        let mut out: usize = 0;
        try_unsafe!(sys::ov_genai_sd_perf_metrics_get_num_generated_tokens(
            self.parent.ptr,
            self.side,
            &mut out
        ))?;
        Ok(out)
    }

    fn mean_std(
        &self,
        f: unsafe extern "C" fn(
            *const sys::ov_genai_sd_perf_metrics,
            i32,
            *mut f32,
            *mut f32,
        ) -> sys::ov_status_e,
    ) -> Result<(f32, f32)> {
        let mut mean: f32 = 0.0;
        let mut std: f32 = 0.0;
        try_unsafe!(f(self.parent.ptr, self.side, &mut mean, &mut std))?;
        Ok((mean, std))
    }
}

// Helpers --------------------------------------------------------------------

fn flatten_properties(properties: &[(&str, &str)]) -> Vec<CString> {
    properties
        .iter()
        .flat_map(|(k, v)| [cstr!(*k), cstr!(*v)])
        .collect()
}

fn pointer_slice(properties: &[CString]) -> Vec<*const c_char> {
    properties.iter().map(|s| s.as_ptr()).collect()
}
