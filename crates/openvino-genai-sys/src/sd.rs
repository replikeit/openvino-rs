//! Hand-written FFI for the speculative-decoding shim compiled in `shim/sd_pipeline.cpp`.
//!
//! The public OpenVINO GenAI C API does not expose draft-model speculative decoding, so this
//! module's symbols come from the static `ov_genai_sd_shim` library built by `build.rs`.
//! The shim itself depends on `libopenvino_genai` (the C++ library), so this feature requires
//! `dynamic-linking` and is incompatible with `runtime-linking`.

#![cfg(feature = "speculative-decoding")]

#[cfg(feature = "runtime-linking")]
compile_error!(
    "the `speculative-decoding` feature requires `dynamic-linking`; \
     it is incompatible with `runtime-linking`"
);

use crate::ov_status_e;
use crate::streamer_callback;

/// Opaque speculative-decoding generation config.
#[repr(C)]
pub struct ov_genai_sd_generation_config {
    _unused: [u8; 0],
}

/// Opaque speculative-decoding LLM pipeline.
#[repr(C)]
pub struct ov_genai_sd_llm_pipeline {
    _unused: [u8; 0],
}

/// Opaque results returned from a speculative-decoding generate call.
#[repr(C)]
pub struct ov_genai_sd_decoded_results {
    _unused: [u8; 0],
}

/// Opaque speculative-decoding per-model performance metrics.
#[repr(C)]
pub struct ov_genai_sd_perf_metrics {
    _unused: [u8; 0],
}

unsafe extern "C" {
    pub fn ov_genai_sd_generation_config_create(
        out: *mut *mut ov_genai_sd_generation_config,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_free(cfg: *mut ov_genai_sd_generation_config);

    pub fn ov_genai_sd_generation_config_set_max_new_tokens(
        cfg: *mut ov_genai_sd_generation_config,
        value: usize,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_max_length(
        cfg: *mut ov_genai_sd_generation_config,
        value: usize,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_temperature(
        cfg: *mut ov_genai_sd_generation_config,
        value: f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_top_p(
        cfg: *mut ov_genai_sd_generation_config,
        value: f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_top_k(
        cfg: *mut ov_genai_sd_generation_config,
        value: usize,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_do_sample(
        cfg: *mut ov_genai_sd_generation_config,
        value: ::std::os::raw::c_int,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_num_beams(
        cfg: *mut ov_genai_sd_generation_config,
        value: usize,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_repetition_penalty(
        cfg: *mut ov_genai_sd_generation_config,
        value: f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_presence_penalty(
        cfg: *mut ov_genai_sd_generation_config,
        value: f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_frequency_penalty(
        cfg: *mut ov_genai_sd_generation_config,
        value: f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_rng_seed(
        cfg: *mut ov_genai_sd_generation_config,
        value: usize,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_num_assistant_tokens(
        cfg: *mut ov_genai_sd_generation_config,
        value: usize,
    ) -> ov_status_e;
    pub fn ov_genai_sd_generation_config_set_assistant_confidence_threshold(
        cfg: *mut ov_genai_sd_generation_config,
        value: f32,
    ) -> ov_status_e;

    pub fn ov_genai_sd_llm_pipeline_create(
        main_path: *const ::std::os::raw::c_char,
        main_device: *const ::std::os::raw::c_char,
        n_main_props: usize,
        main_kv_flat: *const *const ::std::os::raw::c_char,
        draft_path: *const ::std::os::raw::c_char,
        draft_device: *const ::std::os::raw::c_char,
        n_draft_props: usize,
        draft_kv_flat: *const *const ::std::os::raw::c_char,
        out: *mut *mut ov_genai_sd_llm_pipeline,
    ) -> ov_status_e;
    pub fn ov_genai_sd_llm_pipeline_free(pipe: *mut ov_genai_sd_llm_pipeline);

    pub fn ov_genai_sd_llm_pipeline_generate(
        pipe: *mut ov_genai_sd_llm_pipeline,
        prompt: *const ::std::os::raw::c_char,
        cfg: *const ov_genai_sd_generation_config,
        streamer: *const streamer_callback,
        out_results: *mut *mut ov_genai_sd_decoded_results,
    ) -> ov_status_e;
    pub fn ov_genai_sd_llm_pipeline_start_chat(pipe: *mut ov_genai_sd_llm_pipeline) -> ov_status_e;
    pub fn ov_genai_sd_llm_pipeline_finish_chat(pipe: *mut ov_genai_sd_llm_pipeline) -> ov_status_e;
    pub fn ov_genai_sd_llm_pipeline_set_generation_config(
        pipe: *mut ov_genai_sd_llm_pipeline,
        cfg: *const ov_genai_sd_generation_config,
    ) -> ov_status_e;

    pub fn ov_genai_sd_decoded_results_free(results: *mut ov_genai_sd_decoded_results);
    pub fn ov_genai_sd_decoded_results_get_string(
        results: *const ov_genai_sd_decoded_results,
        buf: *mut ::std::os::raw::c_char,
        size: *mut usize,
    ) -> ov_status_e;
    pub fn ov_genai_sd_decoded_results_get_sd_perf_metrics(
        results: *const ov_genai_sd_decoded_results,
        out: *mut *mut ov_genai_sd_perf_metrics,
    ) -> ov_status_e;

    pub fn ov_genai_sd_perf_metrics_free(metrics: *mut ov_genai_sd_perf_metrics);
    pub fn ov_genai_sd_perf_metrics_get_num_accepted_tokens(
        metrics: *const ov_genai_sd_perf_metrics,
        out: *mut usize,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_ttft(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_ttst(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_tpot(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_latency(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_generate_duration(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_num_generated_tokens(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        out: *mut usize,
    ) -> ov_status_e;
}
