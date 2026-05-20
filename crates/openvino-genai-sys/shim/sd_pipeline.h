// Copyright (C) 2026 openvino-rs contributors
// SPDX-License-Identifier: Apache-2.0
//
// Flat C ABI exposing draft-model speculative decoding from OpenVINO GenAI's
// C++ API. The public OpenVINO GenAI C API (runtime/include/openvino/genai/c)
// does not include any draft-model entry point, so this shim wraps the C++
// `ov::genai::LLMPipeline(main, device, draft_model(...))` constructor and the
// `SDPerModelsPerfMetrics` payload behind `DecodedResults::extended_perf_metrics`.
//
// The shim owns parallel opaque types end-to-end (pipeline / decoded-results /
// perf-metrics / generation-config). It does not reuse the public C-API
// opaques because their internals are not public.

#pragma once

#include <stddef.h>

#include "openvino/c/ov_common.h"               // ov_status_e
#include "openvino/genai/c/llm_pipeline.h"      // streamer_callback

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ov_genai_sd_generation_config_t ov_genai_sd_generation_config;
typedef struct ov_genai_sd_llm_pipeline_t      ov_genai_sd_llm_pipeline;
typedef struct ov_genai_sd_decoded_results_t   ov_genai_sd_decoded_results;
typedef struct ov_genai_sd_perf_metrics_t      ov_genai_sd_perf_metrics;

// ---- generation config -----------------------------------------------------

ov_status_e ov_genai_sd_generation_config_create(ov_genai_sd_generation_config** out);
void        ov_genai_sd_generation_config_free(ov_genai_sd_generation_config* cfg);

ov_status_e ov_genai_sd_generation_config_set_max_new_tokens(ov_genai_sd_generation_config* cfg, size_t value);
ov_status_e ov_genai_sd_generation_config_set_max_length(ov_genai_sd_generation_config* cfg, size_t value);
ov_status_e ov_genai_sd_generation_config_set_temperature(ov_genai_sd_generation_config* cfg, float value);
ov_status_e ov_genai_sd_generation_config_set_top_p(ov_genai_sd_generation_config* cfg, float value);
ov_status_e ov_genai_sd_generation_config_set_top_k(ov_genai_sd_generation_config* cfg, size_t value);
ov_status_e ov_genai_sd_generation_config_set_do_sample(ov_genai_sd_generation_config* cfg, int value);
ov_status_e ov_genai_sd_generation_config_set_num_beams(ov_genai_sd_generation_config* cfg, size_t value);
ov_status_e ov_genai_sd_generation_config_set_repetition_penalty(ov_genai_sd_generation_config* cfg, float value);
ov_status_e ov_genai_sd_generation_config_set_presence_penalty(ov_genai_sd_generation_config* cfg, float value);
ov_status_e ov_genai_sd_generation_config_set_frequency_penalty(ov_genai_sd_generation_config* cfg, float value);
ov_status_e ov_genai_sd_generation_config_set_rng_seed(ov_genai_sd_generation_config* cfg, size_t value);
ov_status_e ov_genai_sd_generation_config_set_num_assistant_tokens(ov_genai_sd_generation_config* cfg, size_t value);
ov_status_e ov_genai_sd_generation_config_set_assistant_confidence_threshold(ov_genai_sd_generation_config* cfg, float value);

// ---- pipeline --------------------------------------------------------------

// Construct a speculative-decoding LLMPipeline.
//
// Properties are flat key/value pairs: `[k0, v0, k1, v1, ..., kN-1, vN-1]`,
// length = 2 * n_props. All keys and values are NUL-terminated UTF-8.
//
// `main_device` / `draft_device` may be NULL (treated as empty string, letting
// OpenVINO pick the default).
ov_status_e ov_genai_sd_llm_pipeline_create(
    const char* main_path,
    const char* main_device,
    size_t n_main_props,    const char* const* main_kv_flat,
    const char* draft_path,
    const char* draft_device,
    size_t n_draft_props,   const char* const* draft_kv_flat,
    ov_genai_sd_llm_pipeline** out);

void ov_genai_sd_llm_pipeline_free(ov_genai_sd_llm_pipeline* pipe);

// Generate. `cfg` and `streamer` may both be NULL.
ov_status_e ov_genai_sd_llm_pipeline_generate(
    ov_genai_sd_llm_pipeline* pipe,
    const char* prompt,
    const ov_genai_sd_generation_config* cfg,
    const streamer_callback* streamer,
    ov_genai_sd_decoded_results** out_results);

ov_status_e ov_genai_sd_llm_pipeline_start_chat(ov_genai_sd_llm_pipeline* pipe);
ov_status_e ov_genai_sd_llm_pipeline_finish_chat(ov_genai_sd_llm_pipeline* pipe);
ov_status_e ov_genai_sd_llm_pipeline_set_generation_config(ov_genai_sd_llm_pipeline* pipe, const ov_genai_sd_generation_config* cfg);

// ---- results ---------------------------------------------------------------

void ov_genai_sd_decoded_results_free(ov_genai_sd_decoded_results* results);

// Two-call string getter (matches the public C-API convention): pass `buf =
// NULL` first to read the required size, then a buffer of `*size` bytes.
ov_status_e ov_genai_sd_decoded_results_get_string(
    const ov_genai_sd_decoded_results* results,
    char* buf,
    size_t* size);

// Returns `*out = NULL` and `OK` if the result has no SD perf metrics attached
// (i.e. the underlying `extended_perf_metrics` was not an SDPerModelsPerfMetrics).
ov_status_e ov_genai_sd_decoded_results_get_sd_perf_metrics(
    const ov_genai_sd_decoded_results* results,
    ov_genai_sd_perf_metrics** out);

// ---- perf metrics ----------------------------------------------------------

void ov_genai_sd_perf_metrics_free(ov_genai_sd_perf_metrics* metrics);

ov_status_e ov_genai_sd_perf_metrics_get_num_accepted_tokens(
    const ov_genai_sd_perf_metrics* metrics,
    size_t* out);

// `side`: 0 = main model, 1 = draft model.
ov_status_e ov_genai_sd_perf_metrics_get_ttft     (const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_ttst     (const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_tpot     (const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_latency  (const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_generate_duration(const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_num_generated_tokens(const ov_genai_sd_perf_metrics* m, int side, size_t* out);

#ifdef __cplusplus
}
#endif
