// Copyright (C) 2026 openvino-rs contributors
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of the speculative-decoding shim. See sd_pipeline.h.

#include "sd_pipeline.h"

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// Opaque wrappers ------------------------------------------------------------

struct ov_genai_sd_generation_config_t {
    ov::genai::GenerationConfig cfg;
};

struct ov_genai_sd_llm_pipeline_t {
    std::unique_ptr<ov::genai::LLMPipeline> pipe;
};

struct ov_genai_sd_decoded_results_t {
    ov::genai::DecodedResults results;
};

struct ov_genai_sd_perf_metrics_t {
    std::shared_ptr<ov::genai::SDPerModelsPerfMetrics> metrics;
};

// Helpers --------------------------------------------------------------------

namespace {

template <typename F>
ov_status_e guarded(F&& fn) {
    try {
        fn();
        return OK;
    } catch (const std::exception&) {
        return GENERAL_ERROR;
    } catch (...) {
        return UNKNOW_EXCEPTION;
    }
}

ov::AnyMap make_anymap(size_t n_pairs, const char* const* kv_flat) {
    ov::AnyMap m;
    if (kv_flat == nullptr) {
        return m;
    }
    for (size_t i = 0; i < n_pairs; ++i) {
        const char* k = kv_flat[2 * i];
        const char* v = kv_flat[2 * i + 1];
        if (k == nullptr || v == nullptr) {
            continue;
        }
        m[std::string(k)] = std::string(v);
    }
    return m;
}

std::string c_string(const char* s) {
    return s ? std::string(s) : std::string();
}

const ov::genai::SDPerfMetrics* side_metrics(const ov_genai_sd_perf_metrics* m, int side) {
    if (!m || !m->metrics) {
        return nullptr;
    }
    if (side == 0) {
        return &m->metrics->main_model_metrics;
    }
    if (side == 1) {
        return &m->metrics->draft_model_metrics;
    }
    return nullptr;
}

ov_status_e write_mean_std(ov::genai::MeanStdPair pair, float* mean, float* std) {
    if (mean) {
        *mean = pair.mean;
    }
    if (std) {
        *std = pair.std;
    }
    return OK;
}

}  // namespace

// Generation config ----------------------------------------------------------

extern "C" ov_status_e ov_genai_sd_generation_config_create(ov_genai_sd_generation_config** out) {
    if (!out) {
        return INVALID_C_PARAM;
    }
    return guarded([&] { *out = new ov_genai_sd_generation_config_t{}; });
}

extern "C" void ov_genai_sd_generation_config_free(ov_genai_sd_generation_config* cfg) {
    delete cfg;
}

#define DEFINE_CFG_SETTER(suffix, field, c_type) \
    extern "C" ov_status_e ov_genai_sd_generation_config_set_##suffix( \
        ov_genai_sd_generation_config* cfg, c_type value) { \
        if (!cfg) return INVALID_C_PARAM; \
        return guarded([&] { cfg->cfg.field = value; }); \
    }

DEFINE_CFG_SETTER(max_new_tokens,                max_new_tokens,                size_t)
DEFINE_CFG_SETTER(max_length,                    max_length,                    size_t)
DEFINE_CFG_SETTER(temperature,                   temperature,                   float)
DEFINE_CFG_SETTER(top_p,                         top_p,                         float)
DEFINE_CFG_SETTER(top_k,                         top_k,                         size_t)
DEFINE_CFG_SETTER(num_beams,                     num_beams,                     size_t)
DEFINE_CFG_SETTER(repetition_penalty,            repetition_penalty,            float)
DEFINE_CFG_SETTER(presence_penalty,              presence_penalty,              float)
DEFINE_CFG_SETTER(frequency_penalty,             frequency_penalty,             float)
DEFINE_CFG_SETTER(rng_seed,                      rng_seed,                      size_t)
DEFINE_CFG_SETTER(num_assistant_tokens,          num_assistant_tokens,          size_t)
DEFINE_CFG_SETTER(assistant_confidence_threshold, assistant_confidence_threshold, float)

#undef DEFINE_CFG_SETTER

extern "C" ov_status_e ov_genai_sd_generation_config_set_do_sample(ov_genai_sd_generation_config* cfg, int value) {
    if (!cfg) {
        return INVALID_C_PARAM;
    }
    return guarded([&] { cfg->cfg.do_sample = value != 0; });
}

// Pipeline -------------------------------------------------------------------

extern "C" ov_status_e ov_genai_sd_llm_pipeline_create(
    const char* main_path,
    const char* main_device,
    size_t n_main_props,    const char* const* main_kv_flat,
    const char* draft_path,
    const char* draft_device,
    size_t n_draft_props,   const char* const* draft_kv_flat,
    ov_genai_sd_llm_pipeline** out)
{
    if (!main_path || !draft_path || !out) {
        return INVALID_C_PARAM;
    }
    return guarded([&] {
        ov::AnyMap main_props  = make_anymap(n_main_props,  main_kv_flat);
        ov::AnyMap draft_props = make_anymap(n_draft_props, draft_kv_flat);

        auto draft_entry = ov::genai::draft_model(
            std::filesystem::path(draft_path),
            c_string(draft_device),
            draft_props);
        main_props.insert(draft_entry);

        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(main_path),
            c_string(main_device),
            main_props);

        *out = new ov_genai_sd_llm_pipeline_t{std::move(pipe)};
    });
}

extern "C" void ov_genai_sd_llm_pipeline_free(ov_genai_sd_llm_pipeline* pipe) {
    delete pipe;
}

extern "C" ov_status_e ov_genai_sd_llm_pipeline_set_generation_config(
    ov_genai_sd_llm_pipeline* pipe,
    const ov_genai_sd_generation_config* cfg)
{
    if (!pipe || !pipe->pipe || !cfg) {
        return INVALID_C_PARAM;
    }
    return guarded([&] { pipe->pipe->set_generation_config(cfg->cfg); });
}

extern "C" ov_status_e ov_genai_sd_llm_pipeline_start_chat(ov_genai_sd_llm_pipeline* pipe) {
    if (!pipe || !pipe->pipe) {
        return INVALID_C_PARAM;
    }
    return guarded([&] { pipe->pipe->start_chat(); });
}

extern "C" ov_status_e ov_genai_sd_llm_pipeline_finish_chat(ov_genai_sd_llm_pipeline* pipe) {
    if (!pipe || !pipe->pipe) {
        return INVALID_C_PARAM;
    }
    return guarded([&] { pipe->pipe->finish_chat(); });
}

extern "C" ov_status_e ov_genai_sd_llm_pipeline_generate(
    ov_genai_sd_llm_pipeline* pipe,
    const char* prompt,
    const ov_genai_sd_generation_config* cfg,
    const streamer_callback* streamer,
    ov_genai_sd_decoded_results** out_results)
{
    if (!pipe || !pipe->pipe || !prompt || !out_results) {
        return INVALID_C_PARAM;
    }
    return guarded([&] {
        ov::genai::OptionalGenerationConfig gc = std::nullopt;
        if (cfg) {
            gc = cfg->cfg;
        }

        ov::genai::StreamerVariant sv = std::monostate{};
        if (streamer && streamer->callback_func) {
            auto cb = streamer->callback_func;
            void* args = streamer->args;
            sv = [cb, args](std::string token) -> ov::genai::StreamingStatus {
                auto status = cb(token.c_str(), args);
                switch (status) {
                    case OV_GENAI_STREAMING_STATUS_STOP:
                        return ov::genai::StreamingStatus::STOP;
                    case OV_GENAI_STREAMING_STATUS_CANCEL:
                        return ov::genai::StreamingStatus::CANCEL;
                    case OV_GENAI_STREAMING_STATUS_RUNNING:
                    default:
                        return ov::genai::StreamingStatus::RUNNING;
                }
            };
        }

        ov::genai::DecodedResults results = pipe->pipe->generate(
            ov::genai::StringInputs(std::string(prompt)), gc, sv);
        *out_results = new ov_genai_sd_decoded_results_t{std::move(results)};
    });
}

// Decoded results ------------------------------------------------------------

extern "C" void ov_genai_sd_decoded_results_free(ov_genai_sd_decoded_results* results) {
    delete results;
}

extern "C" ov_status_e ov_genai_sd_decoded_results_get_string(
    const ov_genai_sd_decoded_results* results,
    char* buf,
    size_t* size)
{
    if (!results || !size) {
        return INVALID_C_PARAM;
    }
    return guarded([&] {
        std::string text = static_cast<std::string>(results->results);
        size_t needed = text.size() + 1;  // include trailing NUL
        if (buf == nullptr) {
            *size = needed;
            return;
        }
        if (*size < needed) {
            *size = needed;
            throw std::runtime_error("buffer too small");  // mapped to GENERAL_ERROR
        }
        std::memcpy(buf, text.data(), text.size());
        buf[text.size()] = '\0';
        *size = needed;
    });
}

extern "C" ov_status_e ov_genai_sd_decoded_results_get_sd_perf_metrics(
    const ov_genai_sd_decoded_results* results,
    ov_genai_sd_perf_metrics** out)
{
    if (!results || !out) {
        return INVALID_C_PARAM;
    }
    return guarded([&] {
        auto ep = results->results.extended_perf_metrics;
        auto sd = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(ep);
        if (!sd) {
            *out = nullptr;
            return;
        }
        *out = new ov_genai_sd_perf_metrics_t{std::move(sd)};
    });
}

// Perf metrics ---------------------------------------------------------------

extern "C" void ov_genai_sd_perf_metrics_free(ov_genai_sd_perf_metrics* metrics) {
    delete metrics;
}

extern "C" ov_status_e ov_genai_sd_perf_metrics_get_num_accepted_tokens(
    const ov_genai_sd_perf_metrics* metrics,
    size_t* out)
{
    if (!metrics || !metrics->metrics || !out) {
        return INVALID_C_PARAM;
    }
    return guarded([&] { *out = metrics->metrics->get_num_accepted_tokens(); });
}

#define DEFINE_SIDE_MEAN_STD(suffix, expr)                                                  \
    extern "C" ov_status_e ov_genai_sd_perf_metrics_get_##suffix(                           \
        const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std) {             \
        auto* side_m = side_metrics(m, side);                                               \
        if (!side_m) return INVALID_C_PARAM;                                                \
        return guarded([&] {                                                                \
            auto& mutable_m = const_cast<ov::genai::SDPerfMetrics&>(*side_m);               \
            write_mean_std(expr, mean, std);                                                \
        });                                                                                  \
    }

DEFINE_SIDE_MEAN_STD(ttft,               mutable_m.get_ttft())
DEFINE_SIDE_MEAN_STD(ttst,               mutable_m.get_ttst())
DEFINE_SIDE_MEAN_STD(tpot,               mutable_m.get_tpot())
DEFINE_SIDE_MEAN_STD(latency,            mutable_m.get_latency())
DEFINE_SIDE_MEAN_STD(generate_duration,  mutable_m.get_generate_duration())

#undef DEFINE_SIDE_MEAN_STD

extern "C" ov_status_e ov_genai_sd_perf_metrics_get_num_generated_tokens(
    const ov_genai_sd_perf_metrics* m, int side, size_t* out)
{
    auto* side_m = side_metrics(m, side);
    if (!side_m || !out) {
        return INVALID_C_PARAM;
    }
    return guarded([&] {
        auto& mutable_m = const_cast<ov::genai::SDPerfMetrics&>(*side_m);
        *out = mutable_m.get_num_generated_tokens();
    });
}
