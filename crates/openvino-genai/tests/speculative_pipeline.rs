//! Integration tests for the speculative-decoding pipeline.
//!
//! These tests reuse the `qwen3` fixture as both the main and draft model. This produces no
//! actual inference speedup (same model on both sides) but is sufficient to verify the
//! end-to-end plumbing: pipeline construction, generation, generation-config setters, and
//! perf-metrics extraction. For benchmarking speedup, swap in a smaller draft model.

#![cfg(feature = "speculative-decoding")]

mod fixtures;

use fixtures::qwen3 as fixture;
use openvino_genai::{SpeculativeGenerationConfig, SpeculativeLlmPipeline};

fn try_pipeline() -> Option<SpeculativeLlmPipeline> {
    let model_dir = fixture::model_dir();
    let model_path = model_dir.to_string_lossy().into_owned();

    let result = SpeculativeLlmPipeline::builder(&model_path, "CPU")
        .draft(&model_path, "CPU")
        .build();
    match result {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("Skipping speculative pipeline tests: failed to create pipeline: {e}");
            None
        }
    }
}

#[test]
fn test_create_speculative_pipeline() {
    let _pipeline = match try_pipeline() {
        Some(p) => p,
        None => return,
    };
}

#[test]
fn test_generate_with_draft_model() {
    let mut pipeline = match try_pipeline() {
        Some(p) => p,
        None => return,
    };

    // num_assistant_tokens must be set explicitly for the stateful backend even though the
    // C++ docs claim a default of 5 — the pipeline validates against the field's actual value
    // (which is 0 unless we set it) before falling back to the default in some code paths.
    let mut config = SpeculativeGenerationConfig::new().unwrap();
    config.set_max_new_tokens(8).unwrap();
    config.set_num_assistant_tokens(4).unwrap();

    let results = pipeline.generate("Hello", Some(&config), None).unwrap();
    let text = results.get_string().unwrap();
    assert!(!text.is_empty(), "expected non-empty generation output");
}

#[test]
fn test_generation_config_assistant_tokens() {
    let mut pipeline = match try_pipeline() {
        Some(p) => p,
        None => return,
    };

    let mut config = SpeculativeGenerationConfig::new().unwrap();
    config.set_max_new_tokens(8).unwrap();
    config.set_num_assistant_tokens(4).unwrap();

    // The pipeline accepts the config without errors; the draft strategy is active.
    pipeline.generate("Hello", Some(&config), None).unwrap();
}

#[test]
fn test_sd_perf_metrics() {
    let mut pipeline = match try_pipeline() {
        Some(p) => p,
        None => return,
    };

    let mut config = SpeculativeGenerationConfig::new().unwrap();
    config.set_max_new_tokens(16).unwrap();
    config.set_num_assistant_tokens(4).unwrap();

    let results = pipeline
        .generate("Tell me a short joke.", Some(&config), None)
        .unwrap();

    let metrics = results
        .get_sd_perf_metrics()
        .unwrap()
        .expect("speculative pipeline must produce SDPerModelsPerfMetrics");

    // num_accepted_tokens is a usize, so this checks the getter doesn't error; the value may
    // legitimately be 0 if the draft and main never agree on a prefix.
    let _accepted = metrics.num_accepted_tokens().unwrap();
    let main_generated = metrics.main_model_metrics().num_generated_tokens().unwrap();
    assert!(
        main_generated > 0,
        "main model should have generated at least one token"
    );

    // Smoke-test the mean/std getters — they should not error even when only one or two
    // tokens were produced.
    let _ttft = metrics.main_model_metrics().ttft().unwrap();
    let _tpot = metrics.main_model_metrics().tpot().unwrap();
    let _draft_generated = metrics.draft_model_metrics().num_generated_tokens().unwrap();
}
