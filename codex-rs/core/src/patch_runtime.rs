//! Narrow, effect-free-at-the-boundary runtime entry point for Patch.
//!
//! This facade intentionally owns no Codex session, thread manager, MCP runtime, or tools. Each
//! invocation resolves normal persisted configuration and authentication, obtains a fresh effective
//! model catalog entry, and performs exactly one full-input Responses WebSocket request.

use std::path::PathBuf;
use std::sync::Arc;

use codex_login::AuthManager;
use codex_login::auth::AgentIdentityAuthPolicy;
use codex_model_provider::create_model_provider;
use codex_models_manager::manager::RefreshStrategy;
use codex_otel::SessionTelemetry;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::SessionSource;

use crate::ModelClient;
use crate::Prompt;
use crate::ResponseStream;
use crate::config::ConfigBuilder;
use crate::installation_id::resolve_installation_id;
use crate::responses_metadata::CodexResponsesMetadata;

/// Inputs for one isolated Patch-to-Codex request.
///
/// `input` is the caller's canonical, already-ordered conversation input. The facade never adds
/// tools, performs continuation, retries, prewarming, or HTTP fallback. `session_id` and `turn_id`
/// are Patch logical identifiers and are copied into ordinary Responses metadata.
pub struct PatchRuntimeRequest {
    /// Codex home containing the normal persisted config and auth files.
    pub codex_home: PathBuf,
    /// Exact model slug requested by Patch.
    pub model: String,
    /// Canonical conversation input, without facade-generated items.
    pub input: Vec<ResponseItem>,
    /// Base instructions for this request.
    pub base_instructions: BaseInstructions,
    /// Explicit reasoning effort, if any.
    pub effort: Option<ReasoningEffort>,
    /// Explicit reasoning summary mode.
    pub summary: ReasoningSummary,
    /// Stable Patch logical session identifier.
    pub session_id: String,
    /// Stable Patch logical turn identifier.
    pub turn_id: String,
    /// Sink receiving raw request/output protocol values.
    pub raw_sink: Arc<dyn codex_api::RawResponseSink>,
}

/// Resolve normal Codex runtime state and make exactly one strict full-input WebSocket request.
///
/// Configuration and authentication are loaded lazily inside this invocation. The model catalog is
/// refreshed with authenticated normal provider behavior and the requested slug must be present as
/// an exact effective catalog entry; bundled/fallback model metadata is rejected. Only Responses
/// Lite models on WebSocket-capable Responses providers are accepted.
pub async fn stream_patch_runtime_once(request: PatchRuntimeRequest) -> Result<ResponseStream> {
    let config = ConfigBuilder::default()
        .codex_home(request.codex_home)
        .build()
        .await
        .map_err(|err| CodexErr::Fatal(format!("failed to load Codex config: {err}")))?;

    let auth_manager =
        AuthManager::shared_from_config(&config, /*enable_codex_api_key_env*/ false)
            .await
            .map_err(|err| CodexErr::Fatal(format!("failed to initialize Codex auth: {err}")))?;

    // Loading auth here makes the persisted-auth refresh path part of catalog resolution while
    // retaining support for providers whose credentials come from their configured environment.
    let _ = auth_manager.auth().await;

    let provider_info = config.model_provider.clone();
    if provider_info.wire_api != codex_model_provider_info::WireApi::Responses {
        return Err(CodexErr::UnsupportedOperation(
            "Patch runtime requires the Responses API provider".to_string(),
        ));
    }
    if !provider_info.supports_websockets {
        return Err(CodexErr::UnsupportedOperation(
            "Patch runtime requires a WebSocket-capable provider".to_string(),
        ));
    }

    let provider = create_model_provider(provider_info.clone(), Some(Arc::clone(&auth_manager)));
    let models_manager = provider.models_manager_without_cache(config.model_catalog.clone());
    let catalog = models_manager
        .raw_model_catalog(RefreshStrategy::Online, config.http_client_factory())
        .await;
    let model_info = catalog
        .models
        .into_iter()
        .find(|candidate| candidate.slug == request.model)
        .ok_or_else(|| {
            CodexErr::InvalidRequest(format!(
                "requested model `{}` is not present in the effective authenticated catalog",
                request.model
            ))
        })?;
    if model_info.used_fallback_model_metadata {
        return Err(CodexErr::InvalidRequest(format!(
            "requested model `{}` resolved to fallback metadata",
            request.model
        )));
    }
    if !model_info.supported_in_api || !model_info.use_responses_lite {
        return Err(CodexErr::UnsupportedOperation(format!(
            "requested model `{}` does not support Responses Lite",
            request.model
        )));
    }

    let thread_id = ThreadId::new();
    let installation_id = resolve_installation_id(&config.codex_home)
        .await
        .map_err(|err| {
            CodexErr::Fatal(format!("failed to resolve Codex installation ID: {err}"))
        })?;
    let responses_metadata = CodexResponsesMetadata::new(
        installation_id,
        request.session_id,
        thread_id.to_string(),
        "patchwork".to_string(),
    )
    .with_turn_id(request.turn_id);
    let telemetry = SessionTelemetry::new(
        thread_id,
        model_info.slug.as_str(),
        model_info.slug.as_str(),
        /*account_id*/ None,
        /*account_email*/ None,
        /*auth_mode*/ None,
        "patch".to_string(),
        /*log_user_prompts*/ false,
        "patchwork".to_string(),
        SessionSource::Custom("patch".to_string()),
    );
    let client = ModelClient::new(
        Some(auth_manager),
        AgentIdentityAuthPolicy::JwtOnly,
        thread_id,
        provider_info,
        SessionSource::Custom("patch".to_string()),
        "patch".to_string(),
        /*model_verbosity*/ None,
        /*enable_request_compression*/ false,
        /*include_timing_metrics*/ false,
        /*beta_features_header*/ None,
        /*concurrent_reasoning_summaries_enabled*/ false,
        /*attestation_provider*/ None,
        config.http_client_factory(),
    );
    let prompt = Prompt::new_no_tools(request.input, request.base_instructions);
    client
        .new_fresh_session()
        .stream_full_input_websocket_once_with_raw_sink(
            &prompt,
            &model_info,
            &telemetry,
            request.effort,
            request.summary,
            /*service_tier*/ None,
            &responses_metadata,
            request.raw_sink,
        )
        .await
}
