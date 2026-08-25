//! Narrow, effect-free-at-the-boundary runtime entry point for Patch.
//!
//! This facade intentionally owns no Codex session, thread manager, MCP runtime, or tools. A
//! process-local owner retains one ModelClient while each Patch logical turn gets a fresh
//! ModelClientSession. Typed turns use official cached-WebSocket continuation machinery; raw turns
//! frame retained native JSON directly and may use the maintained HTTP fallback.
//!
//! The legacy one-shot entry point below remains strict full-input WebSocket-only for compatibility.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::value::RawValue;
use sha2::Digest;
use sha2::Sha256;

use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_login::auth::AgentIdentityAuthPolicy;
use codex_model_provider::create_model_provider;
use codex_model_provider_info::ModelProviderInfo;
use codex_models_manager::manager::ModelCatalogProvenance;
use codex_models_manager::manager::ModelCatalogResolution;
use codex_models_manager::manager::RefreshStrategy;
use codex_otel::SessionTelemetry;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::SessionSource;
use codex_rollout_trace::InferenceTraceContext;

use crate::ModelClient;
use crate::Prompt;
use crate::ResponseStream;
use crate::client::FullInputRoute;
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

/// The stream and the authenticated compatibility metadata that produced it.
pub struct PatchRuntimeStream {
    pub stream: ResponseStream,
    pub route: PatchRuntimeRoute,
    pub auth_profile_scope: String,
    pub capability_revision: String,
}

/// Resolve normal Codex runtime state and make exactly one strict full-input WebSocket request.
///
/// Configuration and authentication are loaded lazily inside this invocation. The model catalog
/// uses Codex's normal cache-aware refresh behavior: only a fresh authenticated remote catalog or
/// Codex's still-valid version-matched cache is accepted as capability evidence. The requested
/// slug must be present as an exact effective catalog entry; bundled, configured, and fallback
/// metadata is rejected. Only Responses Lite models on WebSocket-capable Responses providers are
/// accepted.
pub async fn stream_patch_runtime_once(request: PatchRuntimeRequest) -> Result<PatchRuntimeStream> {
    let config = ConfigBuilder::default()
        .codex_home(request.codex_home)
        .build()
        .await
        .map_err(|err| CodexErr::Fatal(format!("failed to load Codex config: {err}")))?;

    let auth_manager =
        AuthManager::shared_from_config(&config, /*enable_codex_api_key_env*/ false)
            .await
            .map_err(|err| CodexErr::Fatal(format!("failed to initialize Codex auth: {err}")))?;

    // Loading auth here makes the persisted-auth refresh path part of catalog resolution.
    let auth = auth_manager.auth().await;

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
    if !provider_info.is_openai() || !provider_info.requires_openai_auth {
        return Err(CodexErr::UnsupportedOperation(
            "Patch runtime requires the first-party OpenAI provider".to_string(),
        ));
    }
    let auth_profile_scope = auth
        .as_ref()
        .filter(|auth| auth.uses_codex_backend())
        .and_then(CodexAuth::get_account_id)
        .map(|account_id| digest_compatibility_value("chatgpt-account", account_id.as_bytes()))
        .ok_or_else(|| {
            CodexErr::InvalidRequest(
                "Patch runtime requires authenticated ChatGPT account scope".to_string(),
            )
        })?;

    if config.model_catalog.is_some() {
        return Err(CodexErr::InvalidRequest(
            "Patch runtime does not accept configured static model metadata as capability evidence"
                .to_string(),
        ));
    }

    let provider = create_model_provider(provider_info.clone(), Some(Arc::clone(&auth_manager)));
    let models_manager = provider.models_manager(
        config.codex_home.to_path_buf(),
        config.model_catalog.clone(),
    );
    let catalog = models_manager
        .raw_model_catalog_with_provenance(
            RefreshStrategy::OnlineIfUncached,
            config.http_client_factory(),
        )
        .await
        .map_err(|err| {
            CodexErr::Fatal(format!(
                "failed to resolve the effective authenticated model catalog: {err}"
            ))
        })?;
    let capability_revision = serde_json::to_vec(&catalog.catalog.models)
        .map(|catalog| digest_compatibility_value("catalog", &catalog))
        .map_err(|err| {
            CodexErr::Fatal(format!(
                "failed to serialize effective model catalog: {err}"
            ))
        })?;
    let model_info =
        validate_patch_model_catalog(catalog, &request.model, request.effort.as_ref())?;

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
    let stream = client
        .new_fresh_session()
        .stream_full_input_websocket_once_with_raw_sink(
            &prompt,
            &model_info,
            &telemetry,
            request.effort,
            request.summary,
            /*service_tier*/ None,
            &responses_metadata,
            &auth_profile_scope,
            &capability_revision,
            request.raw_sink,
        )
        .await?;
    Ok(PatchRuntimeStream {
        stream,
        route: PatchRuntimeRoute::WebSocket,
        auth_profile_scope,
        capability_revision,
    })
}

/// Stable process-local Patch runtime configuration. The resulting owner is safe to retain for
/// multiple logical turns; it owns one Codex [`ModelClient`] and never owns tools or effects.
pub struct PatchRuntimeConfig {
    pub codex_home: PathBuf,
    pub model: String,
    pub effort: Option<ReasoningEffort>,
}

/// Route selected for one completed Patch request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchRuntimeRoute {
    WebSocket,
    HttpFallback,
}

/// Terminal failure classes that an adapter may use to choose canonical replay recovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchRuntimeFailure {
    NoCompletedResponse,
    ReplayRejected,
}

/// Result of checking whether a successive turn may reuse this runtime lease.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PatchRuntimeCompatibility {
    Compatible,
    Incompatible(String),
}

/// Classifies only failures that require Patch recovery policy; unrelated provider errors remain
/// unclassified so the adapter can surface them without treating them as replay decisions.
pub fn classify_patch_runtime_error(error: &CodexErr) -> Option<PatchRuntimeFailure> {
    use codex_protocol::error::CodexErrorDetails;
    match error.details() {
        CodexErrorDetails::Stream(message)
            if message.contains("before response.completed")
                || message.contains("before completion") =>
        {
            Some(PatchRuntimeFailure::NoCompletedResponse)
        }
        CodexErrorDetails::InvalidRequest(message)
            if message.contains("previous response")
                || message.contains("replay")
                || message.contains("strict prefix") =>
        {
            Some(PatchRuntimeFailure::ReplayRejected)
        }
        _ => None,
    }
}

/// Input mode for one Patch logical turn.
pub enum PatchRuntimeInput {
    /// Canonical typed input. The maintained ModelClientSession may use its cached WebSocket
    /// previous-response-id and strict-prefix machinery when this baseline is faithful.
    Typed(Vec<ResponseItem>),
    /// Lossless native replay input. Raw tokens are sent directly, without typed preparation.
    Raw {
        retained_prefix: Vec<Box<RawValue>>,
        fresh_suffix: Vec<Box<RawValue>>,
    },
}

/// Request-specific input and Patch metadata for a fresh logical turn.
pub struct PatchRuntimeTurnRequest {
    pub input: PatchRuntimeInput,
    pub base_instructions: BaseInstructions,
    pub summary: ReasoningSummary,
    pub session_id: String,
    pub turn_id: String,
    pub raw_sink: Arc<dyn codex_api::RawResponseSink>,
}

/// One fresh ModelClientSession. Construct one per Patch logical turn.
pub struct PatchRuntimeTurn {
    runtime: Arc<PatchRuntimeInner>,
    request: PatchRuntimeTurnRequest,
    session: crate::client::ModelClientSession,
}

struct PatchRuntimeInner {
    client: ModelClient,
    model_info: ModelInfo,
    requested_model: String,
    requested_effort: Option<ReasoningEffort>,
    provider_info: ModelProviderInfo,
    codex_home: PathBuf,
    thread_id: ThreadId,
    installation_id: String,
    auth_profile_scope: String,
    capability_revision: String,
}

/// Long-lived process-local Patch runtime owner.
pub struct PatchRuntime {
    inner: Arc<PatchRuntimeInner>,
}

impl PatchRuntime {
    /// Resolves Codex configuration, authenticated capability metadata, and one process-local
    /// ModelClient. Catalog capability resolution follows Codex's normal authenticated cache/remote
    /// path; model-generation requests begin only when a turn is streamed.
    pub async fn new(config: PatchRuntimeConfig) -> Result<Self> {
        let PatchRuntimeConfig {
            codex_home,
            model,
            effort,
        } = config;
        let config_state = ConfigBuilder::default()
            .codex_home(codex_home)
            .build()
            .await
            .map_err(|err| CodexErr::Fatal(format!("failed to load Codex config: {err}")))?;
        let auth_manager =
            AuthManager::shared_from_config(&config_state, /*enable_codex_api_key_env*/ false)
                .await
                .map_err(|err| {
                    CodexErr::Fatal(format!("failed to initialize Codex auth: {err}"))
                })?;
        let auth = auth_manager.auth().await;
        let provider_info = config_state.model_provider.clone();
        if provider_info.wire_api != codex_model_provider_info::WireApi::Responses {
            return Err(CodexErr::UnsupportedOperation(
                "Patch runtime requires the Responses API provider".to_string(),
            ));
        }
        let auth_profile_scope = auth
            .as_ref()
            .filter(|auth| auth.uses_codex_backend())
            .and_then(CodexAuth::get_account_id)
            .map(|account_id| digest_compatibility_value("chatgpt-account", account_id.as_bytes()))
            .ok_or_else(|| {
                CodexErr::InvalidRequest(
                    "Patch runtime requires authenticated ChatGPT account scope".to_string(),
                )
            })?;
        if config_state.model_catalog.is_some() {
            return Err(CodexErr::InvalidRequest(
                "Patch runtime does not accept configured static model metadata as capability evidence"
                    .to_string(),
            ));
        }
        let provider =
            create_model_provider(provider_info.clone(), Some(Arc::clone(&auth_manager)));
        let models_manager = provider.models_manager(
            config_state.codex_home.to_path_buf(),
            config_state.model_catalog.clone(),
        );
        let catalog = models_manager
            .raw_model_catalog_with_provenance(
                RefreshStrategy::OnlineIfUncached,
                config_state.http_client_factory(),
            )
            .await
            .map_err(|err| {
                CodexErr::Fatal(format!(
                    "failed to resolve the effective authenticated model catalog: {err}"
                ))
            })?;
        let capability_revision = serde_json::to_vec(&catalog.catalog.models)
            .map(|catalog| digest_compatibility_value("catalog", &catalog))
            .map_err(|err| {
                CodexErr::Fatal(format!(
                    "failed to serialize effective model catalog: {err}"
                ))
            })?;
        let model_info = validate_patch_model_catalog(catalog, &model, effort.as_ref())?;
        let thread_id = ThreadId::new();
        let installation_id = resolve_installation_id(&config_state.codex_home)
            .await
            .map_err(|err| {
                CodexErr::Fatal(format!("failed to resolve Codex installation ID: {err}"))
            })?;
        let client = ModelClient::new(
            Some(auth_manager),
            AgentIdentityAuthPolicy::JwtOnly,
            thread_id,
            provider_info.clone(),
            SessionSource::Custom("patch".to_string()),
            "patch".to_string(),
            None,
            false,
            false,
            None,
            false,
            None,
            config_state.http_client_factory(),
        );
        Ok(Self {
            inner: Arc::new(PatchRuntimeInner {
                client,
                model_info,
                requested_model: model,
                requested_effort: effort,
                provider_info: provider_info.clone(),
                codex_home: config_state.codex_home.clone().to_path_buf(),
                thread_id,
                installation_id,
                auth_profile_scope,
                capability_revision,
            }),
        })
    }

    /// Re-resolves normal config, authenticated account scope, provider route, and effective
    /// model catalog before a successive lease turn. An incompatible result means the adapter must
    /// discard this owner and construct a new one.
    pub async fn check_compatibility(&self) -> Result<PatchRuntimeCompatibility> {
        let config = ConfigBuilder::default()
            .codex_home(self.inner.codex_home.clone())
            .build()
            .await
            .map_err(|err| CodexErr::Fatal(format!("failed to refresh Codex config: {err}")))?;
        if config.model_provider != self.inner.provider_info {
            return Ok(PatchRuntimeCompatibility::Incompatible(
                "effective provider route changed".to_string(),
            ));
        }
        if config.model_catalog.is_some() {
            return Ok(PatchRuntimeCompatibility::Incompatible(
                "configured model metadata became active".to_string(),
            ));
        }
        let auth_manager =
            AuthManager::shared_from_config(&config, /*enable_codex_api_key_env*/ false)
                .await
                .map_err(|err| CodexErr::Fatal(format!("failed to refresh Codex auth: {err}")))?;
        let auth_profile_scope = auth_manager
            .auth()
            .await
            .as_ref()
            .filter(|auth| auth.uses_codex_backend())
            .and_then(CodexAuth::get_account_id)
            .map(|account_id| digest_compatibility_value("chatgpt-account", account_id.as_bytes()));
        if auth_profile_scope.as_deref() != Some(self.inner.auth_profile_scope.as_str()) {
            return Ok(PatchRuntimeCompatibility::Incompatible(
                "authenticated account scope changed".to_string(),
            ));
        }
        let provider = create_model_provider(
            config.model_provider.clone(),
            Some(Arc::clone(&auth_manager)),
        );
        let catalog = provider
            .models_manager(
                config.codex_home.to_path_buf(),
                config.model_catalog.clone(),
            )
            .raw_model_catalog_with_provenance(
                RefreshStrategy::OnlineIfUncached,
                config.http_client_factory(),
            )
            .await
            .map_err(|err| {
                CodexErr::Fatal(format!("failed to refresh effective model catalog: {err}"))
            })?;
        let capability_revision = serde_json::to_vec(&catalog.catalog.models)
            .map(|catalog| digest_compatibility_value("catalog", &catalog))
            .map_err(|err| {
                CodexErr::Fatal(format!(
                    "failed to serialize effective model catalog: {err}"
                ))
            })?;
        if capability_revision != self.inner.capability_revision {
            return Ok(PatchRuntimeCompatibility::Incompatible(
                "effective model catalog changed".to_string(),
            ));
        }
        match validate_patch_model_catalog(
            catalog,
            &self.inner.requested_model,
            self.inner.requested_effort.as_ref(),
        ) {
            Ok(_) => Ok(PatchRuntimeCompatibility::Compatible),
            Err(error) => Ok(PatchRuntimeCompatibility::Incompatible(error.to_string())),
        }
    }

    /// Returns the nonsecret compatibility envelope bound at runtime construction.
    /// Callers use it only to reject incompatible opaque checkpoint custody.
    pub fn compatibility_envelope(&self) -> (&str, &str) {
        (
            &self.inner.auth_profile_scope,
            &self.inner.capability_revision,
        )
    }

    /// Creates a fresh turn session while retaining the process-local client owner.
    pub fn new_turn(&self, request: PatchRuntimeTurnRequest) -> PatchRuntimeTurn {
        PatchRuntimeTurn {
            runtime: Arc::clone(&self.inner),
            request,
            session: self.inner.client.new_session(),
        }
    }
}

impl PatchRuntimeTurn {
    /// Streams one raw full-input turn. The route is explicit and is either WebSocket or the
    /// maintained supported HTTP fallback; no unbounded retry policy is enabled here.
    pub async fn stream(mut self) -> Result<PatchRuntimeStream> {
        let PatchRuntimeTurnRequest {
            input,
            base_instructions,
            summary,
            session_id,
            turn_id,
            raw_sink,
        } = self.request;
        let responses_metadata = CodexResponsesMetadata::new(
            self.runtime.installation_id.clone(),
            session_id,
            self.runtime.thread_id.to_string(),
            "patchwork".to_string(),
        )
        .with_turn_id(turn_id);
        let telemetry = SessionTelemetry::new(
            self.runtime.thread_id,
            self.runtime.model_info.slug.as_str(),
            self.runtime.model_info.slug.as_str(),
            None,
            None,
            None,
            "patch".to_string(),
            false,
            "patchwork".to_string(),
            SessionSource::Custom("patch".to_string()),
        );
        let (stream, route) = match input {
            PatchRuntimeInput::Typed(items) => {
                let websocket_was_enabled = self.runtime.client.responses_websocket_enabled();
                let prompt = Prompt::new_no_tools(items, base_instructions);
                let stream = self
                    .session
                    .stream_with_raw_sink(
                        &prompt,
                        &self.runtime.model_info,
                        &telemetry,
                        None,
                        summary,
                        None,
                        &responses_metadata,
                        &InferenceTraceContext::disabled(),
                        Some(Arc::clone(&raw_sink)),
                    )
                    .await?;
                let route =
                    if websocket_was_enabled && self.runtime.client.responses_websocket_enabled() {
                        PatchRuntimeRoute::WebSocket
                    } else {
                        PatchRuntimeRoute::HttpFallback
                    };
                (stream, route)
            }
            PatchRuntimeInput::Raw {
                retained_prefix,
                fresh_suffix,
            } => {
                let prompt = Prompt::new_no_tools(Vec::new(), base_instructions);
                let result = self
                    .session
                    .stream_full_input_with_raw_items(
                        &prompt,
                        &self.runtime.model_info,
                        &telemetry,
                        None,
                        summary,
                        None,
                        &responses_metadata,
                        &retained_prefix,
                        &fresh_suffix,
                        raw_sink,
                    )
                    .await?;
                let route = match result.route {
                    FullInputRoute::WebSocket => PatchRuntimeRoute::WebSocket,
                    FullInputRoute::Http => PatchRuntimeRoute::HttpFallback,
                };
                (result.stream, route)
            }
        };
        Ok(PatchRuntimeStream {
            stream,
            route,
            auth_profile_scope: self.runtime.auth_profile_scope.clone(),
            capability_revision: self.runtime.capability_revision.clone(),
        })
    }
}

fn validate_patch_model_catalog(
    resolution: ModelCatalogResolution,
    requested_model: &str,
    effort: Option<&ReasoningEffort>,
) -> Result<ModelInfo> {
    if resolution.provenance == ModelCatalogProvenance::Untrusted {
        return Err(CodexErr::InvalidRequest(
            "Patch runtime requires fresh official model metadata or a valid official Codex cache"
                .to_string(),
        ));
    }
    let model_info = resolution
        .catalog
        .models
        .into_iter()
        .find(|candidate| candidate.slug == requested_model)
        .ok_or_else(|| {
            CodexErr::InvalidRequest(format!(
                "requested model `{requested_model}` is not present in the trusted effective catalog"
            ))
        })?;
    if model_info.used_fallback_model_metadata {
        return Err(CodexErr::InvalidRequest(format!(
            "requested model `{requested_model}` resolved to fallback metadata"
        )));
    }
    if !model_info.supported_in_api || !model_info.use_responses_lite {
        return Err(CodexErr::UnsupportedOperation(format!(
            "requested model `{requested_model}` does not support Responses Lite"
        )));
    }
    if let Some(effort) = effort
        && !model_info
            .supported_reasoning_levels
            .iter()
            .any(|preset| preset.effort == *effort)
    {
        return Err(CodexErr::InvalidRequest(format!(
            "requested reasoning effort is not supported by the effective model `{requested_model}`"
        )));
    }
    Ok(model_info)
}

pub(crate) fn digest_compatibility_value(domain: &str, value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(value);
    let digest = hasher.finalize();
    format!("sha256:{digest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::openai_models::ModelsResponse;
    use serde_json::json;

    fn sol_model() -> ModelInfo {
        serde_json::from_value(json!({
            "slug": "gpt-5.6-sol",
            "display_name": "Sol",
            "description": "fixture",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "low"},
                {"effort": "medium", "description": "medium"}
            ],
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": true,
            "priority": 0,
            "upgrade": null,
            "model_messages": null,
            "support_verbosity": false,
            "default_verbosity": null,
            "apply_patch_tool_type": null,
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "supports_image_detail_original": false,
            "context_window": 272000,
            "max_context_window": 272000,
            "experimental_supported_tools": [],
            "use_responses_lite": true
        }))
        .expect("valid Sol fixture")
    }

    fn resolution(provenance: ModelCatalogProvenance, model: ModelInfo) -> ModelCatalogResolution {
        ModelCatalogResolution {
            catalog: ModelsResponse {
                models: vec![model],
            },
            provenance,
        }
    }

    #[test]
    fn accepts_fresh_official_remote_catalog() {
        let model = validate_patch_model_catalog(
            resolution(ModelCatalogProvenance::FreshRemote, sol_model()),
            "gpt-5.6-sol",
            Some(&ReasoningEffort::Medium),
        )
        .expect("fresh authenticated catalog is accepted");

        assert_eq!(model.slug, "gpt-5.6-sol");
    }

    #[test]
    fn accepts_valid_official_cached_catalog() {
        let model = validate_patch_model_catalog(
            resolution(ModelCatalogProvenance::ValidCache, sol_model()),
            "gpt-5.6-sol",
            Some(&ReasoningEffort::Low),
        )
        .expect("Codex-validated catalog cache is accepted");

        assert!(model.use_responses_lite);
    }

    #[test]
    fn rejects_untrusted_catalog_metadata() {
        let error = validate_patch_model_catalog(
            resolution(ModelCatalogProvenance::Untrusted, sol_model()),
            "gpt-5.6-sol",
            None,
        )
        .expect_err("bundled, configured, and fallback metadata is not production evidence");

        assert!(error.to_string().contains("fresh official model metadata"));
    }

    #[test]
    fn rejects_unsupported_reasoning_effort_before_dispatch() {
        let error = validate_patch_model_catalog(
            resolution(ModelCatalogProvenance::FreshRemote, sol_model()),
            "gpt-5.6-sol",
            Some(&ReasoningEffort::High),
        )
        .expect_err("unsupported effort must not reach dispatch");

        assert!(error.to_string().contains("reasoning effort"));
    }

    #[test]
    fn retains_exact_model_and_responses_lite_checks() {
        let missing_model = validate_patch_model_catalog(
            resolution(ModelCatalogProvenance::FreshRemote, sol_model()),
            "gpt-5.6-sol-other",
            None,
        )
        .expect_err("effective model slug must match exactly");
        assert!(missing_model.to_string().contains("not present"));

        let mut no_lite = sol_model();
        no_lite.use_responses_lite = false;
        let error = validate_patch_model_catalog(
            resolution(ModelCatalogProvenance::FreshRemote, no_lite),
            "gpt-5.6-sol",
            None,
        )
        .expect_err("Responses Lite remains mandatory");
        assert!(error.to_string().contains("Responses Lite"));

        let mut fallback = sol_model();
        fallback.used_fallback_model_metadata = true;
        let error = validate_patch_model_catalog(
            resolution(ModelCatalogProvenance::FreshRemote, fallback),
            "gpt-5.6-sol",
            None,
        )
        .expect_err("fallback metadata remains rejected");
        assert!(error.to_string().contains("fallback metadata"));
    }
}
