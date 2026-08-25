use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::responses_websocket::RawResponseSink;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::requests::headers::build_session_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Serialize;
use serde_json::Value;
use serde_json::value::RawValue;
use std::sync::Arc;
use std::sync::OnceLock;
use tracing::instrument;

pub struct ResponsesClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn crate::telemetry::SseTelemetry>>,
}

#[derive(Default)]
pub struct ResponsesOptions {
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub session_source: Option<SessionSource>,
    pub extra_headers: HeaderMap,
    pub compression: Compression,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}

#[derive(Serialize)]
struct RawResponsesApiRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    instructions: &'a str,
    input: Vec<&'a RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a crate::common::ResponsesApiTools>,
    tool_choice: &'a str,
    parallel_tool_calls: bool,
    reasoning: Option<&'a crate::common::Reasoning>,
    store: bool,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<&'a crate::common::StreamOptions>,
    include: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a crate::common::TextControls>,
    #[serde(skip_serializing_if = "Option::is_none")]
    client_metadata: Option<&'a std::collections::HashMap<String, String>>,
}

impl<'a> RawResponsesApiRequest<'a> {
    fn new(
        request: &'a ResponsesApiRequest,
        retained_prefix: &'a [Box<RawValue>],
        fresh_suffix: &'a [Box<RawValue>],
    ) -> Self {
        let mut input = Vec::with_capacity(retained_prefix.len() + fresh_suffix.len());
        input.extend(retained_prefix.iter().map(std::convert::AsRef::as_ref));
        input.extend(fresh_suffix.iter().map(std::convert::AsRef::as_ref));
        Self {
            model: &request.model,
            instructions: &request.instructions,
            input,
            tools: request.tools.as_ref(),
            tool_choice: &request.tool_choice,
            parallel_tool_calls: request.parallel_tool_calls,
            reasoning: request.reasoning.as_ref(),
            store: request.store,
            stream: request.stream,
            stream_options: request.stream_options.as_ref(),
            include: &request.include,
            service_tier: request.service_tier.as_deref(),
            prompt_cache_key: request.prompt_cache_key.as_deref(),
            text: request.text.as_ref(),
            client_metadata: request.client_metadata.as_ref(),
        }
    }
}

#[cfg(test)]
fn encode_raw_body_for_test(
    request: &ResponsesApiRequest,
    retained_prefix: &[Box<RawValue>],
    fresh_suffix: &[Box<RawValue>],
) -> String {
    let body = EncodedJsonBody::encode(&RawResponsesApiRequest::new(
        request,
        retained_prefix,
        fresh_suffix,
    ))
    .expect("raw request should encode");
    String::from_utf8(body.as_bytes().to_vec()).expect("encoded body should be UTF-8")
}

impl<T: HttpTransport> ResponsesClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn crate::telemetry::SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    #[instrument(
        name = "responses.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "responses_http",
            http.method = "POST",
            api.path = "responses"
        )
    )]
    pub async fn stream_request(
        &self,
        request: ResponsesApiRequest,
        options: ResponsesOptions,
    ) -> Result<ResponseStream, ApiError> {
        let ResponsesOptions {
            session_id,
            thread_id,
            session_source,
            extra_headers,
            compression,
            turn_state,
        } = options;

        let body = EncodedJsonBody::encode(&request)
            .map_err(|e| ApiError::Stream(format!("failed to encode responses request: {e}")))?;

        let mut headers = extra_headers;
        if let Some(thread_id) = &thread_id {
            insert_header(&mut headers, "x-client-request-id", thread_id);
        }
        headers.extend(build_session_headers(session_id, thread_id));
        if let Some(subagent) = subagent_header(&session_source) {
            insert_header(&mut headers, "x-openai-subagent", &subagent);
        }

        self.stream_encoded(body, headers, compression, turn_state)
            .await
    }

    /// Streams a Responses request with an exact raw input prefix followed by a fresh raw suffix.
    /// Retained items are serialized directly as [`RawValue`] tokens and are never typed-decoded.
    pub async fn stream_request_with_raw_input(
        &self,
        request: &ResponsesApiRequest,
        retained_prefix: &[Box<RawValue>],
        fresh_suffix: &[Box<RawValue>],
        options: ResponsesOptions,
        raw_sink: Option<Arc<dyn RawResponseSink>>,
    ) -> Result<ResponseStream, ApiError> {
        let body = EncodedJsonBody::encode(&RawResponsesApiRequest::new(
            request,
            retained_prefix,
            fresh_suffix,
        ))
        .map_err(|e| ApiError::Stream(format!("failed to encode raw responses request: {e}")))?;
        let ResponsesOptions {
            session_id,
            thread_id,
            session_source,
            mut extra_headers,
            compression,
            turn_state,
        } = options;
        if let Some(raw_sink) = raw_sink.as_ref() {
            let mut items = retained_prefix
                .iter()
                .chain(fresh_suffix)
                .map(|item| {
                    RawValue::from_string(item.get().to_owned()).map_err(|err| {
                        ApiError::Stream(format!("failed to retain raw HTTP input: {err}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            raw_sink
                .record_request_input(std::mem::take(&mut items))
                .await?;
            raw_sink.authorize_transport_dispatch().await?;
        }
        if let Some(thread_id) = &thread_id {
            insert_header(&mut extra_headers, "x-client-request-id", thread_id);
        }
        extra_headers.extend(build_session_headers(session_id, thread_id));
        if let Some(subagent) = subagent_header(&session_source) {
            insert_header(&mut extra_headers, "x-openai-subagent", &subagent);
        }
        self.stream_encoded_with_raw_sink(body, extra_headers, compression, turn_state, raw_sink)
            .await
    }

    fn path() -> &'static str {
        "responses"
    }

    #[instrument(
        name = "responses.stream",
        level = "info",
        skip_all,
        fields(
            transport = "responses_http",
            http.method = "POST",
            api.path = "responses",
            turn.has_state = turn_state.is_some()
        )
    )]
    pub async fn stream(
        &self,
        body: Value,
        extra_headers: HeaderMap,
        compression: Compression,
        turn_state: Option<Arc<OnceLock<String>>>,
    ) -> Result<ResponseStream, ApiError> {
        let body = EncodedJsonBody::encode(&body)
            .map_err(|e| ApiError::Stream(format!("failed to encode responses request: {e}")))?;
        self.stream_encoded(body, extra_headers, compression, turn_state)
            .await
    }

    async fn stream_encoded(
        &self,
        body: EncodedJsonBody,
        extra_headers: HeaderMap,
        compression: Compression,
        turn_state: Option<Arc<OnceLock<String>>>,
    ) -> Result<ResponseStream, ApiError> {
        self.stream_encoded_with_raw_sink(body, extra_headers, compression, turn_state, None)
            .await
    }

    async fn stream_encoded_with_raw_sink(
        &self,
        body: EncodedJsonBody,
        extra_headers: HeaderMap,
        compression: Compression,
        turn_state: Option<Arc<OnceLock<String>>>,
        raw_sink: Option<Arc<dyn RawResponseSink>>,
    ) -> Result<ResponseStream, ApiError> {
        let request_compression = match compression {
            Compression::None => RequestCompression::None,
            Compression::Zstd => RequestCompression::Zstd,
        };

        let stream_response = self
            .session
            .stream_encoded_json_with(
                Method::POST,
                Self::path(),
                extra_headers,
                Some(body),
                |req| {
                    req.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    req.compression = request_compression;
                },
            )
            .await?;

        Ok(crate::sse::spawn_response_stream_with_raw_sink(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            turn_state,
            raw_sink,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn raw_http_body_preserves_unknown_item_tokens() {
        let request = ResponsesApiRequest {
            model: "gpt-test".to_string(),
            instructions: String::new(),
            input: Vec::new(),
            tools: None,
            tool_choice: "auto".to_string(),
            parallel_tool_calls: false,
            reasoning: None,
            store: false,
            stream: true,
            stream_options: None,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };
        let prefix = vec![
            RawValue::from_string(r#"{ "type":"future_item", "opaque": {"a":1} }"#.to_string())
                .unwrap(),
        ];
        let suffix = vec![
            RawValue::from_string(r#"{"type":"future_variant","x":[1,2]}"#.to_string()).unwrap(),
        ];
        let body = encode_raw_body_for_test(&request, &prefix, &suffix);
        let value: Value = serde_json::from_str(&body).expect("valid HTTP body");
        assert_eq!(
            value["input"][0],
            json!({"type":"future_item", "opaque":{"a":1}})
        );
        assert_eq!(
            value["input"][1],
            json!({"type":"future_variant", "x":[1,2]})
        );
        assert!(body.contains(r#"{ "type":"future_item", "opaque": {"a":1} }"#));
    }
}
