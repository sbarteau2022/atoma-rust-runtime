//! Production HTTP/TLS edge for the Rust control plane.
//!
//! This intentionally keeps parsing and authorization at the one network boundary. Requests
//! enter as bounded HTTP/1.1 frames, API keys are compared as SHA-256 digests, and map writes
//! advance the worker-cache generation before another governed forward pass can consume atoms.

use crate::MonsterHybridServer;
use atoma_immune::{RoleId, Sha256Digest};
use rustls::{ServerConfig, ServerConnection, StreamOwned};
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub bind_addr: String,
    pub api_key_digests: Vec<Sha256Digest>,
    pub tls_cert_path: Option<PathBuf>,
    pub tls_key_path: Option<PathBuf>,
    pub backup_dir: PathBuf,
    pub backup_retention: usize,
    pub max_requests: usize,
}

impl ApiConfig {
    /// Reads comma-separated API keys from `ATOMA_API_KEYS`; plaintext keys are never retained.
    pub fn from_env() -> Result<Self, String> {
        let raw_keys = std::env::var("ATOMA_API_KEYS")
            .map_err(|_| "ATOMA_API_KEYS must contain at least one API key".to_string())?;
        let api_key_digests: Vec<_> = raw_keys
            .split(',')
            .filter(|k| !k.trim().is_empty())
            .map(|k| Sha256Digest::compute(k.trim().as_bytes()))
            .collect();
        if api_key_digests.is_empty() {
            return Err("ATOMA_API_KEYS must contain at least one non-empty API key".to_string());
        }
        let cert = std::env::var_os("ATOMA_TLS_CERT").map(PathBuf::from);
        let key = std::env::var_os("ATOMA_TLS_KEY").map(PathBuf::from);
        if cert.is_some() != key.is_some() {
            return Err("ATOMA_TLS_CERT and ATOMA_TLS_KEY must be configured together".to_string());
        }
        Ok(Self {
            bind_addr: std::env::var("ATOMA_BIND").unwrap_or_else(|_| "127.0.0.1:8443".into()),
            api_key_digests,
            tls_cert_path: cert,
            tls_key_path: key,
            backup_dir: std::env::var_os("ATOMA_BACKUP_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("backups")),
            backup_retention: std::env::var("ATOMA_BACKUP_RETENTION")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(7),
            max_requests: std::env::var("ATOMA_MAX_REQUESTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(usize::MAX),
        })
    }

    fn tls_config(&self) -> Result<Option<Arc<ServerConfig>>, String> {
        let (Some(cert_path), Some(key_path)) = (&self.tls_cert_path, &self.tls_key_path) else {
            return Ok(None);
        };
        let mut cert_reader =
            BufReader::new(File::open(cert_path).map_err(|e| {
                format!("cannot open TLS certificate {}: {e}", cert_path.display())
            })?);
        let certs: Vec<_> = rustls_pemfile::certs(&mut cert_reader)
            .collect::<Result<_, _>>()
            .map_err(|e| format!("invalid TLS certificate: {e}"))?;
        let mut key_reader = BufReader::new(
            File::open(key_path)
                .map_err(|e| format!("cannot open TLS key {}: {e}", key_path.display()))?,
        );
        let key = rustls_pemfile::private_key(&mut key_reader)
            .map_err(|e| format!("invalid TLS key: {e}"))?
            .ok_or_else(|| "TLS key file contains no private key".to_string())?;
        let config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| format!("invalid TLS key/certificate pair: {e}"))?;
        Ok(Some(Arc::new(config)))
    }
}

#[derive(Debug, Default)]
pub struct ApiMetrics {
    started_at: Option<Instant>,
    requests_total: u64,
    auth_failures_total: u64,
    client_errors_total: u64,
    server_errors_total: u64,
    query_requests_total: u64,
    query_latency_us_total: u128,
    backups_total: u64,
}

impl ApiMetrics {
    fn prometheus(&self, server: &MonsterHybridServer) -> String {
        let uptime = self.started_at.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        format!(
            "# TYPE atoma_http_requests_total counter\natoma_http_requests_total {}\n# TYPE atoma_auth_failures_total counter\natoma_auth_failures_total {}\n# TYPE atoma_query_requests_total counter\natoma_query_requests_total {}\n# TYPE atoma_query_latency_microseconds_total counter\natoma_query_latency_microseconds_total {}\n# TYPE atoma_backups_total counter\natoma_backups_total {}\n# TYPE atoma_map_generation gauge\natoma_map_generation {}\n# TYPE atoma_global_atoms gauge\natoma_global_atoms {}\n# TYPE atoma_reasoning_arcs gauge\natoma_reasoning_arcs {}\n# TYPE atoma_uptime_seconds gauge\natoma_uptime_seconds {}\n",
            self.requests_total,
            self.auth_failures_total,
            self.query_requests_total,
            self.query_latency_us_total,
            self.backups_total,
            server.map_generation,
            server.swarm_map.global_atom_count(),
            server.swarm_map.reasoning_arcs.len(),
            uptime
        )
    }
}

#[derive(Deserialize)]
struct QueryRequest {
    query: String,
    role: Option<String>,
    top_k: Option<usize>,
    max_hops: Option<usize>,
    learn: Option<bool>,
}

/// Runs the authenticated HTTP server. TLS is mandatory for non-loopback binds.
pub fn serve(server: &mut MonsterHybridServer, config: ApiConfig) -> Result<(), String> {
    let tls = config.tls_config()?;
    if tls.is_none() && !is_loopback(&config.bind_addr) {
        return Err(
            "refusing plaintext non-loopback bind; configure ATOMA_TLS_CERT and ATOMA_TLS_KEY"
                .into(),
        );
    }
    std::fs::create_dir_all(&config.backup_dir)
        .map_err(|e| format!("cannot create backup directory: {e}"))?;
    let listener = TcpListener::bind(&config.bind_addr).map_err(|e| e.to_string())?;
    let mut metrics = ApiMetrics {
        started_at: Some(Instant::now()),
        ..ApiMetrics::default()
    };
    for incoming in listener.incoming().take(config.max_requests) {
        let Ok(stream) = incoming else { continue };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(15)));
        match &tls {
            Some(tls_config) => {
                let conn =
                    ServerConnection::new(Arc::clone(tls_config)).map_err(|e| e.to_string())?;
                handle_connection(
                    StreamOwned::new(conn, stream),
                    server,
                    &config,
                    &mut metrics,
                )?;
            }
            None => handle_connection(stream, server, &config, &mut metrics)?,
        }
    }
    Ok(())
}

fn handle_connection<S: Read + Write>(
    mut stream: S,
    server: &mut MonsterHybridServer,
    config: &ApiConfig,
    metrics: &mut ApiMetrics,
) -> Result<(), String> {
    let response = match read_request(&mut stream) {
        Ok(request) => route(request, server, config, metrics),
        Err(e) => {
            metrics.client_errors_total += 1;
            response(400, json!({"error": e}))
        }
    };
    stream.write_all(&response).map_err(|e| e.to_string())
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn read_request<S: Read>(stream: &mut S) -> Result<Request, String> {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 2048];
    let header_end = loop {
        if bytes.len() > MAX_HEADER_BYTES {
            return Err("headers exceed limit".into());
        }
        if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let read = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if read == 0 {
            return Err("unexpected EOF".into());
        }
        bytes.extend_from_slice(&chunk[..read]);
    };
    let header = std::str::from_utf8(&bytes[..header_end]).map_err(|_| "headers are not UTF-8")?;
    let mut lines = header.split("\r\n");
    let start = lines.next().ok_or("missing request line")?;
    let mut parts = start.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_string();
    let path = parts
        .next()
        .ok_or("missing target")?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();
    if parts.next().is_none() {
        return Err("malformed request line".into());
    }
    let mut headers = Vec::new();
    let mut content_length = 0usize;
    let mut saw_content_length = false;
    for line in lines.filter(|l| !l.is_empty()) {
        let (name, value) = line.split_once(':').ok_or("malformed header")?;
        if name.eq_ignore_ascii_case("transfer-encoding")
            && !value.trim().eq_ignore_ascii_case("identity")
        {
            return Err("transfer-encoding is not supported".into());
        }
        if name.eq_ignore_ascii_case("content-length") {
            if saw_content_length {
                return Err("duplicate content-length".into());
            }
            saw_content_length = true;
            content_length = value.trim().parse().map_err(|_| "invalid content-length")?;
        }
        headers.push((name.to_ascii_lowercase(), value.trim().to_string()));
    }
    if content_length > MAX_BODY_BYTES {
        return Err("body exceeds limit".into());
    }
    let mut body = bytes[header_end..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("truncated body".into());
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

fn route(
    req: Request,
    server: &mut MonsterHybridServer,
    config: &ApiConfig,
    metrics: &mut ApiMetrics,
) -> Vec<u8> {
    metrics.requests_total += 1;
    if req.method == "GET" && req.path == "/health/live" {
        return response(200, json!({"status":"live"}));
    }
    if !authorized(&req, &config.api_key_digests) {
        metrics.auth_failures_total += 1;
        return response(401, json!({"error":"unauthorized"}));
    }
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health/ready") => response(
            200,
            json!({"status":"ready", "map_generation":server.map_generation, "global_atoms":server.swarm_map.global_atom_count()}),
        ),
        ("GET", "/metrics") => {
            text_response(200, "text/plain; version=0.0.4", metrics.prometheus(server))
        }
        ("POST", "/v1/query") => query(req.body, server, metrics),
        ("POST", "/v1/admin/backup") => match snapshot(server, config, metrics) {
            Ok(path) => response(201, json!({"backup":path})),
            Err(e) => {
                metrics.server_errors_total += 1;
                response(500, json!({"error":e}))
            }
        },
        _ => {
            metrics.client_errors_total += 1;
            response(404, json!({"error":"not found"}))
        }
    }
}

fn query(body: Vec<u8>, server: &mut MonsterHybridServer, metrics: &mut ApiMetrics) -> Vec<u8> {
    let request: QueryRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return response(400, json!({"error":"invalid JSON query request"})),
    };
    if request.query.trim().is_empty() || request.query.len() > 16_384 {
        return response(400, json!({"error":"query must be 1..16384 bytes"}));
    }
    let role = request.role.as_deref().and_then(parse_role);
    if request.role.is_some() && role.is_none() {
        return response(400, json!({"error":"unknown role"}));
    }
    let started = Instant::now();
    match server.query_swarm_map(
        &request.query,
        role,
        request.top_k.unwrap_or(5).clamp(1, 64),
        request.max_hops.unwrap_or(6).clamp(0, 64),
        request.learn.unwrap_or(false),
    ) {
        Ok(result) => {
            let latency = started.elapsed().as_micros();
            metrics.query_requests_total += 1;
            metrics.query_latency_us_total += latency;
            response(
                200,
                json!({"query":result.query_text,"routed_role":result.routed_role.id_slug(),"sector":result.query_sector,"global_atoms":result.global_atom_count,"reasoning_arcs":result.global_reasoning_arc_count,"hits":result.hits.iter().map(|h| json!({"atom_id":h.atom_id.0,"distance":h.cosh_distance_q32.to_f64()})).collect::<Vec<_>>(),"latency_us":latency}),
            )
        }
        Err(e) => {
            metrics.server_errors_total += 1;
            response(422, json!({"error":e}))
        }
    }
}

fn snapshot(
    server: &MonsterHybridServer,
    config: &ApiConfig,
    metrics: &mut ApiMetrics,
) -> Result<String, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    let final_path = config.backup_dir.join(format!("map-{stamp}.atoma_map"));
    let temp_path = config.backup_dir.join(format!(".map-{stamp}.tmp"));
    server.swarm_map.save_to_binary_vault(&temp_path)?;
    std::fs::rename(&temp_path, &final_path)
        .map_err(|e| format!("atomic backup rename failed: {e}"))?;
    let mut files: Vec<_> = std::fs::read_dir(&config.backup_dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "atoma_map"))
        .collect();
    files.sort();
    while files.len() > config.backup_retention {
        if let Some(old) = files.first().cloned() {
            std::fs::remove_file(old).map_err(|e| e.to_string())?;
            files.remove(0);
        }
    }
    metrics.backups_total += 1;
    Ok(final_path.display().to_string())
}

fn authorized(req: &Request, allowed: &[Sha256Digest]) -> bool {
    let key = req
        .headers
        .iter()
        .find(|(name, _)| name == "x-api-key")
        .map(|(_, value)| value.as_bytes())
        .unwrap_or_default();
    let candidate = Sha256Digest::compute(key);
    // Evaluate every configured digest to avoid exposing which key matched through early exit.
    let mut match_byte = 0u8;
    for digest in allowed {
        let mut diff = 0u8;
        for (a, b) in digest.0.iter().zip(candidate.0.iter()) {
            diff |= a ^ b;
        }
        match_byte |= (diff == 0) as u8;
    }
    match_byte == 1
}

fn parse_role(value: &str) -> Option<RoleId> {
    RoleId::ALL_13.into_iter().find(|r| {
        r.id_slug().eq_ignore_ascii_case(value)
            || r.short_code().eq_ignore_ascii_case(value)
            || r.name().eq_ignore_ascii_case(value)
    })
}
fn is_loopback(bind: &str) -> bool {
    bind.starts_with("127.") || bind.starts_with("[::1]") || bind.starts_with("localhost:")
}
fn response(status: u16, body: Value) -> Vec<u8> {
    text_response(status, "application/json", body.to_string())
}
fn text_response(status: u16, content_type: &str, body: String) -> Vec<u8> {
    let reason = match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        422 => "Unprocessable Content",
        _ => "Internal Server Error",
    };
    format!("HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\nCache-Control: no-store\r\n\r\n{body}", body.len()).into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loopback_plaintext_is_permitted_but_public_is_not() {
        assert!(is_loopback("127.0.0.1:8443"));
        assert!(!is_loopback("0.0.0.0:8443"));
    }
    #[test]
    fn auth_requires_exact_key() {
        let allowed = vec![Sha256Digest::compute(b"correct")];
        let req = Request {
            method: "GET".into(),
            path: "/".into(),
            headers: vec![("x-api-key".into(), "correct".into())],
            body: vec![],
        };
        assert!(authorized(&req, &allowed));
        let denied = Request {
            headers: vec![],
            ..req
        };
        assert!(!authorized(&denied, &allowed));
    }

    #[test]
    fn parser_rejects_ambiguous_body_framing() {
        let duplicate =
            b"POST /v1/query HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx";
        assert!(read_request(&mut std::io::Cursor::new(duplicate)).is_err());
        let chunked = b"POST /v1/query HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
        assert!(read_request(&mut std::io::Cursor::new(chunked)).is_err());
    }
}
