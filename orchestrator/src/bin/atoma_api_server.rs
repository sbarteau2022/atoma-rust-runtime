//! Production entrypoint. Configure `ATOMA_API_KEYS` and, for public binds, TLS paths.
use atoma_immune::FrozenBaseWeights;
use atoma_orchestrator::{
    MonsterHybridServer,
    http_api::{ApiConfig, serve},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ApiConfig::from_env()?;
    let weights =
        FrozenBaseWeights::freeze("production-control-plane-bootstrap", 1, vec![0.0; 128]);
    let mut server = MonsterHybridServer::boot_default(weights)?;
    if let Some(vault) = std::env::var_os("ATOMA_VAULT") {
        server.load_or_init_persistent_vault(std::path::Path::new(&vault))?;
    }
    serve(&mut server, config).map_err(Into::into)
}
