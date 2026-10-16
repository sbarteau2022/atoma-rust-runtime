//! Safely merge the static, operator, Glimmer, and Qwen vaults into one map.
//!
//! Defaults target `design/unified_one_mind.atoma_map`; pass `--help` for overrides.

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{Atlas13, CarvedSwarmMap, FrozenBaseWeights, OverlapPolicyMatrix};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
struct Args {
    inputs: [PathBuf; 4],
    output: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let root = PathBuf::from("design");
    let mut paths = [
        root.join("global_100pc.atoma_map"),
        root.join("glimmer_52l_map_native.atoma_map"),
        root.join("glimmer_token_trajectories.atoma_map"),
        root.join("qwen_token_trajectories.atoma_map"),
    ];
    let flags = ["--static-vault", "--operator-vault", "--glimmer-traj", "--qwen-traj"];
    let mut output = root.join("unified_one_mind.atoma_map");
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            println!("Usage: merge_unified_mind [--static-vault PATH] [--operator-vault PATH] [--glimmer-traj PATH] [--qwen-traj PATH] [--out PATH]");
            std::process::exit(0);
        }
        let value = it.next().ok_or_else(|| format!("missing value for {flag}"))?;
        if flag == "--out" {
            output = value.into();
        } else if let Some(index) = flags.iter().position(|candidate| *candidate == flag) {
            paths[index] = value.into();
        } else {
            return Err(format!("unknown argument: {flag}"));
        }
    }
    Ok(Args { inputs: paths, output })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    for input in &args.inputs {
        if !input.is_file() { return Err(format!("input vault not found: {}", input.display())); }
    }
    let output = normalized(&args.output)?;
    for input in &args.inputs {
        if output == normalized(input)? { return Err(format!("output must not replace input vault {}", input.display())); }
    }
    if args.output.exists() { return Err(format!("output already exists; refusing to overwrite: {}", args.output.display())); }
    let tmp = args.output.with_extension("atoma_map.partial");
    if tmp.exists() { return Err(format!("temporary output already exists; refusing to overwrite: {}", tmp.display())); }

    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let plug = FrozenBaseWeights::freeze("Unified-One-Mind-Merger", 52, vec![0.1; 64]);
    let mut map = CarvedSwarmMap::new(atlas, plug, Q32_31::from_f64(1.22));
    let started = Instant::now();
    for (i, input) in args.inputs.iter().enumerate() {
        let t = Instant::now();
        let (atoms, arcs) = map.merge_vault_from(input, &policy, i >= 2)?;
        println!("[{} / 4] merged {}: +{atoms} atoms, +{arcs} arcs ({:.2}s)", i + 1, input.display(), t.elapsed().as_secs_f64());
    }
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("create output directory: {e}"))?;
    }
    let (atoms, arcs, digest) = map.save_to_binary_vault(&tmp)?;
    if let Err(error) = std::fs::rename(&tmp, &args.output) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("publish output vault: {error}"));
    }
    let bytes = std::fs::metadata(&args.output).map_err(|e| e.to_string())?.len();
    println!("[saved] {} atoms={atoms} arcs={arcs} bytes={bytes} sha256={} elapsed={:.2}s", args.output.display(), digest.to_hex(), started.elapsed().as_secs_f64());
    Ok(())
}

fn normalized(path: &Path) -> Result<PathBuf, String> {
    if path.exists() { std::fs::canonicalize(path).map_err(|e| e.to_string()) }
    else {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let parent = std::fs::canonicalize(parent).map_err(|e| format!("resolve {}: {e}", parent.display()))?;
        Ok(parent.join(path.file_name().ok_or("invalid output path")?))
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("merge_unified_mind: {error}");
        std::process::exit(1);
    }
}
