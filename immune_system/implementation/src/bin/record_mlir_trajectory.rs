//! Write token-trajectory arcs into a global map.
//!
//! Each step plants an atom on the well of the last prefix ID, an atom on the
//! well of the next ID, and an arc whose `compiled_mlir_op` is that next ID.
//! The stored frame is the prefix gyroscope the reader recomputes.
//!
//! Usage:
//!   record_mlir_trajectory --out PATH --prefix ID,ID (--next ID | --generated ID,ID)
//!                          [--vault PATH] [--hidden F,F] [--certainty 1]
//!                          [--base-vocab 248320] [--origin TEXT]
//!   record_mlir_trajectory --out PATH --steps FILE
//!                          [--vault PATH] [--certainty 1] [--base-vocab 248320] [--origin TEXT]
//!
//! `--vault` is loaded first when it is present. `--out` is where the map is written.
//! `--steps` reads many continuations from one `ATOMA_TJ1` file and writes the vault once.
//! A step the reader will not play is reported and kept; the rest of the file is still written.

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    Atlas13, CarvedSwarmMap, FrozenBaseWeights, MlirAtomCatalog, MlirOpId, NativeReader,
    OverlapPolicyMatrix, Sha256Digest,
};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn parse_ids(raw: &str) -> Vec<u32> {
    raw.split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect()
}

fn parse_f32s(raw: &str) -> Vec<f32> {
    raw.split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect()
}

fn main() {
    if let Err(err) = run() {
        eprintln!("record_mlir_trajectory: {err}");
        std::process::exit(1);
    }
}

/// Weight vaults already on disk. A trajectory run must not replace them.
const WEIGHT_VAULTS: &[&str] = &[
    "global_100pc.atoma_map",
    "glimmer_100pc.atoma_map",
    "qwen_100pc.atoma_map",
    "global_swarm_map.atoma_map",
    "glimmer_52l_map_native.atoma_map",
];

struct Continuation {
    prefix: Vec<MlirOpId>,
    generated: Vec<MlirOpId>,
    hidden_at_step: Vec<Vec<f32>>,
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: record_mlir_trajectory --out PATH (--prefix ID,ID (--next ID | --generated ID,ID) | --steps FILE) [--vault PATH] [--hidden F,F] [--certainty 1] [--base-vocab 248320] [--origin TEXT]";
    let Some(out) = arg_value(&args, "--out").map(PathBuf::from) else {
        return Err("--out is required".to_string());
    };
    refuse_weight_vault(&out)?;
    if arg_value(&args, "--steps").is_some() {
        return run_steps(&args, &out);
    }
    let Some(prefix_raw) = arg_value(&args, "--prefix") else {
        return Err(usage.to_string());
    };
    let prefix_ids = parse_ids(prefix_raw);
    if prefix_ids.is_empty() {
        return Err("no ML-IR IDs in --prefix".to_string());
    }
    let generated: Vec<MlirOpId> = if let Some(next) = arg_value(&args, "--next") {
        let id: u32 = next
            .parse()
            .map_err(|_| format!("--next is not an ID: {next}"))?;
        vec![MlirOpId(id)]
    } else if let Some(raw) = arg_value(&args, "--generated") {
        let ids = parse_ids(raw);
        if ids.is_empty() {
            return Err("no ML-IR IDs in --generated".to_string());
        }
        ids.into_iter().map(MlirOpId).collect()
    } else {
        return Err(usage.to_string());
    };

    let base_vocab: u32 = arg_value(&args, "--base-vocab")
        .and_then(|s| s.parse().ok())
        .unwrap_or(248_320);
    let certainty = Q32_31::from_f64(
        arg_value(&args, "--certainty")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
    );
    let origin = Sha256Digest::compute(
        arg_value(&args, "--origin")
            .unwrap_or("token-trajectory")
            .as_bytes(),
    );
    let hidden = arg_value(&args, "--hidden")
        .map(parse_f32s)
        .unwrap_or_default();
    let hidden_at_step: Vec<Vec<f32>> = if hidden.is_empty() {
        Vec::new()
    } else if generated.len() == 1 {
        vec![hidden]
    } else if hidden.len() == generated.len() {
        hidden.into_iter().map(|v| vec![v]).collect()
    } else {
        return Err("--hidden is one list for --next, or one value per --generated ID".to_string());
    };

    let catalog = MlirAtomCatalog::new_canonical(base_vocab);
    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let plug = FrozenBaseWeights::freeze("unused-backend-plug", 1, vec![0.0; 8]);
    let mut map = CarvedSwarmMap::new(atlas, plug, Q32_31::from_f64(1.22));
    if let Some(vault) = arg_value(&args, "--vault").map(PathBuf::from) {
        refuse_weight_vault(&vault)?;
        let (atoms, arcs, _sha) = map.load_from_binary_vault(&vault, &policy)?;
        println!("loaded {}  atoms={atoms} arcs={arcs}", vault.display());
    }

    let prefix: Vec<MlirOpId> = prefix_ids.into_iter().map(MlirOpId).collect();
    let wrote = NativeReader::record_continuation(
        &mut map,
        &catalog,
        &policy,
        &prefix,
        &generated,
        &hidden_at_step,
        origin,
        certainty,
    )?;
    for (i, step) in wrote.iter().enumerate() {
        println!(
            "wrote {i} next={} arc={} novel={} from_atom={} to_atom={}",
            step.read.next_id.0,
            step.read.arc_id,
            step.arc_was_novel,
            step.read.from_atom.0,
            step.read.to_atom.0,
        );
    }
    save_vault(&map, &out)?;
    Ok(())
}

fn run_steps(args: &[String], out: &Path) -> Result<(), String> {
    let Some(steps_path) = arg_value(args, "--steps").map(PathBuf::from) else {
        return Err("--steps is required".to_string());
    };
    let continuations = read_steps(&steps_path)?;
    if continuations.is_empty() {
        return Err("steps file has no continuations".to_string());
    }
    let (catalog, policy, origin, certainty, mut map) = open_map(args)?;
    let mut conflicts = 0u64;
    let mut written = 0u64;
    for (i, continuation) in continuations.iter().enumerate() {
        let (steps, step_conflicts) = match write_continuation(
            &mut map,
            &catalog,
            &policy,
            continuation,
            origin,
            certainty,
        ) {
            Ok(pair) => pair,
            Err(err) => {
                save_vault(&map, out)?;
                return Err(err);
            }
        };
        conflicts += step_conflicts;
        written += steps;
        println!(
            "continuation {i} steps={steps} conflicts={step_conflicts} prefix={} generated={}",
            continuation.prefix.len(),
            continuation.generated.len()
        );
    }
    save_vault(&map, out)?;
    println!("steps_written {written} step_conflicts {conflicts}");
    Ok(())
}

fn open_map(
    args: &[String],
) -> Result<
    (
        MlirAtomCatalog,
        OverlapPolicyMatrix,
        Sha256Digest,
        Q32_31,
        CarvedSwarmMap,
    ),
    String,
> {
    let base_vocab: u32 = arg_value(args, "--base-vocab")
        .and_then(|s| s.parse().ok())
        .unwrap_or(248_320);
    let certainty = Q32_31::from_f64(
        arg_value(args, "--certainty")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
    );
    let origin = Sha256Digest::compute(
        arg_value(args, "--origin")
            .unwrap_or("token-trajectory")
            .as_bytes(),
    );
    let catalog = MlirAtomCatalog::new_canonical(base_vocab);
    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let plug = FrozenBaseWeights::freeze("unused-backend-plug", 1, vec![0.0; 8]);
    let mut map = CarvedSwarmMap::new(atlas, plug, Q32_31::from_f64(1.22));
    if let Some(vault) = arg_value(args, "--vault").map(PathBuf::from) {
        refuse_weight_vault(&vault)?;
        let (atoms, arcs, _sha) = map.load_from_binary_vault(&vault, &policy)?;
        println!("loaded {}  atoms={atoms} arcs={arcs}", vault.display());
    }
    Ok((catalog, policy, origin, certainty, map))
}

/// Write one continuation. A step another arc wins is counted and is not fatal.
fn write_continuation(
    map: &mut CarvedSwarmMap,
    catalog: &MlirAtomCatalog,
    policy: &OverlapPolicyMatrix,
    continuation: &Continuation,
    origin: Sha256Digest,
    certainty: Q32_31,
) -> Result<(u64, u64), String> {
    let mut prefix = continuation.prefix.clone();
    let mut steps = 0u64;
    let mut conflicts = 0u64;
    for (i, next) in continuation.generated.iter().copied().enumerate() {
        let hidden = continuation
            .hidden_at_step
            .get(i)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        match NativeReader::record_step(
            map, catalog, policy, &prefix, next, hidden, origin, certainty,
        ) {
            Ok(step) => {
                println!(
                    "wrote {i} next={} arc={} novel={} from_atom={} to_atom={}",
                    step.read.next_id.0,
                    step.read.arc_id,
                    step.arc_was_novel,
                    step.read.from_atom.0,
                    step.read.to_atom.0,
                );
                steps += 1;
            }
            Err(err) if err.starts_with("recorded mlir_op#") => {
                eprintln!("conflict {i} {err}");
                conflicts += 1;
                steps += 1;
            }
            Err(err) => return Err(err),
        }
        prefix.push(next);
    }
    Ok((steps, conflicts))
}

fn save_vault(map: &CarvedSwarmMap, out: &Path) -> Result<(), String> {
    let tmp = out.with_extension("atoma_map.partial");
    let (atoms, arcs, sha) = map.save_to_binary_vault(&tmp)?;
    std::fs::rename(&tmp, out)
        .map_err(|err| format!("rename {} to {}: {err}", tmp.display(), out.display()))?;
    println!(
        "saved {}  atoms={atoms} arcs={arcs} sha={}",
        out.display(),
        sha.to_hex()
    );
    Ok(())
}

fn refuse_weight_vault(path: &Path) -> Result<(), String> {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if WEIGHT_VAULTS.contains(&name) {
        return Err(format!(
            "{name} is a weight vault; trajectory output goes to a different file"
        ));
    }
    Ok(())
}

/// `ATOMA_TJ1` then repeated records:
/// `u32 n_prefix`, `u32 n_gen`, `u32 hidden_dim`, prefix ids, generated ids,
/// and `n_gen * hidden_dim` little-endian f32 values when `hidden_dim` is not 0.
fn read_steps(path: &Path) -> Result<Vec<Continuation>, String> {
    let mut file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut magic = [0u8; 9];
    file.read_exact(&mut magic)
        .map_err(|err| format!("steps magic: {err}"))?;
    if &magic != b"ATOMA_TJ1" {
        return Err("steps file does not start with ATOMA_TJ1".to_string());
    }
    let mut out = Vec::new();
    loop {
        let mut len_buf = [0u8; 12];
        if !read_full(&mut file, &mut len_buf)? {
            break;
        }
        let n_prefix = u32::from_le_bytes(len_buf[0..4].try_into().unwrap());
        let n_gen = u32::from_le_bytes(len_buf[4..8].try_into().unwrap());
        let hidden_dim = u32::from_le_bytes(len_buf[8..12].try_into().unwrap());
        if n_prefix == 0 || n_prefix > 4096 || n_gen == 0 || n_gen > 4096 || hidden_dim > 16384 {
            return Err(format!(
                "steps record out of range prefix={n_prefix} gen={n_gen} hidden={hidden_dim}"
            ));
        }
        let prefix = read_ids(&mut file, n_prefix)?;
        let generated = read_ids(&mut file, n_gen)?;
        let hidden_at_step = if hidden_dim == 0 {
            Vec::new()
        } else {
            let mut hidden_at_step = Vec::with_capacity(n_gen as usize);
            for _ in 0..n_gen {
                hidden_at_step.push(read_f32s(&mut file, hidden_dim)?);
            }
            hidden_at_step
        };
        out.push(Continuation {
            prefix,
            generated,
            hidden_at_step,
        });
    }
    Ok(out)
}

fn read_full(file: &mut File, buf: &mut [u8]) -> Result<bool, String> {
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err("truncated steps record".to_string()),
            Ok(n) => filled += n,
            Err(err) => return Err(format!("read steps: {err}")),
        }
    }
    Ok(true)
}

fn read_ids(file: &mut File, n: u32) -> Result<Vec<MlirOpId>, String> {
    let mut buf = vec![0u8; n as usize * 4];
    file.read_exact(&mut buf)
        .map_err(|err| format!("steps ids: {err}"))?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| MlirOpId(u32::from_le_bytes(c.try_into().unwrap())))
        .collect())
}

fn read_f32s(file: &mut File, n: u32) -> Result<Vec<f32>, String> {
    let mut buf = vec![0u8; n as usize * 4];
    file.read_exact(&mut buf)
        .map_err(|err| format!("steps hidden: {err}"))?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}
