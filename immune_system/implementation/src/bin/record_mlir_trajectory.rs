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
//!
//! `--vault` is loaded first when it is present. `--out` is where the map is written.

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    Atlas13, CarvedSwarmMap, FrozenBaseWeights, MlirAtomCatalog, MlirOpId, NativeReader,
    OverlapPolicyMatrix, Sha256Digest,
};
use std::path::PathBuf;

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

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: record_mlir_trajectory --out PATH --prefix ID,ID (--next ID | --generated ID,ID) [--vault PATH] [--hidden F,F] [--certainty 1] [--base-vocab 248320] [--origin TEXT]";
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
    let Some(out) = arg_value(&args, "--out").map(PathBuf::from) else {
        return Err("--out is required".to_string());
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
    let (atoms, arcs, sha) = map.save_to_binary_vault(&out)?;
    println!(
        "saved {}  atoms={atoms} arcs={arcs} sha={}",
        out.display(),
        sha.to_hex()
    );
    Ok(())
}
