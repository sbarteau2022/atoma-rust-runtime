//! Play path for a distilled global map with Active Inference & Superposition Holding
//! (`design/superpositionholding.md`).
//!
//! ML-IR IDs in, the next ML-IR ID out. The reader does not load a checkpoint,
//! an embedding matrix, or a layer stack. The vault is the model.
//!
//! Usage:
//!   read_mlir_native --vault design/global_swarm_map.atoma_map --mlir-ids 13796,26,1573
//!                    [--max-new N] [--eos 1,2] [--base-vocab 248320] [--active-inference]
//!
//! `--base-vocab` only places the nine structural opcodes, matching the catalog
//! distillation writes. IDs below that base are token carriers.

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    ActiveInferenceConfig, ActiveInferenceStop, Atlas13, CarvedSwarmMap, FrozenBaseWeights,
    MlirAtomCatalog, MlirOpId, NativeReader, NativeStop, OverlapPolicyMatrix,
};
use std::path::PathBuf;
use std::time::Instant;

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn parse_ids(raw: &str) -> Vec<u32> {
    raw.split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect()
}

fn main() {
    if let Err(err) = run() {
        eprintln!("read_mlir_native: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(id_raw) = arg_value(&args, "--mlir-ids") else {
        return Err(
            "usage: read_mlir_native --vault PATH --mlir-ids ID,ID [--max-new N] [--eos ID,ID] [--base-vocab N] [--active-inference]"
                .to_string(),
        );
    };
    let prefix_ids = parse_ids(id_raw);
    if prefix_ids.is_empty() {
        return Err("no ML-IR IDs in --mlir-ids".to_string());
    }
    let Some(vault) = arg_value(&args, "--vault").map(PathBuf::from) else {
        return Err("--vault is required".to_string());
    };
    let max_new: usize = arg_value(&args, "--max-new")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let eos: Vec<MlirOpId> = arg_value(&args, "--eos")
        .map(parse_ids)
        .unwrap_or_default()
        .into_iter()
        .map(MlirOpId)
        .collect();
    let base_vocab: u32 = arg_value(&args, "--base-vocab")
        .and_then(|s| s.parse().ok())
        .unwrap_or(248_320);
    let active_mode = has_flag(&args, "--active-inference");

    let catalog = MlirAtomCatalog::new_canonical(base_vocab);
    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    // The map constructor requires a backend plug. The reader does not consult it.
    let plug = FrozenBaseWeights::freeze("unused-backend-plug", 1, vec![0.0; 8]);
    let mut map = CarvedSwarmMap::new(atlas, plug, Q32_31::from_f64(1.22));

    let t_load = Instant::now();
    let (atoms, arcs, _sha) = map.load_from_binary_vault(&vault, &policy)?;
    println!(
        "vault {}  atoms={atoms} arcs={arcs} load_s={:.3}",
        vault.display(),
        t_load.elapsed().as_secs_f64()
    );

    let prefix: Vec<MlirOpId> = prefix_ids.into_iter().map(MlirOpId).collect();
    let t_read = Instant::now();

    if active_mode {
        let mut config = ActiveInferenceConfig::default();
        config.predicted_token_envelope = (max_new as u64).saturating_mul(2).max(64);
        let walk = NativeReader::continue_ids_active(
            &map,
            &catalog,
            &policy,
            &prefix,
            max_new,
            &eos,
            &config,
        )?;
        let read_s = t_read.elapsed().as_secs_f64();

        for (i, step) in walk.steps.iter().enumerate() {
            let read = &step.read;
            let held_tag = if let Some(ref st) = step.superposition_held {
                format!(
                    " superposition=held(branches={},iters={})",
                    st.branches.len(),
                    st.iteration_traces.len()
                )
            } else {
                String::new()
            };
            println!(
                "read {i} next={} arc={} from_atom={} to_atom={} query_cosh={:.4} action_cosh={:.4} align={:.4} free_energy={:.4} omega={:.4}{}",
                read.next_id.0,
                read.arc_id,
                read.from_atom.0,
                read.to_atom.0,
                read.query_cosh_q32.to_f64(),
                read.cosh_action_q32.to_f64(),
                read.frame_alignment_q32.to_f64(),
                step.free_energy.total_free_energy_q32.to_f64(),
                step.omega_readiness_q32.to_f64(),
                held_tag,
            );
        }
        match &walk.stop {
            ActiveInferenceStop::MaxNew => println!("stop max_new"),
            ActiveInferenceStop::Eos(id) => println!("stop eos {}", id.0),
            ActiveInferenceStop::TokenEnvelopeExhausted {
                spent_tokens,
                envelope_tokens,
            } => println!(
                "stop token_envelope_exhausted spent={} envelope={}",
                spent_tokens, envelope_tokens
            ),
            ActiveInferenceStop::SuperpositionFailClosed { at, reason, .. } => {
                println!("stop superposition_fail_closed at {} ({reason})", at.0);
            }
        }
        let produced: Vec<u32> = walk.steps.iter().map(|s| s.read.next_id.0).collect();
        println!(
            "next_ids {:?} read_s={:.6} count={} total_free_energy={:.4} superposition_interventions={}",
            produced,
            read_s,
            produced.len(),
            walk.total_free_energy_q32.to_f64(),
            walk.superposition_interventions,
        );
        if walk.steps.is_empty() {
            return Err("map did not continue the prefix".to_string());
        }
        return Ok(());
    }

    let walk = NativeReader::continue_ids(&map, &catalog, &prefix, max_new, &eos)?;
    let read_s = t_read.elapsed().as_secs_f64();

    for (i, read) in walk.reads.iter().enumerate() {
        println!(
            "read {i} next={} arc={} from_atom={} to_atom={} query_cosh={:.4} action_cosh={:.4} align={:.4}",
            read.next_id.0,
            read.arc_id,
            read.from_atom.0,
            read.to_atom.0,
            read.query_cosh_q32.to_f64(),
            read.cosh_action_q32.to_f64(),
            read.frame_alignment_q32.to_f64(),
        );
    }
    match &walk.stop {
        NativeStop::MaxNew => println!("stop max_new"),
        NativeStop::Eos(id) => println!("stop eos {}", id.0),
        NativeStop::FixedPoint(id) => println!("stop fixed_point {}", id.0),
        NativeStop::NoGeodesic { at, detail } => {
            println!("stop no_geodesic at {} ({detail})", at.0);
        }
    }
    let produced: Vec<u32> = walk.reads.iter().map(|r| r.next_id.0).collect();
    println!(
        "next_ids {:?} read_s={:.6} count={}",
        produced,
        read_s,
        produced.len()
    );
    if walk.reads.is_empty() {
        return Err("map did not continue the prefix".to_string());
    }
    Ok(())
}
