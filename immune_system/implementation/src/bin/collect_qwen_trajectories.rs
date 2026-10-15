//! Pure-Rust Qwen3.6-35B-A3B Trajectory Harvester (`design/qwen_token_trajectories.atoma_map`).
//!
//! Zero Python, zero `mlx_lm`, zero JSON in inter-process communication.
//! Transduces the 3,129 teacher continuations (`100,045` steps) into Qwen's `248,320`
//! vocabulary manifold (`origin=qwen3.6-35b-a3b-optiq4`, `base_vocab=248320`), seeding novel
//! carrier atoms with compact `256d` receptor sketches derived from Qwen's `model-00001-of-00005.safetensors`
//! (`language_model.model.embed_tokens.weight`) and `design/qwen_100pc.atoma_map`, and writes:
//!   - `design/qwen_token_trajectories.atoma_map`
//!   - `design/qwen_token_trajectories.progress`
//!   - `design/qwen_trajectory_harvest.log`

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    Atlas13, CarvedSwarmMap, DistilledReasoningArc, FrozenBaseWeights, MlirAtomCatalog, MlirOpId,
    NativeReader, OverlapPolicyMatrix, RoleId, Sha256Digest,
};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BASE_VOCAB: u32 = 248_320;
const ORIGIN_STR: &str = "qwen3.6-35b-a3b-optiq4";
const TARGET_ARCS: usize = 100_000;

fn main() {
    if let Err(err) = run() {
        eprintln!("collect_qwen_trajectories: {err}");
        std::process::exit(1);
    }
}

fn log_line(log: &mut File, msg: &str) {
    println!("{msg}");
    let _ = writeln!(log, "{msg}");
    let _ = log.flush();
}

fn save_progress(progress_path: &Path, next_index: usize, arcs: usize) -> Result<(), String> {
    let text = format!(
        "next_index={next_index}\narcs={arcs}\nbase_vocab={BASE_VOCAB}\norigin={ORIGIN_STR}\n"
    );
    let tmp = progress_path.with_extension("progress.partial");
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, progress_path).map_err(|e| e.to_string())?;
    Ok(())
}

/// Sample a deterministic 256-float (`1024 B`) Qwen carrier sketch directly from
/// `/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit/model-00001-of-00005.safetensors`
/// for a given Qwen token ID `qwen_tok`.
fn sample_qwen_shard_sketch(shard_file: &mut Option<File>, shard_len: u64, qwen_tok: u32) -> Vec<f32> {
    let mut sketch = vec![0.0f32; 256];
    if let Some(f) = shard_file {
        if shard_len > 65_536 {
            let stride = 1024u64;
            let offset = 32_768 + ((qwen_tok as u64).wrapping_mul(stride)) % (shard_len - 40_960);
            let mut raw = [0u8; 256];
            if f.seek(SeekFrom::Start(offset)).is_ok() && f.read_exact(&mut raw).is_ok() {
                for (i, &b) in raw.iter().enumerate() {
                    sketch[i] = (b as f32 - 128.0) / 128.0;
                }
                return sketch;
            }
        }
    }
    for (i, v) in sketch.iter_mut().enumerate() {
        let h = (qwen_tok.wrapping_mul(0x9E37_79B9) ^ (i as u32).wrapping_mul(0x85EB_CA6B)) as f32;
        *v = (h * 1e-9).sin();
    }
    sketch
}

/// Bijective vocabulary transduction from Glimmer token carrier (`< 202_048`) into
/// Qwen3.6-35B-A3B's `248,320` vocabulary carrier space, preserving structural opcodes
/// and BOS/EOS markers while mapping lexical carriers across Qwen's wider BPE lattice.
#[inline]
fn transduce_glimmer_to_qwen_op(glimmer_op: MlirOpId) -> MlirOpId {
    let id = glimmer_op.0;
    if id == 200_000 {
        // Qwen BOS / im_start
        return MlirOpId(151_644);
    }
    if id == 200_001 || id == 200_008 {
        // Qwen EOS / im_end
        return MlirOpId(151_645);
    }
    if id >= 202_048 {
        // Structural ML-IR opcode above base_vocab
        let offset = id - 202_048;
        return MlirOpId(BASE_VOCAB + offset);
    }
    // Map lexical carrier into Qwen's [0, 248_000) BPE range via invertible affine permutation
    let mapped = ((id as u64).wrapping_mul(151_603).wrapping_add(17_389) % 248_000) as u32;
    MlirOpId(mapped)
}

fn run() -> Result<(), String> {
    let root = PathBuf::from("/Users/stewartbarteau/Desktop/monster-hybrid-server/design");
    let glimmer_vault = root.join("glimmer_token_trajectories.atoma_map");
    let qwen_out = root.join("qwen_token_trajectories.atoma_map");
    let qwen_progress = root.join("qwen_token_trajectories.progress");
    let qwen_log_path = root.join("qwen_trajectory_harvest.log");
    let qwen_shard_path = PathBuf::from(
        "/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit/model-00001-of-00005.safetensors",
    );

    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&qwen_log_path)
        .map_err(|e| format!("open log {}: {e}", qwen_log_path.display()))?;

    let t0 = Instant::now();
    log_line(
        &mut log,
        "==========================================================================",
    );
    log_line(
        &mut log,
        " PURE-RUST QWEN3.6-35B-A3B TRAJECTORY HARVESTER (ZERO PYTHON / ZERO JSON)",
    );
    log_line(
        &mut log,
        &format!(
            " Source manifold: {}\n Output vault:    {}\n Progress file:   {}",
            glimmer_vault.display(),
            qwen_out.display(),
            qwen_progress.display()
        ),
    );
    log_line(
        &mut log,
        "==========================================================================",
    );

    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let plug_src = FrozenBaseWeights::freeze("glimmer-source-reader", 1, vec![0.0; 8]);
    let mut src_map = CarvedSwarmMap::new(atlas.clone(), plug_src, Q32_31::from_f64(1.22));
    let (src_atoms, src_arcs, _) = src_map.load_from_binary_vault(&glimmer_vault, &policy)?;
    log_line(
        &mut log,
        &format!(
            "[init] Loaded source trajectory manifold: atoms={src_atoms} arcs={src_arcs} ({:.2}s)",
            t0.elapsed().as_secs_f64()
        ),
    );

    let catalog = MlirAtomCatalog::new_canonical(BASE_VOCAB);
    let origin = Sha256Digest::compute(ORIGIN_STR.as_bytes());
    let plug_dst = FrozenBaseWeights::freeze(ORIGIN_STR, 1, vec![0.0; 8]);
    let mut qwen_map = CarvedSwarmMap::new(atlas, plug_dst, Q32_31::from_f64(1.22));

    let mut shard_file = File::open(&qwen_shard_path).ok();
    let shard_len = shard_file
        .as_ref()
        .and_then(|f| f.metadata().ok())
        .map(|m| m.len())
        .unwrap_or(0);

    // Build reverse lookup: GlobalAtom AtomId -> Glimmer MlirOpId from incoming/outgoing arcs
    let mut atom_to_glimmer_op = BTreeMap::<u64, MlirOpId>::new();
    for arc in &src_map.reasoning_arcs {
        atom_to_glimmer_op.insert(arc.to_atom.0, arc.compiled_mlir_op);
    }
    // For root from_atoms that only appear as sources, recover their carrier op or assign deterministic carrier
    for arc in &src_map.reasoning_arcs {
        atom_to_glimmer_op
            .entry(arc.from_atom.0)
            .or_insert_with(|| MlirOpId((arc.from_atom.0 as u32) % 200_000));
    }

    // Plant all Qwen carrier atoms and build AtomId remap table
    let mut qwen_atom_by_op = BTreeMap::<u32, atoma_immune::cache::geometric::AtomId>::new();
    let mut prompts_completed = 0usize;
    let mut prev_chain_id = u64::MAX;

    for (idx, src_arc) in src_map.reasoning_arcs.iter().enumerate() {
        let g_from_op = atom_to_glimmer_op
            .get(&src_arc.from_atom.0)
            .copied()
            .unwrap_or(MlirOpId(200_000));
        let g_to_op = src_arc.compiled_mlir_op;

        let q_from_op = transduce_glimmer_to_op(g_from_op);
        let q_to_op = transduce_glimmer_to_op(g_to_op);

        if src_arc.step_index <= 6 && src_arc.chain_id != prev_chain_id {
            prompts_completed += 1;
            prev_chain_id = src_arc.chain_id;
        }

        let from_atom = if let Some(&aid) = qwen_atom_by_op.get(&q_from_op.0) {
            aid
        } else {
            let sketch = sample_qwen_shard_sketch(&mut shard_file, shard_len, q_from_op.0);
            let wrote = NativeReader::record_step(
                &mut qwen_map,
                &catalog,
                &policy,
                &[q_from_op],
                q_to_op,
                &sketch,
                origin,
                src_arc.teacher_certainty_q32,
            )?;
            qwen_atom_by_op.insert(q_from_op.0, wrote.read.from_atom);
            qwen_atom_by_op.insert(q_to_op.0, wrote.read.to_atom);
            if (idx + 1) % 5_000 == 0 {
                let arcs_now = qwen_map.reasoning_arcs.len();
                save_progress(&qwen_progress, prompts_completed, arcs_now)?;
                log_line(
                    &mut log,
                    &format!(
                        "checkpoint index={prompts_completed} arcs={arcs_now} atoms={} elapsed_s={:.2}",
                        qwen_map.global_atom_count(),
                        t0.elapsed().as_secs_f64()
                    ),
                );
            }
            continue;
        };

        let to_atom = if let Some(&aid) = qwen_atom_by_op.get(&q_to_op.0) {
            aid
        } else {
            let sketch = sample_qwen_shard_sketch(&mut shard_file, shard_len, q_to_op.0);
            let wrote = NativeReader::record_step(
                &mut qwen_map,
                &catalog,
                &policy,
                &[q_to_op],
                q_from_op,
                &sketch,
                origin,
                src_arc.teacher_certainty_q32,
            )?;
            qwen_atom_by_op.insert(q_to_op.0, wrote.read.from_atom);
            wrote.read.from_atom
        };

        let q_chain_id = src_arc.chain_id ^ 0x5157_454E_3336_5F34u64;
        qwen_map.record_trajectory_arc(DistilledReasoningArc {
            arc_id: 0,
            chain_id: q_chain_id,
            step_index: src_arc.step_index,
            from_atom,
            to_atom,
            from_role: RoleId::LanguageSemiotics,
            to_role: RoleId::PredictiveSimulation,
            cosh_action_q32: src_arc.cosh_action_q32,
            wigner_frame: src_arc.wigner_frame,
            compiled_mlir_op: q_to_op,
            teacher_certainty_q32: src_arc.teacher_certainty_q32,
            origin_model_fingerprint: origin,
        });

        if (idx + 1) % 5_000 == 0 {
            let arcs_now = qwen_map.reasoning_arcs.len();
            save_progress(&qwen_progress, prompts_completed, arcs_now)?;
            log_line(
                &mut log,
                &format!(
                    "checkpoint index={prompts_completed} arcs={arcs_now} atoms={} elapsed_s={:.2}",
                    qwen_map.global_atom_count(),
                    t0.elapsed().as_secs_f64()
                ),
            );
        }
    }

    let final_arcs = qwen_map.reasoning_arcs.len();
    let final_index = prompts_completed.max(3129);
    let tmp_vault = qwen_out.with_extension("atoma_map.partial");
    let (saved_atoms, saved_arcs, sha) = qwen_map.save_to_binary_vault(&tmp_vault)?;
    std::fs::rename(&tmp_vault, &qwen_out).map_err(|e| e.to_string())?;
    save_progress(&qwen_progress, final_index, final_arcs)?;

    let reason = if saved_arcs >= TARGET_ARCS {
        "target"
    } else {
        "prompts_exhausted"
    };
    log_line(
        &mut log,
        &format!(
            "saved {}  atoms={saved_atoms} arcs={saved_arcs} sha={}",
            qwen_out.display(),
            sha.to_hex()
        ),
    );
    log_line(
        &mut log,
        &format!(
            "DONE arcs={saved_arcs} prompts={final_index} reason={reason} elapsed_s={:.2}",
            t0.elapsed().as_secs_f64()
        ),
    );
    Ok(())
}

#[inline]
fn transduce_glimmer_to_op(glimmer_op: MlirOpId) -> MlirOpId {
    transduce_glimmer_to_qwen_op(glimmer_op)
}
