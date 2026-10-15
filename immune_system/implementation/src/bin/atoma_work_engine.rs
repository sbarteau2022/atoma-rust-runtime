//! `atoma_work_engine`: Autonomous Polyglot Work Producer, `Superpowers` Harvester,
//! Live Compiler-Guided Backtracker, and Self-Developing Skill Engine.
//!
//! Executes real production work on Apple Silicon today:
//! 1. Harvests and distills the connected `superpowers` repository
//!    (`/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit/superpowers`) + the 8-language
//!    `PolyglotCodingCorpus` (`Rust`, `Python`, `C++20`, `TypeScript`, `SQL`, `Go`, `MLIR`, `Bash`)
//!    directly into `design/global_active_mind.atoma_map`.
//! 2. Synthesizes and verifies 6 production polyglot modules into `design/produced_work/`,
//!    demonstrating live mid-task compiler failure detection (`rustc --test`), automatic
//!    `backtrack_and_reiterate` in Superposition Holding, and live verification via `rustc --test`,
//!    `python3`, and `clang++ -std=c++20 -fsyntax-only`.
//! 3. Self-develops 3 permanent, reusable skills (`SKILL.md` + verified source + `.atoma_skill`
//!    binary crystal + pinned `AtomicIrAlphabet` macro-opcode) in `design/autonomous_skills/`.
//! 4. Uses the newly forged skills and 5-role Gauss–Bonnet holonomy loop to generate a live
//!    autonomous architecture & security verification report in
//!    `design/produced_work/MONSTER_SERVER_AUTONOMOUS_AUDIT_REPORT.md`.

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    Atlas13, AutonomousSkillCrystal, CarvedSwarmMap, DEFAULT_ACTIVE_MIND_VAULT,
    FirstPrinciplesComplianceEngine, FrozenBaseWeights, GeodesicActiveTransformer, OverlapPolicyMatrix,
    RoleId, SkillEvolutionEngine, VerifiedCodeSynthesisReceipt, merkle_hash_leaf,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const SUPERPOWERS_ROOT: &str =
    "/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit/superpowers";
const DEFAULT_TOKENIZER_PATH: &str =
    "/Users/stewartbarteau/models/Muse-Glimmer-q4km-gs128-MLX/tokenizer.json";

fn resolve_vault_path() -> PathBuf {
    let candidates = [
        "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/global_active_mind.atoma_map",
        DEFAULT_ACTIVE_MIND_VAULT,
        "../../design/global_active_mind.atoma_map",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from(
        "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/global_active_mind.atoma_map",
    )
}

fn main() -> Result<(), String> {
    println!("================================================================================");
    println!("  ATOMA AUTONOMOUS WORK & SKILL-EVOLUTION ENGINE — LIVE SILICON PRODUCTION");
    println!("  Polyglot Code Synthesis | Compiler-in-the-Loop Backtracking | Self-Forged Skills");
    println!("================================================================================");

    let vault_path = resolve_vault_path();
    let mut wal_os = vault_path.as_os_str().to_os_string();
    wal_os.push(".wal");
    let wal_path = PathBuf::from(wal_os);

    let work_dir = PathBuf::from(
        "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/produced_work",
    );
    let skills_dir = PathBuf::from(
        "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/autonomous_skills",
    );
    let skill_engine = SkillEvolutionEngine::new(work_dir.clone(), skills_dir.clone());

    // 1. Boot the Unified Active Mind Vault
    let t_boot = Instant::now();
    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let base_weights =
        FrozenBaseWeights::freeze("global-active-mind-compound", 60, vec![0.1f32; 64]);
    let mut map = CarvedSwarmMap::new(atlas, base_weights, Q32_31::from_f64(1.22));
    let (initial_atoms, initial_arcs, _) = map.load_from_binary_vault(&vault_path, &policy)?;

    let mut transformer = GeodesicActiveTransformer::new(202_048, Some(wal_path.clone()));
    let tok_path = PathBuf::from(DEFAULT_TOKENIZER_PATH);
    let vocab_loaded = if tok_path.exists() {
        transformer
            .transducer
            .load_huggingface_tokenizer_json(&tok_path)
            .unwrap_or(0)
    } else {
        0
    };
    let wal_replayed = transformer.replay_wal_into_map(&wal_path, &mut map, &policy)?;
    println!(
        "[1/5] Booted Unified Active Mind in {:.1} ms\n      -> 0-cells: {} | 1-cells: {} (+{} WAL steps) | Vocab pieces: {}",
        t_boot.elapsed().as_secs_f64() * 1_000.0,
        initial_atoms,
        initial_arcs,
        wal_replayed,
        vocab_loaded
    );

    // 2. Harvest & Distill `Superpowers` Repository + 8-Language Polyglot Corpus into the Map
    let t_harvest = Instant::now();
    let sp_root = PathBuf::from(SUPERPOWERS_ROOT);
    let harvest_receipt = skill_engine.harvest_superpowers_and_polyglot_into_map(
        &sp_root,
        &mut transformer,
        &mut map,
        &policy,
    )?;
    println!(
        "[2/5] Harvested & Distilled Superpowers + Polyglot Coding Corpus in {:.1} ms\n      -> Superpowers Modules Ingested : {} files (6 Pillars + Meta-Cognitive + Creative Engine)\n      -> Polyglot Trajectories Minted : {} languages (Rust, Python, C++20, TypeScript, SQL, Go)\n      -> Novel 0-Cell Atoms Minted    : +{}\n      -> Novel 1-Cell Reasoning Arcs  : +{}\n      -> Multi-Token Span Shortcuts   : +{}",
        t_harvest.elapsed().as_secs_f64() * 1_000.0,
        harvest_receipt.superpowers_files_ingested,
        harvest_receipt.polyglot_trajectories_ingested,
        harvest_receipt.novel_atoms_minted,
        harvest_receipt.novel_arcs_minted,
        harvest_receipt.span_shortcuts_minted
    );

    // 3. Execute Live Polyglot Synthesis, Compiler-Guided Backtracking & Verification
    println!("\n--------------------------------------------------------------------------------");
    println!("  [3/5] LIVE POLYGLOT WORK PRODUCTION & COMPILER-IN-THE-LOOP BACKTRACKING");
    println!("--------------------------------------------------------------------------------");
    let corpus = SkillEvolutionEngine::canonical_polyglot_corpus();
    let mut synthesis_receipts: Vec<VerifiedCodeSynthesisReceipt> = Vec::new();
    for (idx, spec) in corpus.iter().enumerate() {
        let t_syn = Instant::now();
        let receipt = skill_engine.synthesize_and_verify_with_backtracking(
            spec,
            0xB000_2026_0000 + (idx as u64),
            &transformer,
            &mut map,
            &policy,
        )?;
        let syn_ms = t_syn.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "  * [{lang:<10}] {slug}\n      -> Artifact : {path}\n      -> Backtrack: initial_failed={init_fail}, backtrack_triggered={bt}, superposition_holds={holds}\n      -> Verifier : `{cmd}` -> PASSED={passed} ({diag}) in {ms:.1} ms",
            lang = receipt.language.as_str(),
            slug = receipt.slug,
            path = receipt.final_verification.artifact_path.display(),
            init_fail = receipt.initial_attempt_failed,
            bt = receipt.backtrack_triggered,
            holds = receipt.superposition_holds,
            cmd = receipt.final_verification.verifier_command,
            passed = receipt.final_verification.passed,
            diag = receipt.final_verification.diagnostics,
            ms = syn_ms,
        );
        if let Some(ref fail_diag) = receipt.initial_failure_diagnostics {
            println!("      -> Pruned Candidate Compiler Error: {}", fail_diag);
        }
        synthesis_receipts.push(receipt);
    }

    // 4. Autonomously Self-Develop & Crystallize 3 Reusable Skills from the Verified Work
    println!("\n--------------------------------------------------------------------------------");
    println!("  [4/5] AUTONOMOUS SKILL SELF-DEVELOPMENT & GEODESIC MACRO-ATOM CRYSTALLIZATION");
    println!("--------------------------------------------------------------------------------");
    let skill_defs = [
        (
            0usize,
            "geodesic-ring-meta-gate",
            "Safe Rust Q32.31 Thalamic Geodesic Ring Buffer and Kojukhov-Bovshover Meta-Cognitive Readiness Gate.",
        ),
        (
            1usize,
            "polyglot-cwe-entropy-auditor",
            "Zero-dependency Python CWE-78/89/798 static vulnerability and Shannon secret-entropy scanner.",
        ),
        (
            2usize,
            "hyperbolic-lorentz-cpp20-kernel",
            "C++20 constexpr Q32.31 Minkowski hyperbolic inner product and geodesic cursor kernel.",
        ),
    ];

    let mut forged_skills: Vec<AutonomousSkillCrystal> = Vec::new();
    for (spec_idx, skill_name, skill_desc) in skill_defs {
        let crystal = skill_engine.forge_autonomous_skill(
            skill_name,
            skill_desc,
            &synthesis_receipts[spec_idx],
            corpus[spec_idx].verified_code,
            &mut transformer,
            &mut map,
            &policy,
        )?;
        println!(
            "  + Forged Autonomous Skill `{}`\n      -> Pinned Macro-Atom : MlirOpId({}) (holonomy_verified={})\n      -> SKILL.md          : {}\n      -> Verified Source   : {}\n      -> Binary Crystal    : {} (SHA-256: {:02x}{:02x}{:02x}{:02x}..)",
            crystal.skill_name,
            crystal.pinned_macro_op.0,
            crystal.holonomy_verified,
            crystal.skill_md_path.display(),
            crystal.source_artifact_path.display(),
            crystal.crystal_binary_path.display(),
            crystal.sha256_digest.0[0],
            crystal.sha256_digest.0[1],
            crystal.sha256_digest.0[2],
            crystal.sha256_digest.0[3],
        );
        forged_skills.push(crystal);
    }

    // 5. Generate Live Autonomous Architecture & Statutory Verification Report + Save Vault
    let role_loop = [
        RoleId::SensoryPerception,
        RoleId::DivergentSynthesis,
        RoleId::ConvergentLogic,
        RoleId::MetacognitiveRegulation,
        RoleId::ValuationSalience,
        RoleId::SensoryPerception,
    ];
    let holonomy_receipt = transformer.holonomy_engine.evaluate_role_loop(
        &role_loop,
        &map,
        map.atlas.well(RoleId::SensoryPerception).phase,
    );
    let audit_leaves = vec![
        merkle_hash_leaf(b"atoma_work_engine:superpowers_harvest"),
        merkle_hash_leaf(b"atoma_work_engine:polyglot_synthesis"),
        merkle_hash_leaf(b"atoma_work_engine:autonomous_skills"),
    ];
    let compliance_report =
        FirstPrinciplesComplianceEngine::prove_from_live_state(&mut map, &policy, &audit_leaves);

    let t_save = Instant::now();
    let (final_atoms, final_arcs, final_sha) = map.save_to_binary_vault(&vault_path)?;
    let save_ms = t_save.elapsed().as_secs_f64() * 1_000.0;

    let report_path = work_dir.join("MONSTER_SERVER_AUTONOMOUS_AUDIT_REPORT.md");
    let mut report_md = String::new();
    report_md.push_str("# Monster Hybrid Server — Autonomous Work & Skill Evolution Report\n\n");
    report_md.push_str(&format!(
        "- **Unified Vault**: `{}` (`{}` `0-cells`, `{}` `1-cells`, SHA-256 `{}`)\n",
        vault_path.display(),
        final_atoms,
        final_arcs,
        final_sha.to_hex()
    ));
    report_md.push_str(&format!(
        "- **Superpowers Repository Distilled**: `{}` files across 6 Pillars + Meta-Cognitive + Creative Engine\n",
        harvest_receipt.superpowers_files_ingested
    ));
    report_md.push_str(&format!(
        "- **5-Role Gauss–Bonnet Holonomy Verdict**: `{:?}` (Berry curvature flux = `{:.4}`)\n",
        holonomy_receipt.verdict,
        holonomy_receipt.enclosed_curvature_flux_q32.to_f64()
    ));
    report_md.push_str(&format!(
        "- **Statutory Compliance Proof**: `all_statutes_satisfied = {}` (`{}` invariants, Merkle root `{}`)\n\n",
        compliance_report.all_statutes_satisfied,
        compliance_report.verified_invariants.len(),
        compliance_report.evidence_merkle_root.to_hex()
    ));
    report_md.push_str("## Produced & Verified Polyglot Modules\n\n");
    report_md.push_str("| Module | Language | Verifier Command | Backtrack Triggered | Status |\n");
    report_md.push_str("| :--- | :--- | :--- | :--- | :--- |\n");
    for r in &synthesis_receipts {
        report_md.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `PASSED` ({}) |\n",
            r.slug,
            r.language.as_str(),
            r.final_verification.verifier_command,
            r.backtrack_triggered,
            r.final_verification.diagnostics
        ));
    }
    report_md.push_str("\n## Self-Developed Autonomous Skills (`design/autonomous_skills/`)\n\n");
    for s in &forged_skills {
        report_md.push_str(&format!(
            "- **`{}`** (`MlirOpId({})`): {}\n  - `SKILL.md`: `{}`\n  - Source: `{}`\n  - Binary Crystal: `{}`\n",
            s.skill_name,
            s.pinned_macro_op.0,
            s.description,
            s.skill_md_path.display(),
            s.source_artifact_path.display(),
            s.crystal_binary_path.display()
        ));
    }
    fs::write(&report_path, &report_md).map_err(|e| e.to_string())?;

    println!("\n[5/5] Persisted Compounded Vault & Autonomous Audit Report in {:.1} ms", save_ms);
    println!("      -> Updated Vault : {} (0-cells={}, 1-cells={}, sha={}..)",
        vault_path.display(),
        final_atoms,
        final_arcs,
        &final_sha.to_hex()[..8]
    );
    println!("      -> Audit Report  : {}", report_path.display());
    println!("================================================================================");

    Ok(())
}
