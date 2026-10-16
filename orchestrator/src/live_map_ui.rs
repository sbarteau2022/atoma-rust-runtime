//! Interactive Live Qwen-Connected Map Visualizer & Multi-Basin Specialist Carving UI.
//!
//! Serves a self-contained WebGL / HTML5 Canvas 3D Volumetric Poincaré Ball (`B³`, rendering all
//! `221,989` 0-cells in 3D radial cones `r ∈ [0.09, 0.93]` + 3D hyperbolic geodesic arcs) +
//! 2D Torus (`T²` 144-sector) + 12 Attractor Basin (`L1..L12`) interactive map connected to the
//! live `MonsterHybridServer` and `Qwen3.6-35B-A3B-OptiQ-4bit` / `global_active_mind.atoma_map`.

#![forbid(unsafe_code)]

use crate::MonsterHybridServer;
use crate::runtime_model::LiveOperationalTurnReceipt;
use atoma_immune::{
    CANONICAL_DOMAIN_PRESETS, CarvedSpecialistProfile, RoleId, volumetric_cone_ball_coord,
};
use std::path::Path;

pub const DEFAULT_QWEN_SHARD_PATH: &str =
    "/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit/model-00001-of-00005.safetensors";

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

impl MonsterHybridServer {
    /// Run a live domain or prompt probe against the Qwen-connected map and return the
    /// empirically carved multi-basin [`CarvedSpecialistProfile`].
    pub fn probe_and_carve_live_specialist(
        &self,
        domain_or_prompt: &str,
        explicit_basins: Option<&[RoleId]>,
    ) -> CarvedSpecialistProfile {
        let qwen_path = Path::new(DEFAULT_QWEN_SHARD_PATH);
        let qwen_opt = if qwen_path.exists() {
            Some(qwen_path)
        } else {
            None
        };

        let trimmed = domain_or_prompt.trim();
        for preset in &CANONICAL_DOMAIN_PRESETS {
            if trimmed.eq_ignore_ascii_case(preset.domain_id) {
                return self.swarm_map.probe_and_carve_specialist(
                    preset.domain_id,
                    preset.display_name,
                    preset.canonical_prompt,
                    preset.sub_probes,
                    &self.active_transformer,
                    &self.immune.policy,
                    qwen_opt,
                    explicit_basins,
                );
            }
        }

        // Custom live prompt probe
        self.swarm_map.probe_and_carve_specialist(
            "custom_live_probe",
            "Live Qwen + Active-Mind Trajectory Probe",
            trimmed,
            &[],
            &self.active_transformer,
            &self.immune.policy,
            qwen_opt,
            explicit_basins,
        )
    }

    /// Pre-build a compact 16-byte-per-atom binary WebGL point-cloud buffer (`ATOMA_PT` slab)
    /// containing **100% of the global `0-cell` atoms** (`221,989` atoms = `3.55 MB`) mapped into
    /// their true 3D volumetric icosahedral cones (`r ∈ [0.09, 0.93]`).
    ///
    /// Binary layout per atom (`16 bytes` little-endian):
    /// - `x: f32` (`4 bytes`)
    /// - `y: f32` (`4 bytes`)
    /// - `z: f32` (`4 bytes`)
    /// - `role_idx: u8` (`0..11`)
    /// - `sector_144: u8` (`0..143`)
    /// - `radial_u8: u8` (`0..255` representing `r ∈ [0.0, 1.0]`)
    /// - `salience_u8: u8` (`1..255` log-scaled salience + out-degree)
    pub fn render_full_point_cloud_binary(&self) -> Vec<u8> {
        let count = self.swarm_map.global_atom_count();
        let mut buf = Vec::with_capacity(count * 16);
        for (_aid, ga) in self.swarm_map.iter_global_atoms() {
            let out_deg = self.swarm_map.outgoing_arc_indices(ga.id).len();
            let (vb, r_depth, role) =
                volumetric_cone_ball_coord(&self.swarm_map.atlas, ga, out_deg);
            let role_idx = role.carved_index().unwrap_or(0) as u8;
            let sec = ga.invariant.n;
            let rad_u8 = (r_depth.clamp(0.0, 1.0) * 255.0) as u8;
            let sal_u8 = (ga.salience_count.saturating_add(out_deg as u32)).min(255) as u8;
            buf.extend_from_slice(&vb[0].to_le_bytes());
            buf.extend_from_slice(&vb[1].to_le_bytes());
            buf.extend_from_slice(&vb[2].to_le_bytes());
            buf.push(role_idx);
            buf.push(sec);
            buf.push(rad_u8);
            buf.push(sal_u8);
        }
        buf
    }

    /// Serialize the full 12-basin map geometry, overlap matrix, sampled 0-cell cloud, and
    /// canonical domain specialist profiles to JSON for the interactive visualizer.
    pub fn render_map_state_json(&self) -> String {
        let mut json = String::with_capacity(65_536);
        json.push_str("{");
        json.push_str(&format!(
            "\"global_0cells\":{},\"global_1cell_arcs\":{},\"ingested_tensors\":{},\"model_name\":\"{}\",\"qwen_shard_connected\":{},",
            self.swarm_map.global_atom_count(),
            self.swarm_map.reasoning_arcs.len(),
            self.swarm_map.ingested_tensor_digests.len(),
            escape_json(&self.swarm_map.base_weights.model_name),
            Path::new(DEFAULT_QWEN_SHARD_PATH).exists()
        ));

        // 1. 12 Attractor Basins (Information Silos L1..L12)
        json.push_str("\"basins\":[");
        for (i, &role) in RoleId::CARVED_12.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let well = self.swarm_map.atlas.well(role);
            let silo_count = self
                .swarm_map
                .silo(role)
                .map(|s| s.atom_count())
                .unwrap_or(0);
            let neighbors: Vec<&str> = RoleId::CARVED_12
                .iter()
                .copied()
                .filter(|&peer| {
                    peer != role && self.immune.policy.overlap_allowance(role, peer).0 > 0
                })
                .map(|p| p.short_code())
                .collect();
            let n_str = neighbors
                .iter()
                .map(|s| format!("\"{}\"", s))
                .collect::<Vec<_>>()
                .join(",");
            let axis_label = role.duality().map(|d| d.label()).unwrap_or("coordinator");
            let pole_sign: i32 = if role.is_pole_a() { 1 } else { -1 };
            let antipode_code = role
                .opposite_pole()
                .unwrap_or(RoleId::Coordinator)
                .short_code();
            // Place the visual attractor anchor at r = 0.62 along its exact icosahedral unit_dir
            // so the 3D cone radiates from r=0.10 through the well core (0.62) out to r=0.93
            let ux = well.unit_dir[0].to_f64();
            let uy = well.unit_dir[1].to_f64();
            let uz = well.unit_dir[2].to_f64();
            json.push_str(&format!(
                "{{\"index\":{},\"code\":\"{}\",\"name\":\"{}\",\"axis\":\"{}\",\"pole\":{},\"ball\":[{:.5},{:.5},{:.5}],\"unit_dir\":[{:.5},{:.5},{:.5}],\"theta_f\":{:.5},\"theta_s\":{:.5},\"atom_count\":{},\"antipode\":\"{}\",\"neighbors\":[{}]}}",
                i,
                role.short_code(),
                escape_json(role.name()),
                escape_json(axis_label),
                pole_sign,
                ux * 0.62,
                uy * 0.62,
                uz * 0.62,
                ux,
                uy,
                uz,
                well.phase.theta_f.to_f64(),
                well.phase.theta_s.to_f64(),
                silo_count,
                antipode_code,
                n_str
            ));
        }
        json.push_str("],");

        // 2. Sampled background 0-cell atoms for instant fallback before binary stream finishes
        json.push_str("\"background_atoms\":[");
        let total_atoms = self.swarm_map.global_atom_count().max(1);
        let stride = (total_atoms / 600).max(1);
        let mut emitted = 0usize;
        for (idx, (&aid, ga)) in self.swarm_map.iter_global_atoms().enumerate() {
            if idx % stride != 0 || emitted >= 600 {
                continue;
            }
            let out_deg = self.swarm_map.outgoing_arc_indices(ga.id).len();
            let (vb, r_depth, nearest_role) =
                volumetric_cone_ball_coord(&self.swarm_map.atlas, ga, out_deg);
            if emitted > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":{},\"role\":\"{}\",\"ball\":[{:.4},{:.4},{:.4}],\"r\":{:.3},\"sector\":{},\"salience\":{}}}",
                aid,
                nearest_role.short_code(),
                vb[0],
                vb[1],
                vb[2],
                r_depth,
                ga.invariant.n,
                ga.salience_count
            ));
            emitted += 1;
        }
        json.push_str("],");

        // 3. Presets list
        json.push_str("\"presets\":[");
        for (i, p) in CANONICAL_DOMAIN_PRESETS.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":\"{}\",\"name\":\"{}\",\"prompt\":\"{}\"}}",
                p.domain_id,
                escape_json(p.display_name),
                escape_json(p.canonical_prompt)
            ));
        }
        json.push_str("]}");
        json
    }

    /// Serialize a [`CarvedSpecialistProfile`] to JSON for `/api/probe`.
    pub fn render_specialist_profile_json(
        &self,
        profile: &CarvedSpecialistProfile,
        elapsed_us: u128,
    ) -> String {
        let mut json = String::with_capacity(32_768);
        let lit_codes: Vec<String> = profile
            .lit_basins
            .iter()
            .map(|r| format!("\"{}\"", r.short_code()))
            .collect();
        let lit_names: Vec<String> = profile
            .lit_basins
            .iter()
            .map(|r| format!("\"{} ({})\"", r.short_code(), escape_json(r.name())))
            .collect();

        json.push_str("{");
        json.push_str(&format!(
            "\"domain_id\":\"{}\",\"display_name\":\"{}\",\"probe_prompt\":\"{}\",\"qwen_shard_connected\":{},\"latency_us\":{},\"captured_activation_share\":{:.4},\"carved_0cells\":{},\"carved_1cells\":{},\"subtracted_0cells\":{},\"subtracted_1cells\":{},\"holonomy_flux_rad\":{:.6},\"decoded_continuation\":\"{}\",\"lit_basins\":[{}],\"lit_basin_labels\":[{}],",
            escape_json(&profile.domain_id),
            escape_json(&profile.display_name),
            escape_json(&profile.probe_prompt),
            profile.qwen_shard_connected,
            elapsed_us,
            profile.captured_activation_share,
            profile.carved_0cell_count,
            profile.carved_1cell_count,
            profile.subtracted_0cell_count,
            profile.subtracted_1cell_count,
            profile.holonomy_flux_rad,
            escape_json(&profile.decoded_continuation),
            lit_codes.join(","),
            lit_names.join(",")
        ));

        json.push_str("\"basin_metrics\":[");
        for (i, bm) in profile.basin_metrics.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"code\":\"{}\",\"name\":\"{}\",\"total_silo_atoms\":{},\"lit_0cells\":{},\"lit_1cells\":{},\"qwen_affinity\":{:.3},\"activation_mass\":{:.2},\"activation_share\":{:.4},\"selected\":{}}}",
                bm.role.short_code(),
                escape_json(bm.role.name()),
                bm.total_silo_atoms,
                bm.lit_0cell_atoms,
                bm.lit_1cell_arcs,
                bm.qwen_mode_affinity,
                bm.activation_mass,
                bm.activation_share,
                bm.selected_for_specialist
            ));
        }
        json.push_str("],");

        json.push_str("\"lit_atoms\":[");
        for (i, a) in profile.lit_atoms.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":{},\"role\":\"{}\",\"ball\":[{:.5},{:.5},{:.5}],\"theta_f\":{:.4},\"theta_s\":{:.4},\"sector\":{},\"salience\":{},\"op_id\":{},\"waypoint\":{}}}",
                a.atom_id,
                a.role.short_code(),
                a.poincare_ball[0],
                a.poincare_ball[1],
                a.poincare_ball[2],
                a.theta_f,
                a.theta_s,
                a.sector_144,
                a.salience,
                a.op_id,
                a.is_trajectory_waypoint
            ));
        }
        json.push_str("],");

        json.push_str("\"lit_arcs\":[");
        for (i, arc) in profile.lit_arcs.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":{},\"step\":{},\"from_atom\":{},\"to_atom\":{},\"from_role\":\"{}\",\"to_role\":\"{}\",\"from_ball\":[{:.5},{:.5},{:.5}],\"to_ball\":[{:.5},{:.5},{:.5}],\"cosh_action\":{:.5},\"certainty\":{:.4},\"op_id\":{}}}",
                arc.arc_id,
                arc.step_index,
                arc.from_atom,
                arc.to_atom,
                arc.from_role.short_code(),
                arc.to_role.short_code(),
                arc.from_ball[0],
                arc.from_ball[1],
                arc.from_ball[2],
                arc.to_ball[0],
                arc.to_ball[1],
                arc.to_ball[2],
                arc.cosh_action,
                arc.certainty,
                arc.op_id
            ));
        }
        json.push_str("]}");
        json
    }

    /// Serialize a [`LiveOperationalTurnReceipt`] to JSON for `/api/turn` (includes the full
    /// `specialist_profile` payload plus live `turn_receipt` and `.wal` real-time learning stats).
    pub fn render_live_operational_turn_json(&self, op: &LiveOperationalTurnReceipt) -> String {
        let mut base = self.render_specialist_profile_json(&op.specialist_profile, op.latency_us);
        if base.ends_with('}') {
            base.pop();
        }
        let ops_str = op
            .turn
            .emitted_ops
            .iter()
            .take(32)
            .map(|id| id.0.to_string())
            .collect::<Vec<_>>()
            .join(",");
        base.push_str(&format!(
            ",\"turn_receipt\":{{\"session_id\":{},\"turn_index\":{},\"tier\":\"{}\",\"teacher_source\":\"{}\",\"response_text\":\"{}\",\"emitted_op_count\":{},\"emitted_ops\":[{}],\"superposition_holds\":{},\"free_energy\":{:.6},\"omega_readiness\":{:.6},\"atoms_before\":{},\"atoms_after\":{},\"novel_atoms_minted\":{},\"arcs_before\":{},\"arcs_after\":{},\"novel_arcs_minted\":{},\"wal_bytes_appended\":{},\"compliance_verified\":{},\"latency_us\":{}}}}}",
            op.session_id,
            op.turn_index,
            op.turn.tier.as_str(),
            escape_json(&op.teacher_source),
            escape_json(&op.turn.response_text),
            op.turn.emitted_ops.len(),
            ops_str,
            op.turn.tier2_superposition_interventions,
            op.turn.final_free_energy_q32.to_f64(),
            op.turn.final_readiness_omega_q32.to_f64(),
            op.atoms_before,
            op.atoms_after,
            op.novel_atoms_minted,
            op.arcs_before,
            op.arcs_after,
            op.novel_arcs_minted,
            op.wal_bytes_appended,
            op.turn.receipt.compliance_verified,
            op.latency_us
        ));
        base
    }
}

/// Self-contained Interactive WebGL Volumetric 3D Poincaré Ball (`221,989` Atoms) & Live Real-Time Learning Console.
pub const INTERACTIVE_MAP_VISUALIZER_HTML: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MONSTER HYBRID SERVER :: Live Model + Brain Real-Time Execution &amp; 3D Volumetric Map</title>
<style>
  :root {
    --bg: #05080f;
    --panel: #0b101b;
    --panel-border: #1b2538;
    --text: #e4ecf7;
    --muted: #7f91b0;
    --accent: #38bdf8;
    --gold: #f59e0b;
    --emerald: #10b981;
    --rose: #f43f5e;
    --violet: #a855f7;
    --mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    padding: 9px 18px;
    background: #080c14;
    border-bottom: 1px solid var(--panel-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-badge {
    background: linear-gradient(135deg, #0284c7, #7c3aed);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    padding: 4px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
  }
  .brand h1 {
    font-size: 13.5px;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #f8fafc;
  }
  .brand-sub {
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
  }
  .status-pills {
    display: flex;
    gap: 8px;
    font-family: var(--mono);
    font-size: 11px;
  }
  .pill {
    padding: 4px 10px;
    border-radius: 4px;
    background: #111827;
    border: 1px solid #1f2937;
    color: #cbd5e1;
  }
  .pill.live {
    border-color: rgba(16, 185, 129, 0.45);
    color: #34d399;
    background: rgba(16, 185, 129, 0.08);
  }
  .workspace {
    flex: 1;
    display: grid;
    grid-template-columns: 375px 1fr 405px;
    min-height: 0;
  }
  .sidebar, .rightbar {
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 12px;
    gap: 11px;
  }
  .rightbar {
    border-right: none;
    border-left: 1px solid var(--panel-border);
  }
  .card {
    background: #070b12;
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 11px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .card-title {
    font-size: 11px;
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .preset-grid {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .preset-btn {
    background: #0f1626;
    border: 1px solid #1e293b;
    color: var(--text);
    padding: 6px 9px;
    border-radius: 5px;
    text-align: left;
    cursor: pointer;
    transition: all 0.14s ease;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .preset-btn:hover, .preset-btn.active {
    border-color: var(--accent);
    background: rgba(56, 189, 248, 0.11);
  }
  .preset-btn .p-title {
    font-size: 11.5px;
    font-weight: 600;
    color: #f1f5f9;
  }
  .preset-btn .p-sub {
    font-size: 10px;
    font-family: var(--mono);
    color: var(--muted);
  }
  textarea {
    width: 100%;
    min-height: 58px;
    background: #05080f;
    border: 1px solid #1e293b;
    border-radius: 5px;
    color: #f8fafc;
    font-family: var(--mono);
    font-size: 11px;
    padding: 7px;
    resize: vertical;
  }
  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .btn-row {
    display: flex;
    gap: 6px;
  }
  .btn-learn {
    flex: 1.4;
    background: linear-gradient(135deg, #059669, #0284c7);
    color: #fff;
    border: none;
    padding: 8px 10px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
  }
  .btn-learn:hover { filter: brightness(1.12); }
  .btn-primary {
    flex: 1;
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    padding: 7px 9px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-primary:hover { border-color: var(--accent); }
  .btn-secondary {
    background: #111827;
    color: #94a3b8;
    border: 1px solid #1f2937;
    padding: 7px 8px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 10px;
    cursor: pointer;
  }
  .center-stage {
    position: relative;
    display: flex;
    flex-direction: column;
    background: radial-gradient(circle at 50% 48%, #0c1528 0%, #03050a 78%);
    overflow: hidden;
  }
  #glCanvas, #overlayCanvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }
  #overlayCanvas {
    cursor: grab;
  }
  #overlayCanvas:active { cursor: grabbing; }
  .hud-overlay {
    position: absolute;
    top: 12px;
    left: 14px;
    pointer-events: none;
    display: flex;
    flex-direction: column;
    gap: 6px;
    z-index: 5;
  }
  .hud-tag {
    background: rgba(7, 11, 18, 0.86);
    border: 1px solid rgba(56, 189, 248, 0.32);
    padding: 5px 10px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 11px;
    color: #e2e8f0;
  }
  .depth-controls {
    position: absolute;
    top: 12px;
    right: 14px;
    background: rgba(7, 11, 18, 0.9);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 9px 11px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    z-index: 6;
    font-family: var(--mono);
    font-size: 10.5px;
    width: 245px;
  }
  .ctrl-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .ctrl-row input[type="range"] {
    width: 115px;
    accent-color: var(--accent);
  }
  .torus-box {
    position: absolute;
    bottom: 14px;
    left: 14px;
    background: rgba(7, 11, 18, 0.9);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    z-index: 5;
  }
  .torus-box span {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
  }
  .shell-legend {
    position: absolute;
    bottom: 14px;
    right: 14px;
    background: rgba(7, 11, 18, 0.9);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 8px 10px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    display: flex;
    flex-direction: column;
    gap: 3px;
    z-index: 5;
  }
  .basin-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .basin-row {
    display: grid;
    grid-template-columns: 34px 1fr 52px;
    align-items: center;
    gap: 7px;
    padding: 5px 7px;
    border-radius: 4px;
    background: #0c121e;
    border: 1px solid #162032;
    font-size: 11px;
    cursor: pointer;
  }
  .basin-row.lit {
    border-color: var(--gold);
    background: rgba(245, 158, 11, 0.13);
  }
  .basin-code {
    font-family: var(--mono);
    font-weight: 700;
    color: #93c5fd;
  }
  .basin-row.lit .basin-code {
    color: #fbbf24;
  }
  .bar-wrap {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .bar-bg {
    width: 100%;
    height: 5px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    transition: width 0.25s ease;
  }
  .basin-row.lit .bar-fill {
    background: linear-gradient(90deg, #f59e0b, #10b981);
  }
  .carved-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px;
  }
  .stat-cell {
    background: #0c1322;
    border: 1px solid #1e293b;
    padding: 7px;
    border-radius: 5px;
  }
  .stat-val {
    font-family: var(--mono);
    font-size: 13.5px;
    font-weight: 700;
    color: #38bdf8;
  }
  .stat-val.gold { color: #fbbf24; }
  .stat-val.emerald { color: #34d399; }
  .stat-lbl {
    font-size: 9.5px;
    color: var(--muted);
    margin-top: 2px;
  }
  .sphere-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .sphere-chip {
    background: rgba(245, 158, 11, 0.16);
    border: 1px solid #f59e0b;
    color: #fde68a;
    font-family: var(--mono);
    font-size: 10.5px;
    font-weight: 600;
    padding: 3px 7px;
    border-radius: 4px;
  }
  .output-box {
    font-family: var(--mono);
    font-size: 10.5px;
    line-height: 1.42;
    color: #d1fae5;
    background: #05080f;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 8px;
    max-height: 185px;
    overflow-y: auto;
    white-space: pre-wrap;
  }
  .tier-badge {
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-badge">(H³×T²)/Γ₁₂</span>
    <div>
      <h1>MONSTER HYBRID SERVER :: LIVE MODEL + BRAIN REAL-TIME EXECUTION &amp; LEARNING</h1>
      <div class="brand-sub">Tier 1 Geodesic Walk (&lt;1ms) &rarr; Tier 2 Superposition Holding &rarr; Tier 3 Live Teacher + Real-Time .wal Crystallization</div>
    </div>
  </div>
  <div class="status-pills">
    <div class="pill live" id="qwenBadge">MODEL + BRAIN LIVE</div>
    <div class="pill" id="gpuPointBadge">WebGL Point Cloud: Loading...</div>
    <div class="pill" id="vaultStatsPill">Loading map...</div>
  </div>
</header>

<div class="workspace">
  <!-- LEFT SIDEBAR -->
  <div class="sidebar">
    <div class="card">
      <div class="card-title">
        <span>1. Live Model + Brain Execution</span>
        <span id="probeLatency">-- ms</span>
      </div>
      <textarea id="customPromptInput" placeholder="Ask any question or enter any novel prompt (e.g. 'The capital of France is', or a brand-new problem to learn in real time)...">The capital of France is</textarea>
      <div class="btn-row">
        <button class="btn-learn" id="runLiveTurnBtn">EXECUTE &amp; LEARN (LIVE)</button>
        <button class="btn-primary" id="runProbeBtn">PROBE ONLY</button>
        <button class="btn-secondary" id="resetSessionBtn" title="Reset session history">RESET</button>
      </div>
      <div class="carved-stat-grid" style="margin-top:2px;">
        <div class="stat-cell">
          <div class="stat-val emerald" id="statLiveTier">Tier 1 / 2 / 3</div>
          <div class="stat-lbl">Active Inference Tier</div>
        </div>
        <div class="stat-cell">
          <div class="stat-val gold" id="statLiveLearned">+0 0c / +0 1c</div>
          <div class="stat-lbl">Real-Time Minted (.wal)</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>2. Live Brain Response &amp; Learning Log</span>
        <span class="tier-badge" id="teacherSourceBadge">96B GeodesicCursor</span>
      </div>
      <div class="output-box" id="decodedOutputBox">Click "EXECUTE &amp; LEARN (LIVE)" to run a real-time turn through the Brain + Model...</div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>3. Empirical Domain Specialist Presets</span>
        <span style="color:var(--gold)">Auto-Carve</span>
      </div>
      <div class="preset-grid" id="presetContainer"></div>
    </div>
  </div>

  <!-- CENTER STAGE: DUAL-LAYER WEBGL (221,989 3D ATOMS) + 3D GEODESIC OVERLAY -->
  <div class="center-stage">
    <canvas id="glCanvas"></canvas>
    <canvas id="overlayCanvas"></canvas>
    <div class="hud-overlay">
      <div class="hud-tag" id="hudDomainTitle">Active Specialist: Initializing...</div>
      <div class="hud-tag" id="hudCarvedFormula">Carved Basins: --</div>
      <div class="hud-tag" id="hudDepthSlice">3D Volumetric Shell Peeling: r &isin; [0.00, 1.00] (All 221,989 Atoms Rendered on GPU)</div>
    </div>
    <div class="depth-controls">
      <div style="color:var(--accent);font-weight:700;margin-bottom:2px;">3D VOLUMETRIC CAMERA &amp; SHELL PEELER</div>
      <div class="ctrl-row">
        <span>Zoom (FOV):</span>
        <input type="range" id="zoomSlider" min="0.55" max="2.6" step="0.02" value="1.12" />
      </div>
      <div class="ctrl-row">
        <span>Inner Shell r_min:</span>
        <input type="range" id="rMinSlider" min="0.0" max="0.85" step="0.01" value="0.0" />
      </div>
      <div class="ctrl-row">
        <span>Outer Shell r_max:</span>
        <input type="range" id="rMaxSlider" min="0.15" max="1.0" step="0.01" value="1.0" />
      </div>
      <div class="ctrl-row">
        <span>Pruned Silo Alpha:</span>
        <input type="range" id="prunedAlphaSlider" min="0.0" max="0.45" step="0.01" value="0.08" />
      </div>
      <div class="ctrl-row" style="margin-top:2px;">
        <label style="cursor:pointer;"><input type="checkbox" id="showShellsChk" checked /> 3D Radial Shells</label>
        <label style="cursor:pointer;"><input type="checkbox" id="autoRotateChk" checked /> Auto-Orbit</label>
      </div>
    </div>
    <div class="torus-box">
      <span>T² Torus Quotient (144-Sector Phase Grid &psi;*)</span>
      <canvas id="torusCanvas" width="170" height="170"></canvas>
    </div>
    <div class="shell-legend">
      <div style="color:#f8fafc;font-weight:700;">3D HYPERBOLIC RADIAL DEPTH (r = tanh(d/2))</div>
      <div>&bull; Inner Core (r &isin; [0.09, 0.35]): High-Salience Hubs &amp; Early Sensory/Lexical Modes</div>
      <div>&bull; Mid Shell (r &isin; [0.35, 0.65]): Operator Wells &amp; Cross-Layer Geodesic Bridges</div>
      <div>&bull; Outer Boundary (r &isin; [0.65, 0.93]): Deep Specialist Leaf Trajectories &amp; Code Spans</div>
      <div style="color:var(--accent);">Drag to 3D orbit &bull; Scroll wheel to zoom into interior shells</div>
    </div>
  </div>

  <!-- RIGHT SIDEBAR -->
  <div class="rightbar">
    <div class="card">
      <div class="card-title">
        <span>Carved Multi-Basin Specialist</span>
        <span style="color:var(--gold)" id="carvedSphereCount">0 Spheres</span>
      </div>
      <div style="font-size:10.5px;color:var(--muted);">
        Lit Attractor Cones Carved Together (click any basin below to toggle custom multi-cone carving):
      </div>
      <div class="sphere-chips" id="carvedSphereChips"></div>
      <div class="carved-stat-grid">
        <div class="stat-cell">
          <div class="stat-val gold" id="statCarved0Cells">--</div>
          <div class="stat-lbl">Carved 0-Cell Atoms Kept</div>
        </div>
        <div class="stat-cell">
          <div class="stat-val emerald" id="statCarved1Cells">--</div>
          <div class="stat-lbl">Carved 1-Cell Arcs Kept</div>
        </div>
        <div class="stat-cell">
          <div class="stat-val" id="statSubtracted0Cells">--</div>
          <div class="stat-lbl">Out-of-Basin 0-Cells Subtracted</div>
        </div>
        <div class="stat-cell">
          <div class="stat-val" id="statHolonomyFlux">--</div>
          <div class="stat-lbl">2-Cell Gauss-Bonnet Flux (rad)</div>
        </div>
      </div>
    </div>

    <div class="card" style="flex:1;">
      <div class="card-title">
        <span>12 Information Silos (3D Attractor Cones)</span>
        <span id="capturedShareBadge">Share: --%</span>
      </div>
      <div class="basin-list" id="basinList"></div>
    </div>
  </div>
</div>

<script>
let mapState = null;
let currentProfile = null;
let manualBasinSet = new Set();
let currentSessionId = 1;
let rotY = 0.52;
let rotX = 0.34;
let zoomLevel = 1.12;
let rMinCut = 0.0;
let rMaxCut = 1.0;
let prunedAlpha = 0.08;
let showShells = true;
let autoRotate = true;
let isDragging = false;
let lastMouse = { x: 0, y: 0 };
let animPhase = 0;

// WebGL full point cloud state (all 221,989 atoms)
let gl = null;
let glProg = null;
let glVbo = null;
let glPointCount = 0;
let activeMaskArray = new Float32Array(12).fill(1.0);

const ROLE_CODES = ["L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","L12"];
const ROLE_COLORS = {
  L1: "#38bdf8", L2: "#f43f5e", L3: "#10b981", L4: "#f59e0b",
  L5: "#a855f7", L6: "#06b6d4", L7: "#3b82f6", L8: "#ec4899",
  L9: "#eab308", L10: "#14b8a6", L11: "#6366f1", L12: "#ef4444"
};
const ROLE_RGB = [
  [0.22, 0.74, 0.97], [0.96, 0.25, 0.37], [0.06, 0.73, 0.51], [0.96, 0.62, 0.04],
  [0.66, 0.33, 0.97], [0.02, 0.71, 0.83], [0.23, 0.51, 0.96], [0.93, 0.28, 0.60],
  [0.92, 0.70, 0.03], [0.08, 0.72, 0.65], [0.39, 0.40, 0.95], [0.94, 0.27, 0.27]
];

async function init() {
  setupCanvasInteraction();
  initWebGL();

  const res = await fetch("/api/map-state");
  mapState = await res.json();
  document.getElementById("vaultStatsPill").textContent =
    `0-Cells: ${mapState.global_0cells.toLocaleString()} | 1-Cell Arcs: ${mapState.global_1cell_arcs.toLocaleString()}`;
  document.getElementById("qwenBadge").textContent = mapState.qwen_shard_connected
    ? "QWEN3.6-35B-A3B + BRAIN LIVE"
    : "ACTIVE BRAIN LIVE";

  renderPresets();
  await runProbe("coding_intensive", true);
  requestAnimationFrame(renderLoop);

  // Stream the full 221,989-atom 3D volumetric point cloud binary buffer in the background
  loadFullPointCloudBinary();
}

function initWebGL() {
  const canvas = document.getElementById("glCanvas");
  gl = canvas.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: false });
  if (!gl) return;

  const vsSource = `
    attribute vec3 aPos;
    attribute vec4 aMeta; // x: role_idx (0..11), y: sector, z: radial_u8 (0..255), w: salience_u8
    uniform float uRotX;
    uniform float uRotY;
    uniform float uZoom;
    uniform float uAspect;
    uniform float uRMin;
    uniform float uRMax;
    uniform float uPrunedAlpha;
    uniform float uActiveMask[12];
    uniform vec3 uRoleColors[12];
    varying vec4 vColor;

    void main() {
      float rNorm = aMeta.z / 255.0;
      if (rNorm < uRMin || rNorm > uRMax) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        vColor = vec4(0.0);
        return;
      }

      int rIdx = int(aMeta.x + 0.5);
      float isCarved = 0.0;
      vec3 baseCol = vec3(0.35, 0.65, 0.95);
      for (int i = 0; i < 12; i++) {
        if (i == rIdx) {
          isCarved = uActiveMask[i];
          baseCol = uRoleColors[i];
        }
      }

      // 3D rotation (yaw uRotY, pitch uRotX)
      float cy = cos(uRotY);
      float sy = sin(uRotY);
      float cx = cos(uRotX);
      float sx = sin(uRotX);

      float x1 = aPos.x * cy - aPos.z * sy;
      float z1 = aPos.x * sy + aPos.z * cy;
      float y2 = aPos.y * cx - z1 * sx;
      float z2 = aPos.y * sx + z1 * cx;

      // True 3D perspective camera with focal depth
      float camDist = 2.15;
      float persp = camDist / max(0.35, camDist - z2 * 1.15);
      float scale = 0.78 * uZoom * persp;

      gl_Position = vec4((x1 * scale) / uAspect, y2 * scale, -z2 * 0.5, 1.0);

      // Depth fog + radial shell luminance
      float depthBoost = clamp((z2 + 1.0) * 0.5, 0.18, 1.0);
      float salBoost = clamp(aMeta.w / 12.0, 0.0, 1.0);
      float alpha = isCarved > 0.5
        ? (0.28 + 0.42 * depthBoost + 0.25 * salBoost)
        : (uPrunedAlpha * depthBoost);

      // Brighten inner high-salience hubs slightly toward gold/white
      vec3 finalCol = mix(baseCol, vec3(1.0, 0.92, 0.65), (1.0 - rNorm) * 0.35 + salBoost * 0.25);
      vColor = vec4(finalCol, alpha);
      gl_PointSize = isCarved > 0.5
        ? clamp((1.6 + salBoost * 2.2) * persp * sqrt(uZoom), 1.2, 6.5)
        : clamp(1.15 * persp, 0.8, 2.5);
    }
  `;

  const fsSource = `
    precision mediump float;
    varying vec4 vColor;
    void main() {
      vec2 c = gl_PointCoord - vec2(0.5);
      float r2 = dot(c, c);
      if (r2 > 0.25) discard;
      float soft = 1.0 - smoothstep(0.08, 0.25, r2);
      gl_FragColor = vec4(vColor.rgb, vColor.a * soft);
    }
  `;

  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  };
  const vs = compile(gl.VERTEX_SHADER, vsSource);
  const fs = compile(gl.FRAGMENT_SHADER, fsSource);
  glProg = gl.createProgram();
  gl.attachShader(glProg, vs);
  gl.attachShader(glProg, fs);
  gl.linkProgram(glProg);
}

async function loadFullPointCloudBinary() {
  try {
    const res = await fetch("/api/map-points.bin");
    const buf = await res.arrayBuffer();
    const totalBytes = buf.byteLength;
    const numPoints = Math.floor(totalBytes / 16);
    if (!gl || numPoints === 0) return;

    const dv = new DataView(buf);
    const floats = new Float32Array(numPoints * 7);
    for (let i = 0; i < numPoints; i++) {
      const off = i * 16;
      const base = i * 7;
      floats[base + 0] = dv.getFloat32(off + 0, true);
      floats[base + 1] = dv.getFloat32(off + 4, true);
      floats[base + 2] = dv.getFloat32(off + 8, true);
      floats[base + 3] = dv.getUint8(off + 12); // role_idx
      floats[base + 4] = dv.getUint8(off + 13); // sector_144
      floats[base + 5] = dv.getUint8(off + 14); // radial_u8
      floats[base + 6] = dv.getUint8(off + 15); // salience_u8
    }

    glVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, glVbo);
    gl.bufferData(gl.ARRAY_BUFFER, floats, gl.STATIC_DRAW);
    glPointCount = numPoints;
    document.getElementById("gpuPointBadge").textContent =
      `GPU 3D Point Cloud: ${numPoints.toLocaleString()} Atoms (100% Vault)`;
  } catch (e) {
    console.error("Failed to load full point cloud:", e);
  }
}

function renderPresets() {
  const container = document.getElementById("presetContainer");
  container.innerHTML = "";
  mapState.presets.forEach(p => {
    const btn = document.createElement("button");
    btn.className = "preset-btn";
    btn.dataset.id = p.id;
    btn.innerHTML = `<span class="p-title">${p.name}</span><span class="p-sub">${p.prompt.slice(0, 68)}...</span>`;
    btn.onclick = () => {
      manualBasinSet.clear();
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("customPromptInput").value = p.prompt;
      runProbe(p.id, true);
    };
    container.appendChild(btn);
  });
}

async function runProbe(queryOrDomain, isPreset = false) {
  const basinsParam = manualBasinSet.size > 0 ? `&basins=${Array.from(manualBasinSet).join(",")}` : "";
  const url = `/api/probe?q=${encodeURIComponent(queryOrDomain)}${basinsParam}`;
  const res = await fetch(url);
  currentProfile = await res.json();
  updateUI(currentProfile);
}

async function runLiveTurn(promptText) {
  const basinsParam = manualBasinSet.size > 0 ? `&basins=${Array.from(manualBasinSet).join(",")}` : "";
  const url = `/api/turn?q=${encodeURIComponent(promptText)}&session=${currentSessionId}&max_new=32${basinsParam}`;
  const res = await fetch(url);
  currentProfile = await res.json();
  updateUI(currentProfile);
}

function updateUI(prof) {
  const litSet = new Set(prof.lit_basins || []);
  for (let i = 0; i < 12; i++) {
    activeMaskArray[i] = litSet.has(ROLE_CODES[i]) ? 1.0 : 0.0;
  }

  document.getElementById("probeLatency").textContent = `${(prof.latency_us / 1000).toFixed(2)} ms`;
  document.getElementById("hudDomainTitle").textContent = `Specialist: ${prof.display_name}`;
  document.getElementById("hudCarvedFormula").textContent =
    `Carved from ${prof.lit_basins.length} Lit 3D Attractor Cones: [ ${prof.lit_basins.join(" + ")} ] (${(prof.captured_activation_share * 100).toFixed(1)}% mass)`;
  document.getElementById("carvedSphereCount").textContent = `${prof.lit_basins.length} / 12 Cones`;
  document.getElementById("statCarved0Cells").textContent = prof.carved_0cells.toLocaleString();
  document.getElementById("statCarved1Cells").textContent = prof.carved_1cells.toLocaleString();
  document.getElementById("statSubtracted0Cells").textContent = prof.subtracted_0cells.toLocaleString();
  document.getElementById("statHolonomyFlux").textContent = `${prof.holonomy_flux_rad.toFixed(4)} rad`;
  document.getElementById("capturedShareBadge").textContent =
    `Captured: ${(prof.captured_activation_share * 100).toFixed(1)}%`;

  const chips = document.getElementById("carvedSphereChips");
  chips.innerHTML = "";
  prof.lit_basin_labels.forEach(lbl => {
    const s = document.createElement("span");
    s.className = "sphere-chip";
    s.textContent = lbl;
    chips.appendChild(s);
  });

  const outBox = document.getElementById("decodedOutputBox");
  if (prof.turn_receipt) {
    const tr = prof.turn_receipt;
    document.getElementById("statLiveTier").textContent = tr.tier.replace("Tier", "Tier ");
    document.getElementById("statLiveLearned").textContent =
      `+${tr.novel_atoms_minted} 0c / +${tr.novel_arcs_minted} 1c`;
    document.getElementById("teacherSourceBadge").textContent = tr.teacher_source;
    document.getElementById("vaultStatsPill").textContent =
      `0-Cells: ${tr.atoms_after.toLocaleString()} | 1-Cell Arcs: ${tr.arcs_after.toLocaleString()}`;

    outBox.textContent =
      `[Turn #${tr.turn_index} | Session ${tr.session_id} | ${(tr.latency_us / 1000).toFixed(2)} ms]\n` +
      `Tier: ${tr.tier} (${tr.teacher_source})\n` +
      `Prompt: "${prof.probe_prompt}"\n\n` +
      `Response:\n${tr.response_text}\n\n` +
      `Real-Time Brain Learning:\n` +
      `  • 0-Cells: ${tr.atoms_before.toLocaleString()} -> ${tr.atoms_after.toLocaleString()} (+${tr.novel_atoms_minted} novel)\n` +
      `  • 1-Cell Arcs: ${tr.arcs_before.toLocaleString()} -> ${tr.arcs_after.toLocaleString()} (+${tr.novel_arcs_minted} novel)\n` +
      `  • WAL Appended: ${tr.wal_bytes_appended} B | Readiness Ω=${tr.omega_readiness.toFixed(4)}`;
  } else {
    document.getElementById("statLiveTier").textContent = "Probe Mode";
    document.getElementById("statLiveLearned").textContent = "Read-Only";
    document.getElementById("teacherSourceBadge").textContent = "96B GeodesicCursor";
    outBox.textContent = prof.decoded_continuation
      ? `Prompt: "${prof.probe_prompt}"\n\nDecoded Continuation:\n${prof.decoded_continuation}\n\nLit Atoms: ${prof.lit_atoms.length} | Lit Geodesic Arcs: ${prof.lit_arcs.length}`
      : `Prompt: "${prof.probe_prompt}"\n\n[Geodesic Receptor Projection Complete - ${prof.lit_atoms.length} 0-cell atoms & ${prof.lit_arcs.length} 1-cell trajectory arcs illuminated across ${prof.lit_basins.join(", ")}]`;
  }

  const list = document.getElementById("basinList");
  list.innerHTML = "";
  const sortedMetrics = [...prof.basin_metrics].sort((a, b) => b.activation_share - a.activation_share);
  sortedMetrics.forEach(bm => {
    const pct = (bm.activation_share * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = `basin-row ${bm.selected ? "lit" : ""}`;
    row.innerHTML = `
      <div class="basin-code">${bm.code}</div>
      <div class="bar-wrap">
        <div style="display:flex;justify-content:space-between;">
          <span>${bm.name}</span>
          <span style="color:var(--muted);font-family:var(--mono);font-size:10px;">0c:${bm.lit_0cells} 1c:${bm.lit_1cells}</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width:${Math.min(100, bm.activation_share * 260)}%"></div></div>
      </div>
      <div style="text-align:right;font-family:var(--mono);font-weight:700;color:${bm.selected ? '#fbbf24' : '#94a3b8'}">${pct}%</div>
    `;
    row.onclick = () => {
      if (manualBasinSet.size === 0) {
        prof.lit_basins.forEach(c => manualBasinSet.add(c));
      }
      if (manualBasinSet.has(bm.code)) {
        if (manualBasinSet.size > 1) manualBasinSet.delete(bm.code);
      } else {
        manualBasinSet.add(bm.code);
      }
      const q = document.getElementById("customPromptInput").value || prof.domain_id;
      runProbe(q, false);
    };
    list.appendChild(row);
  });

  drawTorus(prof);
}

function drawTorus(prof) {
  const c = document.getElementById("torusCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const cell = c.width / 12;
  for (let r = 0; r < 12; r++) {
    for (let col = 0; col < 12; col++) {
      ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
      ctx.strokeRect(col * cell, r * cell, cell, cell);
    }
  }
  if (!prof) return;
  prof.lit_atoms.forEach(a => {
    const sec = a.sector % 144;
    const row = Math.floor(sec / 12);
    const col = sec % 12;
    ctx.fillStyle = a.waypoint ? "#fbbf24" : (ROLE_COLORS[a.role] || "#38bdf8");
    ctx.fillRect(col * cell + 2, row * cell + 2, cell - 4, cell - 4);
  });
}

function setupCanvasInteraction() {
  const glCanvas = document.getElementById("glCanvas");
  const overlay = document.getElementById("overlayCanvas");
  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    glCanvas.width = glCanvas.clientWidth * dpr;
    glCanvas.height = glCanvas.clientHeight * dpr;
    overlay.width = overlay.clientWidth * dpr;
    overlay.height = overlay.clientHeight * dpr;
    if (gl) gl.viewport(0, 0, glCanvas.width, glCanvas.height);
  };
  window.addEventListener("resize", resize);
  resize();

  overlay.addEventListener("mousedown", e => {
    isDragging = true;
    lastMouse = { x: e.clientX, y: e.clientY };
  });
  window.addEventListener("mouseup", () => { isDragging = false; });
  window.addEventListener("mousemove", e => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    lastMouse = { x: e.clientX, y: e.clientY };
    rotY += dx * 0.0075;
    rotX = Math.max(-1.48, Math.min(1.48, rotX + dy * 0.0075));
  });
  overlay.addEventListener("wheel", e => {
    e.preventDefault();
    zoomLevel = Math.max(0.55, Math.min(2.6, zoomLevel * (e.deltaY < 0 ? 1.07 : 0.93)));
    document.getElementById("zoomSlider").value = zoomLevel.toFixed(2);
  }, { passive: false });

  document.getElementById("zoomSlider").oninput = e => { zoomLevel = parseFloat(e.target.value); };
  document.getElementById("rMinSlider").oninput = e => {
    rMinCut = parseFloat(e.target.value);
    if (rMinCut >= rMaxCut) rMinCut = Math.max(0, rMaxCut - 0.05);
    updateDepthHud();
  };
  document.getElementById("rMaxSlider").oninput = e => {
    rMaxCut = parseFloat(e.target.value);
    if (rMaxCut <= rMinCut) rMaxCut = Math.min(1.0, rMinCut + 0.05);
    updateDepthHud();
  };
  document.getElementById("prunedAlphaSlider").oninput = e => { prunedAlpha = parseFloat(e.target.value); };
  document.getElementById("showShellsChk").onchange = e => { showShells = e.target.checked; };
  document.getElementById("autoRotateChk").onchange = e => { autoRotate = e.target.checked; };

  document.getElementById("runLiveTurnBtn").onclick = () => {
    const q = document.getElementById("customPromptInput").value.trim();
    if (q) runLiveTurn(q);
  };
  document.getElementById("runProbeBtn").onclick = () => {
    const q = document.getElementById("customPromptInput").value.trim();
    if (q) runProbe(q, false);
  };
  document.getElementById("resetSessionBtn").onclick = () => {
    currentSessionId += 1;
    manualBasinSet.clear();
    document.getElementById("statLiveTier").textContent = `Session #${currentSessionId}`;
  };
}

function updateDepthHud() {
  document.getElementById("hudDepthSlice").textContent =
    `3D Volumetric Shell Peeling: r ∈ [${rMinCut.toFixed(2)}, ${rMaxCut.toFixed(2)}] (${glPointCount > 0 ? glPointCount.toLocaleString() : '221,989'} Atoms on GPU)`;
}

function project3D(pt, cx, cy, baseScale) {
  let [x, y, z] = pt;
  const cyRot = Math.cos(rotY), syRot = Math.sin(rotY);
  const cxRot = Math.cos(rotX), sxRot = Math.sin(rotX);
  let x1 = x * cyRot - z * syRot;
  let z1 = x * syRot + z * cyRot;
  let y2 = y * cxRot - z1 * sxRot;
  let z2 = y * sxRot + z1 * cxRot;
  const camDist = 2.15;
  const persp = camDist / Math.max(0.35, camDist - z2 * 1.15);
  return {
    sx: cx + x1 * baseScale * zoomLevel * persp,
    sy: cy - y2 * baseScale * zoomLevel * persp,
    z: z2,
    scale: persp * Math.sqrt(zoomLevel)
  };
}

// Draw a 3D wireframe latitude/longitude ring at hyperbolic radius `rShell`
function draw3DShellRing(ctx, rShell, cx, cy, baseScale, color, dashed = false) {
  const steps = 64;
  ctx.save();
  if (dashed) ctx.setLineDash([4, 4]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.1;

  // Equatorial XZ ring
  ctx.beginPath();
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * Math.PI * 2;
    const p = project3D([rShell * Math.cos(a), 0, rShell * Math.sin(a)], cx, cy, baseScale);
    if (i === 0) ctx.moveTo(p.sx, p.sy);
    else ctx.lineTo(p.sx, p.sy);
  }
  ctx.stroke();

  // Meridian XY ring
  ctx.beginPath();
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * Math.PI * 2;
    const p = project3D([rShell * Math.cos(a), rShell * Math.sin(a), 0], cx, cy, baseScale);
    if (i === 0) ctx.moveTo(p.sx, p.sy);
    else ctx.lineTo(p.sx, p.sy);
  }
  ctx.stroke();
  ctx.restore();
}

// Sample a true 3D hyperbolic geodesic arc in B³ (curving inward toward the origin in 3D before projecting)
function draw3DHyperbolicGeodesic(ctx, pFrom3D, pTo3D, cx, cy, baseScale, color, width) {
  const segs = 14;
  const mid3D = [
    (pFrom3D[0] + pTo3D[0]) * 0.34,
    (pFrom3D[1] + pTo3D[1]) * 0.34,
    (pFrom3D[2] + pTo3D[2]) * 0.34
  ];
  ctx.beginPath();
  for (let s = 0; s <= segs; s++) {
    const t = s / segs;
    const omt = 1 - t;
    const gx = omt * omt * pFrom3D[0] + 2 * omt * t * mid3D[0] + t * t * pTo3D[0];
    const gy = omt * omt * pFrom3D[1] + 2 * omt * t * mid3D[1] + t * t * pTo3D[1];
    const gz = omt * omt * pFrom3D[2] + 2 * omt * t * mid3D[2] + t * t * pTo3D[2];
    const proj = project3D([gx, gy, gz], cx, cy, baseScale);
    if (s === 0) ctx.moveTo(proj.sx, proj.sy);
    else ctx.lineTo(proj.sx, proj.sy);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function renderWebGLPointCloud(w, h) {
  if (!gl || !glProg || !glVbo || glPointCount === 0) return;
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

  gl.useProgram(glProg);
  gl.bindBuffer(gl.ARRAY_BUFFER, glVbo);

  const posLoc = gl.getAttribLocation(glProg, "aPos");
  const metaLoc = gl.getAttribLocation(glProg, "aMeta");
  gl.enableVertexAttribArray(posLoc);
  gl.enableVertexAttribArray(metaLoc);
  gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 28, 0);
  gl.vertexAttribPointer(metaLoc, 4, gl.FLOAT, false, 28, 12);

  gl.uniform1f(gl.getUniformLocation(glProg, "uRotX"), rotX);
  gl.uniform1f(gl.getUniformLocation(glProg, "uRotY"), rotY);
  gl.uniform1f(gl.getUniformLocation(glProg, "uZoom"), zoomLevel);
  gl.uniform1f(gl.getUniformLocation(glProg, "uAspect"), w / Math.max(1, h));
  gl.uniform1f(gl.getUniformLocation(glProg, "uRMin"), rMinCut);
  gl.uniform1f(gl.getUniformLocation(glProg, "uRMax"), rMaxCut);
  gl.uniform1f(gl.getUniformLocation(glProg, "uPrunedAlpha"), prunedAlpha);
  gl.uniform1fv(gl.getUniformLocation(glProg, "uActiveMask[0]"), activeMaskArray);

  const flatCols = new Float32Array(36);
  for (let i = 0; i < 12; i++) {
    flatCols[i * 3 + 0] = ROLE_RGB[i][0];
    flatCols[i * 3 + 1] = ROLE_RGB[i][1];
    flatCols[i * 3 + 2] = ROLE_RGB[i][2];
  }
  gl.uniform3fv(gl.getUniformLocation(glProg, "uRoleColors[0]"), flatCols);

  gl.drawArrays(gl.POINTS, 0, glPointCount);
}

function renderLoop() {
  animPhase += 0.022;
  if (!isDragging && autoRotate) rotY += 0.0028;

  const overlay = document.getElementById("overlayCanvas");
  const ctx = overlay.getContext("2d");
  const w = overlay.width;
  const h = overlay.height;
  ctx.clearRect(0, 0, w, h);

  // 1. Render all 221,989 3D volumetric atoms on the GPU WebGL canvas
  renderWebGLPointCloud(w, h);

  const cx = w * 0.5;
  const cy = h * 0.5;
  const baseScale = Math.min(w, h) * 0.39;

  ctx.save();
  // 2. Render 3D concentric hyperbolic shells (r = 0.35 Core, r = 0.65 Mid-Shell, r = 0.93 Boundary)
  if (showShells) {
    draw3DShellRing(ctx, 0.35, cx, cy, baseScale, "rgba(56, 189, 248, 0.14)", true);
    draw3DShellRing(ctx, 0.65, cx, cy, baseScale, "rgba(129, 140, 248, 0.16)", true);
    draw3DShellRing(ctx, 0.93, cx, cy, baseScale, "rgba(56, 189, 248, 0.28)", false);
  }

  // Draw Coordinator L0 Origin Singularity at (0, 0, 0)
  const pOrigin = project3D([0, 0, 0], cx, cy, baseScale);
  ctx.beginPath();
  ctx.arc(pOrigin.sx, pOrigin.sy, 4.5 * pOrigin.scale, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();

  if (mapState) {
    const litSet = new Set(currentProfile ? currentProfile.lit_basins : []);
    const shareMap = {};
    if (currentProfile) {
      currentProfile.basin_metrics.forEach(m => { shareMap[m.code] = m.activation_share; });
    }

    // Fallback sample if WebGL point buffer hasn't finished streaming yet
    if (glPointCount === 0) {
      mapState.background_atoms.forEach(a => {
        if (a.r < rMinCut || a.r > rMaxCut) return;
        const p = project3D(a.ball, cx, cy, baseScale);
        const isCarved = litSet.has(a.role);
        ctx.fillStyle = isCarved ? (ROLE_COLORS[a.role] || "#38bdf8") : "rgba(71, 85, 105, 0.18)";
        ctx.fillRect(p.sx - 1.5, p.sy - 1.5, 3, 3);
      });
    }

    // 3. Draw 3D radial cone rays from L0 Origin (0,0,0) through each of the 12 Attractor Basins out to r=0.92
    const basinByCode = {};
    mapState.basins.forEach(b => { basinByCode[b.code] = b; });
    mapState.basins.forEach(b => {
      const isLit = litSet.has(b.code);
      const u = b.unit_dir;
      const pTip = project3D([u[0] * 0.91, u[1] * 0.91, u[2] * 0.91], cx, cy, baseScale);
      ctx.beginPath();
      ctx.moveTo(pOrigin.sx, pOrigin.sy);
      ctx.lineTo(pTip.sx, pTip.sy);
      ctx.strokeStyle = isLit ? "rgba(245, 158, 11, 0.42)" : "rgba(71, 85, 105, 0.14)";
      ctx.lineWidth = isLit ? 1.8 : 0.8;
      ctx.stroke();

      // Icosahedral neighbor geodesic struts in 3D
      b.neighbors.forEach(ncode => {
        if (b.code < ncode && basinByCode[ncode]) {
          const bothLit = isLit && litSet.has(ncode);
          if (bothLit) {
            draw3DHyperbolicGeodesic(
              ctx,
              b.ball,
              basinByCode[ncode].ball,
              cx,
              cy,
              baseScale,
              "rgba(245, 158, 11, 0.62)",
              2.0
            );
          }
        }
      });
    });

    // 4. Draw illuminated 1-cell trajectory arcs as true 3D hyperbolic geodesics
    if (currentProfile && currentProfile.lit_arcs) {
      currentProfile.lit_arcs.forEach(arc => {
        draw3DHyperbolicGeodesic(
          ctx,
          arc.from_ball,
          arc.to_ball,
          cx,
          cy,
          baseScale,
          "rgba(16, 185, 129, 0.86)",
          2.2
        );
      });
    }

    // 5. Draw illuminated probe 0-cell waypoints depth-sorted in 3D
    if (currentProfile && currentProfile.lit_atoms) {
      const sortedLit = currentProfile.lit_atoms.map(a => {
        const r = Math.hypot(a.ball[0], a.ball[1], a.ball[2]);
        return { a, r, p: project3D(a.ball, cx, cy, baseScale) };
      }).filter(item => item.r >= rMinCut && item.r <= rMaxCut)
        .sort((x, y) => x.p.z - y.p.z);

      sortedLit.forEach(({ a, p }) => {
        const rad = (a.waypoint ? 4.2 : 2.8) * p.scale;
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, rad, 0, Math.PI * 2);
        ctx.fillStyle = a.waypoint ? "#fbbf24" : (ROLE_COLORS[a.role] || "#38bdf8");
        ctx.fill();
      });
    }

    // 6. Draw the 12 Attractor Basin anchors sorted back-to-front by 3D z-depth
    const projectedBasins = mapState.basins.map(b => ({
      b,
      p: project3D(b.ball, cx, cy, baseScale),
      isLit: litSet.has(b.code),
      share: shareMap[b.code] || 0
    })).sort((a, b) => a.p.z - b.p.z);

    projectedBasins.forEach(({ b, p, isLit, share }) => {
      const depthAlpha = Math.max(0.28, Math.min(1.0, (p.z + 1.1) * 0.55));
      const baseRadius = (isLit ? 18 + share * 46 : 10) * p.scale;
      if (isLit) {
        const pulse = 1 + 0.12 * Math.sin(animPhase * 2.5);
        const grad = ctx.createRadialGradient(p.sx, p.sy, 2, p.sx, p.sy, baseRadius * 1.85 * pulse);
        grad.addColorStop(0, "rgba(245, 158, 11, 0.72)");
        grad.addColorStop(0.5, "rgba(16, 185, 129, 0.26)");
        grad.addColorStop(1, "rgba(245, 158, 11, 0.0)");
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, baseRadius * 1.85 * pulse, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
      }

      ctx.save();
      ctx.globalAlpha = isLit ? 1.0 : depthAlpha * 0.75;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, baseRadius * 0.52, 0, Math.PI * 2);
      ctx.fillStyle = isLit ? "#f59e0b" : "#111827";
      ctx.fill();
      ctx.lineWidth = isLit ? 2.4 : 1.3;
      ctx.strokeStyle = isLit ? "#fef08a" : (ROLE_COLORS[b.code] || "#64748b");
      ctx.stroke();

      ctx.fillStyle = isLit ? "#ffffff" : "#94a3b8";
      ctx.font = `${isLit ? "700" : "600"} ${Math.round(10.5 * Math.min(1.3, p.scale))}px "JetBrains Mono", monospace`;
      ctx.textAlign = "center";
      ctx.fillText(`${b.code} ${(share * 100).toFixed(0)}%`, p.sx, p.sy - baseRadius * 0.65 - 4);
      ctx.font = `500 ${Math.round(9 * Math.min(1.25, p.scale))}px -apple-system, sans-serif`;
      ctx.fillStyle = isLit ? "#fde68a" : "#64748b";
      ctx.fillText(b.name, p.sx, p.sy + baseRadius * 0.65 + 11);
      ctx.restore();
    });
  }

  ctx.restore();
  requestAnimationFrame(renderLoop);
}

init();
</script>
</body>
</html>
"####;
