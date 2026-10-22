//! Tier 4–6 (DESIGN.md): Layer 2 — Erlang/OTP-Style Supervision Tree, Carved Role Processes,
//! Supervised `core_lm` Inference Workers, and Ephemeral Agent Dissolution + Map Deposits.
//!
//! Enforces `SPEC.md` §1 (Layer 2 — Governor) & `design/QUALIFICATION_GATES.md` §3:
//! - Spawns role instances (`L0 Coordinator` + `L1..L12 Specialists`), each owning a supervised
//!   `CoreLmInferenceWorker` (`core_lm -> orchestrator`) and gated by Layer 3 (`orchestrator -> immune_system`).
//! - Enforces a mandatory Layer 3 `CompliancePermit` check on every `core_lm` forward pass
//!   (`core_lm -> immune_system`).
//! - When an agent completes its task, it **dissolves** (`ProcessState::Dissolved`) and surrenders
//!   its local `GeometricMemoryMap` and `3×3` Wigner gyroscope frame back to the `Governor` as an
//!   [`EphemeralMapDeposit`], which the `Governor` compiles into the centralized `(H³ × T²) / Γ`
//!   memory architecture under Layer 3 (`authorize_deposit_compilation`).

use crate::routing::{DeliveredHandoff, GovernorRouter, HandoffRequest};
use atoma_immune::cache::geometric::{AtomId, GeometricMemoryMap};
use atoma_immune::geometry::bridge::{BridgeLossReport, f32_to_atom_address};
use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::geometry::lorentz::LorentzVector;
use atoma_immune::ops::{matvec, softmax};
use atoma_immune::{
    AdaptedKvRecord, CarvedSwarmMap, CompliancePermit, ComplianceViolation,
    DepositCompilationReceipt, DepositedAtomRecord, EphemeralMapDeposit, IDENTITY_FRAME_Q32,
    ImmuneSystem, ModelAdapterLens, RoleId, SiloAtom,
};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Restart only the crashed/timed-out child process (Erlang `one_for_one`).
    OneForOne,
    /// Restart the failed child's whole restart group (Erlang `one_for_all`
    /// within a group). Groups: Coordination = {Coordinator},
    /// Specialists = {L1..L12}. The Coordinator is supervised by the Governor's
    /// root loop independently (CS-NATIVE-04 D2): a Coordinator failure restarts
    /// only the Coordination group and never triggers a specialist restart, and
    /// a specialist failure never leaves the Coordinator unrestarded.
    OneForAll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartIntensity {
    pub max_restarts: u32,
    pub within_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    Running,
    /// Ephemeral agent finished its task, surrendered its `EphemeralMapDeposit` to the Governor,
    /// and dissolved cleanly (ready to be respawned on demand when next needed).
    Dissolved,
    Crashed,
    TimedOut,
    EscalatedIntensityExceeded,
}

/// Receipt emitted by a supervised `RoleProcess` after executing a governed `core_lm` forward step.
#[derive(Debug, Clone)]
pub struct GovernedForwardReceipt {
    pub role: RoleId,
    pub worker_pid: u64,
    pub token_id: u32,
    pub pos: usize,
    pub atom_id: AtomId,
    pub forward_permit: CompliancePermit,
    pub bridge_loss: BridgeLossReport,
    pub output_state: Vec<f32>,
}

/// Receipt emitted when hydrating a specialist worker's local `GeometricMemoryMap` from its
/// carved `SiloView` across model epochs using [`ModelAdapterLens`].
#[derive(Debug, Clone)]
pub struct CrossEpochHydrationReceipt {
    pub role: RoleId,
    pub worker_pid: u64,
    pub target_head_dim: usize,
    pub adapted_records: Vec<AdaptedKvRecord>,
    pub adapter_permits: Vec<CompliancePermit>,
}

/// Live `core_lm` inference worker owned and supervised by a `RoleProcess` (`core_lm -> orchestrator`).
pub struct CoreLmInferenceWorker {
    pub head_dim: usize,
    pub memory_map: GeometricMemoryMap,
    pub forward_steps: u64,
    pub last_permit_sequence: u64,
    pub last_bridge_loss: Option<BridgeLossReport>,
    /// Central map generation from which this local atom cache was hydrated.
    pub map_generation: u64,
}

impl Clone for CoreLmInferenceWorker {
    fn clone(&self) -> Self {
        let mut fresh = GeometricMemoryMap::new(self.memory_map.max_atoms);
        fresh.use_picard = self.memory_map.use_picard;
        for (&k, v) in &self.memory_map.atoms {
            fresh.atoms.insert(k, v.clone());
        }
        fresh.next_id = self.memory_map.next_id;
        fresh.clock = self.memory_map.clock;
        fresh.audit_log = self.memory_map.audit_log.clone();
        Self {
            head_dim: self.head_dim,
            memory_map: fresh,
            forward_steps: self.forward_steps,
            last_permit_sequence: self.last_permit_sequence,
            last_bridge_loss: self.last_bridge_loss,
            map_generation: self.map_generation,
        }
    }
}

impl std::fmt::Debug for CoreLmInferenceWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreLmInferenceWorker")
            .field("head_dim", &self.head_dim)
            .field("atoms", &self.memory_map.atoms.len())
            .field("forward_steps", &self.forward_steps)
            .field("last_permit_sequence", &self.last_permit_sequence)
            .finish()
    }
}

impl CoreLmInferenceWorker {
    pub fn new(head_dim: usize, max_atoms: usize) -> Self {
        let mut memory_map = GeometricMemoryMap::new(max_atoms);
        memory_map.use_picard = true;
        Self {
            head_dim,
            memory_map,
            forward_steps: 0,
            last_permit_sequence: 0,
            last_bridge_loss: None,
            map_generation: 0,
        }
    }
}

/// A supervised role instance (`Agent` in `SPEC.md` §4).
///
/// Holds its `RoleId`, process metadata, live `3×3` Wigner gyroscope frame, supervised
/// `CoreLmInferenceWorker`, and inbox of Governor-delivered handoffs.
#[derive(Debug, Clone)]
pub struct RoleProcess {
    pub pid: u64,
    pub role: RoleId,
    pub state: ProcessState,
    pub spawn_permit: CompliancePermit,
    pub last_heartbeat_tick: u64,
    pub restart_count: u32,
    pub last_recovery_duration: Option<Duration>,
    pub wigner_frame: [[Q32_31; 3]; 3],
    pub inbox: Vec<DeliveredHandoff>,
    pub worker: CoreLmInferenceWorker,
}

impl RoleProcess {
    /// Query this agent's carved `SiloView` for the `top_k` nearest atoms to `query_global`.
    pub fn query_my_silo<'a>(
        &self,
        map: &'a CarvedSwarmMap,
        query_global: &LorentzVector,
        top_k: usize,
    ) -> Result<Vec<&'a SiloAtom>, &'static str> {
        if self.state != ProcessState::Running {
            return Err("RoleProcess is not in Running state");
        }
        let silo = map
            .silo(self.role)
            .ok_or("Coordinator uses global map; query_my_silo is for carved specialists")?;
        Ok(silo.retrieve_in_silo(query_global, top_k))
    }

    /// Check whether a specific `AtomId` exists in this agent's carved `SiloView`.
    pub fn can_see_atom(&self, map: &CarvedSwarmMap, id: AtomId) -> bool {
        if self.role == RoleId::Coordinator {
            return map.global_atom(id).is_some();
        }
        map.silo(self.role)
            .map(|s| s.contains_atom(id))
            .unwrap_or(false)
    }
}

/// Layer 2 — The Governor (`SPEC.md` §1 Layer 2).
pub struct Governor {
    pub strategy: RestartStrategy,
    pub intensity: RestartIntensity,
    pub heartbeat_timeout_ticks: u64,
    pub current_tick: u64,
    next_pid: u64,
    processes: [Option<RoleProcess>; 13],
    restart_history_ticks: Vec<u64>,
    pub router: GovernorRouter,
}

impl Governor {
    /// Refresh an inference worker before it consumes atoms from the central map. A generation
    /// mismatch discards its derived cache and rehydrates it through the existing Layer 3 gate.
    pub fn synchronize_worker_memory(
        &mut self,
        role: RoleId,
        map_generation: u64,
        map: &CarvedSwarmMap,
        immune: &mut ImmuneSystem,
    ) -> Result<(), ComplianceViolation> {
        let stale = self
            .process(role)
            .map(|p| p.worker.map_generation != map_generation)
            .unwrap_or(true);
        if !stale {
            return Ok(());
        }
        if let Some(proc) = self.processes[role.level() as usize].as_mut() {
            proc.worker.memory_map = GeometricMemoryMap::new(proc.worker.memory_map.max_atoms);
            proc.worker.memory_map.use_picard = true;
        }
        self.hydrate_worker_from_silo_with_adapter(role, 128, map, immune)?;
        if let Some(proc) = self.processes[role.level() as usize].as_mut() {
            proc.worker.map_generation = map_generation;
        }
        Ok(())
    }
    pub fn new(
        strategy: RestartStrategy,
        intensity: RestartIntensity,
        heartbeat_timeout_ticks: u64,
    ) -> Self {
        Self {
            strategy,
            intensity,
            heartbeat_timeout_ticks,
            current_tick: 0,
            next_pid: 100,
            processes: std::array::from_fn(|_| None),
            restart_history_ticks: Vec::new(),
            router: GovernorRouter::new(256),
        }
    }

    /// Spawn all 13 role processes (`L0 Coordinator` + `L1..L12 Specialists`), obtaining a
    /// Layer 3 `CompliancePermit` for every single spawn (`SPEC.md` §1 & §4).
    pub fn spawn_swarm(
        &mut self,
        map: &CarvedSwarmMap,
        immune: &mut ImmuneSystem,
    ) -> Result<(), ComplianceViolation> {
        for role in RoleId::ALL_13 {
            let permit = immune.authorize_role_spawn(role, map)?;
            let pid = self.next_pid;
            self.next_pid += 1;
            self.processes[role.level() as usize] = Some(RoleProcess {
                pid,
                role,
                state: ProcessState::Running,
                spawn_permit: permit,
                last_heartbeat_tick: self.current_tick,
                restart_count: 0,
                last_recovery_duration: None,
                wigner_frame: IDENTITY_FRAME_Q32,
                inbox: Vec::new(),
                worker: CoreLmInferenceWorker::new(128, 4096),
            });
        }
        Ok(())
    }

    pub fn process(&self, role: RoleId) -> Option<&RoleProcess> {
        self.processes[role.level() as usize].as_ref()
    }

    pub fn record_heartbeat(&mut self, role: RoleId) -> Result<(), &'static str> {
        let proc = self.processes[role.level() as usize]
            .as_mut()
            .ok_or("Role process not spawned")?;
        if proc.state != ProcessState::Running {
            return Err("Cannot heartbeat non-running process");
        }
        proc.last_heartbeat_tick = self.current_tick;
        Ok(())
    }

    /// Simulate a role crash for fault-tolerance qualification (`QUALIFICATION_GATES.md` §3).
    pub fn simulate_crash(&mut self, role: RoleId) {
        if let Some(proc) = self.processes[role.level() as usize].as_mut() {
            proc.state = ProcessState::Crashed;
        }
    }

    /// Dissolve a specialist agent upon task completion, surrendering its local `GeometricMemoryMap`
    /// and accumulated `3×3` Wigner gyroscope frame back to the `Governor` as an [`EphemeralMapDeposit`],
    /// and compiling that deposit into the centralized model-agnostic `CarvedSwarmMap` under Layer 3
    /// (`ImmuneSystem::authorize_deposit_compilation`).
    pub fn dissolve_and_compile_deposit(
        &mut self,
        role: RoleId,
        map: &mut CarvedSwarmMap,
        immune: &mut ImmuneSystem,
    ) -> Result<
        (
            EphemeralMapDeposit,
            DepositCompilationReceipt,
            CompliancePermit,
        ),
        ComplianceViolation,
    > {
        let proc = self.processes[role.level() as usize]
            .as_mut()
            .ok_or(ComplianceViolation::InvalidSpecialistRole(role))?;

        // Sort local atoms deterministically by AtomId so compilation is bit-reproducible
        let mut sorted_keys: Vec<u64> = proc.worker.memory_map.atoms.keys().copied().collect();
        sorted_keys.sort_unstable();

        let mut records = Vec::with_capacity(sorted_keys.len());
        for k in sorted_keys {
            if let Some(atom) = proc.worker.memory_map.atoms.remove(&k) {
                records.push(DepositedAtomRecord {
                    local_id: atom.id,
                    lorentz: atom.lorentz,
                    phase: atom.phase,
                    payload: atom.payload,
                });
            }
        }

        let deposit = EphemeralMapDeposit {
            from_role: role,
            dissolved_pid: proc.pid,
            origin_model_fingerprint: map.base_weights.fingerprint,
            forward_steps_completed: proc.worker.forward_steps,
            final_wigner_frame: proc.wigner_frame,
            records,
        };

        // Transition agent to Dissolved state and reset its local ephemeral worker state
        proc.state = ProcessState::Dissolved;
        proc.worker.forward_steps = 0;
        proc.wigner_frame = IDENTITY_FRAME_Q32;

        // Layer 3 Immune Gate: Verify deposit integrity before compiling into centralized vault
        let permit = immune.authorize_deposit_compilation(&deposit, map)?;
        let receipt = map.compile_deposit(&deposit, &immune.policy);

        Ok((deposit, receipt, permit))
    }

    /// Advance supervisor clock by 1 tick, obtain Layer 3 supervision authorization
    /// (`orchestrator -> immune_system`), detect crashed/timed-out/dissolved role processes,
    /// and execute automatic supervision recovery gated by Layer 3 (`ImmuneSystem`).
    pub fn step_supervision(
        &mut self,
        map: &CarvedSwarmMap,
        immune: &mut ImmuneSystem,
    ) -> Result<Vec<RoleId>, ComplianceViolation> {
        self.current_tick += 1;
        let cutoff = self
            .current_tick
            .saturating_sub(self.intensity.within_ticks);
        self.restart_history_ticks.retain(|&t| t >= cutoff);

        let _sup_permit = immune.authorize_governor_supervision(
            self.current_tick,
            self.restart_history_ticks.len() as u32,
            self.intensity.max_restarts,
            map,
        )?;

        // Failures (Crashed/TimedOut) are strategy-governed; Dissolved is the
        // normal ephemeral lifecycle and always respawns individually.
        let mut failed: Vec<RoleId> = Vec::new();
        let mut dissolved: Vec<RoleId> = Vec::new();

        for role in RoleId::ALL_13 {
            if let Some(proc) = self.processes[role.level() as usize].as_mut() {
                if proc.state == ProcessState::Running
                    && self.current_tick.saturating_sub(proc.last_heartbeat_tick)
                        > self.heartbeat_timeout_ticks
                {
                    proc.state = ProcessState::TimedOut;
                }
                match proc.state {
                    ProcessState::Crashed | ProcessState::TimedOut => failed.push(role),
                    ProcessState::Dissolved => dissolved.push(role),
                    _ => {}
                }
            }
        }

        if failed.is_empty() && dissolved.is_empty() {
            return Ok(Vec::new());
        }

        // CS-NATIVE-04 D2 (section-13 fix): explicit restart-group membership.
        // The previous OneForAll arm restarted CARVED_12 unconditionally, so a
        // crashed Coordinator was never restarted while 12 healthy specialists
        // were. Now: OneForAll expands each failure to its whole group; the
        // Coordination group is {Coordinator} alone, supervised by this root
        // loop independently of the specialist group.
        let mut restart_set: Vec<RoleId> = dissolved;
        match self.strategy {
            RestartStrategy::OneForOne => restart_set.extend(failed.iter().copied()),
            RestartStrategy::OneForAll => {
                for role in &failed {
                    match role {
                        RoleId::Coordinator => restart_set.push(RoleId::Coordinator),
                        _ => restart_set.extend(RoleId::CARVED_12.iter().copied()),
                    }
                }
            }
        }
        let roles_to_restart: Vec<RoleId> = RoleId::ALL_13
            .into_iter()
            .filter(|r| restart_set.contains(r))
            .collect();

        let mut recovered = Vec::new();
        for role in roles_to_restart {
            let t0 = Instant::now();
            let was_dissolved = self.processes[role.level() as usize]
                .as_ref()
                .map(|p| p.state == ProcessState::Dissolved)
                .unwrap_or(false);

            if !was_dissolved {
                let cutoff = self
                    .current_tick
                    .saturating_sub(self.intensity.within_ticks);
                self.restart_history_ticks.retain(|&t| t >= cutoff);

                if self.restart_history_ticks.len() as u32 >= self.intensity.max_restarts {
                    if let Some(proc) = self.processes[role.level() as usize].as_mut() {
                        proc.state = ProcessState::EscalatedIntensityExceeded;
                    }
                    continue;
                }
            }

            let permit = immune.authorize_role_spawn(role, map)?;
            let new_pid = self.next_pid;
            self.next_pid += 1;
            if !was_dissolved {
                self.restart_history_ticks.push(self.current_tick);
            }

            if let Some(proc) = self.processes[role.level() as usize].as_mut() {
                proc.pid = new_pid;
                proc.state = ProcessState::Running;
                proc.spawn_permit = permit;
                proc.last_heartbeat_tick = self.current_tick;
                if !was_dissolved {
                    proc.restart_count += 1;
                }
                proc.wigner_frame = IDENTITY_FRAME_Q32;
                proc.last_recovery_duration = Some(t0.elapsed());
                recovered.push(role);
            }
        }

        Ok(recovered)
    }

    /// Execute a governed `core_lm` forward step on `role`'s supervised `CoreLmInferenceWorker`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_governed_forward(
        &mut self,
        role: RoleId,
        token_id: u32,
        pos: usize,
        q_head: &[f32],
        k_head: &[f32],
        v_head: &[f32],
        o_proj_matrix: &[f32],
        map: &CarvedSwarmMap,
        immune: &mut ImmuneSystem,
    ) -> Result<GovernedForwardReceipt, ComplianceViolation> {
        let needs_recovery = self.processes[role.level() as usize]
            .as_ref()
            .map(|p| p.state != ProcessState::Running)
            .unwrap_or(true);
        if needs_recovery {
            self.step_supervision(map, immune)?;
        }

        let (k_lv, k_phase, bridge_loss) = f32_to_atom_address(k_head).map_err(|_| {
            ComplianceViolation::SilentInformationLoss {
                stage: "core_lm::geometry::bridge::f32_to_atom_address(k_head)",
            }
        })?;
        let (q_lv, q_phase, _) = f32_to_atom_address(q_head).map_err(|_| {
            ComplianceViolation::SilentInformationLoss {
                stage: "core_lm::geometry::bridge::f32_to_atom_address(q_head)",
            }
        })?;

        let forward_permit =
            immune.authorize_forward_pass(role, token_id, pos, map, &bridge_loss)?;

        let proc = self.processes[role.level() as usize]
            .as_mut()
            .ok_or(ComplianceViolation::InvalidSpecialistRole(role))?;
        if proc.state != ProcessState::Running {
            return Err(ComplianceViolation::InvalidSpecialistRole(role));
        }

        let hd = k_head.len();
        let mut kv_payload = Vec::with_capacity(2 * hd * 4);
        for &f in k_head {
            kv_payload.extend_from_slice(&f.to_le_bytes());
        }
        for &f in v_head {
            kv_payload.extend_from_slice(&f.to_le_bytes());
        }

        let atom_id = proc
            .worker
            .memory_map
            .write(k_lv, k_phase, kv_payload)
            .map_err(|_| ComplianceViolation::SilentInformationLoss {
                stage: "GeometricMemoryMap::write",
            })?;

        let top_k = 64.min(proc.worker.memory_map.atoms.len()).max(1);
        let neighbors = proc
            .worker
            .memory_map
            .retrieve_5d(&q_lv, &q_phase, top_k, false)
            .map_err(|_| ComplianceViolation::SilentInformationLoss {
                stage: "GeometricMemoryMap::retrieve_5d",
            })?;

        let inv_sqrt_hd = 1.0 / (hd as f32).sqrt();
        let mut scores = Vec::with_capacity(neighbors.len());
        let mut v_list = Vec::with_capacity(neighbors.len());
        for atom in neighbors {
            if atom.payload.len() >= 2 * hd * 4 {
                let kh: Vec<f32> = atom.payload[0..hd * 4]
                    .chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let vh: Vec<f32> = atom.payload[hd * 4..2 * hd * 4]
                    .chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let dot: f32 = q_head.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot * inv_sqrt_hd);
                v_list.push(vh);
            }
        }

        let probs = softmax(&scores)
            .unwrap_or_else(|_| vec![1.0 / scores.len().max(1) as f32; scores.len()]);
        let mut attn_vec = vec![0.0f32; hd];
        for (p, vh) in probs.iter().zip(v_list.iter()) {
            for d in 0..hd.min(vh.len()) {
                attn_vec[d] += p * vh[d];
            }
        }

        let out_rows = if hd > 0 { o_proj_matrix.len() / hd } else { 0 };
        let output_state = if out_rows > 0 && o_proj_matrix.len() == out_rows * hd {
            matvec(o_proj_matrix, out_rows, hd, &attn_vec).unwrap_or(attn_vec)
        } else {
            attn_vec
        };

        proc.worker.forward_steps += 1;
        proc.worker.last_permit_sequence = forward_permit.sequence();
        proc.worker.last_bridge_loss = Some(bridge_loss);
        proc.last_heartbeat_tick = self.current_tick;

        Ok(GovernedForwardReceipt {
            role,
            worker_pid: proc.pid,
            token_id,
            pos,
            atom_id,
            forward_permit,
            bridge_loss,
            output_state,
        })
    }

    /// Dispatch an inter-role handoff (`from_role -> to_role`) through Layer 3 (`ImmuneSystem`),
    /// parallel-transporting the sender's `3×3` Wigner gyroscope frame across the hop and updating
    /// the receiver's `wigner_frame`.
    pub fn dispatch_handoff(
        &mut self,
        mut req: HandoffRequest,
        map: &CarvedSwarmMap,
        immune: &mut ImmuneSystem,
    ) -> Result<DeliveredHandoff, ComplianceViolation> {
        if let Some(sender_proc) = self.processes[req.from_role.level() as usize].as_ref() {
            req.incoming_wigner_frame = sender_proc.wigner_frame;
        }
        let receipt = self.router.route_handoff(req, map, immune)?;
        if let Some(target_proc) = self.processes[receipt.to_role.level() as usize].as_mut() {
            target_proc.wigner_frame = receipt.transported_wigner_frame;
            target_proc.inbox.push(receipt.clone());
        }
        Ok(receipt)
    }

    /// Hydrate `role`'s supervised `CoreLmInferenceWorker` at `target_head_dim` from its carved
    /// `SiloView` in `map`, automatically projecting any cross-epoch KV payloads deposited by
    /// older model epochs (`origin_model_fingerprint`) into `target_head_dim` via [`ModelAdapterLens`]
    /// and verifying each projection under Layer 3 (`authorize_cross_epoch_adaptation`).
    pub fn hydrate_worker_from_silo_with_adapter(
        &mut self,
        role: RoleId,
        target_head_dim: usize,
        map: &CarvedSwarmMap,
        immune: &mut ImmuneSystem,
    ) -> Result<CrossEpochHydrationReceipt, ComplianceViolation> {
        let needs_recovery = self.processes[role.level() as usize]
            .as_ref()
            .map(|p| p.state != ProcessState::Running)
            .unwrap_or(true);
        if needs_recovery {
            self.step_supervision(map, immune)?;
        }

        let silo = map
            .silo(role)
            .ok_or(ComplianceViolation::InvalidSpecialistRole(role))?;
        let target_fp = map.base_weights.fingerprint;

        let mut adapted_records = Vec::new();
        let mut adapter_permits = Vec::new();
        let mut local_entries = Vec::new();

        for silo_atom in silo.atoms() {
            // Only adapt KV payloads that carry valid even-head-dim f32 K+V pairs (>= 16 bytes, multiple of 8)
            if silo_atom.payload.len() >= 16 && silo_atom.payload.len() % 8 == 0 {
                let adapted = ModelAdapterLens::adapt_kv_payload(
                    silo_atom.id,
                    &silo_atom.payload,
                    silo_atom.origin_model_fingerprint,
                    target_fp,
                    silo_atom.phase,
                    target_head_dim,
                )
                .map_err(|_| ComplianceViolation::SilentInformationLoss {
                    stage: "ModelAdapterLens::adapt_kv_payload",
                })?;

                let permit = immune.authorize_cross_epoch_adaptation(
                    role,
                    silo_atom.id,
                    &adapted.loss_report,
                    map,
                )?;

                let packed = ModelAdapterLens::pack_kv_payload(&adapted.k_head, &adapted.v_head);
                local_entries.push((silo_atom.local_lorentz, silo_atom.phase, packed));
                adapted_records.push(adapted);
                adapter_permits.push(permit);
            }
        }

        let proc = self.processes[role.level() as usize]
            .as_mut()
            .ok_or(ComplianceViolation::InvalidSpecialistRole(role))?;
        proc.worker.head_dim = target_head_dim;
        for (lv, phase, packed) in local_entries {
            proc.worker
                .memory_map
                .write(lv, phase, packed)
                .map_err(|_| ComplianceViolation::SilentInformationLoss {
                    stage: "GeometricMemoryMap::write (CrossEpochHydration)",
                })?;
        }

        Ok(CrossEpochHydrationReceipt {
            role,
            worker_pid: proc.pid,
            target_head_dim,
            adapted_records,
            adapter_permits,
        })
    }
}
