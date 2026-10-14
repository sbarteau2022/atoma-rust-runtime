//! Tier 4–5 (DESIGN.md): Inter-Role Handoff Routing via Governor & Layer 3 Immune Gate,
//! with Live Levi-Civita Parallel Transport of the `3×3` Wigner Gyroscope Frame on Every Hop.

use crate::bridge::{AtomicRingBuffer, FlatHandoffPacket};
use atoma_immune::cache::geometric::AtomId;
use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::geometry::lorentz::LorentzVector;
use atoma_immune::{
    CarvedSwarmMap, CompliancePermit, ComplianceViolation, HyperbolicHolonomyEngine,
    IDENTITY_FRAME_Q32, ImmuneSystem, RoleId,
};

/// Request from a carved role instance to hand off work to another role via a shared boundary atom.
#[derive(Debug, Clone)]
pub struct HandoffRequest {
    pub from_role: RoleId,
    pub to_role: RoleId,
    /// The `AtomId` in the boundary overlap `V_{from} ∩ V_{to}` anchoring the handoff.
    pub shared_atom_id: AtomId,
    /// Incoming `3×3` Wigner gyroscope frame (defaults to `IDENTITY_FRAME_Q32` on the first hop).
    pub incoming_wigner_frame: [[Q32_31; 3]; 3],
    /// Additional task context bytes carried with the handoff.
    pub message_bytes: Vec<u8>,
}

/// Verified delivery receipt produced only when Layer 3 `ImmuneSystem` authorizes the handoff
/// and the packet traverses the `AtomicRingBuffer` into the target role's inbox.
#[derive(Debug, Clone)]
pub struct DeliveredHandoff {
    pub envelope_id: u64,
    pub from_role: RoleId,
    pub to_role: RoleId,
    pub shared_atom_id: AtomId,
    /// The atom's coordinate in the **receiver's local moving frame** (`SiloView::local_lorentz`).
    pub receiver_local_lorentz: LorentzVector,
    /// Parallel-transported `3×3` Wigner gyroscope frame after crossing `from_role -> to_role`.
    pub transported_wigner_frame: [[Q32_31; 3]; 3],
    pub permit: CompliancePermit,
    pub wire_packet: FlatHandoffPacket,
}

/// Central message router owned by the Layer 2 `Governor`.
pub struct GovernorRouter {
    next_envelope_id: u64,
    pub ring_buffer: AtomicRingBuffer,
    pub delivered_log: Vec<DeliveredHandoff>,
}

impl GovernorRouter {
    pub fn new(ring_capacity: usize) -> Self {
        Self {
            next_envelope_id: 1,
            ring_buffer: AtomicRingBuffer::with_capacity(ring_capacity),
            delivered_log: Vec::new(),
        }
    }

    /// Route an inter-role handoff request through Layer 3 (`ImmuneSystem::authorize_handoff`),
    /// parallel-transport its `3×3` Levi-Civita Wigner gyroscope frame across `from_role -> to_role`,
    /// and transmit across the zero-copy `AtomicRingBuffer`.
    pub fn route_handoff(
        &mut self,
        req: HandoffRequest,
        map: &CarvedSwarmMap,
        immune: &mut ImmuneSystem,
    ) -> Result<DeliveredHandoff, ComplianceViolation> {
        let permit =
            immune.authorize_handoff(req.from_role, req.to_role, req.shared_atom_id, map)?;

        let global_atom = map.global_atom(req.shared_atom_id).ok_or(
            ComplianceViolation::AtomInvisibleToSender {
                sender: req.from_role,
                atom_id: req.shared_atom_id,
            },
        )?;

        let receiver_local_lorentz = if req.to_role == RoleId::Coordinator {
            global_atom.lorentz
        } else {
            let receiver_silo = map
                .silo(req.to_role)
                .ok_or(ComplianceViolation::InvalidSpecialistRole(req.to_role))?;
            receiver_silo
                .get_atom(req.shared_atom_id)
                .ok_or(ComplianceViolation::AtomOutsideBoundaryOverlap {
                    from: req.from_role,
                    to: req.to_role,
                    atom_id: req.shared_atom_id,
                })?
                .local_lorentz
        };

        // Parallel-transport the 3x3 Wigner gyroscope frame along well(from_role) -> well(to_role)
        let from_well = map.atlas.well(req.from_role).lorentz_well;
        let to_well = map.atlas.well(req.to_role).lorentz_well;
        let transported_wigner_frame = HyperbolicHolonomyEngine::transport_frame_hop(
            &from_well,
            &to_well,
            &req.incoming_wigner_frame,
        );

        let envelope_id = self.next_envelope_id;
        self.next_envelope_id += 1;

        let packet = FlatHandoffPacket::new_authorized_with_frame(
            envelope_id,
            req.from_role,
            req.to_role,
            req.shared_atom_id,
            global_atom.lorentz,
            global_atom.phase,
            transported_wigner_frame,
            &permit,
            req.message_bytes,
        );

        self.ring_buffer
            .push_packet(&packet)
            .expect("ring buffer capacity sufficient for synchronous dispatch");
        let popped = self
            .ring_buffer
            .pop_packet()
            .expect("valid FlatHandoffPacket wire format")
            .expect("packet was just pushed");

        let receipt = DeliveredHandoff {
            envelope_id,
            from_role: req.from_role,
            to_role: req.to_role,
            shared_atom_id: req.shared_atom_id,
            receiver_local_lorentz,
            transported_wigner_frame,
            permit,
            wire_packet: popped,
        };
        self.delivered_log.push(receipt.clone());
        Ok(receipt)
    }

    pub fn default_identity_frame() -> [[Q32_31; 3]; 3] {
        IDENTITY_FRAME_Q32
    }
}
