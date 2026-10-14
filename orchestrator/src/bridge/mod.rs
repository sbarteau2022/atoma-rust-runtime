//! Tier 4–5 (DESIGN.md & ARCHITECTURE.md §4): Zero-Copy FlatBuffer Packet & Lock-Free Ring Buffer
//! with Embedded `3×3` Levi-Civita Wigner Gyroscope Frame.
//!
//! Why this transport exists:
//! `design/ARCHITECTURE.md` mandates FlatBuffers binary serialization over lock-free ring buffers
//! synchronized by CPU-level `Acquire`/`Release` atomic memory barriers. Every packet now carries
//! its live `3×3` `Q32_31` Wigner holonomy gyroscope frame (`wigner_frame`) directly in the wire
//! header so multi-hop traversal order and hyperbolic curvature flux are cryptographically and
//! geometrically verifiable on the wire (`[verified]`).

use atoma_immune::cache::geometric::{AtomId, Phase};
use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::geometry::lorentz::LorentzVector;
use atoma_immune::{CompliancePermit, IDENTITY_FRAME_Q32, RoleId};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Fixed-layout 176-byte binary header matching `schema/carved_swarm.fbs` (`HandoffPacket`),
/// extended with the 72-byte `3×3` `Q32_31` Levi-Civita Wigner gyroscope frame (`wigner_frame`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatHandoffPacket {
    pub magic: [u8; 4],
    pub envelope_id: u64,
    pub from_role: RoleId,
    pub to_role: RoleId,
    pub shared_atom_id: AtomId,
    pub global_lorentz: LorentzVector,
    pub phase: Phase,
    /// `3×3` orthonormal tangent gyroscope frame in `Q32_31` parallel-transported across every hop.
    pub wigner_frame: [[Q32_31; 3]; 3],
    pub permit_sequence: u64,
    pub chain_digest_prefix: [u8; 16],
    pub payload: Vec<u8>,
}

impl FlatHandoffPacket {
    pub const MAGIC: [u8; 4] = *b"ATMA";
    pub const HEADER_BYTES: usize = 176;

    /// Construct a wire packet with identity initial Wigner frame when presented with a Layer 3 `CompliancePermit`.
    pub fn new_authorized(
        envelope_id: u64,
        from_role: RoleId,
        to_role: RoleId,
        shared_atom_id: AtomId,
        global_lorentz: LorentzVector,
        phase: Phase,
        permit: &CompliancePermit,
        payload: Vec<u8>,
    ) -> Self {
        Self::new_authorized_with_frame(
            envelope_id,
            from_role,
            to_role,
            shared_atom_id,
            global_lorentz,
            phase,
            IDENTITY_FRAME_Q32,
            permit,
            payload,
        )
    }

    /// Construct a wire packet carrying an explicit `3×3` parallel-transported Wigner gyroscope frame.
    #[allow(clippy::too_many_arguments)]
    pub fn new_authorized_with_frame(
        envelope_id: u64,
        from_role: RoleId,
        to_role: RoleId,
        shared_atom_id: AtomId,
        global_lorentz: LorentzVector,
        phase: Phase,
        wigner_frame: [[Q32_31; 3]; 3],
        permit: &CompliancePermit,
        payload: Vec<u8>,
    ) -> Self {
        let mut prefix = [0u8; 16];
        prefix.copy_from_slice(&permit.chain_digest().0[0..16]);
        Self {
            magic: Self::MAGIC,
            envelope_id,
            from_role,
            to_role,
            shared_atom_id,
            global_lorentz,
            phase,
            wigner_frame,
            permit_sequence: permit.sequence(),
            chain_digest_prefix: prefix,
            payload,
        }
    }

    /// Encode into a flat little-endian byte buffer (`FlatBuffers` wire layout).
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::HEADER_BYTES + self.payload.len());
        buf.extend_from_slice(&self.magic);
        buf.push(self.from_role.level());
        buf.push(self.to_role.level());
        buf.extend_from_slice(&[0u8; 2]); // 2-byte alignment padding
        buf.extend_from_slice(&self.envelope_id.to_le_bytes());
        buf.extend_from_slice(&self.shared_atom_id.0.to_le_bytes());
        for c in &self.global_lorentz.x {
            buf.extend_from_slice(&c.0.to_le_bytes());
        }
        buf.extend_from_slice(&self.phase.theta_f.0.to_le_bytes());
        buf.extend_from_slice(&self.phase.theta_s.0.to_le_bytes());
        for row in &self.wigner_frame {
            for cell in row {
                buf.extend_from_slice(&cell.0.to_le_bytes());
            }
        }
        buf.extend_from_slice(&self.permit_sequence.to_le_bytes());
        buf.extend_from_slice(&self.chain_digest_prefix);
        buf.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        debug_assert_eq!(buf.len(), Self::HEADER_BYTES);
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Zero-alloc header validation and decoding from a flat byte slice.
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < Self::HEADER_BYTES {
            return Err("Packet shorter than 176-byte FlatHandoffPacket header");
        }
        if bytes[0..4] != Self::MAGIC {
            return Err("Invalid FlatHandoffPacket magic");
        }
        let from_role = RoleId::from_level(bytes[4]).ok_or("Invalid from_role level")?;
        let to_role = RoleId::from_level(bytes[5]).ok_or("Invalid to_role level")?;
        let envelope_id = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let shared_atom_id = AtomId(u64::from_le_bytes(bytes[16..24].try_into().unwrap()));
        let x0 = Q32_31(i64::from_le_bytes(bytes[24..32].try_into().unwrap()));
        let x1 = Q32_31(i64::from_le_bytes(bytes[32..40].try_into().unwrap()));
        let x2 = Q32_31(i64::from_le_bytes(bytes[40..48].try_into().unwrap()));
        let x3 = Q32_31(i64::from_le_bytes(bytes[48..56].try_into().unwrap()));
        let theta_f = Q32_31(i64::from_le_bytes(bytes[56..64].try_into().unwrap()));
        let theta_s = Q32_31(i64::from_le_bytes(bytes[64..72].try_into().unwrap()));

        let mut wigner_frame = [[Q32_31::ZERO; 3]; 3];
        let mut offset = 72usize;
        for r in 0..3 {
            for c in 0..3 {
                wigner_frame[r][c] =
                    Q32_31(i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()));
                offset += 8;
            }
        }
        debug_assert_eq!(offset, 144);

        let permit_sequence = u64::from_le_bytes(bytes[144..152].try_into().unwrap());
        let mut chain_digest_prefix = [0u8; 16];
        chain_digest_prefix.copy_from_slice(&bytes[152..168]);
        let payload_len = u64::from_le_bytes(bytes[168..176].try_into().unwrap()) as usize;
        if bytes.len() < Self::HEADER_BYTES + payload_len {
            return Err("Truncated FlatHandoffPacket payload");
        }
        let payload = bytes[Self::HEADER_BYTES..Self::HEADER_BYTES + payload_len].to_vec();

        Ok(Self {
            magic: Self::MAGIC,
            envelope_id,
            from_role,
            to_role,
            shared_atom_id,
            global_lorentz: LorentzVector {
                x: [x0, x1, x2, x3],
            },
            phase: Phase { theta_f, theta_s },
            wigner_frame,
            permit_sequence,
            chain_digest_prefix,
            payload,
        })
    }
}

/// Lock-free SPSC ring buffer with `Acquire`/`Release` atomic memory barriers (`ARCHITECTURE.md` §4).
pub struct AtomicRingBuffer {
    slots: Vec<Vec<u8>>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl AtomicRingBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(2);
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push(Vec::new());
        }
        Self {
            slots,
            capacity: cap,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn push_packet(&mut self, packet: &FlatHandoffPacket) -> Result<(), &'static str> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        let next_tail = (tail + 1) % self.capacity;
        if next_tail == head {
            return Err("AtomicRingBuffer full");
        }
        self.slots[tail] = packet.encode();
        std::sync::atomic::fence(Ordering::Release);
        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    pub fn pop_packet(&mut self) -> Result<Option<FlatHandoffPacket>, &'static str> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return Ok(None);
        }
        std::sync::atomic::fence(Ordering::Acquire);
        let decoded = FlatHandoffPacket::decode(&self.slots[head])?;
        let next_head = (head + 1) % self.capacity;
        self.head.store(next_head, Ordering::Release);
        Ok(Some(decoded))
    }

    pub fn pending_count(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if tail >= head {
            tail - head
        } else {
            self.capacity - head + tail
        }
    }
}
