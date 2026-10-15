// Autonomous Monster-Hybrid Synthesized Client: `atoma_binary_wire_client.ts`
// Zero-JSON `ATOMA_WR` Binary Wire Protocol Codec using native ArrayBuffer + DataView.

export interface NativeWireRequestFrame {
  readonly activeInference: boolean;
  readonly crystallizeOnSuccess: boolean;
  readonly maxNew: number;
  readonly baseVocab: number;
  readonly prefix: Uint32Array;
  readonly eos: Uint32Array;
}

export interface NativeWireResponseFrame {
  readonly totalFreeEnergyQ32: bigint;
  readonly minOmegaReadinessQ32: bigint;
  readonly superpositionInterventions: number;
  readonly crystallizedArcs: number;
  readonly emittedOps: Uint32Array;
  readonly expandedTokens: Uint32Array;
}

const ATOMA_WIRE_MAGIC = new Uint8Array([0x41, 0x54, 0x4f, 0x4d, 0x41, 0x5f, 0x57, 0x52]); // "ATOMA_WR"

export function encodeAtomaWireRequest(req: NativeWireRequestFrame): Uint8Array {
  const totalBytes = 26 + (req.prefix.length + req.eos.length) * 4;
  const buf = new ArrayBuffer(totalBytes);
  const u8 = new Uint8Array(buf);
  const view = new DataView(buf);
  u8.set(ATOMA_WIRE_MAGIC, 0);
  view.setUint8(8, 1);
  const flags = (req.activeInference ? 1 : 0) | (req.crystallizeOnSuccess ? 2 : 0);
  view.setUint8(9, flags);
  view.setUint32(10, req.maxNew, true);
  view.setUint32(14, req.baseVocab, true);
  view.setUint32(18, req.prefix.length, true);
  view.setUint32(22, req.eos.length, true);
  let offset = 26;
  for (let i = 0; i < req.prefix.length; i++, offset += 4) {
    view.setUint32(offset, req.prefix[i], true);
  }
  for (let i = 0; i < req.eos.length; i++, offset += 4) {
    view.setUint32(offset, req.eos[i], true);
  }
  return u8;
}
