// Autonomous Monster-Hybrid Synthesized Dispatcher: `icosahedral_swarm_supervisor.go`
package swarm

import (
	"encoding/binary"
	"errors"
)

const AtomaWireMagic = "ATOMA_WR"

type HandoffPacket struct {
	FromRole uint8
	ToRole   uint8
	AtomID   uint64
	Payload  []byte
}

func EncodeHandoffFrame(pkt HandoffPacket) ([]byte, error) {
	if pkt.FromRole >= 12 || pkt.ToRole >= 12 {
		return nil, errors.New("invalid Atlas13 specialist role index")
	}
	buf := make([]byte, 18+len(pkt.Payload))
	copy(buf[0:8], AtomaWireMagic)
	buf[8] = pkt.FromRole
	buf[9] = pkt.ToRole
	binary.LittleEndian.PutUint64(buf[10:18], pkt.AtomID)
	copy(buf[18:], pkt.Payload)
	return buf, nil
}
