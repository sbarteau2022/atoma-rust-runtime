# Qualification Gates: Performance & Safety

No component may be promoted to the Global Harness until it passes the following gates.

## 1. Numerical Parity (Correctness)
- **Test**: Input the same prompt to the current oMLX baseline and the new implementation.
- **Requirement**: $100\%$ token parity for the first $128$ tokens. Any divergence is a failure.

## 2. Throughput Gates (Performance)
- **TTFT (Time to First Token)**: Must be $\le$ the current baseline for the same batch size.
- **Decode Speed**: Must maintain $\ge 150 \text{ tokens/sec}$ (AGG) on the target hardware.
- **Acceptance Rate**: For speculative modules, a minimum acceptance rate of $30\%$ is required to avoid the "Speculation Tax."

## 3. Adversarial Audit (Safety)
- **Boundary Testing**: Stress test with extreme sequence lengths ($100\text{k}+$).
- **Memory Leak Check**: Run $1,000$ iterations of the loop and verify VRAM usage remains constant (no leaks).
- **Fault Tolerance**: For Erlang actors, simulate a process crash and verify the Supervisor recovers the state within $50\text{ms}$.

## 4. Integration Approval
- **Design Alignment**: The implementation must map exactly to the Tier defined in `DESIGN.md`.
- **Documentation**: README and Skill files must be complete per `DOCUMENTATION_STD.md`.
