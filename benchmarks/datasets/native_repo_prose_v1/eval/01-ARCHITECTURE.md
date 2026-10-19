# System Architecture: The Monster Hybrid Server

## 1. Data Flow Path
**Input Text** $\rightarrow$ **Tokenizer (Rust)** $\rightarrow$ **Embedding Lookup** $\rightarrow$ **FlatBuffer Packet** $\rightarrow$ **mmap Ring Buffer** $\rightarrow$ **coreLM Execution** $\rightarrow$ **mmap Ring Buffer** $\rightarrow$ **Erlang Orchestrator** $\rightarrow$ **De-tokenizer** $\rightarrow$ **Output Text**.

## 2. The Control Plane (Erlang/OTP)
- **Process Model**: Every agent is a lightweight BEAM process.
- **Supervisor Tree**: Root Supervisor $\rightarrow$ Swarm Supervisor $\rightarrow$ Individual Agents.
- **Shared-KV Mapper**: A registry that maps agent IDs to specific KV cache offsets in the `coreLM` memory space.

## 3. The Execution Plane (coreLM - Rust/Metal)
- **Prefill Stage**: 
  - Input tensors dispatched to the **ANE** in batches.
  - KV state generated and stored in the **Paged Cache**.
- **Decode Stage**: 
  - **Predictive Automaton**: Fast, non-weight-based token speculation.
  - **Base Model Verification**: Parallel verification of speculative tokens via fused Metal kernels.
- **Memory Management**: 
  - **Paged Attention**: Non-contiguous KV blocks.
  - **SSD Suspension**: Active paging of KV blocks between VRAM and SSD based on the attention map.

## 4. The Communication Bridge (Common Tongue)
- **Format**: **FlatBuffers**. Zero-copy binary serialization.
- **Transport**: **mmap Ring Buffers**. Lock-free SPSC/MPMC queues in shared memory.
- **Synchronization**: **Atomic Fencing**. CPU-level memory barriers to ensure data visibility between BEAM and Rust.
