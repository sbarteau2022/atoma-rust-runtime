# Documentation Standards

Every component in the Monster Hybrid Server must follow these guidelines to ensure the agent can maintain and evolve the system.

## 1. README.md Requirements
Every directory containing logic must have a `README.md` with the following sections:
- **Purpose**: One sentence describing what this module does.
- **Inputs/Outputs**: Explicit definition of the tensors or packets it consumes and produces.
- **Dependencies**: List of internal modules or external libraries.
- **Qualification**: Link to the test suite that proves this module works.

## 2. Skill Documentation
All tools provided to the agent must be documented in `.md` files within the `/harness/skills/` directory.
- **Plain Language**: Describe the tool's function without jargon.
- **Workflow**: Provide a step-by-step guide on how to use the tool in an e2e sequence.
- **Edge Cases**: Document known failure modes and how to handle them.

## 3. Code Commenting
- Use **Implementation-First comments**. Explain *why* a specific Metal kernel or Erlang supervisor was chosen, not just *what* it does.
- Reference the `DESIGN.md` Tier in the file header to indicate the implementation stage.
