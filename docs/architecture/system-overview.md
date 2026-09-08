# Gene AI System Overview

Gene is a modular AI system composed of a learned neural model and a
larger software architecture surrounding it.

The neural model provides learned language reasoning and generation.

The surrounding system provides:

- context
- memory
- knowledge
- agents
- planning
- tools
- capabilities
- skills
- genomes
- multimodal processing
- browser interaction
- application interaction
- orchestration
- evaluation
- evolution

The architecture therefore separates learned intelligence from
operational intelligence.

## Core Principle

```text
                 GENE
                   |
        +----------+----------+
        |                     |
 Neural Intelligence    System Intelligence
        |                     |
 Transformer             Agent
 Tokenizer               Memory
 Embeddings              Knowledge
 Generation              Skills
                        Tools
                        Planning
                        Capabilities
                        Orchestration
Request Lifecycle
Input
 ↓
Runtime
 ↓
Agent
 ↓
Context
 ↓
Memory / Knowledge / Skills
 ↓
Tokenizer
 ↓
Neural Model
 ↓
Generation
 ↓
Tool / Capability decision
 ↓
Execution
 ↓
Observation
 ↓
Context update
 ↓
Final response

The architecture is intentionally modular so that individual
components can evolve without requiring the entire system to be
rewritten.
