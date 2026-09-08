# Gene Runtime

The runtime is the execution layer connecting applications to the
Gene architecture.

## Responsibilities

The runtime manages:

- model registration
- model selection
- inference
- embeddings
- health state
- backend lifecycle
- request execution
- integration with agents and capabilities

## Backend Abstraction

Gene separates the runtime interface from a specific neural backend.

This allows:

- testing without a trained model
- fallback implementations
- development backends
- production neural backends
- future model versions

A null backend is an architectural/testing mechanism and is not the
production intelligence implementation.

## Runtime Flow

```text
Application
 ↓
Gene Runtime
 ↓
Agent
 ↓
Context
 ↓
Model Backend
 ↓
Generation
 ↓
Response

