# Gene Testing

Gene testing should operate at multiple levels.

## Levels

1. Unit tests
2. Component tests
3. Integration tests
4. Model smoke tests
5. Training tests
6. Checkpoint tests
7. Inference tests
8. Agent end-to-end tests

## Critical Path

```text
Input
 ↓
Agent
 ↓
Context
 ↓
Tokenizer
 ↓
Model
 ↓
Inference
 ↓
Generation
 ↓
Response

Every stage should be independently testable.

Model Scales

200M, 500M, and 700M configurations should not be considered verified
until their architecture and runtime behavior have been tested.
