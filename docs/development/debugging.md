# Gene Debugging

When debugging Gene, identify the failing layer before changing code.

Recommended order:

1. Environment
2. Imports
3. Configuration
4. Tokenizer
5. Model construction
6. Checkpoint loading
7. Inference
8. Agent
9. Tool execution
10. Persistence
11. Output

## Principle

Fix the earliest failing layer first.

A downstream failure may simply be a consequence of an upstream
initialization or integration failure.
