# Gene Model Deployment

A production Gene deployment should contain:

- compatible Gene runtime
- validated model checkpoint
- compatible tokenizer
- model configuration
- model registry metadata
- required dependencies
- security configuration

## Deployment Flow

```text
Validated checkpoint
 ↓
Compatibility verification
 ↓
Model registry
 ↓
Runtime loading
 ↓
Health check
 ↓
Inference smoke test
 ↓
Production traffic

Model artifacts should normally be stored separately from the source
repository.
