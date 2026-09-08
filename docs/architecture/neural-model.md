# Gene Neural Model

The Gene neural model is the learned reasoning and language-generation
core of the system.

## Architecture

The model is based on a transformer architecture containing:

- token embeddings
- transformer layers
- normalization
- language-model head
- autoregressive generation
- training loss
- configurable model dimensions

## Model Scales

The development tree contains configurations targeting:

- approximately 200M parameters
- approximately 500M parameters
- approximately 700M parameters

Each scale must be independently validated.

A working 200M configuration does not automatically prove that the
500M or 700M configurations are correct.

## Model Lifecycle

```text
Configuration
 ↓
Tokenizer
 ↓
Dataset
 ↓
Training
 ↓
Checkpoint
 ↓
Evaluation
 ↓
Registration
 ↓
Inference
 ↓
Deployment
Separation

The neural model does not directly own:

persistent memory
browser execution
filesystem execution
application control
long-term knowledge
agent planning
tool orchestration

Those functions belong to the surrounding Gene architecture.
