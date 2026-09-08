# Gene Training Architecture

Training is deliberately separated from runtime execution.

## Pipeline

```text
Raw datasets
 ↓
Schema inspection
 ↓
Normalization
 ↓
Dataset mixture
 ↓
Tokenizer
 ↓
Token packing
 ↓
Training
 ↓
Checkpoint
 ↓
Evaluation
 ↓
Model registration
 ↓
Inference
Training Components

The development tree contains infrastructure for:

dataset processing
normalization
dataset mixtures
token packing
tokenizer training
production training
long-context training
checkpointing
profiling
evaluation
training resource management
Validation

Training completion alone is not sufficient.

A checkpoint must also be tested for:

loading
tokenizer compatibility
inference
generation quality
memory requirements
context behavior
runtime integration
Model Scale Validation

Each intended model scale should have independent validation.

200M
 ├─ architecture
 ├─ training
 ├─ checkpoint
 └─ inference

500M
 ├─ architecture
 ├─ parameter count
 ├─ memory
 ├─ training
 ├─ checkpoint
 └─ inference

700M
 ├─ architecture
 ├─ parameter count
 ├─ memory
 ├─ training
 ├─ checkpoint
 └─ inference

