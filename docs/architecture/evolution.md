# Gene Evolution

Gene includes an evolution architecture intended to support controlled
improvement of capabilities and system behavior.

## Evolution Loop

```text
Experience
 ↓
Evaluation
 ↓
Candidate improvement
 ↓
Validation
 ↓
Comparison
 ↓
Approval
 ↓
Promotion

Automatic promotion of unvalidated changes should not be permitted in
production.

Evolution must preserve reproducibility and provide a way to identify
which configuration, skill, model, or component produced a behavior.
