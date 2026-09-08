# Gene Memory Architecture

Gene separates short-term request context from persistent memory.

## Context

Context contains information required for the current reasoning
operation.

## Persistent Memory

Persistent memory allows useful information to survive between
sessions.

Conceptually:

```text
Interaction
 ↓
Memory extraction
 ↓
Storage
 ↓
Retrieval
 ↓
Context
 ↓
Reasoning
Production Requirements

Memory systems must implement appropriate:

access control
retention policies
deletion
privacy protection
isolation
auditing

Private interaction logs must never be accidentally included in a
public source release.
