# Gene Capabilities

Capabilities represent higher-level abilities available to Gene.

Examples include:

- filesystem interaction
- browser interaction
- APIs
- retrieval
- application interaction
- multimodal processing
- specialized reasoning

Individual tools implement concrete operations.

```text
Agent
 ↓
Capability Router
 ↓
Tool
 ↓
External System
 ↓
Observation
 ↓
Agent

MCP is one possible tool-provider mechanism and is not equivalent to
the entire Gene capability layer.
