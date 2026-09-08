# Gene Agents

Agents provide operational behavior around the neural model.

An agent can:

1. interpret a goal
2. inspect context
3. plan
4. select capabilities
5. execute actions
6. observe results
7. update its state
8. continue execution
9. verify results
10. return a response

## Agent Loop

```text
Goal
 ↓
Plan
 ↓
Action
 ↓
Observation
 ↓
Context update
 ↓
Decision
 ↓
Action / Final response

The agent layer is what allows Gene to perform multi-step tasks rather
than only generate a single piece of text.
