# Gene Security

Security boundaries must exist between:

- source code
- credentials
- user data
- persistent memory
- datasets
- model weights
- runtime state
- external tools

## Rules

Never commit:

- API keys
- passwords
- private tokens
- private interaction logs
- private databases
- user credentials
- unrestricted model artifacts

External tools should operate under explicit permissions.

Browser, filesystem, application, and API capabilities should be
treated as privileged execution boundaries.

Public releases must be sanitized before publication.
