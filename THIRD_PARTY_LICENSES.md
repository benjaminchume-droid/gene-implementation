# Third-Party Components

Gene AI contains or may integrate with third-party software,
frameworks, libraries, datasets, model implementations, and services.

Those components are NOT covered by the Gene AI proprietary license
when their own licenses grant independent rights.

## Policy

Before distributing Gene AI, verify the license of every dependency,
dataset, pretrained model, and externally sourced component.

Third-party licenses must not be removed or replaced.

## Categories

### Python dependencies

Examples may include:

- PyTorch
- Transformers
- Tokenizers
- NumPy
- Pillow
- SoundFile
- MCP SDK
- Playwright
- Hugging Face libraries
- other runtime dependencies

Each dependency remains governed by its upstream license.

### Models

Downloaded or pretrained models may have licenses that differ from
Gene AI's source-code license.

Model weights must therefore be audited independently.

### Datasets

Training datasets may contain their own:

- copyright restrictions
- attribution requirements
- usage restrictions
- redistribution restrictions
- research/commercial restrictions

Gene AI's proprietary license does not grant rights to redistribute
third-party datasets.

## Compliance

A production release should maintain an accurate inventory of:

1. dependency
2. version
3. source
4. license
5. required attribution
6. redistribution restrictions
7. commercial-use restrictions

This document is a policy placeholder and must be updated as the
dependency and model inventory is finalized.
