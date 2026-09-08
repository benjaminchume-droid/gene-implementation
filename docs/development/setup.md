# Gene Development Setup

Gene development should be performed inside an isolated Python
environment.

## Basic Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Install the project's required dependencies according to its dependency
configuration.

Verification
python -c "import gene; print('Gene import OK')"
python -m pytest tests -q

The exact dependency set should remain synchronized with the source
tree and CI environment.
