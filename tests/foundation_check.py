from pathlib import Path
import importlib
import py_compile
import sys

ROOT = Path("gene").resolve()

print("=" * 70)
print("GENE FOUNDATION — COMPATIBILITY & INTEGRATION CHECK")
print("=" * 70)

failures = []

# ------------------------------------------------------------
# 1. Compile every Python file
# ------------------------------------------------------------
print("\n[1/6] Compiling Python source files...")

py_files = list(ROOT.rglob("*.py"))

for path in py_files:
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        failures.append(f"COMPILE: {path}: {exc}")

print(f"Checked {len(py_files)} Python files.")

# ------------------------------------------------------------
# 2. Import every Python module
# ------------------------------------------------------------
print("\n[2/6] Importing Python modules...")

for path in py_files:
    if "__pycache__" in path.parts:
        continue

    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        continue

    module_name = "gene." + ".".join(parts)

    try:
        importlib.import_module(module_name)
    except Exception as exc:
        failures.append(f"IMPORT: {module_name}: {exc}")

print("Module import pass completed.")

# ------------------------------------------------------------
# 3. Tool runtime
# ------------------------------------------------------------
print("\n[3/6] Testing ToolRuntime...")

try:
    from gene.tools.runtime import ToolRuntime

    runtime = ToolRuntime()

    expected_tools = {
        "filesystem.read",
        "filesystem.write",
        "filesystem.list",
        "filesystem.search",
        "process.execute",
        "apps.find",
        "browser.navigate",
        "browser.execute",
        "mcp.register",
        "mcp.list",
        "mcp.execute",
    }

    actual_tools = set(runtime.capability_status())

    missing = expected_tools - actual_tools

    if missing:
        failures.append(
            "TOOLS: Missing registered tools: "
            + ", ".join(sorted(missing))
        )
    else:
        print(f"Registered tools: {len(actual_tools)}")

    result = runtime.execute(
        "filesystem.list",
        path="gene"
    )

    if not result.get("success"):
        failures.append(
            "TOOLS: filesystem.list failed: "
            + str(result)
        )
    else:
        print("filesystem.list: PASS")

except Exception as exc:
    failures.append(f"RUNTIME: {exc}")

# ------------------------------------------------------------
# 4. Orchestrator
# ------------------------------------------------------------
print("\n[4/6] Testing GeneOrchestrator...")

try:
    from gene.core.orchestrator import GeneOrchestrator

    orchestrator = GeneOrchestrator()

    inspection = orchestrator.inspect()

    required_components = {
        "planner",
        "executor",
        "verifier",
        "memory",
    }

    missing_components = required_components - set(inspection.keys())

    if missing_components:
        failures.append(
            "ORCHESTRATOR: Missing components: "
            + ", ".join(sorted(missing_components))
        )
    else:
        print("Orchestrator components: PASS")

    result = orchestrator.run(
        "list the files in gene"
    )

    if result.get("status") != "completed":
        failures.append(
            "ORCHESTRATOR: Test task failed: "
            + str(result)
        )
    else:
        print("Orchestrator execution: PASS")

except Exception as exc:
    failures.append(f"ORCHESTRATOR: {exc}")

# ------------------------------------------------------------
# 5. Required architecture directories
# ------------------------------------------------------------
print("\n[5/6] Checking Gene architecture...")

required_dirs = [
    "config",
    "context",
    "core",
    "datasets",
    "evolution",
    "gene",
    "genome",
    "knowledge",
    "levels",
    "manifests",
    "memory",
    "model",
    "raw",
    "skills",
    "soul",
    "specializations",
    "tests",
    "tools",
]

for directory in required_dirs:
    path = ROOT / directory

    if not path.exists():
        failures.append(
            f"ARCHITECTURE: Missing directory: {path}"
        )

print(f"Checked {len(required_dirs)} architectural directories.")

# ------------------------------------------------------------
# 6. Final verdict
# ------------------------------------------------------------
print("\n" + "=" * 70)

if failures:
    print("GENE FOUNDATION CHECK: FAILED")
    print("=" * 70)

    for failure in failures:
        print(" -", failure)

    sys.exit(1)

print("GENE FOUNDATION CHECK: PASSED")
print("=" * 70)
print("Python compilation: PASS")
print("Module imports:     PASS")
print("Tool runtime:       PASS")
print("Orchestrator:       PASS")
print("Architecture:       PASS")
print("=" * 70)
print("FOUNDATION IS READY FOR THE NEXT SYSTEM.")
