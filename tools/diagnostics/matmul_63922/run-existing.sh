#!/usr/bin/env bash
set -euo pipefail
: "${MAIN_ROOT:?set MAIN_ROOT to the main checkout}"
: "${PR_ROOT:?set PR_ROOT to the PR checkout}"
task_dir=$(cd "$(dirname "$0")" && pwd)
(cd "$MAIN_ROOT" && env -u TT_MESH_GRAPH_DESC_PATH TT_METAL_HOME="$MAIN_ROOT" TT_METAL_RUNTIME_ROOT="$MAIN_ROOT" TT_METAL_KERNEL_PATH="$MAIN_ROOT" PYTHONPATH="$MAIN_ROOT:$MAIN_ROOT/ttnn:$MAIN_ROOT/tools" LD_LIBRARY_PATH="$MAIN_ROOT/build/lib" "$MAIN_ROOT/python_env/bin/python" "$task_dir/existing-repro.py" collect)
(cd "$PR_ROOT" && env -u TT_MESH_GRAPH_DESC_PATH TT_METAL_HOME="$PR_ROOT" TT_METAL_RUNTIME_ROOT="$PR_ROOT" TT_METAL_KERNEL_PATH="$PR_ROOT" PYTHONPATH="$PR_ROOT:$PR_ROOT/ttnn:$PR_ROOT/tools" LD_LIBRARY_PATH="$PR_ROOT/build/lib" "$PR_ROOT/python_env/bin/python" "$task_dir/existing-repro.py" run)
