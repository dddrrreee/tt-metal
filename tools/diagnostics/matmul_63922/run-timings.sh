#!/usr/bin/env bash
set -euo pipefail
: "${MAIN_ROOT:?set MAIN_ROOT to the main checkout}"
: "${PR_ROOT:?set PR_ROOT to the PR checkout}"
: "${ALT_ROOT:?set ALT_ROOT to the alternative checkout}"
task_dir=$(cd "$(dirname "$0")" && pwd)
cd "$task_dir"
run_phase() {
  phase_name="$1"; run_side="$2"; run_root="$3"; shard="$4"
  mkdir -p "$task_dir/runs/$phase_name/outputs" "$task_dir/hashes/$run_side-prof" "$task_dir/cache/$phase_name"
  export TT_METAL_HOME="$run_root" TT_METAL_RUNTIME_ROOT="$run_root" TT_METAL_KERNEL_PATH="$run_root"
  export TT_MESH_GRAPH_DESC_PATH="$run_root/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
  export PYTHONPATH="$task_dir:$run_root:$run_root/ttnn:$run_root/tools"
  export LD_LIBRARY_PATH="$run_root/build/lib:/opt/openmpi-v5.0.7-ulfm/lib"
  export VALID_LIBRARIES="$task_dir/hashes" VALID_OUT="$task_dir/runs/$phase_name" TT_METAL_CACHE="$task_dir/cache/$phase_name"
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1
  export TK_VALID_CLOCK_OFFSETS="$VALID_OUT/clock-offsets.csv" TK_VALID_CLOCK_MARKERS="$VALID_OUT/clock-markers.csv"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
  "$run_root/python_env/bin/python" - "$VALID_LIBRARIES/$run_side-prof/hashes.json" <<'PYHASH'
import hashlib,json,pathlib,sys,ttnn
paths={line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines()}
names=['_ttnncpp.so','_ttnn.so','libtt_metal.so']
result={n:hashlib.sha256(pathlib.Path(next(p for p in paths if p.endswith('/'+n))).read_bytes()).hexdigest() for n in names}
pathlib.Path(sys.argv[1]).write_text(json.dumps(result)+'\n')
PYHASH
  "$run_root/python_env/bin/python" portable-entry.py --kind perf --side "$run_side" --mode prof --label "$phase_name" --shard "$shard"
}
for shard in ${TIMING_SHARDS:-0 1 2 3 4 5 6}; do
  group="q$shard"; if [ "$shard" = 0 ]; then group=q10; fi
  run_phase "${group}A" main "$MAIN_ROOT" "$shard"
  run_phase "${group}B" pr "$PR_ROOT" "$shard"
  run_phase "${group}A2" main "$MAIN_ROOT" "$shard"
  run_phase "${group}C" ours "$ALT_ROOT" "$shard"
  run_phase "${group}A3" main "$MAIN_ROOT" "$shard"
done
"$MAIN_ROOT/python_env/bin/python" analyze-results.py
"$MAIN_ROOT/python_env/bin/python" summarize-issue.py
"$MAIN_ROOT/python_env/bin/python" summarize-wide.py
