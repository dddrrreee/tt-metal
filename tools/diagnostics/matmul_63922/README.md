# Batched matmul comparison harness

This captures the device-kernel timings and output comparisons used to assess
PR 63922. `plan.json` retains the original shapes, seeds and distributions;
`existing-plan.json` retains the exact previously collected test nodes.
`pins.json` identifies the three compared revisions. The alternative is an
independent fork commit, not a PR.

Use a Blackhole card through the host's normal device admission mechanism.
Prepare separate checkouts and set `MAIN_ROOT`, `PR_ROOT` and `ALT_ROOT` to
main, the proposed PR head and the alternative in `pins.json`. Apply both
instrumentation patches to each checkout and build with the device profiler:

```sh
HARNESS_ROOT=/path/to/this/directory
for tree in "$MAIN_ROOT" "$PR_ROOT" "$ALT_ROOT"; do
  git -C "$tree" apply "$HARNESS_ROOT/clock-observer.patch"
  git -C "$tree" apply --unidiff-zero "$HARNESS_ROOT/selector-observer.patch"
  (cd "$tree" && ./build_metal.sh --release --build-tests)
done
```

`clock-observer.patch` measures per-worker clock offsets before dispatch and
corrects device markers; `selector-observer.patch` records the chosen config.
They instrument measurement only; neither is the proposed matmul fix. The
patches apply to each of the pinned revisions.

```sh
bash "$HARNESS_ROOT/run-timings.sh"
python "$HARNESS_ROOT/verify_clocks.py" q10A q10B q10A2 q10C q10A3
bash "$HARNESS_ROOT/run-existing.sh"
```

`run-timings.sh` uses the original five phases for every case: main A, PR B,
main A2, alternative C, main A3. Each phase uses two warmups, five measured
device-kernel samples, and the original seeded inputs. It reports the 20 issue
shapes, ten wide shapes and two explicit-configuration cases. The same plan's
shard 6 includes three deliberately unsupported sharded inputs, which all
three revisions rejected. Set `TIMING_SHARDS` to a space-separated list to
limit the run. Run all phases of each shard on the same physical card. The
`runs/`, `hashes/` and `cache/` output directories are created under the
harness directory. `analyze-results.py` compares the stored BF16/FP32 bits,
PCC, ULPs and timings; `summarize-issue.py` and `summarize-wide.py` format the
tables. `verify_clocks.py` accepts any list of completed phase labels.

`run-existing.sh` collects the public selectors on main, then runs the same
node list on the PR head in batches of 35. This collection was checked against
`existing-plan.json`; `existing.py --shard N` runs the saved nodes directly
and uses `runtime_provenance.py` to verify the loaded library hashes. The
independent Python tests and C++ selector checks referenced in the comment
come from the alternative commit; they must be overlaid when running those
specific checks on main or the PR.

`repro-subdevice.py` and `sim-transpose-pressure.py` are the standalone
functional reproducers. `sim-grid.py` and `compare-grid.py` reproduce the
explicit-grid simulator comparison; set `VALID_OUT` to `main`, `pr` and
`alternative` respectively before running `sim-grid.py`, then run the
comparison script in their parent directory.
