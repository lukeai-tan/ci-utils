# ci-utils

A test selection and scheduling engine built for the [Blast Radius buildathon](https://www.joinbuilders.org/buildathon/tracks/blast-radius).

Given a batch of patches against a pinned SQLGlot baseline, ci-utils identifies potentially affected tests, refines that selection with bounded verification, and distributes the remaining tests across workers. It aims to reduce simulated CI time while accounting for worker startup costs and the CPU spent making selections.

**Finalized submission:** [`f8ab1d67a6af`](https://github.com/lukeai-tan/ci-utils/commit/f8ab1d67a6af17bc22d245d059a817a7cf5e3bf3), submitted as `BB-C8A95553`.

This commit achieved the best reported practice score: **151.5 s**, with **137.457 simulated seconds**, **14.05 median CPU seconds**, and **zero missed commits in all three runs**.

## How it works

1. **Find affected tests.** Compare patches with baseline coverage and dependency profiles, including statements, imports, class attributes, and fixture reads.
2. **Refine the selection.** Use recorded value observations, query and fixture replay, optimizer checkpoints, and targeted test execution to eliminate unnecessary work.
3. **Schedule the work.** Balance tests by recorded duration across up to eight workers, accounting for startup costs. Additional tests can fill spare capacity without increasing the predicted completion time.

The Docker image enables aggressive selection: some unrun candidates may be omitted after cheaper probes pass. This trades stronger completeness guarantees for speed. Running the selector without the aggressive flags disables those speculative policies.

## Optimization walkthrough

The optimization work addressed two costs together: the simulated time of the scheduled tests and the real CPU time spent deciding what to schedule. An extra verification step is useful only when the work it removes is worth its own execution cost.

### More precise dependency tracking

File-level coverage can select many tests for a small edit. The selector combines several finer-grained observations of the pinned baseline:

- **Executable scopes and statements.** Compiled code objects and line coverage locate the function or statement affected by an edit. Changes to bindings, closures, definitions, or initialization use broader fallbacks.
- **Class and attribute consumers.** Profiles record class usage, inheritance, qualified attribute reads, and forwarded settings. Eligible literal-setting edits can target their readers instead of every user of a class.
- **Dictionary and validator dependencies.** Specialized evidence distinguishes consumers of dictionary keys and validator requirements when an edit preserves the supported construction shape.
- **Fixture and source reads.** Data-file edits retain their readers. Tests that inspect source text or code metadata remain relevant even when an edit leaves the parsed syntax unchanged.
- **Cold execution and subprocess observations.** Profiles include initialization and supported worker execution, rather than relying only on warmed test bodies. Incomplete observations preserve uncertainty.

When independent dependency analyses apply, their candidate sets can be intersected under the implementation's completeness checks. Unknown tests and unsupported changes retain broader selections.

Implementation: [selector.py](selector.py), [compiledcheck.py](compiledcheck.py), [statementcheck.py](statementcheck.py), [fieldcheck.py](fieldcheck.py), and the profile builders in [tools/](tools/).

### Checking whether an edit changes observed behavior

An edited expression can still produce the same result for a test's recorded inputs. The selector handles supported comparison, literal, Boolean, and dictionary-lookup edits using baseline observations:

- Operand types and exact primitive values help evaluate comparison-operator changes.
- Recorded value domains identify literal-comparison edits whose observed outcomes stay unchanged.
- Boolean observations include short-circuit evaluation behavior.
- Dictionary lookup observations preserve relevant lookup and identity behavior.

These checks apply only to supported expression shapes with compatible evidence. Missing observations, initialization effects, source readers, and introspection consumers prevent unrestricted removal.

Implementation: [comparisoncheck.py](comparisoncheck.py), [relationcheck.py](relationcheck.py), [valuecheck.py](valuecheck.py), and [behaviorcheck.py](behaviorcheck.py).

### Replaying affected work inside expensive tests

An expensive test can contain many independent queries or fixture cases. The replay paths reduce the work needed to check eligible patches:

- **Per-query replay.** Dependency and value observations identify affected executor tasks. Replay restores their serialized schema and table inputs and checks results and shared input state.
- **Optimizer checkpoints.** Recorded expression graphs allow eligible query checks to stop at a matching boundary, while optimizer-pass snapshots allow replay to begin at the affected suffix of the pass sequence.
- **Fixture-case filtering.** Eligible optimizer methods execute affected fixture cases through the original test assertions, preserving required setup and shared-state checks.
- **Fixture deltas.** Fixture-only edits can focus verification on changed records. Mixed code/data patches use separate planning for code reachability while preserving data readers and applying the full patch during execution.

Replay has input, source, driver, and state checks. Unsupported or incomplete replay retains the candidate in the ordinary verification path.

Implementation: [queryreplay.py](queryreplay.py), [passcheck.py](passcheck.py), and [fixturecheck.py](fixturecheck.py).

### Spending verification CPU where it helps

Verification uses measured method CPU costs and class setup/teardown costs. The planner compares the predicted schedule saving from removing a candidate bundle with the cost of checking it.

It considers whole bundles, individual tests, and smaller groups that may reduce the schedule's longest worker load. Progressive verification can make several bounded attempts, and observed overhead updates later estimates. Separate budgets and failure counters stop repeatedly unprofitable replay or verification paths.

The ordinary pytest path removes only reported passes with successful setup, call, and teardown. Failed, skipped, timed-out, or incomplete candidates remain selected. Affordable small and medium groups can also be checked using an already-running verification service.

Implementation: [runtime.py](runtime.py).

### Aggressive selection after clean probes

The submitted container also enables a speculative policy. It runs a bounded set of affordable probes against the patch. When all requested probes complete successfully, the policy may omit additional profiled candidates that were not run.

The broader setting extends omission beyond expensive tests. Protected candidates and candidates without timing evidence remain selected; failed or incomplete probes block speculative omission. Passing probes do not establish that every omitted test passes. A missed commit adds 300 simulated seconds, so this policy has a meaningful tradeoff on unseen changes.

The default command enables `--aggressive`, `--aggressive-broad`, `--aggressive-small-groups`, `--verified-medium-groups`, and `--aggressive-medium-probes`, with `--aggressive-min-cpu 0.5`. The small/medium group paths remove actual reported passes; the broad omission policy is the speculative part.

### Scheduling with worker startup costs

The scheduler evaluates between one and eight workers. It places longer tests first onto the least-loaded worker and compares predicted costs using:

```text
predicted cost = longest worker load + 2 seconds × worker count
```

A lower bound based on the longest test and average load avoids evaluating worker counts that cannot improve the current schedule. Optional spare-capacity filling adds tests only when they fit below the existing maximum worker load. `--no-fill` disables this extra coverage.

Implementation: `Selector.schedule` in [selector.py](selector.py).

### Reducing startup, storage, and repeated analysis

- **Lightweight patched trees.** Verification symlinks unchanged baseline files and writes only edited files into temporary trees.
- **Checked bytecode caches.** Baseline bytecode is compiled during the image build. Checked source hashes prevent edited files from silently using stale bytecode.
- **Validated profile caches.** Comparison and behavior profiles are validated and expanded during the build, then loaded from caches tied to the source profile, helper code, and Python version. Stale caches fall back to validation.
- **Lazy profile and replay-state loading.** Large evidence maps load when needed. Query states share compressed storage and use bounded decoding caches.
- **Reused analysis.** Cached baseline ASTs and scope lookups, per-patch AST comparisons, and reuse of an already-computed selection bound avoid repeated parsing and traversals.
- **Controlled runtime overhead.** Automatic pytest plugin loading is disabled, numerical-library thread counts are limited, and verification has CPU/wall budgets and child-process cleanup.

Implementation: [runtime.py](runtime.py), [profilecache.py](profilecache.py), [queryreplay.py](queryreplay.py), and [Dockerfile](Dockerfile).

### Optional verification paths

The repository also implements additional paths that are not enabled by the default container command:

| Option | Purpose |
| --- | --- |
| `--query-checks` | Verify matching frontend boundaries against the original executor assertions. |
| `--stage-replay` | Replay affected parser, SQL-generation, and planning stages. Requires a compatible stage profile, which is not bundled in the current image. |
| `--import-verification` | Compare cold imported state, closures, alias relationships, and dispatch evidence to refine initialization-related candidates. |
| `--warm-verification` | Replace eligible function bodies in a preloaded environment, with cold fallback when warm verification cannot establish a pass. |

These paths require compatible baseline evidence and have their own eligibility checks and budgets.

Implementation: [querycheck.py](querycheck.py), [stagecheck.py](stagecheck.py), [statecheck.py](statecheck.py), [dispatchcheck.py](dispatchcheck.py), and [warmcheck.py](warmcheck.py).

## Reported practice results

Selected milestones from the buildathon practice evaluator:

| Commit | Reported score | Simulated seconds | Median CPU seconds | Missed commits per run |
| --- | ---: | ---: | ---: | ---: |
| `45a77467e8cd` | 336.6 s | 332.549 | 4.00 | 0 |
| `0509fdc2a148` | 193.4 s | 180.711 | 12.70 | 0 |
| `f8ab1d67a6af` | **151.5 s** | 137.457 | 14.05 | 0 |
| `7ea1f6d68e36` | 151.8 s | 137.457 | 14.34 | 0 |

The best reported practice score improved by approximately **55.0%** relative to the earliest result above; simulated time fell by approximately **58.7%**. More verification CPU was worthwhile because it removed substantially more scheduled work.

These results compare complete configurations, rather than isolating each optimization's contribution. They are practice-set measurements; the final evaluation uses a different dataset, and no final-set result is reported here.

## Run with Docker

Build the image:

```sh
docker build -t ci-utils .
```

Place the buildathon input files in a local `data/` directory:

```text
data/
├── tests.json
└── commits/
    ├── <commit-id>.patch
    └── ...
```

Run the default configuration and save its schedules:

```sh
docker run --rm --read-only --network none \
  --tmpfs /tmp:rw,size=64m \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  ci-utils > schedules.jsonl
```

To run without aggressive selection, replace the image's default command:

```sh
docker run --rm --read-only --network none \
  --tmpfs /tmp:rw,size=64m \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  ci-utils python /app/selector.py > schedules.jsonl
```

The build downloads the pinned SQLGlot source and installs locked dependencies. The resulting container runs as a non-root user and uses `/tmp` for temporary patched files.

## Input and output

`tests.json` contains a `tests` array whose entries provide `number`, `id`, and `seconds`. Patches are read from `commits/*.patch` in filename order.

Standard output contains one JSON object per patch. Each worker list contains test numbers from the input manifest:

```json
{"commit":"example-commit","workers":[[12,7],[3]]}
```

Selection diagnostics and CPU timing go to standard error. Use `python /app/selector.py --help` inside the image to inspect the available options.

## Project layout

- `selector.py` — patch analysis, test selection, scheduling, and output.
- `runtime.py` — bounded verification against patched source trees.
- `*check.py` and `queryreplay.py` — specialized analysis and replay helpers.
- `maps/` — compressed profiles of the pinned baseline.
- `tools/` — scripts for collecting and preparing baseline profiles.
- `Dockerfile` and `requirements.lock` — reproducible container setup.

The bundled profiles target SQLGlot revision `eb11aec4be8cb2122afe713a4816662c3f0fb2fa`. Using another revision or codebase requires rebuilding the relevant profiles.

## License

MIT. See [LICENSE](LICENSE). SQLGlot's license is included in [tools/SQLGLOT_LICENSE](tools/SQLGLOT_LICENSE).
