"""Bounded verification with optional speculative selection after clean probes."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import os
import resource
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from importlib.util import MAGIC_NUMBER
from pathlib import Path


REVISION = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"


def load_profile(path):
    try:
        with gzip.open(path, "rt") as source:
            profile = json.load(source)
        if profile["revision"] != REVISION or profile["version"] not in (1, 2):
            return None
        return profile
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_setting_flow(path):
    try:
        with gzip.open(path, "rt") as source:
            profile = json.load(source)
        if (profile["revision"] != REVISION or profile["version"] != 1
                or profile.get("kind") != "setting_flow" or profile["incomplete"]):
            return None
        uninstrumented = set(profile["uninstrumented"])
        if (not uninstrumented <= set(range(len(profile["tests"])))
                or any(name not in profile["cpu"] for i, name in enumerate(profile["tests"])
                       if i not in uninstrumented)
                or not isinstance(profile["known"], list) or not isinstance(profile["blocked"], list)
                or not isinstance(profile["consumers"], dict) or not isinstance(profile["entries"], dict)):
            return None
        return profile
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_timing(path):
    from comparisoncheck import REVISION
    try:
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        if (data["revision"] != REVISION or data["version"] != 1 or data["kind"] != "pristine_timing"
                or data["complete"] is not True or not data["cpu"]):
            return None
        for field in ("cpu", "setup", "teardown"):
            if not isinstance(data[field], dict) or any(type(name) is not str
                    or type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for name, value in data[field].items()):
                return None
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def merge_profile(mapping, profile, setting_flow=None):
    """Add class evidence and distinguish explicit source reads from imports."""
    if not profile or profile.get("incomplete"):
        return
    mapping["attribute_complete"] = profile.get("attribute_complete", False)
    mapping["setting_collection"] = profile.get("collection", [])
    mapping["setting_precise_classes"] = profile.get("setting_precise_classes", [])
    by_id = {name: index for index, name in enumerate(mapping["tests"])}
    numbers = {i: by_id[name] for i, name in enumerate(profile["tests"]) if name in by_id}
    reads = mapping.setdefault("reads", {})
    unprofiled = set(range(len(mapping["tests"]))) - set(numbers.values())
    for path in set(reads) | set(profile.get("reads", {})):
        if path.startswith("sqlglot/") and path.endswith(".py"):
            reads[path] = sorted((set(reads.get(path, [])) & unprofiled)
                                 | {numbers[i] for i in profile.get("reads", {}).get(path, []) if i in numbers})
    for key, consumers in profile.get("classes", {}).items():
        mapping.setdefault("classes", {})[key] = sorted(
            set(mapping.get("classes", {}).get(key, [])) | {numbers[i] for i in consumers if i in numbers}
        )
    for key, consumers in profile.get("attributes", {}).items():
        mapping.setdefault("qualified_attributes", {})[key] = sorted(
            set(mapping.get("qualified_attributes", {}).get(key, [])) | {numbers[i] for i in consumers if i in numbers}
        )
    # A passing parent is insufficient evidence for code executed in a child.
    # Only explicit, successful child traces can resolve baseline uncertainty.
    complete = {i for i in profile.get("child_complete", [])
                if i in numbers and i not in profile.get("child_failed", [])
                and profile["tests"][i] in profile.get("cpu", {})}
    for path, entries in profile.get("child_lines", {}).items():
        lines = mapping.setdefault("lines", {}).setdefault(path, {})
        consumers = set()
        for line, observed in entries.items():
            traced = {numbers[i] for i in observed if i in complete}
            lines[line] = sorted(set(lines.get(line, [])) | traced)
            consumers.update(traced)
        mapping.setdefault("files", {})[path] = sorted(set(mapping.get("files", {}).get(path, [])) | consumers)
    resolved = {numbers[i] for i in complete}
    failed = {numbers[i] for i in profile.get("child_failed", []) if i in numbers}
    mapping["incomplete"] = sorted((set(mapping.get("incomplete", [])) - resolved) | failed)
    if setting_flow:
        mapping["setting_flow"] = setting_flow


class PatchedTree:
    def __init__(self, base, patch):
        from selector import apply_edit, parse_patch

        self.directory = tempfile.TemporaryDirectory(prefix="ci-verify-")
        self.root = Path(self.directory.name)
        self.base = Path(base).resolve()
        try:
            for path in self.base.rglob("*"):
                relative = path.relative_to(self.base)
                if any(part in (".git", ".pytest_cache", ".mypy_cache") for part in relative.parts):
                    continue
                if '__pycache__' in relative.parts:
                    if not path.is_file() or path.suffix != '.pyc':
                        continue
                    with path.open('rb') as cache:
                        header = cache.read(8)
                    # Checked hashes force edited sources to compile afresh,
                    # even when their length and timestamp match the baseline.
                    if header[:4] != MAGIC_NUMBER or int.from_bytes(header[4:8], 'little') != 3:
                        continue
                if path.is_file():
                    target = self.root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(path)
            self.edits = parse_patch(patch)
            for edit in self.edits:
                source = self.base / edit.path
                updated, _ = apply_edit(source.read_text(), edit)
                if edit.path.endswith(".py"):
                    compile(updated, str(self.root / edit.path), "exec")
                target = self.root / edit.path
                target.unlink()
                target.write_text(updated)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Verdicts:
    """A passing call alone is insufficient: retain subtest and fixture failures."""

    def __init__(self):
        self.phases = {}
        self.bad = set()
        self.failed = set()

    def pytest_runtest_logreport(self, report):
        if report.failed or report.skipped:
            self.bad.add(report.nodeid)
        if report.failed:
            self.failed.add(report.nodeid)
        if report.passed:
            self.phases.setdefault(report.nodeid, set()).add(report.when)

    def passed(self):
        return sorted(name for name, phases in self.phases.items()
                      if phases == {"setup", "call", "teardown"} and name not in self.bad)


def run_tests(root, names, cpu_limit, *, fixtures=None, fixture_plans=None):
    import pytest

    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(PYTHONPATH=str(root), SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="2",
                      PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_MAX_THREADS="1")
    verdicts = Verdicts()
    plugins = [verdicts]
    case_filter = None
    if fixtures is not None and fixture_plans:
        from fixturecheck import Filter
        case_filter = Filter(fixtures, fixture_plans)
        case_filter.install()
        plugins.append(case_filter)

    def timed_out(signum, frame):
        raise TimeoutError("verification CPU budget")

    signal.signal(signal.SIGPROF, timed_out)
    signal.setitimer(signal.ITIMER_PROF, cpu_limit)
    try:
        status = pytest.main([*names, "-q", "--maxfail=1", "--import-mode=importlib", "--rootdir=" + str(root),
                     "-p", "no:cacheprovider", "-p", "pytest_timeout",
                     "--timeout=" + str(max(5, min(15, cpu_limit)))], plugins=plugins)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    passed = [name for name in verdicts.passed() if case_filter is None or case_filter.complete(name)]
    failed = sorted(verdicts.failed & set(names))
    return {"passed": passed, "failed": failed,
            "incomplete": sorted(set(names) - set(passed) - set(failed)),
            "completed": status == 0 and set(passed) == set(names)}


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def serve(base, queries=None, task_queries=None, fixtures=None, passes=None, stages=None, warm=False, imported=None):
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      NUMEXPR_MAX_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    import pytest  # Warm third-party imports only; patched packages load in children.

    query_baseline = None
    if queries is not None and queries.is_file():
        from querycheck import load_baseline
        query_baseline = load_baseline(queries)
    task_baseline = None
    if task_queries is not None and task_queries.is_file():
        from queryreplay import load_profile
        task_baseline = load_profile(task_queries)
    fixture_baseline = None
    pass_baseline = None
    stage_baseline = None
    warm_compiled = warm_startup = None
    if warm:
        from compiledcheck import load_profile
        warm_compiled = load_profile(Path(__file__).parent / 'maps/compiled.json.gz')
        with gzip.open(Path(__file__).parent / 'maps/baseline.json.gz', 'rt') as stream:
            warm_startup = json.load(stream)['startup']

    print(json.dumps({"ready": True, "cpu": cpu_seconds()}), flush=True)
    for line in sys.stdin:
        started = cpu_seconds()
        try:
            request = json.loads(line)
            replacements = None
            if warm:
                from warmcheck import plan, preload
                replacements = plan(base, request['patch'], warm_compiled, warm_startup)
                if replacements is None or request.get('kind'):
                    raise ValueError('unsupported warm request')
                preload(base, request['tests'])
            if request.get('kind') == 'stages' and stage_baseline is None:
                from stagecheck import load_profile
                stage_baseline = load_profile(stages)
            if passes is not None and request.get('kind') == 'optimizer_passes' and pass_baseline is None:
                from passcheck import load_profile
                pass_baseline = load_profile(passes)
            if fixtures is not None and request.get('fixture_plans') and fixture_baseline is None:
                from fixturecheck import load_profile
                fixture_baseline = load_profile(fixtures)
            if (request.get('kind') not in {'optimizer_passes', 'stages', 'import_state'}
                    and any("test_executor.py" in name or "test_optimizer.py" in name for name in request.get("tests", []))):
                import duckdb
                import numpy
                import pandas
            with PatchedTree(base, request["patch"]) as tree:
                read_fd, write_fd = os.pipe()
                child = os.fork()
                if child == 0:
                    os.close(read_fd)
                    try:
                        with open(os.devnull, "w") as sink:
                            os.dup2(sink.fileno(), 1)
                            os.dup2(sink.fileno(), 2)
                        if warm:
                            from warmcheck import swap
                            swap(base, replacements)
                        if request.get('kind') == 'import_state':
                            from statecheck import load_all, snapshot, delta
                            signal.signal(signal.SIGPROF, lambda *args: (_ for _ in ()).throw(TimeoutError('import state CPU budget')))
                            signal.setitimer(signal.ITIMER_PROF, request['cpu_limit'])
                            load_all(tree.root)
                            with gzip.open(imported, 'rt') as stream:
                                baseline = json.load(stream)['values']
                            changed, unknown = delta(baseline, snapshot())
                            result = {'passed': [], 'changed': sorted(changed), 'unknown': sorted(unknown)}
                        elif request.get('kind') == 'stages' and stage_baseline is not None:
                            from stagecheck import replay
                            signal.signal(signal.SIGPROF, lambda *args: (_ for _ in ()).throw(TimeoutError('stage replay CPU budget')))
                            signal.setitimer(signal.ITIMER_PROF, request['cpu_limit'])
                            result = replay(tree.root, stage_baseline, request['indices'])
                        elif request.get('kind') == 'optimizer_passes' and pass_baseline is not None:
                            from passcheck import replay
                            signal.signal(signal.SIGPROF, lambda *args: (_ for _ in ()).throw(TimeoutError('pass replay CPU budget')))
                            signal.setitimer(signal.ITIMER_PROF, request['cpu_limit'])
                            result = replay(tree.root, pass_baseline, request['indices'])
                        elif request.get("kind") == "task_queries" and task_baseline is not None:
                            from queryreplay import replay
                            signal.signal(signal.SIGPROF, lambda *args: (_ for _ in ()).throw(TimeoutError("task replay CPU budget")))
                            signal.setitimer(signal.ITIMER_PROF, request["cpu_limit"])
                            result = replay(tree.root, task_baseline, request["indices"], checkpoints=request.get('checkpoints'))
                        elif request.get("kind") == "queries" and query_baseline is not None:
                            from querycheck import run_boundaries
                            signal.signal(signal.SIGPROF, lambda *args: (_ for _ in ()).throw(TimeoutError("query CPU budget")))
                            signal.setitimer(signal.ITIMER_PROF, request["cpu_limit"])
                            result = run_boundaries(tree.root, baseline=query_baseline,
                                                    methods=[name.rsplit("::", 1)[-1] for name in request["tests"]],
                                                    changed_files=[edit.path for edit in tree.edits])
                        else:
                            result = run_tests(tree.root, request["tests"], request["cpu_limit"],
                                               fixtures=fixture_baseline, fixture_plans=request.get('fixture_plans'))
                        payload = json.dumps(result).encode()
                        with os.fdopen(write_fd, "wb") as output:
                            output.write(payload)
                    except BaseException as error:
                        try:
                            os.write(write_fd, json.dumps({'passed': [], 'error': type(error).__name__}).encode())
                        except OSError:
                            pass
                    finally:
                        os._exit(0)
                os.close(write_fd)
                deadline = time.monotonic() + request["wall_limit"]
                payload = bytearray()
                with selectors.DefaultSelector() as selector:
                    selector.register(read_fd, selectors.EVENT_READ)
                    while True:
                        events = selector.select(max(0, deadline - time.monotonic()))
                        if not events:
                            # Kill the complete verification process group, including
                            # spawned workers; the selector retains the static answer.
                            os.killpg(os.getpgrp(), signal.SIGKILL)
                        chunk = os.read(read_fd, 65536)
                        if not chunk:
                            break
                        payload.extend(chunk)
                        if len(payload) > 8 * 1024**2:
                            os.killpg(os.getpgrp(), signal.SIGKILL)
                os.close(read_fd)
                os.waitpid(child, 0)
                result = json.loads(payload) if payload else {"passed": []}
            result["cpu"] = cpu_seconds() - started
        except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
            result = {"passed": [], "cpu": cpu_seconds() - started, "error": type(error).__name__}
        print(json.dumps(result, separators=(",", ":")), flush=True)


class Verifier:
    def __init__(self, base, profile, tests, schedule, *, queries=None, values=None, behavior=None, timing=None, task_queries=None, compiled=None, fixtures=None, passes=None, stages=None, imported=None, dispatch=None, warm=None, fixture_delta=True, progressive=True, aggressive=False, aggressive_min_cpu=4.0, cpu_budget=45, wall_budget=135):
        self.base = Path(base)
        self.profile = profile or {}
        self.timing = timing
        self.tests = {test["number"]: test for test in tests}
        self.by_id = {test["id"]: test["number"] for test in tests}
        self.schedule = schedule
        self.process = None
        self.processes = {}
        self.warm_mode = False
        self.used_cpu = 0.0
        self.cpu_budget = cpu_budget
        self.deadline = time.monotonic() + wall_budget
        self.disabled = not self.base.is_dir()
        self.unprofitable = 0
        self.task_unprofitable = 0
        self.fixture_unprofitable = 0
        self.progressive_unprofitable = 0
        # Reserve cold checkout and collection separately from method work,
        # then raise this estimate from completed verification requests.
        self.progressive_overhead = 1.5
        self.progressive = progressive
        self.aggressive = aggressive
        self.aggressive_min_cpu = aggressive_min_cpu
        self.aggressive_cpu = 0.0
        self.aggressive_attempted = set()
        self.fixture_delta = fixture_delta
        self.queries = Path(queries) if queries is not None else None
        self.query_backend_files = None
        self.values = values
        self.compiled = compiled
        self.behavior = behavior
        self.task_queries = Path(task_queries) if task_queries is not None else None
        self.task_baseline = None
        self.passes = Path(passes) if passes is not None else None
        self.pass_available = self.passes is not None and self.passes.is_file()
        self.pass_baseline = None
        self.pass_unprofitable = 0
        self.stages = Path(stages) if stages is not None else None
        self.stage_baseline = None
        self.stage_unprofitable = 0
        self.imported = Path(imported) if imported is not None else None
        self.import_baseline = None
        self.import_unprofitable = 0
        self.dispatch = dispatch
        self.warm = warm
        self.fixtures = Path(fixtures) if fixtures is not None else None
        self.fixture_baseline = None
        self.fixture_available = self.fixtures is not None and self.fixtures.is_file()
        self.task_available = bool((values or behavior or compiled) and self.task_queries is not None and self.task_queries.is_file())
        if self.queries is not None and self.queries.is_file():
            from querycheck import load_baseline
            try:
                self.query_backend_files = load_baseline(self.queries)["backend_files"]
            except (OSError, ValueError, KeyError, TypeError, EOFError):
                self.queries = None

    def cost(self, selected):
        workers = self.schedule(selected, fill=False)
        return max((sum(math.floor(self.tests[n]["seconds"] * 1000 + 0.5) for n in worker)
                    for worker in workers), default=0) / 1000 + 2 * len(workers)

    def close(self):
        processes = list(self.processes.values()) + ([self.process] if self.process is not None else [])
        seen = set()
        for process in processes:
            if id(process) in seen:
                continue
            seen.add(id(process))
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            process.stdin.close()
            process.stdout.close()
        self.processes.clear()
        self.process = None

    def read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("verification wall budget")
            line = self.process.stdout.readline()
        if not line:
            raise EOFError("verification worker exited")
        return json.loads(line)

    def start(self, warm=False):
        if self.process is not None:
            self.processes[self.warm_mode] = self.process
        self.warm_mode = warm
        self.process = self.processes.get(warm)
        if self.process is None:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--serve", "--source", str(self.base.resolve())]
            if warm:
                command.append('--warm')
            if self.queries is not None:
                command.extend(["--queries", str(self.queries.resolve())])
            if self.task_baseline is not None:
                command.extend(["--task-queries", str(self.task_queries.resolve())])
            if self.fixture_available:
                command.extend(["--fixtures", str(self.fixtures.resolve())])
            if self.pass_available:
                command.extend(['--passes', str(self.passes.resolve())])
            if self.stages is not None and self.stages.is_file():
                command.extend(['--stages', str(self.stages.resolve())])
            if self.imported is not None and self.imported.is_file():
                command.extend(['--imported', str(self.imported.resolve())])
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, start_new_session=True,
                env={**os.environ, "PYTHON_CPU_COUNT": "2", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                     "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1"},
            )
            self.used_cpu += self.read(10)["cpu"]
            self.processes[warm] = self.process

    def import_refine(self, patch, selected, *, protected=()):
        from statecheck import changed_sources, delta, projected
        from comparisoncheck import source_hash
        if (self.disabled or self.imported is None or not self.imported.is_file()
                or not self.compiled or self.import_unprofitable >= 2
                or self.used_cpu >= self.cpu_budget or time.monotonic() >= self.deadline):
            return selected, []
        candidates = selected - set(protected)
        saving = self.cost(selected) - self.cost(selected - candidates)
        if saving < 4:
            return selected, []
        try:
            # Ordinary body edits already have complete code/statement evidence.
            # Spend cold import CPU only on initialization or callback factories.
            from selector import same_ast, parse_patch, apply_edit
            from compiledcheck import changes as code_changes
            relevant = False
            for edit in parse_patch(patch):
                old = (self.base / edit.path).read_text()
                new, _ = apply_edit(old, edit)
                if not same_ast(ast.parse(old), ast.parse(new), structure=True):
                    relevant = True
                    break
                for before, after in code_changes(compile(old, edit.path, 'exec', dont_inherit=True),
                                                  compile(new, edit.path, 'exec', dont_inherit=True)):
                    if before.co_name == '<lambda>' or before.co_cellvars or after.co_cellvars:
                        relevant = True
                        break
            if not relevant:
                return selected, []
            changes = changed_sources(self.base, patch)
            if not changes:
                return selected, []
            if self.import_baseline is None:
                with gzip.open(self.imported, 'rt') as stream:
                    self.import_baseline = json.load(stream)
                data = self.import_baseline
                if (data['revision'] != REVISION or data['version'] != 1
                        or data['python'] != list(sys.version_info[:2])):
                    self.imported = None
                    return selected, []
            if any(self.import_baseline['sources'].get(path) != source_hash(a) for path, (a, b) in changes.items()):
                return selected, []
            self.start()
            budget = min(12, saving * 0.5, self.cpu_budget - self.used_cpu)
            if budget < 8:
                return selected, []
            request = dict(kind='import_state', patch=patch, tests=[], cpu_limit=budget,
                           wall_limit=min(20, max(.1, self.deadline-time.monotonic())))
            self.process.stdin.write(json.dumps(request, separators=(',', ':'))+'\n')
            self.process.stdin.flush()
            result = self.read(min(22, max(.1, self.deadline-time.monotonic())))
            self.used_cpu += result['cpu']
            if 'changed' not in result or 'unknown' not in result:
                self.import_unprofitable += 1
                return selected, ['imported-state verification incomplete; retained candidates']
            changed, unknown = set(result['changed']), set(result['unknown'])
            certified = projected(self.compiled, self.base, changes, changed,
                                  list(self.tests.values()), unknown=unknown, dispatch=self.dispatch)
            retained = selected - ((certified or set()) & candidates)
            net = self.cost(selected) - self.cost(retained) - result['cpu']
            self.import_unprofitable = self.import_unprofitable + 1 if net <= 0 else 0
            return retained, [f'checked imported state and dispatch; removed {len(selected-retained)} candidates']
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, RecursionError, TimeoutError, EOFError):
            self.close()
            self.import_unprofitable += 1
            return selected, ['imported-state evidence unavailable; retained candidates']

    def stage_refine(self, patch, selected, *, protected=()):
        from stagecheck import METHODS, load_profile, plan
        if (self.disabled or self.stages is None or not self.stages.is_file() or self.stage_unprofitable >= 2
                or self.used_cpu >= self.cpu_budget or time.monotonic() >= self.deadline):
            return selected, []
        candidates = {n for n in selected - set(protected)
                      if self.tests[n]['id'].startswith('tests/test_executor.py::TestExecutor::')
                      and self.tests[n]['id'].rsplit('::', 1)[1] in METHODS}
        if not candidates or self.cost(selected) - self.cost(selected - candidates) < 2:
            return selected, []
        try:
            if self.stage_baseline is None:
                self.stage_baseline = load_profile(self.stages)
            plans = plan(self.stage_baseline, self.base, patch, [self.tests[n]['id'].rsplit('::', 1)[1] for n in candidates])
            requested = []
            tested = set()
            for n in sorted(candidates, key=lambda n: -self.tests[n]['seconds']):
                row = plans.get(self.tests[n]['id'].rsplit('::', 1)[1])
                if row is None:
                    continue
                indices, estimate = row
                saving = self.cost(selected) - self.cost(selected - {n})
                if .8 + estimate * 1.4 < min(8, saving * .65, self.cpu_budget - self.used_cpu):
                    requested.extend(indices)
                    tested.add(n)
            if not tested:
                return selected, []
            saving = self.cost(selected) - self.cost(selected - tested)
            budget = min(10, saving * .65, self.cpu_budget - self.used_cpu)
            self.start()
            budget = min(budget, self.cpu_budget-self.used_cpu)
            if budget <= 0:
                return selected, []
            request = dict(kind='stages', patch=patch, tests=[self.tests[n]['id'] for n in sorted(tested)],
                           indices=sorted(set(requested)), cpu_limit=budget,
                           wall_limit=min(20, max(.1, self.deadline-time.monotonic())))
            self.process.stdin.write(json.dumps(request, separators=(',', ':'))+'\n')
            self.process.stdin.flush()
            result = self.read(min(22, max(.1, self.deadline-time.monotonic())))
            self.used_cpu += result['cpu']
            passed = {self.by_id[name] for name in result.get('passed', []) if name in self.by_id} & tested
            if result.get('replayed_stages') != len(request['indices']):
                passed = set()
            net = self.cost(selected) - self.cost(selected-passed) - result['cpu']
            self.stage_unprofitable = self.stage_unprofitable + 1 if net <= 0 else 0
            return selected-passed, [f'verified {len(passed)} executor tests through {len(request["indices"])} affected stages']
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, RecursionError, TimeoutError, EOFError):
            self.close()
            self.stage_unprofitable += 1
            return selected, ['stage replay unavailable; retained executor candidates']

    def task_query_refine(self, patch, selected, *, protected=(), planning_patch=None):
        from queryreplay import TEST, load_profile, plan
        from selector import apply_edit, parse_patch
        from valuecheck import validated_change

        number = self.by_id.get(TEST)
        if (self.disabled or number not in selected or number in protected
                or self.used_cpu >= self.cpu_budget or time.monotonic() >= self.deadline):
            return selected, []
        if ((not self.task_available or self.task_unprofitable >= 2)
                and (not self.pass_available or self.pass_unprofitable >= 2)):
            return selected, []
        saving = self.cost(selected) - self.cost(selected - {number})
        if saving < 8:
            return selected, []
        try:
            # Avoid decoding table states for files outside optimizer replay.
            planning_patch = patch if planning_patch is None else planning_patch
            edits = parse_patch(planning_patch)
            if not edits or any(not e.path.startswith('sqlglot/optimizer/') for e in edits):
                return selected, []
            if self.pass_available and self.pass_unprofitable < 2:
                retained, notes, attempted = self.pass_refine(patch, planning_patch, selected, number, saving)
                if attempted:
                    return retained, notes
            if not self.task_available or self.task_unprofitable >= 2:
                return selected, []
            change = None
            if len(edits) == 1:
                old = (self.base / edits[0].path).read_text()
                updated, _ = apply_edit(old, edits[0])
                change = validated_change(self.values, {edits[0].path: updated}, {edits[0].path: old})
            if len(edits) == 1 and change is None and self.behavior is not None:
                from behaviorcheck import validated_change as behavior_change
                change = behavior_change(self.behavior, {edits[0].path: updated}, {edits[0].path: old})
            if change is None and self.compiled is None:
                return selected, []
            if change is None and self.compiled is not None:
                from compiledcheck import query_reachability
                if query_reachability(self.compiled, self.base, planning_patch) is None:
                    return selected, []
            if self.task_baseline is None:
                self.task_baseline = load_profile(self.task_queries)
                self.close()  # A previously started worker did not load task data.
            request_plan = plan(self.base, planning_patch, self.values, self.task_baseline, self.behavior) if change else None
            checkpoints = {}
            if self.compiled is not None:
                from compiledcheck import query_plan
                compiled_plan = query_plan(self.compiled, self.base, planning_patch, self.task_baseline)
                if compiled_plan is not None:
                    reachable, estimate, available = compiled_plan
                    if request_plan is None:
                        checkpoints = available
                        request_plan = reachable, estimate
                    elif set(request_plan[0]) <= set(reachable):
                        indices = request_plan[0]
                        checkpoints = {str(i): available[str(i)] for i in indices if str(i) in available}
                        estimate = sum(checkpoints[str(i)][0].get('cpu', self.task_baseline['queries'][i]['cpu'])
                                       if str(i) in checkpoints else self.task_baseline['queries'][i]['cpu'] for i in indices)
                        request_plan = indices, estimate
            if request_plan is None:
                return selected, []
            indices, estimate = request_plan
            # Charge startup, checkout and state restoration as well as baseline
            # query work. A timeout or partial task never removes the test.
            budget = min(6.0, saving * 0.3, self.cpu_budget - self.used_cpu)
            if 1.0 + estimate * 1.5 > budget:
                return selected, []
            if not indices:
                return selected - {number}, ["complete baseline task comparisons unchanged"]
            self.start()
            budget = min(budget, self.cpu_budget - self.used_cpu)
            if budget <= 0:
                return selected, []
            request = {"kind": "task_queries", "patch": patch, "tests": [TEST], "indices": indices, 'checkpoints': checkpoints,
                       "cpu_limit": budget, "wall_limit": min(12, max(0.1, self.deadline - time.monotonic()))}
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            result = self.read(min(14, max(0.1, self.deadline - time.monotonic())))
            self.used_cpu += result["cpu"]
            if result.get("passed") == [TEST] and result.get("replayed_queries") == len(indices):
                self.task_unprofitable = self.task_unprofitable + 1 if result["cpu"] >= saving else 0
                return selected - {number}, [f"verified {len(indices)} affected TPC-DS queries and shared input state"]
            self.task_unprofitable += 1
            return selected, ["affected query replay incomplete or changed; retained executor"]
        except (OSError, ValueError, KeyError, TypeError, EOFError, TimeoutError, RecursionError):
            self.close()
            self.task_unprofitable += 1
            return selected, ["task replay unavailable; retained executor"]

    def pass_refine(self, patch, planning_patch, selected, number, saving):
        from passcheck import load_profile, plan
        try:
            if self.pass_baseline is None:
                self.pass_baseline = load_profile(self.passes)
                if self.pass_baseline is None:
                    self.pass_available = False
                    return selected, [], False
            candidate = plan(self.pass_baseline, self.compiled, self.base, planning_patch)
            if candidate is None:
                return selected, [], False
            indices, estimate = candidate
            budget = min(12.0, saving * 0.6, self.cpu_budget - self.used_cpu)
            # Recorded suffix costs include restoration and state comparison.
            # Add fork/import overhead and variance; incomplete work retains
            # the executor even when the actual replay exceeds this estimate.
            if 1.0 + estimate * 1.35 > budget:
                return selected, [], False
            self.start()
            budget = min(budget, self.cpu_budget - self.used_cpu)
            if budget <= 0:
                return selected, [], False
            request = dict(kind='optimizer_passes', patch=patch, tests=[self.tests[number]['id']], indices=indices,
                           cpu_limit=budget, wall_limit=min(20, max(0.1, self.deadline-time.monotonic())))
            self.process.stdin.write(json.dumps(request, separators=(',', ':'))+'\n')
            self.process.stdin.flush()
            result = self.read(min(22, max(0.1, self.deadline-time.monotonic())))
            self.used_cpu += result['cpu']
            if result.get('passed') == [self.tests[number]['id']] and result.get('replayed_passes') == len(indices):
                self.pass_unprofitable = self.pass_unprofitable + 1 if result['cpu'] >= saving else 0
                return selected-{number}, [f'verified {len(indices)} optimizer passes with exact input graphs and shared state'], True
            self.pass_unprofitable += 1
            return selected, ['optimizer pass replay changed or incomplete; retained executor'], True
        except (OSError, ValueError, KeyError, TypeError, EOFError, TimeoutError, RecursionError, SyntaxError):
            self.close()
            self.pass_unprofitable += 1
            return selected, ['optimizer pass replay unavailable; retained executor'], True

    def query_refine(self, patch, selected):
        from querycheck import METHODS, supported_patch

        if (self.disabled or self.queries is None or not self.queries.is_file()
                or self.used_cpu >= self.cpu_budget or time.monotonic() >= self.deadline):
            return selected, []
        names = {"tests/test_executor.py::TestExecutor::" + method for method in METHODS}
        candidates = {n for n in selected if self.tests[n]["id"] in names}
        saving = self.cost(selected) - self.cost(selected - candidates)
        if (not candidates or saving < 8 or self.query_backend_files is None
                or not supported_patch(self.base, patch, self.query_backend_files)):
            return selected, []
        try:
            self.start()
            request = {"kind": "queries", "patch": patch,
                       "tests": [self.tests[n]["id"] for n in sorted(candidates)],
                       "cpu_limit": min(12, self.cpu_budget - self.used_cpu),
                       "wall_limit": min(18, max(0.1, self.deadline - time.monotonic()))}
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            result = self.read(min(20, max(0.1, self.deadline - time.monotonic())))
            self.used_cpu += result["cpu"]
            passed = {self.by_id[name] for name in result["passed"] if name in self.by_id} & candidates
            return selected - passed, [f"verified {len(passed)} executor tests at query boundaries"]
        except (OSError, ValueError, KeyError, TypeError, EOFError, TimeoutError):
            self.close()
            self.disabled = True
            return selected, ["query verification unavailable; retained candidates"]

    def aggressive_refine(self, patch, selected, *, protected=()):
        """Speculate about unrun expensive tests only after a completed clean probe.

        Passing a sample does not certify the omitted tests. This separate,
        opt-in policy trades that risk against their simulated schedule cost.
        Actual failures, skips, missing reports and errors block speculation.
        """
        self.aggressive_attempted = set()
        if (not self.aggressive or self.disabled or not self.timing
                or self.aggressive_cpu >= 20 or self.used_cpu >= self.cpu_budget
                or time.monotonic() >= self.deadline):
            return selected, []
        from selector import parse_patch
        try:
            edits = parse_patch(patch)
            if (not edits or len({e.path for e in edits}) != len(edits)
                    or not any(e.path.startswith('sqlglot/') and e.path.endswith('.py') for e in edits)
                    or any(not (self.base / e.path).is_file() for e in edits)):
                return selected, []
        except (ValueError, OSError):
            return selected, []
        eligible = selected - set(protected)
        prices = self.timing['cpu']
        expensive = {n for n in eligible
                     if prices.get(self.tests[n]['id'], 0) >= self.aggressive_min_cpu}
        if not expensive:
            return selected, []
        saving = self.cost(selected) - self.cost(selected - expensive)
        cheap = sorted((n for n in eligible - expensive
                        if self.tests[n]['id'] in prices and prices[self.tests[n]['id']] <= .35),
                       key=lambda n: (prices[self.tests[n]['id']], n))
        if not cheap or saving < 4:
            return selected, []
        budget = min(2.5, saving * .35, self.cpu_budget - self.used_cpu, 20 - self.aggressive_cpu)
        if budget < 1.5:
            return selected, []
        # Include class setup/teardown when choosing up to eight cheap probes.
        # One costly class should not prevent affordable evidence elsewhere.
        candidates, owners, estimate = [], set(), 1.0
        for number in cheap:
            owner = self.tests[number]['id'].rsplit('::', 1)[0]
            price = max(.002, prices[self.tests[number]['id']])
            if owner not in owners:
                price += self.timing['setup'].get(owner, 0) + self.timing['teardown'].get(owner, 0)
            if estimate + price >= budget:
                continue
            candidates.append(number)
            owners.add(owner)
            estimate += price
            if len(candidates) == 8:
                break
        if not candidates:
            return selected, []
        started = self.used_cpu
        try:
            self.start()
            budget = min(budget, self.cpu_budget - self.used_cpu,
                         20 - self.aggressive_cpu - (self.used_cpu - started))
            if budget <= 0:
                return selected, []
            names = [self.tests[n]['id'] for n in candidates]
            request = dict(patch=patch, tests=names, cpu_limit=budget,
                           wall_limit=min(10, max(.1, self.deadline - time.monotonic())))
            self.process.stdin.write(json.dumps(request, separators=(',', ':')) + '\n')
            self.process.stdin.flush()
            self.aggressive_attempted = set(candidates)
            result = self.read(min(12, max(.1, self.deadline - time.monotonic())))
            self.used_cpu += result['cpu']
            passed = {self.by_id[name] for name in result.get('passed', []) if name in self.by_id} & set(candidates)
            clean = (result.get('completed') is True and not result.get('failed')
                     and not result.get('incomplete') and 'error' not in result
                     and set(result.get('passed', [])) == set(names))
            if clean:
                return selected - passed - expensive, [
                    f'aggressive: {len(passed)} clean probes; speculatively omitted {len(expensive)} unrun expensive tests']
            return selected - passed, [
                f'aggressive: {len(passed)} passed, {len(result.get("failed", []))} failed; '
                'probe failed or incomplete, retained expensive tests']
        except (OSError, ValueError, KeyError, TypeError, EOFError, TimeoutError):
            self.close()
            self.aggressive_cpu = 20  # No trustworthy CPU receipt from this worker.
            return selected, ['aggressive probe unavailable; retained candidates']
        finally:
            self.aggressive_cpu += self.used_cpu - started

    def refine(self, patch, selected, *, data_readers=(), mixed=False):
        if (self.disabled or not selected
                or time.monotonic() >= self.deadline or self.used_cpu >= self.cpu_budget):
            return selected, []
        prices = self.timing["cpu"] if self.timing else self.profile.get("cpu", {})
        fixture_plans = {}
        if (self.fixture_available and self.fixture_unprofitable < 2
                and ('sqlglot/optimizer/' in patch or (self.fixture_delta and 'tests/fixtures/optimizer/' in patch))
                and any(self.tests[n]['id'].startswith('tests/test_optimizer.py::TestOptimizer::') for n in selected)):
            from fixturecheck import load_profile, plan
            try:
                if self.fixture_baseline is None:
                    self.fixture_baseline = load_profile(self.fixtures)
                fixture_plans = plan(self.fixture_baseline, self.base, patch,
                                     [self.tests[n]['id'] for n in selected], allow_delta=self.fixture_delta)
            except (OSError, ValueError, KeyError, TypeError, SyntaxError, RecursionError):
                fixture_plans = {}
        if fixture_plans:
            prices = dict(prices)
            prices.update({name: row['cpu'] for name, row in fixture_plans.items()})
        attempted, notes = set(self.aggressive_attempted), []
        started = self.used_cpu
        for _ in range(3 if self.progressive else 1):
            if (self.disabled or self.used_cpu - started >= 12 or self.used_cpu >= self.cpu_budget
                    or time.monotonic() >= self.deadline):
                break
            eligible = selected - attempted
            if mixed:
                eligible -= {n for n in data_readers if self.tests[n]['id'] not in fixture_plans}
            self.last_probe = set()
            selected, checked = self._probe(patch, selected, eligible, prices, fixture_plans)
            notes.extend(checked)
            if not self.last_probe:
                break
            attempted.update(self.last_probe)
        return selected, notes

    def _probe(self, patch, selected, eligible, prices, fixture_plans):
        # Simulated duration does not predict verification CPU. Use measured
        # real method and fixture costs, and retain anything not completed.
        candidates = {n for n in eligible if self.tests[n]["id"] in prices
                      and prices[self.tests[n]["id"]] <= 4.0}
        expanded = {n for n in eligible if self.tests[n]['id'] in prices
                    and prices[self.tests[n]['id']] <= 10.0}
        progressive = False
        if self.timing:
            # A large bundle's whole saving requires many tests to pass. Price
            # smaller critical bundles as well; unrun tests are always retained.
            ranked = sorted(candidates, key=lambda n: (-self.tests[n]["seconds"], prices[self.tests[n]["id"]], n))
            bundles = [(candidates, False)] + [({n}, False) for n in ranked[:8]]
            bundles += [(set(ranked[:count]), False) for count in (2, 4, 8) if len(ranked) > count]
            if self.progressive and self.progressive_unprofitable < 2:
                cheapest = sorted(expanded, key=lambda n: (prices[self.tests[n]['id']], n))
                counts = {*range(1, min(16, len(cheapest)) + 1), 32, 64, len(cheapest)}
                bundles += [(set(cheapest[:count]), True) for count in sorted(counts)
                            if 0 < count <= len(cheapest)]
            choices = []
            current_cost = self.cost(selected)
            seen = set()
            for bundle, broader in bundles:
                if not bundle:
                    continue
                signature = (frozenset(bundle), broader)
                if signature in seen:
                    continue
                seen.add(signature)
                has_fixtures = any(self.tests[n]['id'] in fixture_plans for n in bundle)
                if not broader and self.unprofitable >= 2 and not has_fixtures:
                    continue
                owners = {self.tests[n]["id"].rsplit("::", 1)[0] for n in bundle}
                estimate = (self.progressive_overhead if broader else 1.0) + sum(
                    max(0.002, prices[self.tests[n]["id"]]) for n in bundle)
                for owner in owners:
                    teardown = self.timing['teardown'].get(owner, 0)
                    methods = [self.tests[n]['id'] for n in bundle
                               if self.tests[n]['id'].rsplit('::', 1)[0] == owner]
                    if all(name in fixture_plans for name in methods):
                        # The pristine class-wide shutdown includes pool work
                        # accumulated across its methods. Restricted replay
                        # starts a fresh pool and reaps only its selected tasks.
                        # Preserve a fixed shutdown allowance; mixed bundles
                        # retain the original estimate for untrimmed methods.
                        share = max(fixture_plans[name]['task_share'] for name in methods)
                        teardown = min(teardown, 0.25 + teardown * share)
                    estimate += self.timing['setup'].get(owner, 0) + teardown
                saving = current_cost - self.cost(selected - bundle)
                # The broader policy treats failure as a property of the edit;
                # the probability only steers probes, never certifies a pass.
                probability = 0.7 if broader else 0.7 ** min(len(bundle), 4)
                deltas = [fixture_plans.get(self.tests[n]['id'], {}) for n in bundle]
                if any(p.get('fixture_delta') for p in deltas):
                    probability = (1.0 if all(p.get('fixture_delta') and not p['indices'] for p in deltas)
                                   else min(probability, 0.5))
                expected = saving * probability - estimate
                if expected > 0.1 and estimate < min(12.0, self.cpu_budget - self.used_cpu):
                    choices.append((expected, saving, estimate, bundle, broader))
            if not choices:
                return selected, []
            # Keep useful established probes ahead of exploratory groups. Their
            # failure counters are independent, so exploration cannot suppress
            # later established probes after two unprofitable results.
            established = [choice for choice in choices if not choice[4]]
            _, possible_saving, estimate, candidates, progressive = max(established or choices,
                key=lambda choice: (choice[0], -len(choice[3])))
        else:
            possible_saving = self.cost(selected) - self.cost(selected - candidates)
            estimate = 3.0 + sum(max(0.002, prices[self.tests[n]["id"]]) for n in candidates)
        if not candidates or possible_saving < estimate + 0.5:
            return selected, []
        budget = min(self.cpu_budget - self.used_cpu, 12.0 if progressive else 8.0, max(3.5, estimate * 1.5))
        try:
            use_warm = False
            if self.warm and not fixture_plans and not (set(candidates) & getattr(self, 'protected', set())):
                from warmcheck import plan
                use_warm = plan(self.base, patch, self.compiled, self.warm) is not None
            if use_warm:
                self.start(warm=True)
            else:
                self.start()
            budget = min(budget, self.cpu_budget - self.used_cpu)
            if budget <= 0:
                return selected, []
            owners = {self.tests[n]['id'].rsplit('::', 1)[0] for n in candidates}
            ordered_owners = sorted(owners, key=lambda owner: (
                self.timing['setup'].get(owner, 0) if self.timing else 0,
                min(prices[self.tests[n]['id']] for n in candidates if self.tests[n]['id'].rsplit('::', 1)[0] == owner), owner))
            ordered = [n for owner in ordered_owners for n in sorted(candidates, key=lambda n: (prices[self.tests[n]['id']], n))
                       if self.tests[n]['id'].rsplit('::', 1)[0] == owner]
            self.last_probe = set(candidates)
            request = {"patch": patch, "tests": [self.tests[n]["id"] for n in ordered],
                       "fixture_plans": {self.tests[n]['id']: fixture_plans[self.tests[n]['id']]
                                         for n in candidates if self.tests[n]['id'] in fixture_plans},
                       "cpu_limit": budget, "wall_limit": min(20 if progressive else 15, max(2, self.deadline - time.monotonic()))}
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            result = self.read(min(22 if progressive else 20, max(0.1, self.deadline - time.monotonic())))
            self.used_cpu += result["cpu"]
            if use_warm and not result.get('passed') and self.used_cpu < self.cpu_budget:
                # Warm eligibility or a live alias can fail independently of
                # the test. Retry once against freshly imported patched files.
                self.start()
                request['cpu_limit'] = min(budget, self.cpu_budget-self.used_cpu)
                self.process.stdin.write(json.dumps(request, separators=(',', ':'))+'\n')
                self.process.stdin.flush()
                cold = self.read(min(22, max(.1, self.deadline-time.monotonic())))
                self.used_cpu += cold['cpu']
                result = dict(cold, cpu=result['cpu']+cold['cpu'])
            if self.timing and not request['fixture_plans']:
                # Whole-suite method timings exclude checkout, collection and
                # cold imports. Price further exploration with their observed
                # cost; the established planner keeps its existing estimates.
                fixed = self.progressive_overhead if progressive else 1.0
                self.progressive_overhead = max(self.progressive_overhead, result['cpu'] - (estimate - fixed))
            passed = {self.by_id[name] for name in result["passed"] if name in self.by_id} & candidates
            net = self.cost(selected) - self.cost(selected - passed) - result["cpu"]
            if progressive:
                self.progressive_unprofitable = self.progressive_unprofitable + 1 if net <= 0 else 0
            elif request['fixture_plans']:
                self.fixture_unprofitable = self.fixture_unprofitable + 1 if net <= 0 else 0
            else:
                self.unprofitable = self.unprofitable + 1 if net <= 0 else 0
            trimmed = sum(self.tests[n]['id'] in request['fixture_plans'] for n in passed)
            note = f" ({trimmed} with affected fixture cases)" if trimmed else ''
            return selected - passed, [f"verified {len(passed)} passing candidates" + note]
        except (OSError, ValueError, KeyError, TypeError, EOFError, TimeoutError):
            self.close()
            self.disabled = True
            return selected, ["verification unavailable; retained candidates"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--task-queries", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument('--passes', type=Path)
    parser.add_argument('--stages', type=Path)
    parser.add_argument('--warm', action='store_true')
    parser.add_argument('--imported', type=Path)
    args = parser.parse_args()
    if args.serve:
        serve(args.source, args.queries, args.task_queries, args.fixtures, args.passes, args.stages, args.warm, args.imported)
