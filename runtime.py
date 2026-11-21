"""Bounded verification on the real patched source; uncertainty retains tests."""

from __future__ import annotations

import argparse
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
                if any(part in (".git", "__pycache__", ".pytest_cache", ".mypy_cache") for part in relative.parts):
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

    def pytest_runtest_logreport(self, report):
        if report.failed or report.skipped:
            self.bad.add(report.nodeid)
        if report.passed:
            self.phases.setdefault(report.nodeid, set()).add(report.when)

    def passed(self):
        return sorted(name for name, phases in self.phases.items()
                      if phases == {"setup", "call", "teardown"} and name not in self.bad)


def run_tests(root, names, cpu_limit):
    import pytest

    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(PYTHONPATH=str(root), SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="2",
                      PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_MAX_THREADS="1")
    verdicts = Verdicts()

    def timed_out(signum, frame):
        raise TimeoutError("verification CPU budget")

    signal.signal(signal.SIGPROF, timed_out)
    signal.setitimer(signal.ITIMER_PROF, cpu_limit)
    try:
        pytest.main([*names, "-q", "--maxfail=1", "--import-mode=importlib", "--rootdir=" + str(root),
                     "-p", "no:cacheprovider", "-p", "pytest_timeout",
                     "--timeout=5"], plugins=[verdicts])
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    return {"passed": verdicts.passed()}


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def serve(base, queries=None):
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      NUMEXPR_MAX_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    import pytest  # Warm third-party imports only; patched packages load in children.

    query_baseline = None
    if queries is not None and queries.is_file():
        from querycheck import load_baseline
        query_baseline = load_baseline(queries)

    print(json.dumps({"ready": True, "cpu": cpu_seconds()}), flush=True)
    for line in sys.stdin:
        started = cpu_seconds()
        try:
            request = json.loads(line)
            if any("test_executor.py" in name or "test_optimizer.py" in name for name in request.get("tests", [])):
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
                        if request.get("kind") == "queries" and query_baseline is not None:
                            from querycheck import run_boundaries
                            signal.signal(signal.SIGPROF, lambda *args: (_ for _ in ()).throw(TimeoutError("query CPU budget")))
                            signal.setitimer(signal.ITIMER_PROF, request["cpu_limit"])
                            result = run_boundaries(tree.root, baseline=query_baseline,
                                                    methods=[name.rsplit("::", 1)[-1] for name in request["tests"]],
                                                    changed_files=[edit.path for edit in tree.edits])
                        else:
                            result = run_tests(tree.root, request["tests"], request["cpu_limit"])
                        payload = json.dumps(result).encode()
                        with os.fdopen(write_fd, "wb") as output:
                            output.write(payload)
                    except BaseException:
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
                        if len(payload) > 2 * 1024**2:
                            os.killpg(os.getpgrp(), signal.SIGKILL)
                os.close(read_fd)
                os.waitpid(child, 0)
                result = json.loads(payload) if payload else {"passed": []}
            result["cpu"] = cpu_seconds() - started
        except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
            result = {"passed": [], "cpu": cpu_seconds() - started, "error": type(error).__name__}
        print(json.dumps(result, separators=(",", ":")), flush=True)


class Verifier:
    def __init__(self, base, profile, tests, schedule, *, queries=None, cpu_budget=45, wall_budget=135):
        self.base = Path(base)
        self.profile = profile or {}
        self.tests = {test["number"]: test for test in tests}
        self.by_id = {test["id"]: test["number"] for test in tests}
        self.schedule = schedule
        self.process = None
        self.used_cpu = 0.0
        self.cpu_budget = cpu_budget
        self.deadline = time.monotonic() + wall_budget
        self.disabled = not self.base.is_dir()
        self.unprofitable = 0
        self.queries = Path(queries) if queries is not None else None
        self.query_backend_files = None
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
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
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

    def start(self):
        if self.process is None:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--serve", "--source", str(self.base.resolve())]
            if self.queries is not None:
                command.extend(["--queries", str(self.queries.resolve())])
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, start_new_session=True,
                env={**os.environ, "PYTHON_CPU_COUNT": "2", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                     "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1"},
            )
            self.used_cpu += self.read(10)["cpu"]

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

    def refine(self, patch, selected):
        if (self.disabled or self.unprofitable >= 2 or not selected
                or time.monotonic() >= self.deadline or self.used_cpu >= self.cpu_budget):
            return selected, []
        candidates = {n for n in selected if self.tests[n]["seconds"] < 3
                      and self.tests[n]["id"] in self.profile.get("cpu", {})}
        possible_saving = self.cost(selected) - self.cost(selected - candidates)
        estimate = 3.0 + sum(max(0.002, self.profile["cpu"][self.tests[n]["id"]]) for n in candidates)
        if not candidates or possible_saving < estimate + 0.5:
            return selected, []
        budget = min(self.cpu_budget - self.used_cpu, 8.0, max(3.5, estimate * 1.5))
        try:
            self.start()
            request = {"patch": patch, "tests": [self.tests[n]["id"] for n in sorted(candidates)],
                       "cpu_limit": budget, "wall_limit": min(15, max(2, self.deadline - time.monotonic()))}
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            result = self.read(min(20, max(0.1, self.deadline - time.monotonic())))
            self.used_cpu += result["cpu"]
            passed = {self.by_id[name] for name in result["passed"] if name in self.by_id} & candidates
            net = self.cost(selected) - self.cost(selected - passed) - result["cpu"]
            self.unprofitable = self.unprofitable + 1 if net <= 0 else 0
            return selected - passed, [f"verified {len(passed)} passing candidates"]
        except (OSError, ValueError, KeyError, TypeError, EOFError, TimeoutError):
            self.close()
            self.disabled = True
            return selected, ["verification unavailable; retained candidates"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--queries", type=Path)
    args = parser.parse_args()
    if args.serve:
        serve(args.source, args.queries)
