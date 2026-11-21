"""Observe primitive comparisons and real executor tasks on the pristine suite.

No patches or failure labels are accepted. Spawned workers start tracing before
source imports, and class fixture observations belong to every method of a class.
"""

from __future__ import annotations

import argparse
import ast
import atexit
import base64
import concurrent.futures
import gzip
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparisoncheck import REVISION, location, source_hash
from queryreplay import TEST, compact_states, result_digest, state_digest
from tools.build_comparison_profile import Recorder as TypeRecorder, cpu_seconds
from valuecheck import OPS, comparison_site


CURRENT = None
LIMIT = 128


def domain(site):
    return {"type": site["type"], "kind": "range" if site["type"] == "int" else "exact",
            "values": [], "unknown": False}


def add(row, value):
    if row["unknown"]:
        return
    if type(value).__name__ != row["type"] or type(value) not in (int, str):
        row["unknown"] = True
    elif row["kind"] == "range":
        if not row["values"]:
            row["values"] = [value, value]
        elif value < row["values"][0]:
            row["values"][0] = value
        elif value > row["values"][1]:
            row["values"][1] = value
    elif value not in row["values"]:
        if len(row["values"]) >= LIMIT or len(value) > 1024:
            row["unknown"] = True
        else:
            row["values"].append(value)


def combine(a, b):
    a["unknown"] |= b["unknown"]
    for value in b["values"]:
        add(a, value)


class Recorder(TypeRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sites = {}
        self.rows = defaultdict(dict)
        self.query_rows = defaultdict(dict)
        self.query = None
        self.outside = set()
        self.tasks = {}
        self.states = {}
        self.cache = {}
        self.query_total = 0

    def comparison(self, key, left, right):
        site = self.sites[key]
        tag = self.current, self.query
        cached = self.cache.get(key)
        if cached is None or cached[0] != tag:
            row = self.rows[key].get(self.current)
            if row is None:
                row = self.rows[key][self.current] = domain(site)
            query_row = None
            is_executor = bool(self.current.isdecimal() and self.tests and self.tests[int(self.current)] == TEST)
            is_executor |= os.environ.get("CI_VALUES_TEST") == TEST and not self.tests
            if is_executor:
                if self.query is None:
                    self.outside.add(key)
                else:
                    query_row = self.query_rows[key].setdefault(self.query, domain(site))
            cached = self.cache[key] = tag, row, query_row
        _, row, query_row = cached
        value = left if site["side"] == "left" else right
        add(row, value)
        if query_row is not None:
            add(query_row, value)
        try:
            result = OPS[site["op"]](left, right)
        except BaseException:
            row["unknown"] = True
            if query_row is not None:
                query_row["unknown"] = True
            raise
        if type(result) is not bool:
            row["unknown"] = True
            if query_row is not None:
                query_row["unknown"] = True
        return result

    def data(self):
        return dict(revision=REVISION, version=1, kind="literal_values", tests=self.tests,
                    sources=self.sources, sites=self.sites, rows=dict(self.rows),
                    complete=sorted(self.complete), cpu=self.cpu, bad=sorted(self.bad),
                    collection_complete=self.collection_complete,
                    query_rows={key: {str(i): row for i, row in rows.items()} for key, rows in self.query_rows.items()},
                    outside=sorted(self.outside), tasks={str(i): task for i, task in self.tasks.items()},
                    states=self.states)

    def merge(self, data):
        for path, digest in data["sources"].items():
            if path in self.sources and self.sources[path] != digest:
                raise ValueError("worker source mismatch")
            self.sources[path] = digest
        for key, site in data["sites"].items():
            if key in self.sites and self.sites[key] != site:
                raise ValueError("worker comparison mismatch")
            self.sites[key] = site
        for key, contexts in data["rows"].items():
            for context, row in contexts.items():
                combine(self.rows[key].setdefault(context, domain(self.sites[key])), row)
        for key, queries in data["query_rows"].items():
            for index, row in queries.items():
                combine(self.query_rows[key].setdefault(int(index), domain(self.sites[key])), row)
        for index, task in data["tasks"].items():
            if int(index) in self.tasks:
                raise ValueError("duplicate executor query")
            self.tasks[int(index)] = task
        self.states.update(data["states"])
        self.outside.update(data["outside"])
        self.bad.update(data["bad"])

    def set_context(self, context, test=None):
        self.current = context
        os.environ["CI_VALUES_CONTEXT"] = context
        os.environ["CI_VALUES_TEST"] = test or ""

    def pytest_runtest_logstart(self, nodeid, location):
        self.set_context(str(self.by_id[nodeid]), nodeid)
        self.started = cpu_seconds()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        previous = self.current
        self.set_context("class:" + item.nodeid.rsplit("::", 1)[0])
        try:
            yield
        finally:
            self.set_context(previous, item.nodeid)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        previous = self.current
        self.set_context("class:" + item.nodeid.rsplit("::", 1)[0])
        try:
            yield
        finally:
            self.set_context(previous, item.nodeid)

    def pytest_runtest_logfinish(self, nodeid, location):
        context = str(self.by_id[nodeid])
        for owner, pids in list(self.expected_children.items()):
            for pid in pids:
                try:
                    data = json.loads((self.directory / f"end-{pid}.json").read_text())
                    if data["context"] != owner or not data["finished"]:
                        raise ValueError("unfinished worker")
                    self.merge(data)
                except (OSError, ValueError, KeyError, TypeError):
                    self.bad.add(owner)
            del self.expected_children[owner]
        self.cpu[nodeid] = round(cpu_seconds() - self.started, 6)
        if context not in self.bad and self.phases[context] == {"setup", "call", "teardown"}:
            self.complete.add(int(context))
        self.set_context("startup")

    def export(self):
        complete = set(self.complete)
        for owner in self.bad:
            if owner == "startup":
                complete.clear()
            elif owner.startswith("class:"):
                complete.difference_update(i for i, name in enumerate(self.tests) if name.startswith(owner[6:] + "::"))
        rows = {}
        for key, contexts in self.rows.items():
            expanded = {}
            for context, row in contexts.items():
                owners = [str(i) for i, name in enumerate(self.tests) if name.startswith(context[6:] + "::")] if context.startswith("class:") else [context]
                for owner in owners:
                    combine(expanded.setdefault(owner, domain(self.sites[key])), row)
            rows[key] = expanded
        profile = self.data()
        for key in ("query_rows", "outside", "tasks", "states", "bad"):
            del profile[key]
        profile.update(complete=sorted(complete), rows=rows)
        return profile


class Instrument(ast.NodeTransformer):
    def __init__(self, path, recorder):
        self.path, self.recorder = path, recorder

    def visit_Compare(self, node):
        self.generic_visit(node)
        site = comparison_site(node)
        if site is None:
            return node
        key = location(self.path, node)
        self.recorder.sites[key] = site
        call = ast.Call(ast.Name("_ci_value_observe", ast.Load()),
                        [ast.Constant(key), node.left, node.comparators[0]], [])
        return ast.copy_location(call, node)


class Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, recorder):
        super().__init__(fullname, path)
        self.recorder = recorder

    def get_code(self, fullname):
        source = self.get_data(self.path).decode("utf-8")
        path = Path(self.path).resolve().relative_to(self.recorder.root).as_posix()
        self.recorder.sources[path] = source_hash(source)
        tree = Instrument(path, self.recorder).visit(ast.parse(source))
        return compile(ast.fix_missing_locations(tree), self.path, "exec")

    def exec_module(self, module):
        module.__dict__["_ci_value_observe"] = self.recorder.comparison
        super().exec_module(module)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, recorder):
        self.recorder = recorder

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin.endswith(".py"):
            if not Path(spec.origin).resolve().is_relative_to(self.recorder.root):
                raise ValueError("source imported outside baseline")
            spec.loader = Loader(fullname, spec.origin, self.recorder)
        return spec


def install(recorder):
    global CURRENT
    if any(name == "sqlglot" or name.startswith("sqlglot.") for name in sys.modules):
        raise ValueError("source imported before tracing")
    CURRENT = recorder
    sys.meta_path.insert(0, Finder(recorder))
    sys.addaudithook(recorder.audit)


def measured_task(context, test, index, payload):
    recorder = CURRENT
    recorder.current = context
    os.environ["CI_VALUES_TEST"] = test
    recorder.query = index if test == TEST else None
    function, args, kwargs = pickle.loads(payload)
    started = time.process_time()
    if recorder.query is not None:
        from tests import test_executor as driver
        if function is not driver.mp_execute:
            raise ValueError("unexpected query function")
        state = pickle.dumps((driver._schema, driver._tables), protocol=5)
        digest = hashlib.sha256(state).hexdigest()
        recorder.states[digest] = base64.b64encode(state).decode()
    try:
        result = function(*args, **kwargs)
        if recorder.query is not None:
            recorder.tasks[index] = {"task": base64.b64encode(payload).decode(), "state": digest,
                                     "result": result_digest(result),
                                     "after": state_digest((driver._schema, driver._tables)),
                                     "cpu": time.process_time() - started}
        return result
    finally:
        recorder.query = None


def child_bootstrap():
    arguments = getattr(sys, "orig_argv", sys.argv)
    if any("from multiprocessing.resource_tracker import main" in arg for arg in arguments):
        return
    directory = Path(os.environ["CI_VALUES_OUTPUT"])
    context = os.environ["CI_VALUES_CONTEXT"]
    recorder = Recorder(os.environ["CI_VALUES_ROOT"], directory=directory, context=context)
    install(recorder)
    (directory / f"start-{os.getpid()}.json").write_text(json.dumps({"context": context}))

    def save():
        data = recorder.data()
        data.update(context=context, finished=True)
        (directory / f"end-{os.getpid()}.json").write_text(json.dumps(data, separators=(",", ":")))

    atexit.register(save)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries-out", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, default=Path(__file__).resolve().parents[1] / "maps/baseline.json.gz")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    root, output, queries_output = args.source.resolve(), args.out.resolve(), args.queries_out.resolve()
    catalog = json.loads(args.tests.resolve().read_text())["tests"]
    with gzip.open(args.baseline_map.resolve(), "rt") as stream:
        baseline = json.load(stream)
    expected = {path: source for path, source in baseline["sources"].items() if path.startswith("sqlglot/")}
    actual = {path.relative_to(root).as_posix() for path in (root / "sqlglot").rglob("*.py")}
    if (baseline["revision"] != REVISION or actual != set(expected)
            or any((root / path).read_text() != source for path, source in expected.items())
            or os.environ.get("PYTHONHASHSEED") != "0"):
        raise ValueError("pristine source and PYTHONHASHSEED=0 required")
    selected = args.only if args.only is not None else [test["id"] for test in catalog]
    if not set(selected) <= {test["id"] for test in catalog}:
        raise ValueError("unknown tests")
    with tempfile.TemporaryDirectory(prefix="ci-value-profile-") as tmp:
        directory = Path(tmp)
        (directory / "sitecustomize.py").write_text("from build_value_profile import child_bootstrap\nchild_bootstrap()\n")
        os.environ.update(SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="2", PYTHONDONTWRITEBYTECODE="1",
                          OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_MAX_THREADS="1",
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CI_VALUES_ROOT=str(root),
                          CI_VALUES_OUTPUT=str(directory), CI_VALUES_CONTEXT="startup", CI_VALUES_TEST="",
                          PYTHONPATH=os.pathsep.join([str(directory), str(Path(__file__).resolve().parent), str(root)]))
        sys.path.insert(0, str(root))
        os.chdir(root)
        recorder = Recorder(root, catalog, directory)
        install(recorder)
        original_pool = concurrent.futures.ProcessPoolExecutor
        from build_value_profile import measured_task as task

        class MeasuredPool(original_pool):
            def submit(self, fn, /, *a, **kw):
                test = os.environ.get("CI_VALUES_TEST", "")
                if test == TEST:
                    recorder.query_total += 1
                index = getattr(self, "query_index", 0)
                self.query_index = index + 1
                return super().submit(task, recorder.current, test, index, pickle.dumps((fn, a, kw), protocol=5))

            def __exit__(self, *a):
                recorder.expected_children[recorder.current].update(self._processes or {})
                return super().__exit__(*a)

        concurrent.futures.ProcessPoolExecutor = MeasuredPool
        status = pytest.main([*selected, "-q", "--maxfail=1", "-p", "no:cacheprovider",
                              "-p", "pytest_timeout", "--timeout=300"], plugins=[recorder])
        if status:
            raise SystemExit(status)
        profile = recorder.export()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_values = output.with_name(output.name + ".building")
        with gzip.GzipFile(filename=str(temporary_values), mode="wb", mtime=0) as stream:
            stream.write(json.dumps(profile, separators=(",", ":"), sort_keys=True).encode())
        temporary_values.replace(output)
        queries = {i: dict(task, task=base64.b64decode(task["task"])) for i, task in recorder.tasks.items()}
        query_profile = {"revision": REVISION, "version": 1, "kind": "executor_tasks", "test": TEST,
                         "hashseed": "0", "driver": hashlib.sha256((root / "tests/test_executor.py").read_bytes()).hexdigest(),
                         "sources": recorder.sources, "rows": dict(recorder.query_rows), "outside": recorder.outside,
                         "queries": queries, "states": {key: base64.b64decode(state) for key, state in recorder.states.items()},
                         "complete": recorder.by_id[TEST] in profile["complete"]
                                     and recorder.query_total > 0
                                     and set(queries) == set(range(recorder.query_total))}
        query_profile = compact_states(query_profile)
        queries_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_queries = queries_output.with_name(queries_output.name + ".building")
        with gzip.GzipFile(filename=str(temporary_queries), mode="wb", mtime=0) as stream:
            pickle.dump(query_profile, stream, protocol=5)
        temporary_queries.replace(queries_output)
        print(f"Literal values: {len(recorder.sites)} sites, {len(profile['complete'])}/{len(catalog)} complete; "
              f"executor tasks: {len(queries)}", file=sys.stderr)
        raise SystemExit(status)


if __name__ == "__main__":
    main()
