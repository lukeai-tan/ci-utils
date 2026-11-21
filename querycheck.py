"""Record pristine executor boundaries and verify matching patched computations."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
import types
import unittest
from pathlib import Path


REVISION = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
METHODS = ("test_execute_tpch", "test_execute_tpcds")


class SerialFuture(concurrent.futures.Future):
    def __init__(self, function, args, kwargs, observer=None, index=0):
        super().__init__()
        self.work = (function, args, kwargs)
        self.observer = observer
        self.index = index

    def result(self, timeout=None):
        if not self.done():
            function, args, kwargs = self.work
            previous = sys.getprofile()
            try:
                if self.observer is not None:
                    self.observer.begin(self.index)
                    if self.observer.recording is not None:
                        sys.setprofile(self.observer.trace)
                self.set_result(pickle.loads(pickle.dumps(function(*args, **kwargs))))
            except BaseException as error:
                self.set_exception(error)
            finally:
                sys.setprofile(previous)
                if self.observer is not None:
                    self.observer.end()
        return super().result(timeout)


class SerialPool:
    stage_observer = None

    def __init__(self, initializer=None, initargs=(), **kwargs):
        self.index = 0
        if initializer:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, function, *args, **kwargs):
        future = SerialFuture(*pickle.loads(pickle.dumps((function, args, kwargs))),
                              observer=self.stage_observer, index=self.index)
        self.index += 1
        return future

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def freeze(value, seen=None):
    """Canonicalize observable data; reject unsupported objects instead of guessing."""
    from sqlglot import exp
    from sqlglot.planner import Step

    if seen is None:
        seen = {}
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, exp.DType):
        return ["dtype", value.value]
    if isinstance(value, types.ModuleType):
        return ["module", value.__name__]
    if isinstance(value, type):
        return ["class", value.__module__, value.__qualname__]
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        code = value.__code__ if isinstance(value, types.FunctionType) else None
        return ["function", value.__module__, value.__qualname__, freeze_code(code) if code else None]
    if isinstance(value, types.MethodType):
        return ["method", freeze(value.__func__), type(value.__self__).__module__, type(value.__self__).__qualname__]
    identity = id(value)
    if isinstance(value, exp.Expr):
        if identity in seen:
            return ["ref", seen[identity]]
        seen[identity] = len(seen)
        arguments = []
        for key, arg in value.args.items():
            children = arg if isinstance(arg, list) else [arg]
            links = [(child.parent is value, child.arg_key, child.index)
                     if isinstance(child, exp.Expr) else None for child in children]
            arguments.append([key, freeze(arg, seen), links])
        return ["expression", type(value).__module__, type(value).__qualname__, arguments,
                freeze(value._type, seen), freeze(value.comments, seen), freeze(value._meta, seen)]
    if isinstance(value, Step):
        fields = {key: item for key, item in vars(value).items() if key not in ("dependencies", "dependents")}
        dependencies = sorted((freeze(dep) for dep in value.dependencies), key=lambda item: json.dumps(item, sort_keys=True))
        if any(value not in dep.dependents for dep in value.dependencies):
            raise ValueError("inconsistent plan graph")
        return ["step", type(value).__module__, type(value).__qualname__, freeze(fields), dependencies]
    if isinstance(value, dict):
        return ["dict", [[freeze(key, seen), freeze(item, seen)] for key, item in value.items()]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [freeze(item, seen) for item in value]]
    if isinstance(value, (set, frozenset)):
        return [type(value).__name__, sorted((freeze(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))]
    raise TypeError("unsupported boundary value: " + type(value).__name__)


def freeze_code(code):
    if code is None:
        return None
    constants = []
    for value in code.co_consts:
        constants.append(freeze_code(value) if isinstance(value, types.CodeType) else freeze(value))
    return [code.co_code.hex(), list(code.co_names), list(code.co_varnames), constants,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags]


class StageObserver:
    """Reject divergent affected stages; matching stages never certify a pass."""

    def __init__(self, root, *, recording=None, baseline=None, changed_files=()):
        from sqlglot.optimizer.optimizer import RULES, optimize

        self.root = str(root) + os.sep
        self.recording = recording
        self.baseline = baseline or {}
        self.changed = set(changed_files)
        self.optimizer = optimize.__code__
        self.rules = {rule.__code__ for rule in RULES}
        self.method = None
        self.query = None
        self.index = 0
        self.frames = {}
        self.checked = 0
        self.monitor_id = None

    def begin(self, query):
        self.query = query
        self.index = 0
        self.frames.clear()

    def end(self):
        self.query = None

    def open(self):
        if self.recording is not None:
            return True
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:
            return False
        identifier = next((i for i in range(5, -1, -1) if monitoring.get_tool(i) is None), None)
        if identifier is None:
            return False
        monitoring.use_tool_id(identifier, "query-stage-rejection")
        self.monitor_id = identifier
        monitoring.register_callback(identifier, monitoring.events.PY_START, self.enter)
        monitoring.register_callback(identifier, monitoring.events.PY_RETURN, self.leave)
        for code in self.rules:
            monitoring.set_local_events(identifier, code, monitoring.events.PY_START | monitoring.events.PY_RETURN)
        return True

    def close(self):
        if self.monitor_id is not None:
            for code in self.rules:
                sys.monitoring.set_local_events(self.monitor_id, code, 0)
            sys.monitoring.register_callback(self.monitor_id, sys.monitoring.events.PY_START, None)
            sys.monitoring.register_callback(self.monitor_id, sys.monitoring.events.PY_RETURN, None)
            sys.monitoring.free_tool_id(self.monitor_id)
            self.monitor_id = None

    def enter(self, code, offset):
        if self.query is not None:
            self.trace(sys._getframe(1), "call", None)

    def leave(self, code, offset, value):
        if self.query is not None:
            self.trace(sys._getframe(1), "return", value)

    def trace(self, frame, event, value):
        if event == "call":
            if (frame.f_code in self.rules and frame.f_back is not None
                    and frame.f_back.f_code is self.optimizer):
                key = (self.method, self.query, self.index)
                self.index += 1
                self.frames[id(frame)] = [key, set(), frame.f_code.co_name]
            if self.recording is not None and self.frames:
                filename = frame.f_code.co_filename
                if filename.startswith(self.root + "sqlglot/"):
                    for _, files, _ in self.frames.values():
                        files.add(filename[len(self.root):])
        elif event == "return" and id(frame) in self.frames:
            key, files, name = self.frames.pop(id(frame))
            reference = self.baseline.get(key)
            if self.recording is None and (reference is None or not self.changed.intersection(reference[0])):
                return
            # This is an additional rejection gate. Input, shared-table state,
            # hash caches and final results still undergo the full verifier.
            digest = hashlib.sha256(json.dumps(freeze(value), separators=(",", ":"),
                                               ensure_ascii=True).encode()).hexdigest()
            if self.recording is not None:
                self.recording[key] = (sorted(files), name, digest)
            elif reference[1] != name or reference[2] != digest:
                raise KeyboardInterrupt("affected optimizer stage changed")
            else:
                self.checked += 1


def boundary(executor, plan):
    from sqlglot import exp
    from sqlglot.executor.table import Table
    from sqlglot.planner import Step

    checked = set()

    def validate_hashes(value):
        if id(value) in checked:
            return
        checked.add(id(value))
        if isinstance(value, exp.Expr):
            originals = list(value.walk())
            copies = list(value.copy().walk())
            for node in copies:
                node._hash = None
            hash(copies[0])
            for original, copied in zip(originals, copies):
                checked.add(id(original))
                if original._hash is not None and original._hash != copied._hash:
                    raise ValueError("inconsistent expression hash cache")
        elif isinstance(value, Step):
            for key, item in vars(value).items():
                if key != "dependents":
                    validate_hashes(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                validate_hashes(key)
                validate_hashes(item)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                validate_hashes(item)

    validate_hashes(plan.expression)
    validate_hashes(plan.ctes)
    validate_hashes(plan.root)

    tables = []
    for name, table in executor.tables.mapping.items():
        if not isinstance(table, Table):
            raise TypeError("unsupported table")
        # Input CSV tables are fixed and the frontend receives no rows. Include
        # columns and a pickle digest of their complete typed data at the boundary.
        tables.append([name, list(table.columns), hashlib.sha256(pickle.dumps(table.rows, protocol=5)).hexdigest()])
    payload = [freeze(plan.expression), freeze(plan.ctes), freeze(plan.root), tables,
               freeze(executor.env), type(executor.generator).__module__, type(executor.generator).__qualname__]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def supported_patch(base, patch, backend_files=()):
    from selector import apply_edit, parse_patch, same_ast

    prefixes = ("sqlglot/optimizer/", "sqlglot/parsers/")
    allowed_files = {"sqlglot/parser.py"}
    forbidden = {"eval", "exec", "compile", "globals", "locals", "setattr", "delattr", "__import__", "getattr",
                 "open", "os", "sys", "builtins", "importlib"}
    try:
        for edit in parse_patch(patch):
            # The executor may call optimizer helpers again for subqueries. A
            # matching input cannot certify edits to code behind the boundary.
            if edit.path in backend_files:
                return False
            if not edit.path.endswith(".py") or not (edit.path.startswith(prefixes) or edit.path in allowed_files):
                return False
            old = (Path(base) / edit.path).read_text()
            updated, _ = apply_edit(old, edit)
            old_tree, new_tree = ast.parse(old), ast.parse(updated)
            if same_ast(old_tree, new_tree):
                continue
            if not same_ast(old_tree, new_tree, structure=True):
                return False
            def outer_functions(tree):
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield node
                    else:
                        yield from outer_functions(node)

            for owner, replacement in zip(outer_functions(old_tree), outer_functions(new_tree)):
                if same_ast(owner, replacement):
                    continue
                for tree in (owner, replacement):
                    local_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
                    local_names.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))

                    def safe_target(target):
                        while isinstance(target, (ast.Attribute, ast.Subscript)):
                            if isinstance(target, ast.Attribute) and target.attr.isupper():
                                return False
                            target = target.value
                        return not isinstance(target, ast.Name) or target.id in local_names and target.id != "cls"

                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)):
                            return False
                        if isinstance(node, ast.Name) and node.id in forbidden:
                            return False
                        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr == "_hash"):
                            return False
                        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)) and not safe_target(node):
                            return False
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                            if node.func.attr in {"update", "clear", "append", "extend", "insert", "pop", "remove", "setdefault", "add", "discard", "sort", "reverse"} and not safe_target(node.func.value):
                                return False
        return True
    except (OSError, ValueError, SyntaxError, KeyError, TypeError, RecursionError):
        return False


def run_boundaries(root, *, recording=None, baseline=None, methods=METHODS, changed_files=()):
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="2", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    import tests.test_executor as tests
    from sqlglot.executor.python import PythonExecutor
    from sqlglot import transpile

    tests.ProcessPoolExecutor = SerialPool
    original_backend = PythonExecutor.execute
    original_expected = tests.TestExecutor.cached_execute
    record_backend = recording is not None and baseline is None
    backend_rows = {} if record_backend else baseline["backend"]
    expected_rows = {} if record_backend else baseline["expected"]
    stage_rows = {} if recording is not None else None
    observer = StageObserver(root, recording=stage_rows,
                             baseline=(baseline or {}).get("stages"), changed_files=changed_files)
    previous_observer = SerialPool.stage_observer
    # Old baselines keep their original replay path and avoid profiling overhead.
    observing = recording is not None or baseline.get("stages") and changed_files
    SerialPool.stage_observer = observer if observing and observer.open() else None
    matched = set()
    backend_files = set()
    source_prefix = str(root) + os.sep

    def backend_calls(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if filename.startswith(source_prefix + "sqlglot/"):
                backend_files.add(filename[len(source_prefix):])

    def execute(executor, plan, outer_scope=None):
        if outer_scope is not None:
            if record_backend:
                return original_backend(executor, plan, outer_scope)
            raise KeyboardInterrupt("nested backend boundary")
        fingerprint = boundary(executor, plan)
        if record_backend:
            before = {name: pickle.dumps(table, protocol=5) for name, table in executor.tables.mapping.items()}
            previous = sys.getprofile()
            sys.setprofile(backend_calls)
            try:
                result = original_backend(executor, plan)
            finally:
                sys.setprofile(previous)
            changes = {name: pickle.dumps(table, protocol=5) for name, table in executor.tables.mapping.items()
                       if pickle.dumps(table, protocol=5) != before[name]}
            backend_rows[fingerprint] = (pickle.dumps(result, protocol=5), changes)
            return result
        if fingerprint not in backend_rows:
            raise KeyboardInterrupt("frontend boundary changed")
        matched.add(fingerprint)
        result, changes = backend_rows[fingerprint]
        # Execution can mutate shared input tables. Replay these outputs as well
        # as the returned table so subsequent queries observe identical inputs.
        for name, state in changes.items():
            table = executor.tables.mapping[name]
            restored = pickle.loads(state)
            table.__dict__.clear()
            table.__dict__.update(restored.__dict__)
            table.range_reader.table = table
        return pickle.loads(result)

    def expected(instance, sql, tpch=True):
        backend_sql = transpile(sql, write="duckdb")[0]
        key = (tpch, sql)
        if record_backend:
            result = original_expected(instance, sql, tpch)
            expected_rows[key] = (backend_sql, pickle.dumps(result, protocol=5))
            return result
        if key not in expected_rows or expected_rows[key][0] != backend_sql:
            raise KeyboardInterrupt("expected DuckDB SQL changed")
        return pickle.loads(expected_rows[key][1])

    PythonExecutor.execute = execute
    tests.TestExecutor.cached_execute = expected
    passed = []
    try:
        tests.TestExecutor.setUpClass()
        for method in methods:
            observer.method = method
            result = unittest.TestResult()
            tests.TestExecutor(method).run(result)
            if result.wasSuccessful() and result.testsRun == 1 and not result.skipped:
                passed.append("tests/test_executor.py::TestExecutor::" + method)
            else:
                for _, traceback in (result.errors + result.failures)[:2]:
                    print(traceback, file=sys.stderr)
                raise ValueError("baseline executor method failed: " + method)
            if recording is not None:
                print("Recorded " + method, file=sys.stderr, flush=True)
    finally:
        tests.TestExecutor.tearDownClass()
        PythonExecutor.execute = original_backend
        tests.TestExecutor.cached_execute = original_expected
        SerialPool.stage_observer = previous_observer
        observer.close()
    if recording is not None:
        recording.update(revision=REVISION, version=3, backend=backend_rows, expected=expected_rows,
                         methods=passed, backend_files=sorted(backend_files if record_backend else baseline["backend_files"]),
                         stages=stage_rows)
    return {"passed": passed, "matched_boundaries": len(matched), "checked_stages": observer.checked}


def load_baseline(path):
    try:
        with gzip.open(path, "rb") as source:
            baseline = pickle.load(source)
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError("invalid query baseline") from error
    if baseline["revision"] != REVISION or baseline["version"] != 3:
        raise ValueError("invalid query baseline")
    return baseline


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
                                       env={**os.environ, "PYTHONHASHSEED": "0", "PYTHON_CPU_COUNT": "2"}).returncode)
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-stages", type=Path, help="augment a verified boundary map using pristine replay")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.build or args.build_stages:
        output = args.out.resolve()
        recording = {}
        result = run_boundaries(args.source.resolve(), recording=recording,
                                baseline=load_baseline(args.build_stages) if args.build_stages else None)
        output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(filename=str(output), mode="wb", mtime=0) as stream:
            pickle.dump(recording, stream, protocol=5)
        print(json.dumps({**result, "boundaries": len(recording["backend"])}))
