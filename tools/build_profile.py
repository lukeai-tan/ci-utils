"""Record baseline class consumers and attribute reads; accepts no proposed patches."""

from __future__ import annotations

import argparse
import ast
import builtins
import concurrent.futures
import gzip
import importlib.abc
import importlib.machinery
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


class InlinePool:
    """Keep worker dependencies in the test context, including serialization."""

    def __init__(self, max_workers=None, initializer=None, initargs=(), **kwargs):
        if initializer:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, function, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            function, args, kwargs = pickle.loads(pickle.dumps((function, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(function(*args, **kwargs))))
        except BaseException as error:
            future.set_exception(error)
        return future

    def map(self, function, *iterables, **kwargs):
        return (self.submit(function, *args).result() for args in zip(*iterables))

    def shutdown(self, wait=True, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


class Recorder:
    def __init__(self, root, tests):
        self.root = root.resolve()
        self.tests = [test["id"] for test in tests]
        self.by_id = {name: index for index, name in enumerate(self.tests)}
        self.current = None
        self.classes = defaultdict(set)
        self.attributes = defaultdict(set)
        self.collection = set()
        self.reads = defaultdict(set)
        self.incomplete = set()
        self.cpu = {}
        self.started = 0
        self.owners = {}
        self.class_keys = {}
        self.entered = {}
        self.seen_attributes = {}
        self.monitored_names = self.setting_names()

    def setting_names(self):
        names = set()
        for path in (self.root / "sqlglot").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ClassDef):
                    for statement in node.body:
                        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                            names.update(target.id for target in targets if isinstance(target, ast.Name))
        return names

    def key(self, cls):
        identity = id(cls)
        if identity not in self.class_keys:
            module = type.__getattribute__(cls, "__module__")
            qualname = type.__getattribute__(cls, "__qualname__")
            key = module + ":" + qualname if module.startswith("sqlglot.") else None
            self.class_keys[identity] = (cls, key)
        return self.class_keys[identity][1]

    def enter(self, obj):
        if self.current is None:
            return
        cls = obj if isinstance(obj, type) else type(obj)
        identity = id(cls)
        if self.entered.get(identity) == self.current:
            return
        self.entered[identity] = self.current
        for base in type.__getattribute__(cls, "__mro__"):
            key = self.key(base)
            if key:
                self.classes[key].add(self.current)

    def attribute(self, obj, name):
        value = getattr(obj, name)
        if isinstance(value, type):
            self.enter(value)
        cls = obj if isinstance(obj, type) else type(obj)
        pair = (id(cls), name)
        if pair in self.seen_attributes and self.seen_attributes[pair] == self.current:
            return value
        self.seen_attributes[pair] = self.current
        if not any(self.key(base) for base in type.__getattribute__(cls, "__mro__")):
            return value
        if pair not in self.owners:
            owner = next((base for base in type.__getattribute__(cls, "__mro__")
                          if name in type.__getattribute__(base, "__dict__")), None)
            key = self.key(owner) if owner is not None else None
            self.owners[pair] = key + ":" + name if key else None
        key = self.owners[pair]
        if key:
            if self.current is None:
                self.collection.add(key)
            else:
                self.attributes[key].add(self.current)
                self.enter(obj)
        return value

    def audit(self, event, args):
        if self.current is None or event not in ("open", "os.listdir", "os.scandir"):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
            relative = path.relative_to(self.root).as_posix()
        except (ValueError, TypeError, OSError):
            return
        # Import machinery reads bytecode and sources; retain separate provenance
        # only when baseline repository code explicitly asks for the file.
        frame = sys._getframe(1)
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename.startswith(str(self.root)):
                self.reads[relative].add(self.current)
                break
            if "importlib" in filename and frame.f_code.co_name in ("_find_and_load", "_find_and_load_unlocked", "_load_unlocked"):
                break
            frame = frame.f_back

    def pytest_runtest_setup(self, item):
        self.current = self.by_id.get(item.nodeid)
        self.started = time.process_time()

    def pytest_runtest_logreport(self, report):
        index = self.by_id.get(report.nodeid)
        if index is None:
            return
        if report.failed or report.skipped:
            self.incomplete.add(index)
        if report.when == "teardown":
            self.cpu[index] = round(time.process_time() - self.started, 6)
            self.current = None

    def data(self):
        return {
            "revision": "eb11aec4be8cb2122afe713a4816662c3f0fb2fa",
            "version": 1,
            "tests": self.tests,
            "classes": {key: sorted(value) for key, value in sorted(self.classes.items())},
            "attributes": {key: sorted(value) for key, value in sorted(self.attributes.items())},
            "collection": sorted(self.collection),
            "reads": {key: sorted(value) for key, value in sorted(self.reads.items())},
            "incomplete": sorted(self.incomplete | (set(range(len(self.tests))) - set(self.cpu))),
            "cpu": {self.tests[index]: value for index, value in self.cpu.items()},
        }

    def restore(self, profile):
        if profile["tests"] != self.tests or profile["revision"] != "eb11aec4be8cb2122afe713a4816662c3f0fb2fa":
            raise ValueError("profile does not match pinned test catalog")
        for name in ("classes", "attributes", "reads"):
            target = getattr(self, name)
            for key, consumers in profile.get(name, {}).items():
                target[key].update(consumers)
        self.collection.update(profile.get("collection", []))
        self.incomplete.update(profile.get("incomplete", []))
        self.cpu.update({self.by_id[name]: value for name, value in profile.get("cpu", {}).items()})

    def conservative_classes(self, baseline, selected):
        """An uninstrumented workload retains every class in each covered module."""
        numbered = {name: index for index, name in enumerate(baseline["tests"])}
        for name in selected:
            old_index = numbered[name]
            index = self.by_id[name]
            for path, consumers in baseline["files"].items():
                if old_index not in consumers or not path.startswith("sqlglot/"):
                    continue
                module = path.removesuffix(".py").replace("/", ".")
                source = baseline["sources"].get(path)
                if source is None:
                    continue

                def classes(tree, prefix=""):
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.ClassDef):
                            key = prefix + node.name
                            self.classes[module + ":" + key].add(index)
                            for statement in node.body:
                                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                                    for target in targets:
                                        if isinstance(target, ast.Name):
                                            self.attributes[module + ":" + key + ":" + target.id].add(index)
                            classes(node, key + ".")
                        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            classes(node, prefix)

                classes(ast.parse(source))


class Instrument(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and node.attr in self.names:
            call = ast.Call(ast.Name("_ci_profile_attribute", ast.Load()), [node.value, ast.Constant(node.attr)], [])
            return ast.copy_location(call, node)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        args = node.args.posonlyargs + node.args.args
        if args and args[0].arg in ("self", "cls"):
            call = ast.Expr(ast.Call(ast.Name("_ci_profile_enter", ast.Load()), [ast.Name(args[0].arg, ast.Load())], []))
            ast.copy_location(call, node.body[0])
            index = int(isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str))
            node.body.insert(index, call)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


class Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, recorder):
        super().__init__(name, path)
        self.recorder = recorder

    def get_code(self, fullname):
        tree = ast.parse(self.get_data(self.path), filename=self.path)
        tree = Instrument(self.recorder.monitored_names).visit(tree)
        return compile(ast.fix_missing_locations(tree), self.path, "exec")

    def exec_module(self, module):
        module.__dict__["_ci_profile_enter"] = self.recorder.enter
        module.__dict__["_ci_profile_attribute"] = self.recorder.attribute
        super().exec_module(module)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, recorder):
        self.recorder = recorder

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin.endswith(".py"):
            spec.loader = Loader(fullname, spec.origin, self.recorder)
        return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--baseline-map", type=Path, default=Path(__file__).resolve().parents[1] / "maps/baseline.json.gz")
    parser.add_argument("--no-instrumentation", action="store_true")
    parser.add_argument("--trace-max-seconds", type=float, default=25)
    args = parser.parse_args()
    args.tests = args.tests.resolve()
    args.out = args.out.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    args.baseline_map = args.baseline_map.resolve()
    os.environ.update(SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="2", PYTHONDONTWRITEBYTECODE="1",
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    catalog = json.loads(args.tests.read_text())["tests"]
    recorder = Recorder(source, catalog)
    if args.resume:
        with gzip.open(args.resume, "rt") as stream:
            recorder.restore(json.load(stream))
    selected = args.only or ([recorder.tests[i] for i in sorted(recorder.incomplete)] if args.resume else recorder.tests)
    untraced = [] if args.no_instrumentation else [test["id"] for test in catalog
                                                  if test["id"] in selected and test["seconds"] > args.trace_max_seconds]
    if untraced:
        with tempfile.TemporaryDirectory(prefix="ci-profile-") as temporary:
            output = Path(temporary) / "profile.json.gz"
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--source", str(source),
                            "--tests", str(args.tests), "--out", str(output), "--baseline-map", str(args.baseline_map),
                            "--no-instrumentation", "--only", *untraced], check=True,
                           env={**os.environ, "PYTHON_CPU_COUNT": "2"})
            with gzip.open(output, "rt") as stream:
                extra = json.load(stream)
            # The child catalog includes tests that it intentionally did not run.
            extra["incomplete"] = [i for i in extra["incomplete"] if recorder.tests[i] in untraced]
            recorder.restore(extra)
            recorder.incomplete.difference_update(recorder.by_id[name] for name in untraced if name in extra["cpu"])
        selected = [name for name in selected if name not in untraced]
    if args.no_instrumentation:
        with gzip.open(args.baseline_map, "rt") as stream:
            recorder.conservative_classes(json.load(stream), selected)
    else:
        sys.meta_path.insert(0, Finder(recorder))
    sys.addaudithook(recorder.audit)
    concurrent.futures.ProcessPoolExecutor = InlinePool
    import pytest
    recorder.incomplete.difference_update(recorder.by_id[name] for name in selected)
    result = pytest.main([*selected, "-q", "--maxfail=1", "-p", "no:cacheprovider", "-p", "pytest_timeout", "--timeout=300"], plugins=[recorder]) if selected else 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(args.out), mode="wb", mtime=0) as output:
        output.write(json.dumps(recorder.data(), separators=(",", ":"), sort_keys=True).encode())
    print("Profile", args.out, "classes", len(recorder.classes), "incomplete", len(recorder.data()["incomplete"]), file=sys.stderr)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
