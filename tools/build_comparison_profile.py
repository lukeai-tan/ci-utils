"""Record pristine comparison operand types, including real spawned pool workers.

Accepts source and test catalogs only. Does not evaluate proposed edits or
record alternate comparison outcomes.
"""

from __future__ import annotations

import argparse
import ast
import atexit
import concurrent.futures
import gzip
import importlib.abc
import importlib.machinery
import json
import operator
import os
import resource
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparisoncheck import COMPARISONS, REVISION, location, operand_types, source_hash


OPERATORS = {"Eq": operator.eq, "NotEq": operator.ne, "Is": operator.is_, "IsNot": operator.is_not}


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def compact_profile(data):
    """Intern repeated baseline type pairs; retain the same per-context evidence."""
    pairs = sorted({tuple(tuple(value) for value in pair) for contexts in data["operands"].values()
                    for observed in contexts.values() for pair in observed})
    indices = {pair: i for i, pair in enumerate(pairs)}
    return dict(data, version=2, pairs=pairs,
                operands={key: {context: sorted(indices[tuple(tuple(value) for value in pair)] for pair in observed)
                                for context, observed in contexts.items()}
                          for key, contexts in data["operands"].items()})


class Recorder:
    def __init__(self, root, catalog=(), directory=None, context="startup"):
        self.root = Path(root).resolve()
        self.tests = [test["id"] for test in catalog]
        self.by_id = {name: i for i, name in enumerate(self.tests)}
        self.current = context
        self.directory = Path(directory) if directory else None
        self.operands = defaultdict(lambda: defaultdict(set))
        self.sources = {}
        self.complete = set()
        self.phases = defaultdict(set)
        self.bad = set()
        self.cpu = {}
        self.expected_children = defaultdict(set)
        self.collection_complete = False
        self.started = 0

    def comparison(self, key, op, left, right):
        self.operands[key][self.current].add(operand_types(left, right))
        return OPERATORS[op](left, right)

    def audit(self, event, args):
        if event in ("subprocess.Popen", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn"):
            # Only the explicitly measured spawn pool is supported.
            self.bad.add(self.current)

    def merge(self, data):
        for path, digest in data["sources"].items():
            if path in self.sources and self.sources[path] != digest:
                raise ValueError("child source mismatch")
            self.sources[path] = digest
        for key, contexts in data["operands"].items():
            for context, pairs in contexts.items():
                self.operands[key][context].update(tuple(tuple(value) for value in pair) for pair in pairs)
        self.bad.update(data.get("bad", []))

    def data(self):
        return dict(revision=REVISION, version=1, kind="comparison_types", tests=self.tests,
                    sources=self.sources, operands={key: {context: sorted(pairs) for context, pairs in contexts.items()}
                                                   for key, contexts in self.operands.items()},
                    complete=sorted(self.complete), cpu=self.cpu, bad=sorted(self.bad),
                    collection_complete=self.collection_complete)

    def pytest_collection_finish(self, session):
        self.collection_complete = not session.testsfailed

    def pytest_runtest_logstart(self, nodeid, location):
        self.current = str(self.by_id[nodeid])
        os.environ["CI_COMPARE_CONTEXT"] = self.current
        self.started = cpu_seconds()

    def pytest_runtest_logreport(self, report):
        context = str(self.by_id[report.nodeid])
        if report.failed or report.skipped:
            self.bad.add(context)
        if report.passed:
            self.phases[context].add(report.when)

    def pytest_runtest_logfinish(self, nodeid, location):
        index = self.by_id[nodeid]
        context = str(index)
        for pid in self.expected_children[context]:
            try:
                data = json.loads((self.directory / f"end-{pid}.json").read_text())
                if data["context"] != context or not data["finished"]:
                    raise ValueError("unfinished child")
                self.merge(data)
            except (OSError, ValueError, KeyError, TypeError):
                self.bad.add(context)
        self.cpu[nodeid] = round(cpu_seconds() - self.started, 6)
        if context not in self.bad and self.phases[context] == {"setup", "call", "teardown"}:
            self.complete.add(index)
        self.current = "startup"
        os.environ["CI_COMPARE_CONTEXT"] = "startup"


class Instrument(ast.NodeTransformer):
    def __init__(self, path):
        self.path = path

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or not isinstance(node.ops[0], COMPARISONS):
            return node
        call = ast.Call(ast.Name("_ci_compare_observe", ast.Load()),
                        [ast.Constant(location(self.path, node)), ast.Constant(type(node.ops[0]).__name__),
                         node.left, node.comparators[0]], [])
        return ast.copy_location(call, node)


class Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, recorder):
        super().__init__(fullname, path)
        self.recorder = recorder

    def get_code(self, fullname):
        text = self.get_data(self.path).decode("utf-8")
        relative = Path(self.path).resolve().relative_to(self.recorder.root).as_posix()
        self.recorder.sources[relative] = source_hash(text)
        tree = Instrument(relative).visit(ast.parse(text, filename=self.path))
        return compile(ast.fix_missing_locations(tree), self.path, "exec")

    def exec_module(self, module):
        module.__dict__["_ci_compare_observe"] = self.recorder.comparison
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
                raise ValueError("sqlglot imported outside pinned source")
            spec.loader = Loader(fullname, spec.origin, self.recorder)
        return spec


def install(recorder):
    if any(name == "sqlglot" or name.startswith("sqlglot.") for name in sys.modules):
        raise ValueError("sqlglot imported before instrumentation")
    sys.meta_path.insert(0, Finder(recorder))
    sys.addaudithook(recorder.audit)


def child_bootstrap():
    """sitecustomize installs measurement before spawn unpickles source objects."""
    arguments = getattr(sys, "orig_argv", sys.argv)
    if any("from multiprocessing.resource_tracker import main" in value for value in arguments):
        return
    root = os.environ["CI_COMPARE_ROOT"]
    directory = Path(os.environ["CI_COMPARE_OUTPUT"])
    context = os.environ["CI_COMPARE_CONTEXT"]
    recorder = Recorder(root, directory=directory, context=context)
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
    parser.add_argument("--baseline-map", type=Path, default=Path(__file__).resolve().parents[1] / "maps/baseline.json.gz")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    root = args.source.resolve()
    output = args.out.resolve()
    catalog = json.loads(args.tests.resolve().read_text())["tests"]
    with gzip.open(args.baseline_map.resolve(), "rt") as stream:
        baseline = json.load(stream)
    expected = {path: text for path, text in baseline["sources"].items() if path.startswith("sqlglot/")}
    actual = {path.relative_to(root).as_posix() for path in (root / "sqlglot").rglob("*.py")}
    if (baseline["revision"] != REVISION or actual != set(expected)
            or any((root / path).read_text() != text for path, text in expected.items())):
        raise ValueError("source differs from the pristine pinned baseline")
    tests = [test["id"] for test in catalog]
    selected = args.only if args.only is not None else tests
    if not set(selected + args.exclude) <= set(tests):
        raise ValueError("unknown test context")
    selected = [name for name in selected if name not in args.exclude]
    with tempfile.TemporaryDirectory(prefix="ci-comparison-profile-") as temporary:
        directory = Path(temporary)
        (directory / "sitecustomize.py").write_text("from build_comparison_profile import child_bootstrap\nchild_bootstrap()\n")
        os.environ.update(SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="2", PYTHONDONTWRITEBYTECODE="1",
                          OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_MAX_THREADS="1",
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CI_COMPARE_ROOT=str(root),
                          CI_COMPARE_OUTPUT=str(directory), CI_COMPARE_CONTEXT="startup",
                          PYTHONPATH=os.pathsep.join([str(directory), str(Path(__file__).resolve().parent), str(root)]))
        sys.path.insert(0, str(root))
        os.chdir(root)
        recorder = Recorder(root, catalog, directory)
        install(recorder)
        original_pool = concurrent.futures.ProcessPoolExecutor

        class MeasuredPool(original_pool):
            def __exit__(self, *args):
                context = recorder.current
                recorder.expected_children[context].update(self._processes or {})
                return super().__exit__(*args)

        concurrent.futures.ProcessPoolExecutor = MeasuredPool
        import pytest
        result = pytest.main([*selected, "-q", "--maxfail=1", "-p", "no:cacheprovider",
                              "-p", "pytest_timeout", "--timeout=300"], plugins=[recorder]) if selected else 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(filename=str(output), mode="wb", mtime=0) as stream:
            stream.write(json.dumps(compact_profile(recorder.data()), separators=(",", ":"), sort_keys=True).encode())
        print("Comparison types", len(recorder.operands), "complete", len(recorder.complete),
              "of", len(catalog), "bad", len(recorder.bad), file=sys.stderr)
        raise SystemExit(result)


if __name__ == "__main__":
    main()
