"""Observe Boolean operands and exact dict identity facts on the pristine suite.

No alternate source or outcome labels are accepted. Real spawn workers record
their own imports and tasks; fixture observations expand to all class methods.
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
import os
import pickle
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behaviorcheck import NONBOOL, boolean_site, lookup_site
from comparisoncheck import REVISION, location, source_hash
from queryreplay import TEST
from tools.build_value_profile import Recorder as ContextRecorder


ACTIVE = None
LOOKUPS_ONLY = False
EXACT_TRUTH = (type(None), bool, int, float, str, bytes, tuple, list, dict, set, frozenset, range)
EXACT_TRUTH_IDS = {id(cls) for cls in EXACT_TRUTH}


def truth_mask(value):
    cls = type(value)
    if id(cls) in EXACT_TRUTH_IDS:
        return (2 if bool(value) else 1) | (0 if cls is bool else NONBOOL)
    # Python objects without either truth special method are always true.
    # Read class dictionaries directly without metaclass or instance hooks.
    if type(cls) is type and all(not {"__bool__", "__len__"}.intersection(type.__getattribute__(base, "__dict__"))
                                 for base in type.__getattribute__(cls, "__mro__")):
        return 2 | NONBOOL
    return None


def empty(site):
    if site["kind"] == "boolean":
        return {"masks": [0] * len(site["atoms"]), "unknown": False}
    return {"different": [], "same": None, "unknown": False, "seen": False}


def merge_row(target, incoming, site):
    target["unknown"] |= incoming["unknown"]
    if site["kind"] == "boolean":
        target["masks"] = [a | b for a, b in zip(target["masks"], incoming["masks"])]
    else:
        target["seen"] |= incoming["seen"]
        target["different"] = sorted(set(target["different"]) | set(incoming["different"]))
        if incoming["same"] is not None:
            target["same"] = (list(incoming["same"]) if target["same"] is None
                              else sorted(set(target["same"]) & set(incoming["same"])))


class Recorder(ContextRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.functions = defaultdict(set)
        self.outside_functions = set()
        self.queries = set()
        self.monitor_id = None
        self.code_keys = {}
        self.source_prefix = str(self.root) + "/"
        self.is_executor = False
        self.context_tag = None
        self.persistent_children = {}

    def executor_context(self):
        tag = self.current, os.environ.get("CI_VALUES_TEST")
        if self.context_tag != tag:
            self.is_executor = bool(tag[1] == TEST or self.current == "class:" + TEST.rsplit("::", 1)[0])
            self.is_executor |= self.current == "startup"
            self.context_tag = tag
        return self.is_executor

    def observed_rows(self, key):
        tag = self.current, self.query
        cached = self.cache.get(key)
        if cached is None or cached[0] != tag:
            site = self.sites[key]
            row = self.rows[key].setdefault(self.current, empty(site))
            query_row = None
            if self.executor_context():
                if self.query is None:
                    self.outside.add(key)
                else:
                    query_row = self.query_rows[key].setdefault(self.query, empty(site))
            cached = self.cache[key] = [tag, row, query_row, 0]
        return cached[1:3]

    def operand(self, key, index, value):
        rows = self.observed_rows(key)
        if rows[0]["unknown"] and (rows[1] is None or rows[1]["unknown"]):
            return value
        # Do not invoke arbitrary __bool__/__len__, including on final operands.
        # The original bytecode still performs its original truth tests.
        mask = truth_mask(value)
        if mask is None:
            for row in rows:
                if row is not None:
                    row["unknown"] = True
        else:
            for row in rows:
                if row is not None:
                    row["masks"][index] |= mask
        return value

    def enter_site(self, key):
        rows = self.observed_rows(key)
        if self.sites[key]["kind"] == "boolean":
            cached = self.cache[key]
            if cached[3] < 4096:
                cached[3] += 1
            else:
                for row in rows:
                    if row is not None:
                        row["unknown"] = True

    def lookup(self, key, function, argument):
        rows = self.observed_rows(key)
        try:
            result = function(argument)
        except BaseException:
            for row in rows:
                if row is not None:
                    row["unknown"] = True
            raise
        for row in rows:
            if row is not None:
                row["seen"] = True
        if (type(function) is not types.BuiltinMethodType or function.__name__ != "get"
                or type(function.__self__) is not dict or any(type(k) is not str for k in function.__self__)):
            for row in rows:
                if row is not None:
                    row["unknown"] = True
            return result
        mapping = function.__self__
        if result is None:
            different = {k for k, value in mapping.items() if value is not None}
            for row in rows:
                if row is not None:
                    row["different"] = sorted(set(row["different"]) | different)
        else:
            same = {k for k, value in mapping.items() if value is result}
            for row in rows:
                if row is not None:
                    row["same"] = sorted(same if row["same"] is None else set(row["same"]) & same)
        return result

    def enter(self, code, offset):
        if not self.is_executor:
            return
        identity = id(code)
        if identity not in self.code_keys:
            path = code.co_filename.removeprefix(self.source_prefix)
            key = f"{path}:{code.co_firstlineno}" if path.startswith("sqlglot/") else None
            self.code_keys[identity] = code, key
        key = self.code_keys[identity][1]
        if key is not None:
            if self.query is None:
                self.outside_functions.add(key)
            else:
                self.functions[key].add(self.query)

    def monitor(self):
        m = sys.monitoring
        self.monitor_id = next(i for i in range(5, -1, -1) if m.get_tool(i) is None)
        m.use_tool_id(self.monitor_id, "pristine-task-functions")
        m.register_callback(self.monitor_id, m.events.PY_START, self.enter)
        m.set_events(self.monitor_id, m.events.PY_START if self.executor_context() else 0)

    def set_context(self, context, test=None):
        super().set_context(context, test)
        if self.monitor_id is not None:
            sys.monitoring.set_events(self.monitor_id, sys.monitoring.events.PY_START if self.executor_context() else 0)

    def data(self):
        return dict(revision=REVISION, version=1, kind="local_behavior", tests=self.tests, sources=self.sources,
                    sites=self.sites, rows=dict(self.rows), complete=sorted(self.complete), cpu=self.cpu,
                    bad=sorted(self.bad), collection_complete=self.collection_complete,
                    query_rows={key: {str(i): row for i, row in rows.items()} for key, rows in self.query_rows.items()},
                    outside=sorted(self.outside), functions={key: sorted(indices) for key, indices in self.functions.items()},
                    outside_functions=sorted(self.outside_functions), queries=sorted(self.queries))

    def merge(self, data):
        for path, digest in data["sources"].items():
            if path in self.sources and self.sources[path] != digest:
                raise ValueError("worker source mismatch")
            self.sources[path] = digest
        for key, site in data["sites"].items():
            if key in self.sites and self.sites[key] != site:
                raise ValueError("worker expression mismatch")
            self.sites[key] = site
        for key, contexts in data["rows"].items():
            for context, row in contexts.items():
                merge_row(self.rows[key].setdefault(context, empty(self.sites[key])), row, self.sites[key])
        for key, contexts in data["query_rows"].items():
            for context, row in contexts.items():
                merge_row(self.query_rows[key].setdefault(int(context), empty(self.sites[key])), row, self.sites[key])
        for key, queries in data["functions"].items():
            self.functions[key].update(queries)
        if self.queries & set(data["queries"]):
            raise ValueError("duplicate query index")
        self.queries.update(data["queries"])
        self.outside.update(data["outside"])
        self.outside_functions.update(data["outside_functions"])
        self.bad.update(data["bad"])

    def export(self):
        # Class-scoped pools shut down only after many methods have finished.
        # Merge their final traces before certifying any of those contexts.
        for pid, owner in self.persistent_children.items():
            try:
                data = json.loads((self.directory / f"end-{pid}.json").read_text())
                if data["context"] != owner or not data["finished"]:
                    raise ValueError("unfinished persistent worker")
                self.merge(data)
            except (OSError, ValueError, KeyError, TypeError):
                self.bad.add("class:" + self.tests[int(owner)].rsplit("::", 1)[0]
                             if owner.isdecimal() else owner)
        complete = set(self.complete)
        for context in self.bad:
            if context == "startup":
                complete.clear()
            elif context.isdecimal():
                complete.discard(int(context))
            elif context.startswith("class:"):
                complete.difference_update(i for i, name in enumerate(self.tests) if name.startswith(context[6:] + "::"))
        rows = {}
        for key, contexts in self.rows.items():
            expanded = {}
            for context, row in contexts.items():
                owners = ([str(i) for i, name in enumerate(self.tests) if name.startswith(context[6:] + "::")]
                          if context.startswith("class:") else [context])
                for owner in owners:
                    merge_row(expanded.setdefault(owner, empty(self.sites[key])), row, self.sites[key])
            rows[key] = expanded
        data = self.data()
        data.update(rows=rows, complete=sorted(complete), query_complete=TEST in self.by_id
                    and self.by_id[TEST] in complete and self.query_total > 0
                    and self.queries == set(range(self.query_total)))
        del data["bad"]
        return data


class Instrument(ast.NodeTransformer):
    def __init__(self, path, recorder):
        self.path, self.recorder = path, recorder
        self.truth = False

    def generic_visit(self, node):
        previous = self.truth
        self.truth = False
        result = super().generic_visit(node)
        self.truth = previous
        return result

    def visit_If(self, node):
        return self.guard(node)

    visit_While = visit_If
    visit_IfExp = visit_If

    def guard(self, node):
        previous = self.truth
        self.truth = True
        node.test = self.visit(node.test)
        self.truth = previous
        for field in ("body", "orelse"):
            value = getattr(node, field)
            setattr(node, field, [self.visit(child) for child in value] if isinstance(value, list) else self.visit(value))
        return node

    def visit_BoolOp(self, node):
        if LOOKUPS_ONLY:
            return self.generic_visit(node)
        site = boolean_site(node, truth=self.truth)
        if site is None:
            return self.generic_visit(node)
        key = location(self.path, node)
        self.recorder.sites[key] = site

        def mark(child):
            if isinstance(child, ast.BoolOp):
                child.values = [mark(value) for value in child.values]
                return child
            if isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.Not):
                child.operand = mark(child.operand)
                return child
            if isinstance(child, ast.Constant) and type(child.value) in (bool, type(None), int, str):
                return child
            index = site["atoms"].index(ast.dump(child))
            previous = self.truth
            self.truth = False
            value = self.visit(child)
            self.truth = previous
            return ast.copy_location(ast.Call(ast.Name("_ci_behavior_operand", ast.Load()),
                                              [ast.Constant(key), ast.Constant(index), value], []), child)

        return self.entered(key, mark(node))

    def entered(self, key, node):
        # Record arrival before evaluating operands, including caught exceptions
        # raised while loading an operand or binding the get method itself.
        begin = ast.Call(ast.Name("_ci_behavior_enter", ast.Load()), [ast.Constant(key)], [])
        return ast.copy_location(ast.Subscript(ast.Tuple([begin, node], ast.Load()), ast.Constant(1), ast.Load()), node)

    def visit_UnaryOp(self, node):
        return self.visit_BoolOp(node) if isinstance(node.op, ast.Not) else self.generic_visit(node)

    def visit_Call(self, node):
        site = lookup_site(node)
        key = location(self.path, node)
        node = self.generic_visit(node)
        if site is None:
            return node
        self.recorder.sites[key] = site
        # Evaluate and bind the original get method exactly once before its key.
        return self.entered(key, ast.copy_location(ast.Call(ast.Name("_ci_behavior_lookup", ast.Load()),
                                                           [ast.Constant(key), node.func, node.args[0]], []), node))


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
        module.__dict__.update(_ci_behavior_operand=self.recorder.operand, _ci_behavior_lookup=self.recorder.lookup,
                               _ci_behavior_enter=self.recorder.enter_site)
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
                raise ValueError("source imported outside pristine baseline")
            spec.loader = Loader(fullname, spec.origin, self.recorder)
        return spec


def install(recorder):
    global ACTIVE
    if any(name == "sqlglot" or name.startswith("sqlglot.") for name in sys.modules):
        raise ValueError("source imported before tracing")
    ACTIVE = recorder
    sys.meta_path.insert(0, Finder(recorder))
    sys.addaudithook(recorder.audit)
    recorder.monitor()


def measured_task(context, test, index, payload):
    recorder = ACTIVE
    recorder.current = context
    os.environ["CI_VALUES_TEST"] = test
    recorder.executor_context()
    recorder.query = index if test == TEST else None
    if recorder.query is not None:
        recorder.queries.add(index)
    try:
        function, args, kwargs = pickle.loads(payload)
        return function(*args, **kwargs)
    finally:
        recorder.query = None


def child_bootstrap():
    global LOOKUPS_ONLY
    LOOKUPS_ONLY = os.environ.get("CI_BEHAVIOR_LOOKUPS_ONLY") == "1"
    if any("from multiprocessing.resource_tracker import main" in arg for arg in getattr(sys, "orig_argv", sys.argv)):
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


def pack_profile(data):
    """Group identical context rows losslessly within the repository size limit."""
    data = dict(data, version=2)
    templates, indices = [], {}
    for field in ("rows", "query_rows"):
        packed = {}
        for key, contexts in data.pop(field).items():
            groups = {}
            for context, row in contexts.items():
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
                if encoded not in indices:
                    indices[encoded] = len(templates)
                    templates.append(row)
                group = groups.setdefault(indices[encoded], [0, False])
                if context == "startup":
                    group[1] = True
                else:
                    group[0] |= 1 << int(context)
            packed[key] = [[index, format(mask, "x"), startup]
                           for index, (mask, startup) in sorted(groups.items())]
        data[field + "_groups"] = packed
    data["row_templates"] = templates
    return data


def merge_profiles(primary, lookups, initialization, supplement=None):
    """Replace all lookup evidence with one complete pass, preserving Boolean data."""
    from behaviorcheck import load_profile
    primary, lookups = load_profile(primary), load_profile(lookups)
    with gzip.open(initialization, "rt") as stream:
        initialization = json.load(stream)
    if (primary is None or lookups is None or primary["tests"] != lookups["tests"]
            or primary["sources"] != lookups["sources"]
            or not primary["query_complete"] or not lookups["query_complete"]
            or primary["queries"] != lookups["queries"]
            or any(site["kind"] != "lookup" for site in lookups["sites"].values())
            or initialization["revision"] != REVISION or initialization["kind"] != "pristine_initialization"
            or initialization["complete"] is not True or initialization["tests"] != primary["tests"]
            or any(primary["sources"].get(path) != digest for path, digest in initialization["sources"].items())
            or any(lookups["sites"].get(key) != site for key, site in initialization["sites"].items())
            or not set(initialization["rows"]) <= set(initialization["sites"])):
        raise ValueError("incompatible or incomplete independent profile passes")
    boolean_keys = {key for key, site in primary["sites"].items() if site["kind"] == "boolean"}
    for field in ("sites", "rows", "query_rows"):
        primary[field] = {key: value for key, value in primary[field].items() if key in boolean_keys}
        primary[field].update(lookups[field])
    primary["complete"] = sorted(set(primary["complete"]) & set(lookups["complete"]))
    for field in ("outside", "outside_functions"):
        primary[field] = sorted(set(primary[field]) | set(lookups[field]))
    for key, queries in lookups["functions"].items():
        primary["functions"][key] = sorted(set(primary["functions"].get(key, [])) | set(queries))
    primary.update(initialization_complete=True, initialization=initialization["rows"])
    if supplement is not None:
        supplement = load_profile(supplement)
        if (supplement is None or supplement["tests"] != primary["tests"]
                or any(primary["sources"].get(path) != digest for path, digest in supplement["sources"].items())
                or not supplement.get("pool_trace_complete") or supplement["queries"]
                or any(primary["sites"].get(key, site) != site for key, site in supplement["sites"].items())):
            raise ValueError("incompatible supplemental worker traces")
        primary["sites"].update(supplement["sites"])
        for field in ("rows", "query_rows"):
            for key, contexts in supplement[field].items():
                for context, observed in contexts.items():
                    row = primary[field].setdefault(key, {}).setdefault(context, empty(primary["sites"][key]))
                    merge_row(row, observed, primary["sites"][key])
        covered = {primary["tests"].index(name) for name in supplement["profiled_tests"]}
        primary["complete"] = sorted(set(primary["complete"]) - (covered - set(supplement["complete"])))
        for field in ("outside", "outside_functions"):
            primary[field] = sorted(set(primary[field]) | set(supplement[field]))
        for key, queries in supplement["functions"].items():
            primary["functions"][key] = sorted(set(primary["functions"].get(key, [])) | set(queries))
        primary["pool_trace_complete"] = True
    return primary


def main():
    global LOOKUPS_ONLY
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, default=Path(__file__).resolve().parents[1] / "maps/baseline.json.gz")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--lookups-only", action="store_true")
    parser.add_argument("--merge-profile", type=Path)
    parser.add_argument("--lookup-profile", type=Path)
    parser.add_argument("--initialization-profile", type=Path)
    parser.add_argument("--supplement-profile", type=Path)
    args = parser.parse_args()
    LOOKUPS_ONLY = args.lookups_only
    root, output = args.source.resolve(), args.out.resolve()
    catalog = json.loads(args.tests.resolve().read_text())["tests"]
    with gzip.open(args.baseline_map.resolve(), "rt") as stream:
        baseline = json.load(stream)
    expected = {path: text for path, text in baseline["sources"].items() if path.startswith("sqlglot/")}
    actual = {path.relative_to(root).as_posix() for path in (root / "sqlglot").rglob("*.py")}
    if (baseline["revision"] != REVISION or actual != set(expected)
            or any((root / path).read_text() != text for path, text in expected.items())
            or os.environ.get("PYTHONHASHSEED") != "0"):
        raise ValueError("pristine source and PYTHONHASHSEED=0 required")
    selected = args.only if args.only is not None else [test["id"] for test in catalog]
    if not set(selected) <= {test["id"] for test in catalog}:
        raise ValueError("unknown test context")
    if args.merge_profile:
        if not args.lookup_profile or not args.initialization_profile:
            raise ValueError("both independent lookup and initialization passes required")
        data = merge_profiles(args.merge_profile, args.lookup_profile, args.initialization_profile,
                              args.supplement_profile)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(output.name + ".building")
        with gzip.GzipFile(filename=str(temporary_output), mode="wb", mtime=0) as stream:
            stream.write(json.dumps(pack_profile(data), separators=(",", ":"), sort_keys=True).encode())
        temporary_output.replace(output)
        return
    with tempfile.TemporaryDirectory(prefix="ci-behavior-profile-") as temporary:
        directory = Path(temporary)
        (directory / "sitecustomize.py").write_text("from build_behavior_profile import child_bootstrap\nchild_bootstrap()\n")
        os.environ.update(SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="2", PYTHONDONTWRITEBYTECODE="1",
                          OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_MAX_THREADS="1",
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CI_VALUES_ROOT=str(root), CI_VALUES_OUTPUT=str(directory),
                          CI_VALUES_CONTEXT="startup", CI_VALUES_TEST="",
                          CI_BEHAVIOR_LOOKUPS_ONLY="1" if LOOKUPS_ONLY else "0",
                          PYTHONPATH=os.pathsep.join([str(directory), str(Path(__file__).resolve().parent), str(root)]))
        sys.path.insert(0, str(root))
        os.chdir(root)
        recorder = Recorder(root, catalog, directory)
        install(recorder)
        original_pool = concurrent.futures.ProcessPoolExecutor
        from build_behavior_profile import measured_task as task
        class Pool(original_pool):
            def submit(self, function, /, *a, **kw):
                test = os.environ.get("CI_VALUES_TEST", "")
                if test == TEST:
                    recorder.query_total += 1
                index = getattr(self, "query_index", 0)
                self.query_index = index + 1
                result = super().submit(task, recorder.current, test, index, pickle.dumps((function, a, kw), protocol=5))
                if test != TEST:
                    for pid in self._processes or {}:
                        recorder.persistent_children.setdefault(pid, recorder.current)
                return result
            def __exit__(self, *a):
                if os.environ.get("CI_VALUES_TEST") == TEST:
                    recorder.expected_children[recorder.current].update(self._processes or {})
                return super().__exit__(*a)
        concurrent.futures.ProcessPoolExecutor = Pool
        status = pytest.main([*selected, "-q", "--maxfail=1", "-p", "no:cacheprovider",
                              "-p", "pytest_timeout", f"--timeout={args.timeout}"], plugins=[recorder])
        if status:
            raise SystemExit(status)
        data = recorder.export()
        data.update(pool_trace_complete=True, profiled_tests=selected)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(output.name + ".building")
        with gzip.GzipFile(filename=str(temporary_output), mode="wb", mtime=0) as stream:
            stream.write(json.dumps(pack_profile(data), separators=(",", ":"), sort_keys=True).encode())
        temporary_output.replace(output)
        print(f"Behavior sites: {len(data['sites'])}; complete: {len(data['complete'])}/{len(catalog)}; "
              f"query contexts: {len(data['queries'])}; query complete: {data['query_complete']}", file=sys.stderr)


if __name__ == "__main__":
    main()
