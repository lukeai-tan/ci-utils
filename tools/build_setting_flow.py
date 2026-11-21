"""Trace pinned tokenizer option forwarding without replacing application values."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import importlib.machinery
import json
import os
import operator
import sys
from collections import defaultdict
from pathlib import Path

try:
    from .build_profile import Finder, InlinePool, Loader, Recorder
except ImportError:
    from build_profile import Finder, InlinePool, Loader, Recorder


REVISION = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"


def forwarding(root):
    """Only certify direct arguments assigned unchanged to plain core slots."""
    source = ast.parse((root / "sqlglot/tokens.py").read_text())
    core = ast.parse((root / "sqlglot/tokenizer_core.py").read_text())
    cls = next(node for node in core.body if isinstance(node, ast.ClassDef) and node.name == "TokenizerCore")
    if cls.bases or cls.decorator_list:
        raise ValueError("unsupported tokenizer core")
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
           and node.name in {"__getattribute__", "__getattr__", "__setattr__", "__delattr__"}
           for node in cls.body):
        raise ValueError("custom core attribute access")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    slots = set()
    for node in init.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "self"
                and isinstance(node.value, ast.Name) and node.targets[0].attr == node.value.id):
            uses = sum(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                       and item.id == node.value.id for statement in init.body for item in ast.walk(statement))
            if uses == 1:
                slots.add(node.value.id)
    method = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "_init_core")
    if len(method.body) != 1 or not isinstance(method.body[0], ast.Return):
        raise ValueError("unsupported core factory")
    call = method.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "TokenizerCore":
        raise ValueError("unsupported core factory")
    result = {}
    for keyword in call.keywords:
        value = keyword.value
        if keyword.arg not in slots or not isinstance(value, ast.Attribute):
            continue
        if isinstance(value.value, ast.Name) and value.value.id == "self":
            result[keyword.arg] = ("self", value.attr)
        elif (isinstance(value.value, ast.Attribute) and value.value.attr == "dialect"
              and isinstance(value.value.value, ast.Name) and value.value.value.id == "self"):
            result[keyword.arg] = ("dialect", value.attr)
    return result


class FlowRecorder(Recorder):
    def __init__(self, root, tests):
        super().__init__(root, tests)
        self.forward = forwarding(self.root)
        self.option_names = {value[1] for value in self.forward.values()}
        self.field_names = set(self.forward)
        self.core_objects = {}
        self.dictionaries = {}
        self.namespaces = {}
        self.consumers = defaultdict(set)
        self.entries = defaultdict(lambda: defaultdict(set))
        self.blocked = set()
        self.known = set()
        self.untrusted = set()
        self.uninstrumented = set()

    def owner(self, obj, name):
        cls = obj if isinstance(obj, type) else type(obj)
        for base in type.__getattribute__(cls, "__mro__"):
            if name in type.__getattribute__(base, "__dict__"):
                key = self.key(base)
                return key + ":" + name if key else None
        return None

    def consume(self, keys, entry=None, mutation=False):
        for key in keys:
            if mutation or self.current is None:
                self.blocked.add(key)
            elif entry is None:
                self.consumers[key].add(self.current)
            elif isinstance(entry, str):
                self.entries[key][entry].add(self.current)
            else:
                self.consumers[key].add(self.current)

    def attribute(self, obj, name, mode="value"):
        value = getattr(obj, name)
        cls = type(obj)
        if cls.__module__ == "sqlglot.tokenizer_core" and cls.__name__ == "TokenizerCore":
            recorded = self.core_objects.get(id(obj))
            if recorded is None or recorded[0] is not obj:
                if self.current is None:
                    self.blocked.update(self.known)
                else:
                    self.untrusted.add(self.current)
                return value
            key = recorded[1].get(name)
        else:
            key = self.owner(obj, name) if name in self.option_names else None
        if key is None:
            return value
        self.known.add(key)
        if type(value) is dict:
            previous = self.dictionaries.get(id(value))
            keys = set(previous[1]) if previous and previous[0] is value else set()
            keys.add(key)
            self.dictionaries[id(value)] = (value, keys)
        if mode != "transfer" and (mode != "borrow" or type(value) is not dict):
            self.consume([key])
        return value

    def construct(self, receiver, constructor, *args, **kwargs):
        result = constructor(*args, **kwargs)
        if (type(result).__module__ != "sqlglot.tokenizer_core"
                or type(result).__name__ != "TokenizerCore" or args):
            raise ValueError("unsupported forwarded core")
        origins = {}
        for field, (kind, name) in self.forward.items():
            obj = receiver.dialect if kind == "dialect" else receiver
            key = self.owner(obj, name)
            if key:
                if field not in kwargs or getattr(result, field) is not kwargs[field]:
                    self.blocked.add(key)
                    continue
                self.known.add(key)
                origins[field] = key
        self.core_objects[id(result)] = (result, origins)
        return result

    def map_operation(self, obj, method, *args, **kwargs):
        namespace = self.namespaces.get(id(obj))
        if namespace and namespace[0] is obj:
            if method == "__getitem__" and args and isinstance(args[0], str):
                key = self.owner(namespace[1], args[0])
                if key:
                    self.consume([key])
            else:
                self.reflect(namespace[1])
        recorded = self.dictionaries.get(id(obj))
        if recorded is not None and recorded[0] is obj:
            keys = recorded[1]
            if method in ("get", "__getitem__"):
                self.consume(keys, args[0] if args else None)
            elif method not in ("keys", "__iter__", "__len__", "__contains__"):
                self.consume(keys, mutation=method in {
                    "update", "clear", "pop", "popitem", "setdefault", "__setitem__", "__delitem__",
                })
        return operator.getitem(obj, args[0]) if method == "__getitem__" else getattr(obj, method)(*args, **kwargs)

    def map_call(self, function, *args, **kwargs):
        obj = getattr(function, "__self__", None)
        method = getattr(function, "__name__", "")
        namespace = self.namespaces.get(id(obj))
        if namespace and namespace[0] is obj:
            if method == "get" and args and isinstance(args[0], str):
                key = self.owner(namespace[1], args[0])
                if key:
                    self.consume([key])
            elif method != "keys":
                self.reflect(namespace[1])
        if type(obj) is dict:
            recorded = self.dictionaries.get(id(obj))
            if recorded and recorded[0] is obj:
                if method == "get":
                    self.consume(recorded[1], args[0] if args else None)
                elif method != "keys":
                    self.consume(recorded[1], mutation=method in {
                        "update", "clear", "pop", "popitem", "setdefault",
                    })
        return function(*args, **kwargs)

    def dictionary(self, obj, mode):
        value = obj.__dict__
        self.namespaces[id(value)] = (value, obj)
        if mode != "borrow":
            self.reflect(obj)
        return value

    def argument(self, value, seen=None):
        if seen is None:
            seen = set()
        if id(value) in seen:
            return value
        seen.add(id(value))
        recorded = self.dictionaries.get(id(value))
        if recorded and recorded[0] is value:
            self.consume(recorded[1])
        core = self.core_objects.get(id(value))
        if core and core[0] is value:
            self.consume(core[1].values())
        # Escaping containers can carry a core into uninstrumented native code.
        if type(value) in (list, tuple, dict, set, frozenset):
            items = value.items() if type(value) is dict else value
            for item in items:
                if (id(item) in self.core_objects or id(item) in self.dictionaries
                        or type(item) in (list, tuple, dict, set, frozenset)):
                    self.argument(item, seen)
        return value

    def writing(self, obj, name=None):
        if name is not None:
            recorded = self.core_objects.get(id(obj))
            key = recorded[1].get(name) if recorded and recorded[0] is obj else self.owner(obj, name)
            if key:
                self.consume([key], mutation=True)
        else:
            namespace = self.namespaces.get(id(obj))
            if namespace and namespace[0] is obj:
                for option in self.option_names:
                    key = self.owner(namespace[1], option)
                    if key:
                        self.consume([key], mutation=True)
            recorded = self.dictionaries.get(id(obj))
            if recorded and recorded[0] is obj:
                self.consume(recorded[1], mutation=True)
        return obj

    def mutation(self, function, obj, name, *args, **kwargs):
        self.writing(obj, name)
        if name in ("__dict__", "__class__"):
            core = self.core_objects.get(id(obj))
            if core and core[0] is obj:
                self.consume(core[1].values(), mutation=True)
            for option in self.option_names:
                key = self.owner(obj, option)
                if key:
                    self.consume([key], mutation=True)
        return function(obj, name, *args, **kwargs)

    def dynamic(self, function, obj, name, *default):
        value = function(obj, name, *default)
        if name in self.option_names or name in self.field_names:
            core = self.core_objects.get(id(obj))
            key = core[1].get(name) if core and core[0] is obj else self.owner(obj, name)
            if key:
                self.consume([key])
        if name in ("__dict__", "__slots__"):
            self.reflect(obj)
        return value

    def reflect(self, obj):
        core = self.core_objects.get(id(obj))
        if core and core[0] is obj:
            self.consume(core[1].values())
        else:
            for name in self.option_names:
                key = self.owner(obj, name)
                if key:
                    self.consume([key])

    def object_vars(self, function, obj):
        self.reflect(obj)
        return function(obj)

    def audit(self, event, args):
        super().audit(event, args)
        if event == "subprocess.Popen" and self.current is not None:
            self.untrusted.add(self.current)

    def restore(self, profile):
        if profile.get("kind") != "setting_flow" or profile.get("version") != 1:
            raise ValueError("invalid setting flow checkpoint")
        super().restore(profile)
        self.known.update(profile["known"])
        self.blocked.update(profile["blocked"])
        self.untrusted.update(set(profile["untrusted"]) - set(profile["uninstrumented"]))
        for key, contexts in profile["consumers"].items():
            self.consumers[key].update(contexts)
        for key, entries in profile["entries"].items():
            for entry, contexts in entries.items():
                self.entries[key][entry].update(contexts)

    def data(self):
        # Runtime selection and resumable evidence need no duplicate class/file maps.
        data = {key: value for key, value in super().data().items()
                if key in {"revision", "tests", "incomplete", "cpu"}}
        data.update(version=1, kind="setting_flow", known=sorted(self.known),
                    blocked=sorted(self.blocked), untrusted=sorted(self.untrusted | self.uninstrumented),
                    uninstrumented=sorted(self.uninstrumented),
                    consumers={key: sorted(self.consumers[key]) for key in sorted(self.known - self.blocked)},
                    entries={key: {entry: sorted(ids) for entry, ids in sorted(values.items())}
                             for key, values in sorted(self.entries.items()) if key not in self.blocked})
        return data


class FlowInstrument(ast.NodeTransformer):
    def __init__(self, recorder, factory=False):
        self.recorder = recorder
        self.factory = factory
        self.parents = {}
        self.function = None

    def visit(self, node):
        if isinstance(node, ast.AST):
            for child in ast.iter_child_nodes(node):
                self.parents[id(child)] = node
        return super().visit(node)

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node
        self.generic_visit(node)
        self.function = previous
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def borrowed(self, node):
        parent = self.parents.get(id(node))
        if isinstance(parent, ast.Subscript) and parent.value is node:
            return True
        if (isinstance(parent, ast.Attribute) and parent.value is node
                and parent.attr in {"get", "keys", "items", "values", "copy", "update", "clear",
                                    "pop", "popitem", "setdefault"}):
            call = self.parents.get(id(parent))
            return isinstance(call, ast.Call) and call.func is parent
        if isinstance(parent, (ast.BoolOp, ast.If, ast.While)):
            return not isinstance(parent, (ast.If, ast.While)) or parent.test is node
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
            return True
        if (isinstance(parent, ast.Compare) and len(parent.ops) == 1
                and isinstance(parent.ops[0], (ast.In, ast.NotIn)) and parent.comparators[0] is node):
            return True
        if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                and parent.func.id in {"bool", "len"} and parent.args == [node] and not parent.keywords):
            return True
        if (isinstance(parent, ast.Assign) and len(parent.targets) == 1
                and isinstance(parent.targets[0], ast.Name) and self.function):
            name = parent.targets[0].id
            tree = self.function
            if any(isinstance(item, ast.Name) and item.id in {"locals", "vars", "eval", "exec"}
                   for item in ast.walk(tree)):
                return False
            for item in ast.walk(tree):
                if not isinstance(item, ast.Name) or item.id != name or not isinstance(item.ctx, ast.Load):
                    continue
                owner = self.parents.get(id(item))
                if isinstance(owner, ast.BoolOp) or isinstance(owner, ast.UnaryOp) and isinstance(owner.op, ast.Not):
                    continue
                if isinstance(owner, (ast.If, ast.While)) and owner.test is item:
                    continue
                if isinstance(owner, ast.Compare) and len(owner.ops) == 1 and isinstance(owner.ops[0], (ast.In, ast.NotIn)) and owner.comparators[0] is item:
                    continue
                if isinstance(owner, ast.Subscript) and owner.value is item:
                    continue
                if isinstance(owner, ast.Attribute) and owner.value is item and owner.attr in {"get", "keys", "items", "values", "copy", "update", "clear", "pop", "popitem", "setdefault"}:
                    call = self.parents.get(id(owner))
                    if isinstance(call, ast.Call) and call.func is owner:
                        continue
                return False
            return True
        return False

    def visit_Attribute(self, node):
        mode = "borrow" if self.borrowed(node) else "value"
        parent = self.parents.get(id(node))
        if self.factory and self.function and self.function.name == "_init_core" and isinstance(parent, ast.keyword):
            expected = self.recorder.forward.get(parent.arg)
            if expected and expected[1] == node.attr:
                mode = "transfer"
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and node.attr == "__dict__":
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_dictionary", ast.Load()),
                                             [node.value, ast.Constant(mode)], []), node)
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.attr in self.recorder.option_names | self.recorder.field_names:
            node.value = ast.copy_location(ast.Call(ast.Name("_ci_flow_write", ast.Load()),
                                                   [node.value, ast.Constant(node.attr)], []), node.value)
        if isinstance(node.ctx, ast.Load) and node.attr in self.recorder.option_names | self.recorder.field_names:
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_attribute", ast.Load()),
                                             [node.value, ast.Constant(node.attr), ast.Constant(mode)], []), node)
        return node

    def visit_Call(self, node):
        core = (self.factory and self.function and self.function.name == "_init_core"
                and isinstance(node.func, ast.Name) and node.func.id == "TokenizerCore")
        self.generic_visit(node)
        if core:
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_construct", ast.Load()),
                                             [ast.Name("self", ast.Load()), node.func, *node.args], node.keywords), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "get", "keys", "items", "values", "copy", "update", "clear", "pop", "popitem", "setdefault",
        }:
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_map_call", ast.Load()),
                                             [node.func, *node.args], node.keywords), node)
        if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "hasattr"} and len(node.args) >= 2:
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_dynamic", ast.Load()),
                                             [node.func, *node.args], node.keywords), node)
        if isinstance(node.func, ast.Name) and node.func.id in {"setattr", "delattr"} and len(node.args) >= 2:
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_mutation", ast.Load()),
                                             [node.func, *node.args], node.keywords), node)
        if isinstance(node.func, ast.Name) and node.func.id == "vars" and len(node.args) == 1 and not node.keywords:
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_vars", ast.Load()),
                                             [node.func, *node.args], []), node)
        if isinstance(node.func, ast.Name) and node.func.id in {"bool", "len"}:
            return node
        def escaped(value):
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_argument", ast.Load()), [value], []), value)
        for index, argument in enumerate(node.args):
            if isinstance(argument, ast.Starred):
                argument.value = escaped(argument.value)
            else:
                node.args[index] = escaped(argument)
        for keyword in node.keywords:
            keyword.value = escaped(keyword.value)
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            node.value = ast.copy_location(ast.Call(ast.Name("_ci_flow_write", ast.Load()), [node.value], []), node.value)
        if isinstance(node.ctx, ast.Load) and not isinstance(node.slice, ast.Slice):
            return ast.copy_location(ast.Call(ast.Name("_ci_flow_map", ast.Load()),
                                             [node.value, ast.Constant("__getitem__"), node.slice], []), node)
        return node


class FlowLoader(Loader):
    def get_code(self, fullname):
        tree = ast.parse(self.get_data(self.path), filename=self.path)
        instrument = FlowInstrument(self.recorder, factory=fullname == "sqlglot.tokens")
        # Populate the complete parent graph before alias-use analysis.
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                instrument.parents[id(child)] = parent
        tree = instrument.visit(tree)
        return compile(ast.fix_missing_locations(tree), self.path, "exec")

    def exec_module(self, module):
        module.__dict__.update(_ci_flow_attribute=self.recorder.attribute,
                              _ci_flow_construct=self.recorder.construct,
                              _ci_flow_map=self.recorder.map_operation,
                              _ci_flow_map_call=self.recorder.map_call,
                              _ci_flow_write=self.recorder.writing,
                              _ci_flow_dictionary=self.recorder.dictionary,
                              _ci_flow_argument=self.recorder.argument,
                              _ci_flow_dynamic=self.recorder.dynamic,
                              _ci_flow_mutation=self.recorder.mutation,
                              _ci_flow_vars=self.recorder.object_vars)
        importlib.machinery.SourceFileLoader.exec_module(self, module)


class FlowFinder(Finder):
    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == "sqlglot" or fullname.startswith(("sqlglot.", "tests."))):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin.endswith(".py"):
            spec.loader = FlowLoader(fullname, spec.origin, self.recorder)
        return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--resume", type=Path, help="retain evidence from completed pristine test contexts")
    parser.add_argument("--exclude", nargs="*", default=[], help="keep conservative selection for unsupported workloads")
    args = parser.parse_args()
    root = args.source.resolve()
    output = args.out.resolve()
    catalog = json.loads(args.tests.resolve().read_text())["tests"]
    recorder = FlowRecorder(root, catalog)
    if args.resume:
        with gzip.open(args.resume, "rt") as stream:
            recorder.restore(json.load(stream))
    requested = set(args.only or [test["id"] for test in catalog if test["seconds"] < 10])
    excluded = set(args.exclude)
    if not requested <= set(recorder.tests) or not excluded <= set(recorder.tests):
        raise ValueError("unknown test context")
    completed = {recorder.tests[i] for i in recorder.cpu if i not in recorder.incomplete}
    recorder.uninstrumented.update(i for i, name in enumerate(recorder.tests)
                                  if name in excluded or name not in requested | completed)
    selected = [name for name in recorder.tests if name in requested - excluded and name not in completed]
    recorder.incomplete.difference_update(recorder.by_id[name] for name in selected)
    os.environ.update(SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="2", PYTHONDONTWRITEBYTECODE="1",
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    sys.path.insert(0, str(root))
    os.chdir(root)
    sys.meta_path.insert(0, FlowFinder(recorder))
    sys.addaudithook(recorder.audit)
    concurrent.futures.ProcessPoolExecutor = InlinePool
    import pytest
    result = pytest.main([*selected, "-q", "--maxfail=1", "-p", "no:cacheprovider",
                          "-p", "pytest_timeout", "--timeout=300"], plugins=[recorder])
    data = recorder.data()
    data["incomplete"] = sorted(set(data["incomplete"]) - recorder.uninstrumented)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(output), mode="wb", mtime=0) as stream:
        stream.write(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    print("Setting flows", len(data["known"]), "blocked", len(data["blocked"]),
          "incomplete", len(data["incomplete"]), file=sys.stderr)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
