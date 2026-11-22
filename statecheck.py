"""Cold imported-state deltas, including closures and resolved dispatch tables."""
from __future__ import annotations

import ast
import collections
import gc
import dis
import enum
import hashlib
import importlib
import inspect
import json
import re
import struct
import sys
import types
import typing
import logging
import unittest  # SQLGlot uses this presence to enable its real-test mode.
from pathlib import Path

from comparisoncheck import source_hash


def graph(value, *, code=True):
    """Encode supported graphs without repr, descriptor calls or order loss."""
    seen = {}
    retained = []
    def encode(item):
        # Typing and code metadata can return temporary tuples. Retain them so
        # a reused object id cannot be mistaken for a real alias later on.
        retained.append(item)
        cls = type(item)
        if item is None or cls is bool:
            return [cls.__name__, item]
        if id(item) in seen:
            return ['ref', seen[id(item)]]
        if len(seen) >= 50000:
            raise ValueError('state graph too large')
        index = seen[id(item)] = len(seen)
        if cls in (int, str):
            return [index, cls.__name__, item]
        if cls is float:
            return [index, 'float', struct.pack('!d', item).hex()]
        if cls is bytes:
            return [index, 'bytes', item.hex()]
        if cls in (tuple, list):
            return [index, cls.__name__, [encode(v) for v in item]]
        if cls in (dict, types.MappingProxyType, collections.OrderedDict, collections.Counter, collections.defaultdict):
            if cls is collections.defaultdict:
                return [index, 'defaultdict', encode(item.default_factory), [[encode(k), encode(v)] for k, v in item.items()]]
            return [index, cls.__name__, [[encode(k), encode(v)] for k, v in item.items()]]
        if cls in (type({}.keys()), type({}.values()), type({}.items())):
            backing = gc.get_referents(item)
            if len(backing) != 1 or type(backing[0]) is not dict:
                raise ValueError('unsupported dictionary view')
            return [index, cls.__name__, encode(backing[0])]
        if cls in (set, frozenset):
            # Iteration is observable. Preserve it even when address-dependent
            # order prevents a certificate in an otherwise equivalent process.
            return [index, cls.__name__, [encode(v) for v in item]]
        if isinstance(item, enum.Enum):
            return [index, 'enum', cls.__module__, cls.__qualname__,
                    encode(object.__getattribute__(item, '_name_')),
                    encode(object.__getattribute__(item, '_value_'))]
        if isinstance(item, type):
            namespace = type.__getattribute__(item, '__dict__')
            members = namespace.get('_member_map_')
            return [index, 'type', type.__getattribute__(item, '__module__'), type.__getattribute__(item, '__qualname__'),
                    encode(members) if isinstance(item, enum.EnumMeta) else None]
        if cls is types.ModuleType:
            return [index, 'module', item.__name__]
        if cls is types.CodeType:
            return [index, 'code', item.co_code.hex(), encode(item.co_consts), encode(item.co_names),
                    encode(item.co_varnames), encode(item.co_freevars), encode(item.co_cellvars),
                    item.co_flags, item.co_argcount, item.co_posonlyargcount, item.co_kwonlyargcount,
                    item.co_exceptiontable.hex(), item.co_name, item.co_qualname]
        if cls is types.CellType:
            try:
                contents = encode(item.cell_contents)
            except ValueError:
                raise ValueError('uninitialized closure') from None
            return [index, 'cell', contents]
        if cls is types.FunctionType:
            owner = item.__globals__.get('__name__')
            if type(owner) is not str or sys.modules.get(owner) is None or vars(sys.modules[owner]) is not item.__globals__:
                raise ValueError('unregistered function globals')
            return [index, 'function', item.__module__, item.__qualname__, owner,
                    encode(item.__code__) if code else None, encode(item.__defaults__),
                    encode(item.__kwdefaults__), encode(item.__closure__), encode(item.__dict__),
                    encode(item.__annotations__), encode(item.__doc__)]
        if cls is types.MethodType:
            return [index, 'method', encode(item.__func__), encode(item.__self__)]
        if cls in (staticmethod, classmethod):
            return [index, cls.__name__, encode(item.__func__)]
        if cls is property:
            return [index, 'property', encode(item.fget), encode(item.fset), encode(item.fdel)]
        if cls in (types.BuiltinFunctionType, types.BuiltinMethodType):
            owner = item.__self__
            if owner is not None and not isinstance(owner, (types.ModuleType, type)):
                raise ValueError('bound native state')
            return [index, 'builtin', item.__module__, item.__qualname__]
        if cls in (types.WrapperDescriptorType, types.MethodDescriptorType, types.ClassMethodDescriptorType):
            owner = item.__objclass__
            return [index, 'native-descriptor', owner.__module__, owner.__qualname__, item.__name__]
        if cls is re.Pattern:
            return [index, 'regex', encode(item.pattern), item.flags]
        if cls in (types.GenericAlias, types.UnionType) or cls.__module__ == 'typing':
            name = getattr(item, '_name', None)
            if type(name) is str and vars(typing).get(name) is item:
                return [index, 'typing-singleton', name]
            if cls is typing.TypeVar:
                return [index, 'typevar', item.__name__, encode(item.__constraints__), encode(item.__bound__),
                        item.__covariant__, item.__contravariant__]
            if cls is typing.ForwardRef:
                return [index, 'forward-ref', item.__forward_arg__, item.__forward_module__,
                        item.__forward_is_argument__, item.__forward_is_class__]
            origin = typing.get_origin(item)
            if origin is not None:
                return [index, 'typing', encode(origin), encode(typing.get_args(item))]
        if cls in (logging.Logger, logging.RootLogger):
            return [index, 'logger', item.name, item.level, item.disabled, item.propagate,
                    encode(item._cache), encode(item.handlers), encode(item.parent)]
        if cls.__module__.startswith(('sqlglot.expressions.', 'sqlglot.schema', 'sqlglot.planner',
                                      'sqlglot.dialects.', 'sqlglot.tokenizer_core')):
            slots = []
            for base in type.__getattribute__(cls, '__mro__'):
                names = type.__getattribute__(base, '__dict__').get('__slots__', ())
                names = (names,) if type(names) is str else names
                for name in names:
                    if name in {'__dict__', '__weakref__'}:
                        continue
                    try:
                        entry = encode(object.__getattribute__(item, name))
                    except AttributeError:
                        entry = ['unset']
                    slots.append([name, entry])
            try:
                namespace = object.__getattribute__(item, '__dict__')
            except AttributeError:
                namespace = None
            if not slots and namespace is None:
                raise ValueError('opaque state object')
            return [index, cls.__module__, cls.__qualname__, slots, encode(namespace)]
        raise ValueError('unsupported state type: ' + cls.__module__ + '.' + cls.__qualname__)
    return hashlib.sha256(json.dumps(encode(value), separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def load_all(root):
    """Use a fixed import order; CLI modules are never imported as libraries."""
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    paths = sorted((root / 'sqlglot').rglob('*.py'))
    for path in paths:
        if path.name == '__main__.py':
            continue
        parts = list(path.relative_to(root).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        importlib.import_module('.'.join(parts))
    if not Path(sys.modules['sqlglot'].__file__).absolute().is_relative_to(root):
        raise ValueError('source imported outside worker')


def roots(*, dispatch=True):
    values, classes = {}, {}
    for name, module in sorted(sys.modules.items()):
        if name != 'sqlglot' and not name.startswith('sqlglot.'):
            continue
        for key, value in sorted(vars(module).items()):
            if key in {'__builtins__', '__loader__', '__spec__', '__file__', '__cached__', '__path__'}:
                continue
            values[name + ':' + key] = value
            if isinstance(value, type) and type.__getattribute__(value, '__module__').startswith('sqlglot.'):
                classes[type.__getattribute__(value, '__module__') + '.' + type.__getattribute__(value, '__qualname__')] = value
    queue = list(classes.items())
    while queue:
        owner, cls = queue.pop()
        namespace = type.__getattribute__(cls, '__dict__')
        values[owner + ':@bases'] = type.__getattribute__(cls, '__bases__')
        for key, value in namespace.items():
            if key in {'__dict__', '__weakref__', '__module__'} or isinstance(value, (types.MemberDescriptorType, types.GetSetDescriptorType)):
                continue
            values[owner + ':' + key] = value
            if isinstance(value, type) and value.__module__.startswith('sqlglot.'):
                nested = value.__module__ + '.' + value.__qualname__
                if nested not in classes:
                    classes[nested] = value
                    queue.append((nested, value))
    if dispatch:
        from sqlglot.generator import Generator, _build_dispatch
        for owner, cls in sorted(classes.items()):
            if issubclass(cls, Generator):
                table = _build_dispatch(cls)
                values[owner + ':@dispatch'] = table
                for target, callback in table.items():
                    if isinstance(target, type):
                        values[owner + ':@entry:' + target.__module__ + '.' + target.__qualname__] = callback
    return values


def alias_topology(values, *, code=True):
    """Follow cross-cell aliases, including nested containers and closures."""
    first, edges = {}, {key: [] for key in values}
    def visit(item, root, path):
        if item is None or type(item) is bool:
            return
        previous = first.get(id(item))
        if previous is not None:
            if previous[0] != root:
                edges[root].append((path, *previous))
            return
        if len(first) > 500000:
            raise ValueError('import alias graph too large')
        first[id(item)] = root, path
        cls = type(item)
        children = []
        if cls in (list, tuple, set, frozenset):
            children = list(item)
        elif cls in (dict, collections.OrderedDict, collections.Counter, collections.defaultdict, types.MappingProxyType):
            children = [entry for pair in item.items() for entry in pair]
            if cls is collections.defaultdict:
                children.append(item.default_factory)
        elif cls in (type({}.keys()), type({}.values()), type({}.items())):
            children = gc.get_referents(item)
        elif cls is types.FunctionType:
            children = [item.__defaults__, item.__kwdefaults__, item.__dict__, item.__annotations__,
                        item.__doc__, item.__closure__]
            if code:
                children.append(item.__code__)
        elif cls is types.MethodType:
            children = [item.__func__, item.__self__]
        elif cls is types.CellType:
            try:
                children = [item.cell_contents]
            except ValueError:
                pass
        elif cls is types.CodeType:
            children = [item.co_consts, item.co_names, item.co_varnames, item.co_freevars, item.co_cellvars]
        elif cls in (staticmethod, classmethod):
            children = [item.__func__]
        elif cls is property:
            children = [item.fget, item.fset, item.fdel]
        elif isinstance(item, enum.Enum):
            children = [object.__getattribute__(item, '_name_'), object.__getattribute__(item, '_value_')]
        elif isinstance(item, enum.EnumMeta):
            children = [type.__getattribute__(item, '__dict__').get('_member_map_')]
        elif cls in (logging.Logger, logging.RootLogger):
            children = [item._cache, item.handlers, item.parent]
        elif cls in (types.GenericAlias, types.UnionType) or cls.__module__ == 'typing':
            if cls is typing.TypeVar:
                children = [item.__constraints__, item.__bound__]
            else:
                children = [typing.get_origin(item), typing.get_args(item)]
        elif cls.__module__.startswith(('sqlglot.expressions.', 'sqlglot.schema', 'sqlglot.planner',
                                        'sqlglot.dialects.', 'sqlglot.tokenizer_core')) and not isinstance(item, type):
            for base in type.__getattribute__(cls, '__mro__'):
                slots = type.__getattribute__(base, '__dict__').get('__slots__', ())
                slots = (slots,) if type(slots) is str else slots
                for name in slots:
                    if name in {'__dict__', '__weakref__'}:
                        continue
                    try:
                        children.append(object.__getattribute__(item, name))
                    except AttributeError:
                        pass
            try:
                children.append(object.__getattribute__(item, '__dict__'))
            except AttributeError:
                pass
        for i, child in enumerate(children):
            visit(child, root, path + '/' + str(i))
    # Keep every traversed object alive: ids must not be reused by ephemeral
    # mapping proxies or code metadata tuples during this traversal.
    retained = []
    original_visit = visit
    def retaining_visit(item, root, path):
        retained.append(item)
        return original_visit(item, root, path)
    visit = retaining_visit
    for key, value in values.items():
        visit(value, key, '')
    return {key: hashlib.sha256(json.dumps(paths, separators=(',', ':')).encode()).hexdigest()
            for key, paths in edges.items() if paths}


def snapshot(*, names=None, code=True, dispatch=True):
    values = roots(dispatch=dispatch)
    if names is not None:
        values = {key: value for key, value in values.items() if key.rsplit(':', 1)[1] in names}
    fingerprints = {}
    for key, value in values.items():
        try:
            fingerprints[key] = graph(value, code=code)
        except (ValueError, TypeError, RecursionError, OverflowError):
            fingerprints[key] = None
    aliases = alias_topology(values, code=code)
    return fingerprints, aliases


def delta(before, after):
    left, a = before
    right, b = after
    unknown = {key for key in left.keys() | right.keys()
               if key in left and left[key] is None or key in right and right[key] is None}
    changed = {key for key in (left.keys() | right.keys()) - unknown
               if left.get(key) != right.get(key) or a.get(key, ()) != b.get(key, ())}
    return changed - unknown, unknown


def changed_sources(base, patch):
    from selector import parse_patch, apply_edit
    changes = {}
    for edit in parse_patch(patch):
        if not edit.path.startswith('sqlglot/') or not edit.path.endswith('.py') or edit.path.endswith('/__main__.py'):
            return None
        old = (Path(base) / edit.path).read_text()
        new, _ = apply_edit(old, edit)
        if ast.dump(ast.parse(old)) != ast.dump(ast.parse(new)):
            changes[edit.path] = old, new
    if not all(safe_initialization(base, old, new, path=path, updates={p: pair[1] for p, pair in changes.items()})
               for path, (old, new) in changes.items()):
        return None
    return changes or None


def safe_initialization(base, old, new, *, path='', updates=None):
    """Allow inert construction and factories returning deferred callbacks.

    A successful import cannot certify arbitrary external side effects. Calls
    with changed arguments must be demonstrated inert from pristine source.
    """
    from selector import same_ast
    pure = set()
    for source in (Path(base) / 'sqlglot').rglob('*.py'):
        relative = source.relative_to(base).as_posix()
        tree = ast.parse((updates or {}).get(relative, source.read_text()))
        module = relative.removesuffix('.py').removesuffix('/__init__').replace('/', '.')
        for n in tree.body:
            if (isinstance(n, ast.FunctionDef) and not n.decorator_list and len(n.body) == 1
                    and isinstance(n.body[0], ast.Return) and isinstance(n.body[0].value, ast.Lambda)
                    and not n.body[0].value.args.defaults and not any(n.body[0].value.args.kw_defaults)):
                pure.add(module + '.' + n.name)
    module = path.removesuffix('.py').removesuffix('/__init__').replace('/', '.')
    factories = {name.rsplit('.', 1)[1] for name in pure if name.rsplit('.', 1)[0] == module}
    expressions = set()
    for n in ast.parse(new).body:
        if isinstance(n, ast.ImportFrom) and n.module:
            for alias in n.names:
                if n.module + '.' + alias.name in pure:
                    factories.add(alias.asname or alias.name)
                if n.module + '.' + alias.name in {'sqlglot.exp', 'sqlglot.expressions'}:
                    expressions.add(alias.asname or alias.name)
        if isinstance(n, ast.Import):
            expressions.update(a.asname for a in n.names if a.name == 'sqlglot.expressions' and a.asname)
    def inert(n):
        if isinstance(n, (ast.Constant, ast.Name)):
            return True
        if isinstance(n, ast.Attribute):
            return isinstance(n.value, ast.Name) and n.value.id in expressions and not n.attr.startswith('_')
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            return all(inert(v) for v in n.elts)
        if isinstance(n, ast.Dict):
            return all(k is None or inert(k) for k in n.keys) and all(inert(v) for v in n.values)
        if isinstance(n, ast.Lambda):
            return all(inert(v) for v in [*n.args.defaults, *[v for v in n.args.kw_defaults if v is not None]])
        if isinstance(n, ast.UnaryOp) and isinstance(n.operand, ast.Constant):
            return isinstance(n.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in factories:
            return all(isinstance(a, ast.Constant) for a in n.args) and not n.keywords
        return False
    def value_change(a, b):
        if same_ast(a, b):
            return True
        if type(a) is type(b) and isinstance(a, ast.Dict):
            return (len(a.keys) == len(b.keys)
                    and all(value_change(x, y) for x, y in zip(a.keys, b.keys))
                    and all(value_change(x, y) for x, y in zip(a.values, b.values)))
        return inert(a) and inert(b)
    def walk(a, b):
        if same_ast(a, b):
            return True
        if type(a) is not type(b):
            return False
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return all(same_ast(v, getattr(b, key)) for key, v in ast.iter_fields(a) if key != 'body')
        if isinstance(a, (ast.Assign, ast.AnnAssign)):
            return all(same_ast(v, getattr(b, key)) for key, v in ast.iter_fields(a) if key != 'value') and value_change(a.value, b.value)
        if isinstance(a, ast.ClassDef):
            return all(walk(v, getattr(b, key)) if key == 'body' else same_ast(v, getattr(b, key)) for key, v in ast.iter_fields(a))
        if isinstance(a, ast.Module):
            return walk(a.body, b.body)
        if isinstance(a, list):
            return len(a) == len(b) and all(walk(x, y) for x, y in zip(a, b))
        return False
    return walk(ast.parse(old), ast.parse(new))


def projected(compiled, base, changes, changed, catalog, *, unknown=(), dispatch=None):
    """Retain unknown/reflection readers and every affected cold code scope."""
    from compiledcheck import changes as code_changes, identity
    from selector import same_ast
    if not compiled or not compiled.get('unseen_scopes'):
        return None
    entries = [key.split(':@entry:', 1) for key in changed if ':@entry:' in key]
    dispatch_readers = None
    if entries and dispatch:
        from dispatchcheck import consumers
        dispatch_readers = consumers(dispatch, entries, catalog, base)
    changed = {key for key in changed if ':@entry:' not in key}
    names = {key.rsplit(':', 1)[1] for key in changed}
    if '@dispatch' in names:
        names.remove('@dispatch')
        names.update({'TRANSFORMS', '_build_dispatch', '_dispatch'})
    # Internal/structural changes have implicit and native readers.
    if any(name.startswith('@') or name.startswith('__') for name in names):
        return None
    names.update(key.rsplit(':', 1)[1] for key in unknown if ':@entry:' not in key)
    if any(':@entry:' in key for key in unknown):
        dispatch_readers = None
    if '@dispatch' in names:
        names.remove('@dispatch')
        names.update({'TRANSFORMS', '_build_dispatch', '_dispatch'})
    masks = 0
    reflection = {'getattr', 'hasattr', 'vars', 'dir', 'eval', 'exec', 'globals', 'locals', '__dict__'}
    for path, (old, new) in changes.items():
        if compiled['sources'].get(path) != source_hash(old):
            return None
        try:
            pairs = list(code_changes(compile(old, path, 'exec', dont_inherit=True),
                                      compile(new, path, 'exec', dont_inherit=True)))
        except ValueError:
            # Initializer edits change module constants/shape. Their final
            # outputs are compared; changed executable scopes still need reach.
            pairs = []
            if same_ast(ast.parse(old), ast.parse(new), structure=True):
                return None
            # Structural code edits can replace or add nested scopes. Retain
            # every entered function in that file rather than losing the parent.
            for key, row in compiled['codes'].get(path, {}).items():
                masks |= int(row.get('entered', '0'), 16)
        for before, after in pairs:
            if before.co_flags & 1:
                row = compiled['codes'].get(path, {}).get(identity(before))
                if row is None and identity(before) not in compiled['universe'].get(path, []):
                    return None
                masks |= int((row or {}).get('entered', '0'), 16)
    def visit(code, path):
        nonlocal masks
        row = compiled['codes'].get(path, {}).get(identity(code))
        if code.co_flags & 1:
            loaded = {str(i.argval) for i in dis.get_instructions(code)
                      if i.opname.startswith(('LOAD_GLOBAL', 'LOAD_NAME', 'LOAD_ATTR', 'LOAD_METHOD'))}
            readers = names | reflection
            if dispatch_readers is not None and path == 'sqlglot/generator.py' and code.co_name in {'sql', '_build_dispatch', '__init__'}:
                readers = readers - {'TRANSFORMS', '_build_dispatch', '_dispatch', '_DISPATCH_CACHE', 'dir', 'getattr'}
            if loaded & readers:
                if row is None and identity(code) not in compiled['universe'].get(path, []):
                    raise ValueError('missing cold scope')
                masks |= int((row or {}).get('entered', '0'), 16)
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                visit(constant, path)
    try:
        for path in compiled['sources']:
            if path.endswith('__main__.py'):
                continue
            visit(compile((Path(base) / path).read_text(), path, 'exec', dont_inherit=True), path)
    except (OSError, ValueError, SyntaxError):
        return None
    kept_ids = {compiled['tests'][i] for i in range(len(compiled['tests'])) if masks & (1 << i)}
    complete_ids = {compiled['tests'][i] for i in compiled['complete']}
    # Test-module bindings and class construction are outside SQLGlot tracing.
    cache = {}
    for test in catalog:
        path = test['file']
        if path not in cache:
            try:
                tree = ast.parse((Path(base) / path).read_text())
                tokens = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                tokens.update(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute))
                tokens.update(a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names)
                cache[path] = bool(tokens & (names | reflection))
            except (OSError, SyntaxError):
                cache[path] = True
        if cache[path]:
            kept_ids.add(test['id'])
    certified = {t['number'] for t in catalog if t['id'] in complete_ids - kept_ids}
    if entries:
        if dispatch_readers is None:
            return None
        certified.difference_update(dispatch_readers)
    return certified


if __name__ == '__main__':
    import argparse
    import gzip
    from comparisoncheck import REVISION
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    load_all(args.source)
    result = snapshot()
    data = dict(revision=REVISION, version=1, python=list(sys.version_info[:2]),
                values=result, sources={p.relative_to(args.source).as_posix(): source_hash(p.read_text())
                                       for p in (args.source / 'sqlglot').rglob('*.py')})
    with gzip.GzipFile(filename=str(args.out), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(data, separators=(',', ':')).encode())
    print('Imported cells:', len(result[0]), 'unknown:', sum(v is None for v in result[0].values()), flush=True)
