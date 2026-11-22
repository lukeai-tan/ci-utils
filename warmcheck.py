"""Fork-safe function-body replacement with cold fallback for initialization."""
import ast
import gc
import importlib
import sys
import types
from pathlib import Path

from comparisoncheck import source_hash
from compiledcheck import changes, identity


def plan(base, patch, compiled, startup):
    from selector import parse_patch, apply_edit, same_ast
    if not compiled or not startup:
        return None
    replacements = {}
    for edit in parse_patch(patch):
        if not edit.path.startswith('sqlglot/') or edit.path not in compiled['sources']:
            return None
        old = (Path(base) / edit.path).read_text()
        if compiled['sources'][edit.path] != source_hash(old):
            return None
        new, _ = apply_edit(old, edit)
        trees = ast.parse(old), ast.parse(new)
        if not same_ast(*trees, structure=True):
            return None
        try:
            pairs = list(changes(compile(old, str(Path(base) / edit.path), 'exec', dont_inherit=True),
                                 compile(new, str(Path(base) / edit.path), 'exec', dont_inherit=True)))
        except ValueError:
            return None
        for before, after in pairs:
            binding = lambda c: (c.co_argcount, c.co_posonlyargcount, c.co_kwonlyargcount,
                                 c.co_flags, c.co_varnames, c.co_freevars, c.co_cellvars)
            if (not before.co_flags & 1 or before.co_freevars or before.co_cellvars
                    or binding(before) != binding(after)
                    or not set(after.co_names) <= set(before.co_names)
                    or set(before.co_names) & {'__file__', '__code__', '__globals__', '__dict__',
                                               '__loader__', '__spec__', 'inspect', 'linecache',
                                               'open', 'getsource', 'getfile', 'exec', 'eval',
                                               'getattr', 'setattr', 'globals', 'locals', 'vars'}):
                return None
            owners = [n for n in ast.walk(trees[0]) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name == before.co_name and min([n.lineno, *[d.lineno for d in n.decorator_list]]) == before.co_firstlineno]
            if len(owners) != 1:
                return None
            first = owners[0].body[0].lineno
            if any(first <= int(n) <= max(line for _, _, line in before.co_lines() if line is not None)
                   for n in startup.get(edit.path, [])):
                return None
            replacements[(edit.path, identity(before))] = before, after
    return replacements or None


def preload(root, names):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    importlib.import_module('sqlglot')
    for name in names:
        module = name.split('::', 1)[0].removesuffix('.py').replace('/', '.')
        importlib.import_module(module)


def swap(base, replacements):
    """Install every live alias of each edited scope in an isolated fork."""
    prefix = str(Path(base).resolve()) + '/'
    found = set()
    updates = []
    for obj in gc.get_objects():
        if type(obj) is not types.FunctionType:
            continue
        code = obj.__code__
        if not code.co_filename.startswith(prefix):
            continue
        key = code.co_filename.removeprefix(prefix), identity(code)
        pair = replacements.get(key)
        if pair is None:
            continue
        before, after = pair
        # The warm copy must be pristine, including constants and local layout.
        if code != before or obj.__closure__:
            raise ValueError('warm scope differs from pristine compilation')
        updates.append((obj, after))
        found.add(key)
    if found != set(replacements):
        raise ValueError('edited scope missing from warm modules')
    for function, code in updates:
        function.__code__ = code
    return len(updates)
