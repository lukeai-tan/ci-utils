"""Replay exact optimizer-pass graphs from complete pristine executor tasks."""

from __future__ import annotations

import ast
import enum
import hashlib
import inspect
import io
import lzma
import pickle
import sys
from pathlib import Path

from comparisoncheck import REVISION, source_hash
from queryreplay import TEST, checkpoint_digest


def blank(cls):
    return object.__new__(cls)


def restore(obj, state):
    slots, namespace = state
    for name, value in slots.items():
        object.__setattr__(obj, name, value)
    if namespace is not None:
        object.__getattribute__(obj, '__dict__').update(namespace)


class Snapshot(pickle.Pickler):
    """Retain expression caches, parent cycles, aliases and unset slots."""
    def reducer_override(self, obj):
        cls = type(obj)
        if (not cls.__module__.startswith(('sqlglot.expressions.', 'sqlglot.schema', 'sqlglot.dialects.'))
                or isinstance(obj, (enum.Enum, type))):
            return NotImplemented
        slots = {}
        for base in cls.__mro__:
            names = vars(base).get('__slots__', ())
            names = (names,) if isinstance(names, str) else names
            for name in names:
                if name not in {'__dict__', '__weakref__'}:
                    try:
                        slots[name] = object.__getattribute__(obj, name)
                    except AttributeError:
                        pass
        try:
            namespace = object.__getattribute__(obj, '__dict__')
        except AttributeError:
            namespace = None
        return blank, (cls,), (slots, namespace), None, None, restore


def snapshot(value):
    import pickletools
    stream = io.BytesIO()
    Snapshot(stream, protocol=5).dump(value)
    return pickletools.optimize(stream.getvalue())


def load_profile(path):
    try:
        with lzma.open(path, 'rb') as stream:
            data = pickle.load(stream)
        if (data['revision'] != REVISION or data['version'] != 1 or data['kind'] != 'optimizer_passes'
                or data['python'] != list(sys.version_info[:2]) or data['hashseed'] != '0'
                or not data['complete'] or not data['calls']):
            return None
        for row in data['calls']:
            if (type(row['query']) is not int or row['query'] < 0 or type(row['input']) is not bytes
                    or hashlib.sha256(row['input']).hexdigest() != row['bytes']
                    or any(type(row[k]) is not str or len(row[k]) != 64 for k in ('initial', 'output', 'state'))
                    or type(row['cpu']) not in (int, float) or not 0 <= row['cpu'] < 60
                    or not row['rule'].startswith('sqlglot.optimizer.') or ':' not in row['rule']
                    or type(row['start']) is not int or row['start'] not in {8, 9, 12}
                    or row['rule'] != {8: 'sqlglot.optimizer.eliminate_joins:eliminate_joins',
                                       9: 'sqlglot.optimizer.eliminate_ctes:eliminate_ctes',
                                       12: 'sqlglot.optimizer.canonicalize:canonicalize'}[row['start']]):
                return None
        for entries in data['codes'].values():
            for row in entries.values():
                if (not row or len(row) != len(set(row))
                        or any(type(i) is not int or not 0 <= i < len(data['calls']) for i in row)):
                    return None
        return data
    except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError, lzma.LZMAError):
        return None


def plan(profile, compiled, base, patch):
    """Require every edited frame to execute only inside certified passes."""
    from compiledcheck import changes, identity, query_reachability
    from fixturecheck import digest, independent_scope
    from selector import apply_edit, parse_patch, same_ast

    if (not profile or not compiled or not compiled.get('query_complete')
            or profile['driver'] != compiled.get('driver')
            or any(digest((Path(base) / p).read_bytes()) != h for p, h in profile['drivers'].items())):
        return None
    edits = parse_patch(patch)
    if len({edit.path for edit in edits}) != len(edits) or any(edit.path not in profile['sources'] for edit in edits):
        return None
    queries = query_reachability(compiled, base, patch)
    if queries is None:
        return None
    requested = set()
    for edit in edits:
        old = (Path(base) / edit.path).read_text()
        if profile['sources'].get(edit.path) != source_hash(old):
            return None
        new, _ = apply_edit(old, edit)
        trees = ast.parse(old), ast.parse(new)
        if not same_ast(*trees, structure=True):
            return None
        for before, after in changes(compile(old, edit.path, 'exec', dont_inherit=True),
                                     compile(new, edit.path, 'exec', dont_inherit=True)):
            binding = lambda c: (c.co_varnames, c.co_flags, c.co_freevars, c.co_cellvars)
            if (not before.co_flags & 1 or binding(before) != binding(after)
                    or not set(after.co_names) <= set(before.co_names)
                    or not independent_scope(trees[0], before) or not independent_scope(trees[1], after)):
                return None
            key = identity(before)
            calls = profile['codes'].get(edit.path, {}).get(key)
            row = compiled['codes'].get(edit.path, {}).get(key)
            if (not calls or row is None or row['query_outside']
                    or {profile['calls'][i]['query'] for i in calls} != set(row['queries'])):
                return None
            requested.update(i for i in calls if profile['calls'][i]['query'] in queries)
    if not requested:
        return None
    earliest = {}
    for i in requested:
        row = profile['calls'][i]
        previous = earliest.get(row['query'])
        if previous is None or row['start'] < profile['calls'][previous]['start']:
            earliest[row['query']] = i
    requested = sorted(earliest.values())
    return requested, sum(profile['calls'][i]['cpu'] for i in requested)


def run_suffix(context, start):
    from sqlglot.optimizer.optimizer import RULES
    possible_kwargs, outer_args, outer_kwargs, expression = context
    for rule in RULES[start:]:
        names = inspect.getfullargspec(rule).args
        kwargs = {name: possible_kwargs[name] for name in names if name in possible_kwargs}
        expression = rule(expression, **kwargs)
    return expression, outer_args, outer_kwargs


def replay(root, profile, indices):
    from fixturecheck import digest, prepare_worker, shared_state
    root = Path(root)
    if any(digest((root / p).read_bytes()) != h for p, h in profile['drivers'].items()):
        raise ValueError('optimizer pass driver changed')
    sys.path.insert(0, str(root))
    prepare_worker()
    for index in indices:
        row = profile['calls'][index]
        if shared_state() != row['state']:
            raise ValueError('optimizer pass shared state changed')
        context = pickle.loads(row['input'])
        if checkpoint_digest(context) != row['initial']:
            raise ValueError('optimizer pass input graph changed')
        result = run_suffix(context, row['start'])
        if (checkpoint_digest(result) != row['output'] or shared_state() != row['state']):
            raise ValueError('optimizer pass output or shared state changed')
    return {'passed': [TEST], 'replayed_passes': len(indices)}
