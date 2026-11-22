"""Replay affected optimizer pool cases through the original test assertions."""

from __future__ import annotations

import ast
import enum
import gzip
import hashlib
import importlib
import json
import logging
import pickle
import pkgutil
import sys
import types
import _thread
from concurrent.futures import Future
from pathlib import Path

from compiledcheck import changes, identity
from comparisoncheck import REVISION, source_hash


OWNER = 'tests/test_optimizer.py::TestOptimizer'
DRIVER = 'tests/test_optimizer.py'


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def shared_state():
    """Fingerprint optimizer data with stable set ordering and graph aliases."""
    values = {}
    lock_type = type(_thread.RLock())
    for name, module in sorted(sys.modules.items()):
        if name == 'sqlglot.optimizer' or name.startswith('sqlglot.optimizer.'):
            namespace = {}
            for key, value in sorted(vars(module).items()):
                if key.startswith('__') or isinstance(value, (types.ModuleType, type,
                        types.FunctionType, types.BuiltinFunctionType)):
                    continue
                if isinstance(value, lock_type):
                    value = ('lock', value._is_owned())
                elif isinstance(value, logging.Logger):
                    value = ('logger', value.name, value.level, value.disabled, value.propagate,
                             tuple(type(h).__module__ + '.' + type(h).__qualname__ for h in value.handlers))
                elif name == 'sqlglot.optimizer.canonicalize' and key == '_CANONICALIZE_TYPES':
                    # The pinned definition converts a set into a tuple, used
                    # solely as isinstance's membership argument. Its order is
                    # determined by class addresses in each fresh interpreter.
                    value = ('membership-types', *sorted(value, key=lambda cls: (cls.__module__, cls.__qualname__)))
                namespace[key] = value
            values[name] = namespace
    seen = {}
    def atom(value):
        cls = type(value)
        if value is None or cls in (str, int, bool):
            return [cls.__name__, value]
        if cls is float:
            return ['float', value.hex()]
        if isinstance(value, enum.Enum):
            return ['enum', cls.__module__, cls.__qualname__, value.name]
        if isinstance(value, (type, types.FunctionType, types.BuiltinFunctionType)):
            return [type(value).__name__, value.__module__, value.__qualname__]
        raise ValueError('unsupported shared state key')
    def encode(value):
        cls = type(value)
        if cls not in (dict, list, tuple, set, frozenset):
            try:
                return atom(value)
            except ValueError:
                # Typing and __future__ declarations are immutable metadata.
                if cls.__module__ in {'typing', '__future__', 'types'}:
                    return ['metadata', pickle.dumps(value, protocol=5).hex()]
                raise
        if id(value) in seen:
            return ['ref', seen[id(value)]]
        index = seen[id(value)] = len(seen)
        if cls is dict:
            items = list(value.items())
            # Dispatch tables built from sets of classes have process-dependent
            # insertion order. Their pinned readers only perform key lookups.
            if items and all(isinstance(k, type) for k, _ in items):
                items.sort(key=lambda item: (item[0].__module__, item[0].__qualname__))
            return [index, 'dict', [[encode(k), encode(v)] for k, v in items]]
        if cls in (set, frozenset):
            ordered = sorted(value, key=lambda item: json.dumps(atom(item)))
            return [index, cls.__name__, [encode(v) for v in ordered]]
        return [index, cls.__name__, [encode(v) for v in value]]
    try:
        return digest(json.dumps(encode(values), separators=(',', ':'), ensure_ascii=True).encode())
    except (TypeError, ValueError, pickle.PickleError, RecursionError):
        return None


def prepare_worker():
    # Give recorded and replayed tasks the same complete optimizer namespace.
    # The profiler marks these imports as startup dependencies of every case.
    package = importlib.import_module('sqlglot.optimizer')
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
        importlib.import_module('sqlglot.optimizer.' + info.name)


def checked_task(payload, expected, after):
    function, args, kwargs = pickle.loads(payload)
    prepare_worker()
    if not expected or shared_state() != expected:
        raise ValueError('optimizer task initial shared state changed')
    result = function(*args, **kwargs)
    pickle.dumps(result, protocol=5)
    if not after or shared_state() != after:
        raise ValueError('optimizer task shared state differs from baseline')
    return result


def load_profile(path):
    try:
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        if (data['revision'] != REVISION or data['version'] not in (1, 2) or data['kind'] != 'optimizer_cases'
                or data.get('state_kind') != 'optimizer_globals_v1'
                or data['python'] != list(sys.version_info[:2]) or data['hashseed'] != '0'):
            return None
        for name, row in data['tests'].items():
            fixtures = row.get('fixtures', [])
            if (fixtures and sum(len(f['records']) for f in fixtures) != len(row['tasks'])) or any(
                    not f['path'].startswith('tests/fixtures/optimizer/') or not f['path'].endswith('.sql')
                    or len(f['hash']) != 64 or any(type(v) is not str or len(v) != 64 for v in f['records'])
                    for f in fixtures):
                return None
            limit = 1 << len(row['tasks'])
            if data['version'] == 2 and any(not 0 <= int(mask, 16) < limit for mask in row['sets']):
                return None
            def valid(cases):
                if data['version'] == 1:
                    return type(cases) is list and all(type(i) is int and 0 <= i < len(row['tasks']) for i in cases)
                return type(cases) is int and 0 <= cases < len(row['sets'])
            if (not name.startswith(OWNER + '::') or not row['tasks'] or row['cpu'] < 0
                    or any(len(task['input']) != 64 or task['cpu'] < 0 for task in row['tasks'])
                    or any(not valid(entry['cases']) or any(not valid(cases) for cases in entry['lines'].values())
                           for entries in row['codes'].values() for entry in entries.values())):
                return None
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def case_numbers(row, value):
    if type(value) is list:
        return set(value)
    mask = int(row['sets'][value], 16)
    found = set()
    while mask:
        bit = mask & -mask
        found.add(bit.bit_length() - 1)
        mask ^= bit
    return found


def compact(data):
    """Intern repeated consumer sets so runtime maps do not inflate fork costs."""
    import copy
    result = copy.deepcopy(data)
    for row in result['tests'].values():
        sets, ids = [], {}
        def index(cases):
            mask = format(sum(1 << i for i in set(cases)), 'x')
            if mask not in ids:
                ids[mask] = len(sets)
                sets.append(mask)
            return ids[mask]
        for entries in row['codes'].values():
            for entry in entries.values():
                entry['cases'] = index(entry['cases'])
                entry['lines'] = {line: index(cases) for line, cases in entry['lines'].items()}
        row['sets'] = sets
    result['version'] = 2
    return result


def fixture_records(text):
    """Decode the pinned pair loader, including metadata and expectations."""
    statements = '\n'.join(line for line in text.splitlines()
                           if line and not line.startswith('--')).split(';')
    records = []
    try:
        for i in range(0, len(statements) - 1, 2):
            lines = statements[i].strip().split('\n')
            meta, n = {}, 0
            while lines[n].startswith('#'):
                key, value = lines[n].split(':', 1)
                meta[key.lstrip('#').strip()] = value.strip()
                n += 1
            record = [meta, '\n'.join(lines[n:]), statements[i + 1].strip()]
            records.append(digest(json.dumps(record, separators=(',', ':'), ensure_ascii=True).encode()))
    except (IndexError, ValueError) as error:
        raise ValueError('unsupported fixture record') from error
    return records


def fixture_inputs(root, method, task_count):
    """Bind task ordinals to unconditional pinned check_file pair loops."""
    calls = []
    def visit(node):
        if node is not method and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'check_file' for n in ast.walk(node)):
                raise ValueError('nested fixture loop')
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.TryStar, ast.Match)):
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'check_file' for n in ast.walk(node)):
                raise ValueError('conditional fixture loop')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'check_file':
            if (not node.args or not isinstance(node.args[0], ast.Constant)
                    or type(node.args[0].value) is not str
                    or any(k.arg in {'only', None} for k in node.keywords)):
                raise ValueError('dynamic fixture loop')
            calls.append(node.args[0].value)
        for child in ast.iter_child_nodes(node):
            visit(child)
    try:
        visit(method)
        fixtures = []
        for name in calls:
            path = 'tests/fixtures/optimizer/' + name + '.sql'
            payload = (Path(root) / path).read_bytes()
            fixtures.append(dict(path=path, hash=digest(payload), records=fixture_records(payload.decode('utf-8'))))
        return fixtures if sum(len(f['records']) for f in fixtures) == task_count else []
    except (OSError, UnicodeError, ValueError):
        return []


def independent_scope(tree, code):
    """Exclude shared writes, closures and aliases of global objects.

    The pinned driver pickles a fresh expression/schema for each task and has no
    pool initializer. Local expression mutations belong to that task. Changes
    that could mutate module/class state cannot reuse other tasks' verdicts.
    """
    if code.co_freevars or code.co_cellvars:
        return False
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == code.co_name
                 and min([n.lineno, *[d.lineno for d in n.decorator_list]]) == code.co_firstlineno]
    if len(functions) != 1:
        return False
    function = functions[0]
    if function.decorator_list:
        return False
    nodes = [n for stmt in function.body for n in ast.walk(stmt)]
    if any(isinstance(n, (ast.Global, ast.Nonlocal, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef, ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await)) for n in nodes):
        return False
    locals_ = set(code.co_varnames)
    def root(node):
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None
    aliases = set()
    while True:
        previous = set(aliases)
        for n in nodes:
            if not isinstance(n, (ast.Assign, ast.AnnAssign)):
                continue
            if isinstance(n.value, (ast.Name, ast.Attribute, ast.Subscript)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if root(n.value) not in locals_ - aliases:
                    aliases.update(t.id for t in targets if isinstance(t, ast.Name))
            elif (isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                  and n.value.func.id in {'type', 'getattr'}):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                aliases.update(t.id for t in targets if isinstance(t, ast.Name))
        if aliases == previous:
            break
    for n in nodes:
        if isinstance(n, ast.Attribute) and n.attr in {'__class__', '__dict__'}:
            return False
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
            if root(n) not in locals_ - aliases:
                return False
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id in {'setattr', 'delattr', 'globals', 'locals', 'vars', 'exec', 'eval'}:
                return False
            if isinstance(n.func, ast.Attribute) and root(n.func.value) in aliases:
                return False
    return True


def plan(profile, base, patch, names, *, allow_delta=True):
    """Price complete methods with only provably unaffected pool cases skipped."""
    from selector import apply_edit, parse_patch, same_ast

    if profile is None or profile['python'] != list(sys.version_info[:2]):
        return {}
    names = [name for name in names if name in profile['tests']]
    if not names:
        return {}
    edits = parse_patch(patch)
    code_edits = [e for e in edits if e.path.startswith('sqlglot/optimizer/')]
    data_edits = [e for e in edits if e.path.startswith('tests/fixtures/optimizer/') and e.path.endswith('.sql')]
    if data_edits and not allow_delta:
        return {}
    if (not edits or len(code_edits) + len(data_edits) != len(edits)
            or len({e.path for e in edits}) != len(edits)):
        return {}
    if any(digest((Path(base) / p).read_bytes()) != h for p, h in profile['drivers'].items()):
        return {}
    scopes = []
    for edit in code_edits:
        old = (Path(base) / edit.path).read_text()
        if profile['sources'].get(edit.path) != source_hash(old):
            return {}
        new, _ = apply_edit(old, edit)
        trees = ast.parse(old), ast.parse(new)
        if not same_ast(*trees, structure=True):
            return {}
        pairs = list(changes(compile(old, edit.path, 'exec', dont_inherit=True),
                             compile(new, edit.path, 'exec', dont_inherit=True)))
        for before, after in pairs:
            binding = lambda c: (c.co_varnames, c.co_flags, c.co_freevars, c.co_cellvars)
            if (not before.co_flags & 1 or binding(before) != binding(after)
                    or not set(after.co_names) <= set(before.co_names)
                    or not independent_scope(trees[0], before) or not independent_scope(trees[1], after)):
                return {}
            from statementcheck import locations
            scopes.append((edit.path, identity(before), locations(*trees, before, after)))
    found = {}
    for name in names:
        row = profile['tests'].get(name)
        if not row or any(not task.get('state') or not task.get('after', task.get('state') if task.get('pure') else None)
                          for task in row['tasks']):
            continue
        # A task that changed shared state on the pinned run is never skipped,
        # even if it did not execute an edited scope.
        affected = {i for i, task in enumerate(row['tasks']) if not task.get('pure')}
        delta_valid, delta_seen = True, False
        offset = 0
        for fixture in row.get('fixtures', []):
            if digest((Path(base) / fixture['path']).read_bytes()) != fixture['hash']:
                delta_valid = False
                break
            changes_here = [e for e in data_edits if e.path == fixture['path']]
            if changes_here:
                delta_seen = True
                old = (Path(base) / fixture['path']).read_text(encoding='utf-8')
                new, _ = apply_edit(old, changes_here[0])
                records = fixture_records(new)
                original = fixture['records']
                if len(records) != len(original):
                    delta_valid = False
                    break
                from collections import Counter
                if records != original and Counter(records) == Counter(original):
                    delta_valid = False
                    break
                affected.update(offset + i for i, (a, b) in enumerate(zip(original, records)) if a != b)
            offset += len(fixture['records'])
        if data_edits and (not delta_valid or not row.get('fixtures')
                           or offset != len(row['tasks']) or (not code_edits and not delta_seen)):
            continue
        for path, key, spans in scopes:
            entry = row['codes'].get(path, {}).get(key)
            if entry is None:
                if key not in profile['universe'].get(path, []):
                    affected.update(range(len(row['tasks'])))
                continue
            if entry['startup']:
                affected.update(range(len(row['tasks'])))
            elif spans:
                # Case tracing stores every executed optimizer line. A whole
                # scope is the fallback when source spans cannot be certified.
                for start, end in spans:
                    observed = False
                    for line, cases in entry['lines'].items():
                        if start <= int(line) <= end:
                            observed = True
                            affected.update(case_numbers(row, cases))
                    if not observed:
                        affected.update(case_numbers(row, entry['cases']))
            else:
                affected.update(case_numbers(row, entry['cases']))
        whole = sum(t['cpu'] for t in row['tasks'])
        share = sum(row['tasks'][i]['cpu'] for i in affected) / whole if whole else 1.0
        # Account for real spawn startup even when very few cases remain. Direct
        # assertions and fixture parsing are always run and included in base_cpu.
        estimate = row['base_cpu'] + max(0.0, row['cpu'] - row['base_cpu']) * share
        if affected:
            estimate += 0.7
        if len(affected) < len(row['tasks']) and estimate < row['cpu'] * 0.7:
            found[name] = {'indices': sorted(affected), 'cpu': estimate, 'task_share': share,
                           'fixture_delta': delta_seen}
    return found


class Filter:
    """Use actual spawned workers, preserving setup, inputs and assertions."""
    def __init__(self, profile, plans):
        self.profile, self.plans = profile, plans
        self.current = None
        self.count = 0
        self.counts = {}
        self.skipped = set()

    def install(self):
        from tests import test_optimizer as driver
        original_pool, completed = driver.ProcessPoolExecutor, driver.as_completed
        owner = self
        class Pool(original_pool):
            def submit(self, function, /, *args, **kwargs):
                name, index = owner.current, owner.count
                owner.count += 1
                row = owner.profile['tests'].get(name)
                selected = owner.plans.get(name)
                if selected is not None and row and index < len(row['tasks']) and index not in selected['indices']:
                    task = row['tasks'][index]
                    # Changed inputs are executed even when baseline coverage
                    # would have excluded the task. Unknown serialization also
                    # executes the real task and can never clear it by a digest.
                    try:
                        matches = digest(pickle.dumps((function, args, kwargs), protocol=5)) == task['input']
                    except (TypeError, ValueError, pickle.PickleError):
                        matches = False
                    if matches and function is driver.parse_and_optimize:
                        future = Future()
                        future.set_result(None)
                        owner.skipped.add(future)
                        return future
                if selected is not None and row and index < len(row['tasks']):
                    payload = pickle.dumps((function, args, kwargs), protocol=5)
                    task = row['tasks'][index]
                    return super().submit(checked_task, payload, task['state'], task.get('after', task['state']))
                return super().submit(function, *args, **kwargs)
        def as_completed(futures, *args, **kwargs):
            return completed([f for f in futures if f not in owner.skipped], *args, **kwargs)
        driver.ProcessPoolExecutor, driver.as_completed = Pool, as_completed

    def pytest_runtest_call(self, item):
        self.current, self.count = item.nodeid, 0

    def pytest_runtest_teardown(self, item):
        self.counts[item.nodeid] = self.count
        self.current = None

    def complete(self, name):
        return name not in self.plans or self.counts.get(name) == len(self.profile['tests'][name]['tasks'])
