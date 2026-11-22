"""Complete affected-stage replay for the original executor workloads."""
from __future__ import annotations

import ast
import dis
import gzip
import hashlib
import importlib
import inspect
import lzma
import os
import pickle
import sys
import time
import types
from pathlib import Path

from comparisoncheck import REVISION, source_hash
from compiledcheck import changes, identity
from passcheck import snapshot as pack_graph
from statecheck import graph, snapshot, load_all

METHODS = ('test_execute_tpcds', 'test_execute_tpch', 'test_optimized_tpch')
TARGETS = {'parse': ('sqlglot', 'parse_one'), 'sql': ('sqlglot', 'transpile'),
           'opt': ('sqlglot.executor', 'optimize'), 'plan': ('sqlglot.executor', 'Plan'),
           'schema': ('sqlglot.executor', 'ensure_schema'),
           'format': ('sqlglot.expressions.core', 'Expression.sql')}


def state(cells):
    values = {}
    for owner, name in cells:
        parts = owner.split('.')
        for end in range(len(parts), 0, -1):
            module = sys.modules.get('.'.join(parts[:end]))
            if module is not None:
                value = module
                for part in parts[end:]:
                    value = inspect.getattr_static(value, part)
                values[owner + ':' + name] = inspect.getattr_static(value, name)
                break
        else:
            raise ValueError('stage owner missing')
    return graph(values, code=False)


def prepare(root):
    """Materialize the unchanged memoized dispatch used by these workloads.

    Only the three original receiver classes are needed. The driver hash in
    the profile prevents edits to cache construction or access from using this
    preparation in place of the original lazy construction.
    """
    load_all(root)
    from sqlglot.generator import Generator, _build_dispatch, _DISPATCH_CACHE
    from sqlglot.generators.python import PythonGenerator
    from sqlglot.generators.duckdb import DuckDBGenerator
    _DISPATCH_CACHE.clear()
    for cls in (Generator, PythonGenerator, DuckDBGenerator):
        _DISPATCH_CACHE[cls] = _build_dispatch(cls)
    # The real executor probes this shared logger before reaching its stages.
    import logging
    logging.getLogger('sqlglot').isEnabledFor(logging.DEBUG)


def execute(row):
    module, name = TARGETS[row['phase']]
    function = importlib.import_module(module)
    for part in name.split('.'):
        function = getattr(function, part)
    args, kwargs = pickle.loads(row['input'])
    if graph((args, kwargs)) != row['initial']:
        raise ValueError('stage input graph changed')
    result = function(*args, **kwargs)
    if row['phase'] == 'parse':
        result = pickle.loads(pickle.dumps(result, protocol=5))
    return graph((result, args, kwargs))


def load_profile(path):
    try:
        with lzma.open(path, 'rb') as stream:
            data = pickle.load(stream)
        if (data['revision'] != REVISION or data['version'] != 1 or not data['complete']
                or data['python'] != list(sys.version_info[:2]) or data['hashseed'] != '0'
                or type(data.get('dispatch_driver')) is not str or len(data['dispatch_driver']) != 64
                or not data['calls'] or not set(data['methods']) <= set(METHODS)):
            return None
        for row in data['calls']:
            if (row['phase'] not in TARGETS or row['method'] not in METHODS
                    or type(row['input']) is not bytes or hashlib.sha256(row['input']).hexdigest() != row['bytes']
                    or not 0 <= row['cpu'] < 60 or not isinstance(row['cells'], list)):
                return None
        return data
    except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.PickleError, lzma.LZMAError):
        return None


def plan(data, base, patch, methods):
    """Every changed scope must be covered inside recorded stages only."""
    from selector import parse_patch, apply_edit, same_ast
    from fixturecheck import independent_scope
    edits = parse_patch(patch)
    if not data or not edits or len({e.path for e in edits}) != len(edits):
        return {}
    if any(source_hash((Path(base) / path).read_text()) != value for path, value in data['drivers'].items()):
        return {}
    scopes = []
    viable = set(methods) & set(data['methods'])
    for edit in edits:
        if edit.path not in data['sources']:
            return {}
        old = (Path(base) / edit.path).read_text()
        if source_hash(old) != data['sources'][edit.path]:
            return {}
        new, _ = apply_edit(old, edit)
        if edit.path == 'sqlglot/generator.py':
            from dispatchcheck import driver_hash
            if driver_hash(new) != data['dispatch_driver']:
                return {}
        trees = ast.parse(old), ast.parse(new)
        if not same_ast(*trees, structure=True):
            return {}
        for before, after in changes(compile(old, edit.path, 'exec', dont_inherit=True), compile(new, edit.path, 'exec', dont_inherit=True)):
            observed = data['codes'].get(edit.path, {}).get(identity(before))
            if observed is None:
                return {}
            viable = {method for method in viable if method not in observed['outside']
                      and all(data['calls'][i]['safe'] for i in observed['calls']
                              if data['calls'][i]['method'] == method)}
            if not viable:
                return {}
            binding = lambda c: (c.co_varnames, c.co_freevars, c.co_cellvars, c.co_flags)
            if (not before.co_flags & 1 or binding(before) != binding(after)
                    or not set(after.co_names) <= set(before.co_names)
                    or not independent_scope(trees[0], before) or not independent_scope(trees[1], after)):
                return {}
            scopes.append((edit.path, identity(before)))
    planned = {}
    for method in viable:
        if method not in data['methods']:
            continue
        indices = set()
        for path, key in scopes:
            row = data['codes'].get(path, {}).get(key)
            if row is None or method in row['outside']:
                break
            indices.update(i for i in row['calls'] if data['calls'][i]['method'] == method)
        else:
            if indices and all(data['calls'][i]['safe'] for i in indices):
                planned[method] = sorted(indices), sum(data['calls'][i]['cpu'] for i in indices)
    return planned


def replay(root, data, indices):
    if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(data['calls']) for i in indices):
        raise ValueError('invalid stage request')
    root = Path(root)
    if any(source_hash((root / path).read_text()) != value for path, value in data['drivers'].items()):
        raise ValueError('executor inputs or driver changed')
    os.chdir(root)
    from dispatchcheck import driver_hash
    if driver_hash((root / 'sqlglot/generator.py').read_text()) != data['dispatch_driver']:
        raise ValueError('dispatch preparation driver changed')
    prepare(root)
    checked = set()
    for index in indices:
        row = data['calls'][index]
        if not row['safe'] or state(row['cells']) != row['state']:
            raise ValueError('stage initial shared state changed')
        output = execute(row)
        if output != row['output'] or state(row['cells']) != row['state']:
            raise ValueError('stage output or shared state changed')
        checked.add(index)
    methods = {data['calls'][i]['method'] for i in checked}
    return {'passed': ['tests/test_executor.py::TestExecutor::' + m for m in sorted(methods)],
            'replayed_stages': len(checked)}


def build(root, boundary_map=None):
    """Trace the complete original tests and validate each saved graph replay."""
    from querycheck import SerialPool
    root = Path(root).resolve()
    os.chdir(root)
    os.environ.update(SKIP_INTEGRATION='0', PYTHON_CPU_COUNT='2', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    data = dict(revision=REVISION, version=1, complete=False, python=list(sys.version_info[:2]),
                hashseed=os.environ.get('PYTHONHASHSEED'), calls=[], codes={}, sources={}, drivers={}, methods=[])
    current = {'method': '', 'index': None}
    cache = {}
    ref_cache = {}
    def trace(frame, event, arg):
        if event != 'call':
            return
        code = frame.f_code
        code_id = id(code)
        if code_id not in cache:
            try:
                path = Path(code.co_filename).relative_to(root).as_posix()
            except ValueError:
                path = ''
            if path.startswith('sqlglot/'):
                scope = identity(code)
                globals_ = {str(i.argval) for i in dis.get_instructions(code) if i.opname.startswith('LOAD_GLOBAL')}
                row = data['codes'].setdefault(path, {}).setdefault(scope, {'calls': set(), 'outside': set(), 'names': set()})
                row['names'].update(code.co_names)
                cache[code_id] = code, (globals_, row)
            else:
                cache[code_id] = code, None
        key = cache[code_id][1]
        if key is None:
            return
        globals_, row = key
        if current['index'] is None:
            row['outside'].add(current['method'])
        else:
            row['calls'].add(current['index'])
            cells = data['calls'][current['index']]['cells']
            receiver = frame.f_locals.get('self', frame.f_locals.get('cls'))
            cls = receiver if isinstance(receiver, type) else type(receiver)
            reference_key = (code_id, id(cls))
            if reference_key in ref_cache:
                cells.update(ref_cache[reference_key][1])
                return
            references = set()
            module = frame.f_globals.get('__name__', '')
            for name in globals_:
                if name in frame.f_globals:
                    references.add((module, name))
                    value = frame.f_globals[name]
                    if type(value) is types.ModuleType and value.__name__.startswith('sqlglot.'):
                        references.update((value.__name__, n) for n in code.co_names if n in vars(value))
            if cls.__module__.startswith('sqlglot.'):
                for name in code.co_names:
                    try:
                        raw = inspect.getattr_static(cls, name)
                    except AttributeError:
                        continue
                    if not isinstance(raw, (types.MemberDescriptorType, types.GetSetDescriptorType)):
                        references.add((cls.__module__ + '.' + cls.__qualname__, name))
            ref_cache[reference_key] = cls, references
            cells.update(references)
    saved = []
    def wrap(module, name, phase):
        original = getattr(module, name)
        saved.append((module, name, original))
        def invoke(*args, **kwargs):
            if current['index'] is not None:
                return original(*args, **kwargs)
            previous = sys.getprofile()
            sys.setprofile(None)
            index = len(data['calls'])
            row = {'phase': phase, 'method': current['method'], 'safe': False, 'cpu': 0, 'names': [],
                   'input': b'', 'initial': '', 'output': '', 'cells': set()}
            try:
                row['input'], row['initial'] = pack_graph((args, kwargs)), graph((args, kwargs))
            except (ValueError, TypeError, RecursionError) as error:
                row['unsupported'] = str(error)
            row['bytes'] = hashlib.sha256(row['input']).hexdigest()
            data['calls'].append(row)
            current['index'] = index
            try:
                sys.setprofile(trace)
                try:
                    result = original(*args, **kwargs)
                finally:
                    sys.setprofile(None)
                delivered = pickle.loads(pickle.dumps(result, protocol=5)) if phase == 'parse' else result
                try:
                    row['output'] = graph((delivered, args, kwargs))
                except (ValueError, TypeError, RecursionError) as error:
                    row['unsupported'] = str(error)
                return result
            finally:
                current['index'] = None
                sys.setprofile(previous)
        setattr(module, name, invoke)
    sys.setprofile(trace)
    prepare(root)
    import tests.test_executor as driver
    import sqlglot
    import sqlglot.executor as executor
    original_pool = driver.ProcessPoolExecutor
    driver.ProcessPoolExecutor = SerialPool
    wrap(driver, 'parse_one', 'parse')
    wrap(driver, 'transpile', 'sql')
    wrap(sqlglot, 'transpile', 'sql')
    wrap(executor, 'optimize', 'opt')
    wrap(executor, 'Plan', 'plan')
    wrap(executor, 'ensure_schema', 'schema')
    wrap(sqlglot.exp.Expression, 'sql', 'format')
    # Changing a class into a wrapper would break PythonExecutor's type checks;
    # its Plan reference is used only to construct a plan in executor.execute.
    sys.setprofile(trace)
    try:
        for method in METHODS:
            current['method'] = method
            import unittest
            verdict = unittest.TestResult()
            driver.TestExecutor.setUpClass()
            try:
                driver.TestExecutor(method).run(verdict)
            finally:
                driver.TestExecutor.tearDownClass()
            if not verdict.wasSuccessful() or verdict.testsRun != 1 or verdict.skipped:
                for _, failure in (verdict.errors + verdict.failures)[:2]:
                    print(failure, file=sys.stderr)
                raise ValueError('original executor stage test incomplete: ' + method)
            data['methods'].append(method)
            print('Recorded', method, 'stages', len(data['calls']), file=sys.stderr, flush=True)
    finally:
        sys.setprofile(None)
        driver.ProcessPoolExecutor = original_pool
        for module, name, original in reversed(saved):
            setattr(module, name, original)
    for path, entries in data['codes'].items():
        for row in entries.values():
            # Startup outside a method is shared by all methods.
            if '' in row['outside']:
                row['outside'].update(METHODS)
            for index in row['calls']:
                data['calls'][index]['names'].extend(row['names'])
            row['calls'], row['outside'] = sorted(row['calls']), sorted(row['outside'])
            del row['names']
    for index, row in enumerate(data['calls']):
        row['names'] = sorted(set(row['names']))
        row['cells'] = sorted(row['cells'])
        started = time.process_time()
        try:
            if not row['initial'] or not row['output']:
                raise ValueError('unsupported stage boundary')
            row['state'] = state(row['cells'])
            row['safe'] = execute(row) == row['output'] and state(row['cells']) == row['state']
        except (ValueError, TypeError, RecursionError) as error:
            row['safe'] = False
            row['unsupported'] = str(error)
        row['cpu'] = time.process_time() - started
        if index % 50 == 0:
            print('Validated stage', index, file=sys.stderr, flush=True)
    for path in (root / 'sqlglot').rglob('*.py'):
        data['sources'][path.relative_to(root).as_posix()] = source_hash(path.read_text())
    from dispatchcheck import driver_hash
    data['dispatch_driver'] = driver_hash((root / 'sqlglot/generator.py').read_text())
    for path in [root / 'tests/test_executor.py', root / 'tests/helpers.py',
                 *(root / 'tests/fixtures/optimizer/tpc-h').glob('*.sql'),
                 *(root / 'tests/fixtures/optimizer/tpc-ds').glob('*.sql')]:
        data['drivers'][path.relative_to(root).as_posix()] = source_hash(path.read_text())
    data['complete'] = True
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--boundaries', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('stage recording requires PYTHONHASHSEED=0')
    result = build(args.source)
    with lzma.open(args.out, 'wb') as stream:
        pickle.dump(result, stream, protocol=5)
    print('Stages:', len(result['calls']), 'replayable:', sum(r['safe'] for r in result['calls']), flush=True)
