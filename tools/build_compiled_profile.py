"""Record pristine code consumers in cold interpreters and real spawn workers."""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import gzip
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiledcheck import identity
from comparisoncheck import REVISION, source_hash
from queryreplay import TEST, checkpoint_digest, state_digest


ACTIVE = None


class Recorder:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.prefix = str(self.root) + '/'
        self.codes = {}
        self.original_codes = {}
        import ast
        self.field_names = set()
        for source_path in (self.root / 'sqlglot').rglob('*.py'):
            tree = ast.parse(source_path.read_text())
            for cls in ast.walk(tree):
                if isinstance(cls, ast.ClassDef):
                    for stmt in cls.body:
                        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.Dict):
                            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                            self.field_names.update(t.id for t in targets if isinstance(t, ast.Name))
        self.group = json.loads(os.environ.get('CI_COMPILED_GROUP', '[]'))
        self.group_mask = sum(1 << t['index'] for t in self.group)
        self.current_mask = self.group_mask
        self.current_test = 'startup'
        from field_recorder import Fields
        self.fields = Fields(lambda: self.current_mask)
        self.cache = {}
        self.local_codes = set()
        self.reads = set()
        self.unsafe = False
        self.tool = None
        self.query = None
        self.region = False
        self.checkpoints = {}

    @property
    def is_executor(self):
        return self.current_test == TEST or self.current_test == 'startup' and any(t['id'] == TEST for t in self.group)

    def row(self, code):
        cached = self.cache.get(id(code))
        if cached is not None:
            return cached[1]
        path = code.co_filename.removeprefix(self.prefix)
        row = None
        if path.startswith('sqlglot/'):
            keys = self.original_codes.get((path, code.co_firstlineno, code.co_qualname), [])
            # Instrumented scopes can share a location (e.g. multiple lambdas).
            # Union all pristine candidates; never assume one is the right one.
            row = [self.codes.setdefault(path, {}).setdefault(key, {'lines': {}, 'entered': 0,
                    'queries': set(), 'query_outside': False}) for key in keys]
        self.cache[id(code)] = code, row
        return row

    def enter(self, code, offset):
        row = self.row(code)
        if row is not None:
            if id(code) not in self.local_codes:
                sys.monitoring.set_local_events(self.tool, code, sys.monitoring.events.LINE)
                self.local_codes.add(id(code))
            for item in row:
                item['entered'] |= self.current_mask
                if self.is_executor:
                    if self.region and self.query is not None:
                        item['queries'].add(self.query)
                    else:
                        item['query_outside'] = True
        else:
            return sys.monitoring.DISABLE
        # Once per code/context is enough. Context and optimizer-boundary
        # transitions restart events so cached hits cannot hide later consumers.
        return sys.monitoring.DISABLE

    def line(self, code, line):
        row = self.row(code)
        if row is not None:
            for item in row:
                item['lines'][line] = item['lines'].get(line, 0) | self.current_mask
        return sys.monitoring.DISABLE

    def audit(self, event, args):
        if event in {'subprocess.Popen', 'os.fork', 'os.forkpty'}:
            self.unsafe = True
        if event == 'open':
            try:
                path = Path(os.fsdecode(args[0])).resolve().relative_to(self.root).as_posix()
                # Import machinery reads .pyc or uses cached bytecode. Explicit
                # source readers are also retained by the existing baseline map.
                if path.startswith('sqlglot/') and path.endswith('.py'):
                    self.reads.add(path)
            except (OSError, TypeError, ValueError):
                pass

    def install(self):
        from field_recorder import Finder
        sys.meta_path.insert(0, Finder(self))
        m = sys.monitoring
        self.tool = next(i for i in range(5, -1, -1) if m.get_tool(i) is None)
        m.use_tool_id(self.tool, 'cold-code-consumers')
        m.register_callback(self.tool, m.events.PY_START, self.enter)
        m.register_callback(self.tool, m.events.PY_RESUME, self.enter)
        m.register_callback(self.tool, m.events.LINE, self.line)
        m.set_events(self.tool, m.events.PY_START | m.events.PY_RESUME)
        sys.addaudithook(self.audit)

    def data(self):
        return {'codes': {path: {key: {'entered': format(row['entered'], 'x'),
                                      'lines': {str(line): format(mask, 'x') for line, mask in row['lines'].items()},
                                      'queries': sorted(row['queries']), 'query_outside': row['query_outside']}
                                for key, row in entries.items()} for path, entries in self.codes.items()},
                'unsafe': self.unsafe, 'fields': self.fields.data(), 'checkpoints': self.checkpoints}

    def optimize(self, function, /, *args, **kwargs):
        if not self.is_executor or self.query is None:
            return function(*args, **kwargs)
        driver = sys.modules['tests.test_executor']
        started = time.process_time()
        before = state_digest((driver._schema, driver._tables))
        previous = self.region
        self.region = True
        sys.monitoring.restart_events()
        try:
            expression = function(*args, **kwargs)
        finally:
            self.region = previous
            sys.monitoring.restart_events()
        try:
            output = checkpoint_digest((expression, args, kwargs))
        except (ValueError, TypeError, RecursionError):
            output = None
        self.checkpoints.setdefault(str(self.query), []).append({
            'kind': 'expression_graph_v1', 'output': output,
            'pure': before == state_digest((driver._schema, driver._tables)),
            'cpu': time.process_time() - started})
        return expression


def child_bootstrap():
    global ACTIVE
    arguments = getattr(sys, 'orig_argv', sys.argv)
    if any('from multiprocessing.resource_tracker import main' in arg for arg in arguments):
        return
    if any(name == 'sqlglot' or name.startswith('sqlglot.') for name in sys.modules):
        raise ValueError('source imported before cold tracing')
    ACTIVE = Recorder(os.environ['CI_COMPILED_ROOT'])
    ACTIVE.install()
    directory = Path(os.environ['CI_COMPILED_OUTPUT'])
    (directory / f'start-{os.getpid()}.json').write_text('{}')
    def save():
        (directory / f'end-{os.getpid()}.json').write_text(json.dumps(ACTIVE.data(), separators=(',', ':')))
    atexit.register(save)


class Verdict:
    def __init__(self):
        self.passed = set()
        self.failed = False

    def pytest_runtest_call(self, item):
        recorder = ACTIVE
        recorder.current_test = item.nodeid
        recorder.current_mask = next(1 << t['index'] for t in recorder.group if t['id'] == item.nodeid)
        sys.monitoring.restart_events()

    def pytest_runtest_teardown(self, item):
        ACTIVE.current_test = 'startup'
        ACTIVE.current_mask = ACTIVE.group_mask
        sys.monitoring.restart_events()

    def pytest_runtest_logreport(self, report):
        self.failed |= report.failed or report.skipped
        if report.passed:
            self.passed.add(report.when)


def measured_task(test, mask, index, payload):
    ACTIVE.current_test, ACTIVE.current_mask = test, mask
    ACTIVE.query = index if test == TEST else None
    sys.monitoring.restart_events()
    try:
        function, args, kwargs = pickle.loads(payload)
        return function(*args, **kwargs)
    finally:
        ACTIVE.query = None


def read_traces(target):
    verdict = target / 'verdict.json'
    files = list(target.glob('start-*.json'))
    if not verdict.exists() or not json.loads(verdict.read_text())['passed']:
        return False, []
    traces = []
    for file in files:
        completed = file.with_name(file.name.replace('start-', 'end-', 1))
        if not completed.exists():
            return False, []
        trace = json.loads(completed.read_text())
        if trace['unsafe']:
            return False, []
        traces.append(trace)
    return bool(traces), traces


def run_one(root, group, directory, reuse=False, timeout=1800):
    target = Path(directory)
    if reuse:
        return read_traces(target)
    target.mkdir()
    hooks = target / 'hooks'
    hooks.mkdir()
    (hooks / 'sitecustomize.py').write_text('from build_compiled_profile import child_bootstrap\nchild_bootstrap()\n')
    env = {**os.environ, 'CI_COMPILED_ROOT': str(root), 'CI_COMPILED_OUTPUT': str(target), 'CI_COMPILED_GROUP': json.dumps(group),
           'PYTHONPATH': os.pathsep.join([str(hooks), str(Path(__file__).resolve().parent), str(root)]),
           'SKIP_INTEGRATION': '0', 'PYTHONHASHSEED': '0', 'PYTHON_CPU_COUNT': '2',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'NUMEXPR_MAX_THREADS': '1'}
    with (target / 'run.log').open('w') as log:
        try:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--one', json.dumps(group),
                                     '--source', str(root), '--out', str(target / 'verdict.json')],
                                    cwd=root, env=env, stdout=log, stderr=log, start_new_session=True)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return False, []
    if process.returncode:
        return False, []
    return read_traces(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tests', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--one')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--owner')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--fields-out', type=Path)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if args.one:
        import pytest
        from build_compiled_profile import ACTIVE as recorder, Verdict as GroupVerdict, measured_task as worker_task
        group = json.loads(args.one)
        original_pool = concurrent.futures.ProcessPoolExecutor
        class Pool(original_pool):
            def submit(self, function, /, *a, **kw):
                index = getattr(self, 'query_index', 0)
                self.query_index = index + 1
                return super().submit(worker_task, recorder.current_test, recorder.current_mask,
                                      index, pickle.dumps((function, a, kw), protocol=5))
        concurrent.futures.ProcessPoolExecutor = Pool
        verdict = GroupVerdict()
        status = pytest.main([*[t['id'] for t in group], '-q', '-p', 'no:cacheprovider', '-p', 'pytest_timeout', '--timeout=1500'], plugins=[verdict])
        args.out.write_text(json.dumps({'passed': status == 0 and not verdict.failed
                                       and verdict.passed == {'setup', 'call', 'teardown'}}))
        return
    catalog = json.loads(args.tests.read_text())['tests']
    root = args.source.resolve()
    with gzip.open(Path(__file__).resolve().parents[1] / 'maps/baseline.json.gz', 'rt') as stream:
        baseline = json.load(stream)
    sources = {path: source for path, source in baseline['sources'].items() if path.startswith('sqlglot/')}
    if (baseline['revision'] != REVISION or any((root / path).read_text() != text for path, text in sources.items())
            or set(sources) != {p.relative_to(root).as_posix() for p in (root / 'sqlglot').rglob('*.py')}):
        raise ValueError('pristine pinned source required')
    indices = [i for i, t in enumerate(catalog) if (args.only is None or t['id'] in args.only)
               and (args.owner is None or t['id'].rsplit('::', 1)[0] == args.owner)]
    if args.only is not None and set(args.only) != {catalog[i]['id'] for i in indices}:
        raise ValueError('unknown test context')
    codes, fields, checkpoints, complete = {}, {}, {}, set()
    groups = defaultdict(list)
    for index in indices:
        groups[catalog[index]['id'].rsplit('::', 1)[0]].append({'id': catalog[index]['id'], 'index': index})
    import contextlib
    if args.work:
        args.work.mkdir(parents=True, exist_ok=True)
    workspace = contextlib.nullcontext(str(args.work)) if args.work else tempfile.TemporaryDirectory(prefix='ci-cold-code-')
    with workspace as temporary:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending = {pool.submit(run_one, root, group, Path(temporary) / str(i), args.reuse): group
                       for i, group in enumerate(groups.values())}
            for count, future in enumerate(concurrent.futures.as_completed(pending), 1):
                group = pending[future]
                passed, traces = future.result()
                if not passed:
                    print(f'Incomplete group: {group[0]["id"]}', flush=True)
                if passed:
                    complete.update(t['index'] for t in group)
                    for trace in traces:
                        for key, observed in trace['checkpoints'].items():
                            if key in checkpoints:
                                raise ValueError('duplicate worker query checkpoint')
                            checkpoints[key] = observed
                        for key, observed in trace['fields'].items():
                            row = fields.setdefault(key, {'whole': 0, 'keys': {}, 'missing': {}})
                            row['whole'] |= int(observed['whole'], 16)
                            for kind in ('keys', 'missing'):
                                for name, mask in observed[kind].items():
                                    row[kind][name] = row[kind].get(name, 0) | int(mask, 16)
                        for path, entries in trace['codes'].items():
                            for key, observed in entries.items():
                                row = codes.setdefault(path, {}).setdefault(key, {'tests': 0, 'startup': False, 'lines': {},
                                                       'queries': set(), 'query_outside': False})
                                row['queries'].update(observed['queries'])
                                row['query_outside'] |= observed['query_outside']
                                row['tests'] |= int(observed['entered'], 16)
                                for line, mask in observed['lines'].items():
                                    row['lines'][line] = row['lines'].get(line, 0) | int(mask, 16)
                print(f'Cold groups {count}/{len(groups)}, complete methods {len(complete)}/{len(indices)}', flush=True)
    for entries in codes.values():
        for row in entries.values():
            row['tests'] = format(row['tests'], 'x')
            row['lines'] = {line: format(mask, 'x') for line, mask in row['lines'].items()}
            row['queries'] = sorted(row['queries'])
    data = {'revision': REVISION, 'version': 1, 'kind': 'compiled_consumers',
            'python': list(sys.version_info[:2]), 'tests': [t['id'] for t in catalog],
            'sources': {path: source_hash(text) for path, text in sources.items()}, 'codes': codes,
            'incomplete': format(sum(1 << i for i in range(len(catalog)) if i not in complete), 'x'),
            'complete': sorted(complete)}
    data['unseen_scopes'] = True
    from compiledcheck import children
    def universe(code):
        yield identity(code)
        for child in children(code):
            yield from universe(child)
    data['universe'] = {path: sorted(set(universe(compile(text, path, 'exec', dont_inherit=True))))
                        for path, text in sources.items()}
    data['checkpoints'] = checkpoints
    data['query_complete'] = TEST in [catalog[i]['id'] for i in complete]
    with gzip.GzipFile(filename=str(args.out), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(data, sort_keys=True, separators=(',', ':')).encode())
    if args.fields_out:
        from fieldcheck import CORE, validator_hash
        import ast
        # Test source also consumes settings directly. Retain static readers,
        # including helpers and shared fixtures, even though only library code
        # is instrumented. Unknown reflection retains every dictionary field.
        by_file = defaultdict(list)
        for index, test in enumerate(catalog):
            by_file[test['id'].split('::')[0]].append(index)
        names = {key.rsplit(':', 1)[-1] for key in fields}
        for path, members in by_file.items():
            tree = ast.parse((root / path).read_text())
            def retain(node, mask):
                attributes = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
                reflection = bool(attributes & {'__dict__', '__code__'})
                for n in ast.walk(node):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                        if n.func.id == 'vars':
                            reflection = True
                        elif n.func.id == 'getattr':
                            if len(n.args) < 2 or not isinstance(n.args[1], ast.Constant) or type(n.args[1].value) is not str:
                                reflection = True
                            else:
                                attributes.add(n.args[1].value)
                if 'required_args' in attributes:
                    attributes.add('arg_types')
                for key, row in fields.items():
                    if reflection or key.rsplit(':', 1)[-1] in attributes & names:
                        row['whole'] |= mask
            for cls in tree.body:
                methods = cls.body if isinstance(cls, ast.ClassDef) else [cls]
                for node in methods:
                    matching = [i for i in members if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and catalog[i]['id'].endswith('::' + node.name)]
                    retain(node, sum(1 << i for i in matching or members))
        # Reflection inside the library can consume a dictionary without an
        # ordinary attribute read. Retain every observed caller of those scopes.
        for path, source in sources.items():
            tree = ast.parse(source)
            scopes = [tree, *[n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]]
            for scope in scopes:
                attributes, reflection = set(), False
                scan = list(ast.walk(scope)) if scope is not tree else [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                for n in scan:
                    if isinstance(n, ast.Attribute) and n.attr == '__dict__':
                        reflection = True
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                        if n.func.id == 'vars':
                            reflection = True
                        elif n.func.id == 'getattr':
                            if len(n.args) < 2 or not isinstance(n.args[1], ast.Constant) or type(n.args[1].value) is not str:
                                reflection = True
                            elif n.args[1].value in {'__dict__', 'required_args'}:
                                reflection = True
                            else:
                                attributes.add(n.args[1].value)
                if reflection or attributes & names:
                    prefix = str(getattr(scope, 'lineno', 1)) + ':'
                    mask = 0
                    for code, observed in data['codes'].get(path, {}).items():
                        if code.startswith(prefix):
                            mask |= int(observed['tests'], 16)
                    for key, row in fields.items():
                        if reflection or key.rsplit(':', 1)[-1] in attributes:
                            row['whole'] |= mask
        for row in fields.values():
            row['whole'] = format(row['whole'], 'x')
            for kind in ('keys', 'missing'):
                row[kind] = {name: format(mask, 'x') for name, mask in row[kind].items()}
        field_data = {k: data[k] for k in ('revision', 'version', 'tests', 'sources', 'incomplete')}
        field_data.update(kind='field_consumers', fields=fields, validator_source=validator_hash(sources[CORE]))
        with gzip.GzipFile(filename=str(args.fields_out), mode='wb', mtime=0) as stream:
            stream.write(json.dumps(field_data, sort_keys=True, separators=(',', ':')).encode())
    print(f'Complete {len(complete)}/{len(catalog)}; profile bytes {args.out.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
