"""Record optimizer fixture-task dependencies from the pinned source only."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiledcheck import children, identity
from comparisoncheck import REVISION, source_hash
from fixturecheck import DRIVER, OWNER, digest, shared_state, prepare_worker, compact, fixture_inputs


class Recorder:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.prefix = str(self.root) + '/'
        self.cache, self.rows = {}, {}
        self.startup = True
        self.tool = 5
        m = sys.monitoring
        m.use_tool_id(self.tool, 'optimizer-fixture-case')
        m.register_callback(self.tool, m.events.PY_START, self.enter)
        m.register_callback(self.tool, m.events.PY_RESUME, self.enter)
        m.register_callback(self.tool, m.events.LINE, self.line)
        m.set_events(self.tool, m.events.PY_START | m.events.PY_RESUME)
        m.restart_events()

    def row(self, code):
        if code in self.cache:
            return self.cache[code]
        path = code.co_filename.removeprefix(self.prefix)
        row = None
        if path.startswith('sqlglot/optimizer/'):
            row = self.rows.setdefault(path, {}).setdefault(identity(code), {'startup': False, 'lines': set()})
        self.cache[code] = row
        return row

    def enter(self, code, offset):
        row = self.row(code)
        if row is None:
            return sys.monitoring.DISABLE
        row['startup'] |= self.startup
        sys.monitoring.set_local_events(self.tool, code, sys.monitoring.events.LINE)
        return sys.monitoring.DISABLE

    def line(self, code, number):
        row = self.row(code)
        if row is not None:
            row['lines'].add(number)
        return sys.monitoring.DISABLE

    def finish(self):
        m = sys.monitoring
        m.set_events(self.tool, 0)
        for code, row in self.cache.items():
            if row is not None:
                m.set_local_events(self.tool, code, 0)
        m.free_tool_id(self.tool)
        return {p: {k: dict(startup=r['startup'], lines=sorted(r['lines'])) for k, r in entries.items()}
                for p, entries in self.rows.items()}


def measured_task(root, target, index, payload):
    recorder = Recorder(root)
    function, args, kwargs = pickle.loads(payload)
    prepare_worker()
    recorder.startup = False
    sys.monitoring.restart_events()
    state = shared_state()
    started = time.process_time()
    try:
        result = function(*args, **kwargs)
        # Real pool output serialization is part of the task's execution path.
        pickle.dumps(result, protocol=5)
        return result
    finally:
        cpu = time.process_time() - started
        after = shared_state()
        data = {'input': digest(payload), 'cpu': cpu, 'state': state, 'after': after,
                'pure': state is not None and after == state, 'codes': recorder.finish()}
        (Path(target) / f'task-{index}.json').write_text(json.dumps(data, separators=(',', ':')))


def one(root, name, target):
    import pytest
    original_pool = concurrent.futures.ProcessPoolExecutor
    submitted = []
    class ParentTiming:
        started = None
        cpu = None
        def pytest_runtest_call(self, item):
            self.started = time.process_time()
        def pytest_runtest_teardown(self, item):
            self.cpu = time.process_time() - self.started if self.started is not None else None
    timing = ParentTiming()
    class Pool(original_pool):
        def __init__(self, *args, **kwargs):
            if kwargs.get('initializer') is not None:
                raise ValueError('optimizer baseline has an initializer')
            super().__init__(*args, **kwargs)
        def submit(self, function, /, *args, **kwargs):
            if function.__module__ != 'tests.test_optimizer' or function.__name__ != 'parse_and_optimize':
                raise ValueError('unknown optimizer task')
            payload = pickle.dumps((function, args, kwargs), protocol=5)
            index = len(submitted)
            submitted.append(digest(payload))
            return super().submit(measured_task, str(root), str(target), index, payload)
    concurrent.futures.ProcessPoolExecutor = Pool
    status = pytest.main([name, '-q', '-p', 'no:cacheprovider', '-p', 'pytest_timeout', '--timeout=600'], plugins=[timing])
    (target / 'verdict.json').write_text(json.dumps({'passed': status == 0, 'tasks': submitted, 'parent_cpu': timing.cpu}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--one')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    root = args.source.resolve()
    if args.one:
        one(root, args.one, args.work)
        return
    repo = Path(__file__).resolve().parents[1]
    baseline = json.load(gzip.open(repo / 'maps/baseline.json.gz', 'rt'))
    sources = {p: s for p, s in baseline['sources'].items() if p.startswith('sqlglot/')}
    if baseline['revision'] != REVISION or any((root / p).read_text() != s for p, s in sources.items()):
        raise ValueError('pristine pinned source required')
    timing = json.load(gzip.open(repo / 'maps/timing.json.gz', 'rt'))
    tree = ast.parse((root / DRIVER).read_text())
    methods_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')
               and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == 'check_file'
                       for c in ast.walk(n))]
    methods = [n.name for n in methods_nodes]
    names = [OWNER + '::' + method for method in methods]
    if args.only is not None:
        if not set(args.only) <= set(names):
            raise ValueError('unknown fixture-loop method')
        names = args.only
    args.work.mkdir(parents=True, exist_ok=True)
    def run(name):
        target = args.work / name.rsplit('::', 1)[-1]
        target.mkdir(exist_ok=True)
        if not args.reuse:
            env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(__file__).resolve().parent), str(repo), str(root)]),
                   'PYTHONHASHSEED': '0', 'PYTHON_CPU_COUNT': '2', 'SKIP_INTEGRATION': '0',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'NUMEXPR_MAX_THREADS': '1'}
            with (target / 'run.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--one', name,
                                          '--source', str(root), '--out', str(args.out), '--work', str(target)],
                                         env=env, cwd=root, stdout=log, stderr=log, start_new_session=True)
                try:
                    child.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                if child.returncode:
                    return name, None
        verdict_path = target / 'verdict.json'
        if not verdict_path.exists():
            return name, None
        verdict = json.loads(verdict_path.read_text())
        if not verdict['passed'] or not verdict['tasks']:
            return name, None
        tasks, codes = [], {}
        for index, expected in enumerate(verdict['tasks']):
            path = target / f'task-{index}.json'
            if not path.exists():
                return name, None
            task = json.loads(path.read_text())
            if task['input'] != expected:
                return name, None
            tasks.append({k: task.get(k) for k in ('input', 'cpu', 'state', 'after', 'pure')})
            for path, entries in task['codes'].items():
                for key, observed in entries.items():
                    row = codes.setdefault(path, {}).setdefault(key, {'cases': [], 'startup': False, 'lines': {}})
                    row['cases'].append(index)
                    row['startup'] |= observed['startup']
                    for line in observed['lines']:
                        row['lines'].setdefault(str(line), []).append(index)
        method = next(n for n in methods_nodes if n.name == name.rsplit('::', 1)[-1])
        return name, {'tasks': tasks, 'codes': codes, 'cpu': timing['cpu'][name],
                      'fixtures': fixture_inputs(root, method, len(tasks)),
                      'base_cpu': min(timing['cpu'][name], 0.2 + (verdict.get('parent_cpu')
                                      if verdict.get('parent_cpu') is not None else timing['cpu'][name] * 0.1))}
    tests = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, name) for name in names]
        for future in concurrent.futures.as_completed(futures):
            name, row = future.result()
            print(name + (' complete' if row else ' incomplete'), flush=True)
            if row:
                tests[name] = row
    def universe(code):
        yield identity(code)
        for child in children(code):
            yield from universe(child)
    data = {'revision': REVISION, 'version': 1, 'kind': 'optimizer_cases', 'python': list(sys.version_info[:2]),
            'state_kind': 'optimizer_globals_v1',
            'hashseed': '0', 'tests': tests, 'sources': {p: source_hash(s) for p, s in sources.items()},
            'drivers': {p: digest((root / p).read_bytes()) for p in (DRIVER, 'tests/helpers.py')},
            'universe': {p: sorted(set(universe(compile(s, p, 'exec', dont_inherit=True))))
                         for p, s in sources.items() if p.startswith('sqlglot/optimizer/')}}
    with gzip.GzipFile(filename=str(args.out), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(compact(data), sort_keys=True, separators=(',', ':')).encode())
    print(f'Complete {len(tests)}/{len(names)}; {args.out.stat().st_size} bytes', flush=True)


if __name__ == '__main__':
    main()
