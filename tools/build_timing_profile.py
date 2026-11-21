"""Measure real pristine test costs without dependency instrumentation."""

import argparse
import gzip
import json
import os
import resource
import sys
from pathlib import Path

import pytest
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparisoncheck import REVISION


def cpu_seconds():
    """Include live pool workers, whose cost is reaped only at class cleanup."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
    total = own.ru_utime + own.ru_stime + reaped.ru_utime + reaped.ru_stime
    for process in psutil.Process().children(recursive=True):
        try:
            cpu = process.cpu_times()
            total += cpu.user + cpu.system + cpu.children_user + cpu.children_system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


class Timings:
    def __init__(self):
        self.cpu = {}
        self.setup = {}
        self.teardown = {}
        self.bad = set()
        self.phases = {}
        self.collection_complete = False
        self.started = 0

    def pytest_collection_finish(self, session):
        self.collection_complete = not session.testsfailed

    def pytest_runtest_logstart(self, nodeid, location):
        self.started = cpu_seconds()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        started = cpu_seconds()
        yield
        owner = item.nodeid.rsplit('::', 1)[0]
        self.setup[owner] = max(self.setup.get(owner, 0), cpu_seconds() - started)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        started = cpu_seconds()
        yield
        owner = item.nodeid.rsplit('::', 1)[0]
        self.teardown[owner] = max(self.teardown.get(owner, 0), cpu_seconds() - started)

    def pytest_runtest_logreport(self, report):
        if report.failed or report.skipped:
            self.bad.add(report.nodeid)
        if report.passed:
            self.phases.setdefault(report.nodeid, set()).add(report.when)

    def pytest_runtest_logfinish(self, nodeid, location):
        self.cpu[nodeid] = cpu_seconds() - self.started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-map', type=Path, default=Path(__file__).resolve().parents[1] / 'maps/baseline.json.gz')
    args = parser.parse_args()
    root, output = args.source.resolve(), args.out.resolve()
    catalog = json.loads(args.tests.resolve().read_text())['tests']
    with gzip.open(args.baseline_map.resolve(), 'rt') as stream:
        baseline = json.load(stream)
    expected = {path: text for path, text in baseline['sources'].items() if path.startswith('sqlglot/')}
    actual = {path.relative_to(root).as_posix() for path in (root / 'sqlglot').rglob('*.py')}
    if baseline['revision'] != REVISION or actual != set(expected) or any((root / path).read_text() != text for path, text in expected.items()):
        raise ValueError('pristine source required')
    os.environ.update(SKIP_INTEGRATION='0', PYTHON_CPU_COUNT='2', PYTHONDONTWRITEBYTECODE='1',
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMEXPR_MAX_THREADS='1',
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    os.chdir(root)
    sys.path.insert(0, str(root))
    times = Timings()
    names = [test['id'] for test in catalog]
    status = pytest.main([*names, '-q', '--maxfail=1', '-p', 'no:cacheprovider',
                          '-p', 'pytest_timeout', '--timeout=180'], plugins=[times])
    if status or not times.collection_complete or times.bad or set(times.cpu) != set(names):
        raise ValueError('baseline timing run was incomplete')
    if any(times.phases.get(name) != {'setup', 'call', 'teardown'} for name in names):
        raise ValueError('baseline phases were incomplete')
    data = dict(revision=REVISION, version=1, kind='pristine_timing', complete=True,
                cpu=times.cpu, setup=times.setup, teardown=times.teardown)
    temporary_output = output.with_name(output.name + '.building')
    with gzip.GzipFile(filename=str(temporary_output), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(data, separators=(',', ':'), sort_keys=True).encode())
    temporary_output.replace(output)
    print(f'Real timing profile: {len(names)} complete tests', file=sys.stderr)


if __name__ == '__main__':
    main()
