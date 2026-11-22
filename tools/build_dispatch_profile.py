"""Record actual receiver/expression pairs in passing pristine generator tests."""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparisoncheck import REVISION
from dispatchcheck import driver_hash


class Recorder:
    def __init__(self, tests):
        self.tests = tests
        self.by_id = {name: i for i, name in enumerate(tests)}
        self.current = None
        self.rows = {}
        self.bad = set()
        self.phases = {}
        self.collection_complete = False

    def observe(self, generator, expression):
        receiver = type(generator).__module__ + '.' + type(generator).__qualname__
        key = type(expression).__module__ + '.' + type(expression).__qualname__
        if not receiver.startswith('sqlglot.'):
            if self.current is not None:
                self.bad.add(self.current)
            return
        mask = (1 << self.current) if self.current is not None else (1 << len(self.tests)) - 1
        rows = self.rows.setdefault(receiver, {})
        rows[key] = rows.get(key, 0) | mask

    def audit(self, event, args):
        if event in {'subprocess.Popen', 'os.fork', 'os.forkpty', 'os.exec', 'os.posix_spawn'}:
            if self.current is not None:
                self.bad.add(self.current)

    def pytest_collection_finish(self, session):
        self.collection_complete = not session.testsfailed

    def pytest_runtest_logstart(self, nodeid, location):
        self.current = self.by_id[nodeid]

    def pytest_runtest_logreport(self, report):
        i = self.by_id[report.nodeid]
        if report.failed or report.skipped:
            self.bad.add(i)
        if report.passed:
            self.phases.setdefault(i, set()).add(report.when)

    def pytest_runtest_logfinish(self, nodeid, location):
        self.current = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve()
    catalog = json.loads(args.tests.read_text())['tests']
    tests = [t['id'] for t in catalog]
    selected = [t['id'] for t in catalog if t['file'].startswith('tests/dialects/') or t['file'] == 'tests/test_generator.py']
    output = args.out.resolve()
    os.environ.update(SKIP_INTEGRATION='0', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    sys.path.insert(0, str(root))
    os.chdir(root)
    from sqlglot.generator import Generator
    original = Generator.sql
    recorder = Recorder(tests)
    def sql(self, expression, *args, **kwargs):
        if expression is not None and not isinstance(expression, str):
            recorder.observe(self, expression)
        return original(self, expression, *args, **kwargs)
    Generator.sql = sql
    sys.addaudithook(recorder.audit)
    import pytest
    result = pytest.main([*selected, '-q', '-p', 'no:cacheprovider', '-p', 'pytest_timeout', '--timeout=120'], plugins=[recorder])
    complete = sorted(i for i, phases in recorder.phases.items() if phases == {'setup', 'call', 'teardown'} and i not in recorder.bad)
    data = dict(revision=REVISION, version=1, tests=tests, complete=complete,
                collection_complete=recorder.collection_complete,
                driver=driver_hash((root / 'sqlglot/generator.py').read_text()),
                receivers={receiver: {key: format(mask, 'x') for key, mask in rows.items()} for receiver, rows in recorder.rows.items()})
    with gzip.GzipFile(filename=str(output), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(data, separators=(',', ':'), sort_keys=True).encode())
    print('Dispatch contexts:', len(complete), 'receivers:', len(recorder.rows), flush=True)
    raise SystemExit(result)
