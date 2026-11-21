"""Observe pristine collection and pytest/unittest fixtures, without test bodies."""

import argparse
import gzip
import json
import os
import sys
import tempfile
import inspect
from pathlib import Path

import pytest
from _pytest.unittest import TestCaseFunction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools.build_behavior_profile as collector
from comparisoncheck import REVISION


class Phases:
    def __init__(self):
        self.passed = {}
        self.bad = set()

    def pytest_collection_modifyitems(self, items):
        # --setup-only omits unittest setUp/tearDown, which execute inside call.
        # Preserve the original unittest lifecycle and suppress just its body.
        for item in items:
            if not isinstance(item, TestCaseFunction) or inspect.iscoroutinefunction(item.obj):
                raise ValueError('unsupported fixture lifecycle')
            item._obj = lambda: None

    def pytest_runtest_logreport(self, report):
        if report.failed or report.skipped:
            self.bad.add(report.nodeid)
        if report.passed:
            self.passed.setdefault(report.nodeid, set()).add(report.when)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.source.resolve(), args.out.resolve()
    catalog = json.loads(args.tests.resolve().read_text())['tests']
    with gzip.open(Path(__file__).resolve().parents[1]/'maps/baseline.json.gz','rt') as stream:
        baseline = json.load(stream)
    expected = {path:text for path,text in baseline['sources'].items() if path.startswith('sqlglot/')}
    actual = {path.relative_to(root).as_posix() for path in (root/'sqlglot').rglob('*.py')}
    if baseline['revision'] != REVISION or actual != set(expected) or any((root/path).read_text()!=text for path,text in expected.items()):
        raise ValueError('pristine source required')
    os.environ.update(SKIP_INTEGRATION='0',PYTHON_CPU_COUNT='2',PYTHONDONTWRITEBYTECODE='1',
                      OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_MAX_THREADS='1',
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    collector.LOOKUPS_ONLY = True
    os.chdir(root)
    sys.path.insert(0,str(root))
    with tempfile.TemporaryDirectory(prefix='ci-initialization-profile-') as directory:
        recorder=collector.Recorder(root,catalog,directory)
        collector.install(recorder)
        phases=Phases()
        names=[test['id'] for test in catalog]
        status=pytest.main([*names,'-q','-p','no:cacheprovider'],plugins=[recorder,phases])
        if (status or phases.bad or recorder.bad or not recorder.collection_complete
                or set(phases.passed)!=set(names)
                or any(phases.passed[name]!={'setup','call','teardown'} for name in names)):
            raise ValueError('incomplete or untraced initialization phases')
        rows={}
        for key,contexts in recorder.rows.items():
            row=collector.empty(recorder.sites[key])
            for observed in contexts.values():
                collector.merge_row(row,observed,recorder.sites[key])
            rows[key]=row
        data=dict(revision=REVISION,kind='pristine_initialization',complete=True,
                  tests=recorder.tests,sources=recorder.sources,sites=recorder.sites,rows=rows)
        with gzip.GzipFile(filename=str(output),mode='wb',mtime=0) as stream:
            stream.write(json.dumps(data,sort_keys=True,separators=(',',':')).encode())
        print(f'Initialization phases: {len(names)} complete methods; {len(rows)} observed lookups',file=sys.stderr)


if __name__=='__main__':
    main()
