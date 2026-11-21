"""Measure pristine spawned executor tasks without source instrumentation."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path


def task(index, payload):
    function, args, kwargs = pickle.loads(payload)
    started = time.process_time()
    result = function(*args, **kwargs)
    (Path(os.environ['CI_QUERY_TIMING_OUTPUT']) / f'{index}.json').write_text(json.dumps(time.process_time() - started))
    return result


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from comparisoncheck import REVISION
    from queryreplay import TEST
    import pytest
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.source.resolve(), args.out.resolve()
    with gzip.open(Path(__file__).resolve().parents[1] / 'maps/baseline.json.gz', 'rt') as stream:
        baseline = json.load(stream)
    expected = {path: text for path, text in baseline['sources'].items() if path.startswith('sqlglot/')}
    actual = {path.relative_to(root).as_posix() for path in (root/'sqlglot').rglob('*.py')}
    if baseline['revision'] != REVISION or actual != set(expected) or any((root/path).read_text() != text for path,text in expected.items()):
        raise ValueError('pristine source required')
    with tempfile.TemporaryDirectory(prefix='ci-query-timing-') as directory:
        os.environ.update(SKIP_INTEGRATION='0', PYTHON_CPU_COUNT='2', PYTHONDONTWRITEBYTECODE='1',
                          OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMEXPR_MAX_THREADS='1',
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', CI_QUERY_TIMING_OUTPUT=directory,
                          PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parent), str(root)]))
        os.chdir(root)
        sys.path.insert(0, str(root))
        from build_query_timing import task as measured
        original = concurrent.futures.ProcessPoolExecutor
        expected_count = 0
        class Pool(original):
            def submit(self, function, /, *a, **kw):
                nonlocal expected_count
                index = expected_count
                expected_count += 1
                return super().submit(measured, index, pickle.dumps((function,a,kw), protocol=5))
        concurrent.futures.ProcessPoolExecutor = Pool
        status = pytest.main([TEST, '-q', '-p', 'no:cacheprovider', '-p', 'pytest_timeout', '--timeout=180'])
        rows = {int(path.stem): json.loads(path.read_text()) for path in Path(directory).glob('*.json')}
        if status or not expected_count or set(rows) != set(range(expected_count)):
            raise ValueError('incomplete pristine query timings')
        output.write_text(json.dumps(dict(revision=REVISION, kind='pristine_query_timing', test=TEST, cpu=rows,
                              driver=hashlib.sha256((root/'tests/test_executor.py').read_bytes()).hexdigest())))
        print(f'Real query work: {sum(rows.values()):.3f} CPU seconds, {len(rows)} tasks', file=sys.stderr)


if __name__ == '__main__':
    main()
