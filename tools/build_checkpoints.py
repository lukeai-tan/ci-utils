"""Record pristine optimizer graphs for actual serialized pool-task inputs."""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparisoncheck import REVISION, source_hash
from queryreplay import checkpoint_digest, load_profile, state_bytes, state_digest


def record(root, tasks):
    if (os.environ.get('PYTHONHASHSEED') != '0'
            or hashlib.sha256((root / 'tests/test_executor.py').read_bytes()).hexdigest() != tasks['driver']
            or any(source_hash((root / path).read_text()) != digest for path, digest in tasks['sources'].items())):
        raise ValueError('pristine source, driver and hash seed required')
    os.chdir(root)
    sys.path.insert(0, str(root))
    from tests import test_executor as driver
    import sqlglot.executor as executor
    original = executor.optimize
    checkpoints = {}

    class Complete(BaseException):
        pass

    for index, query in tasks['queries'].items():
        driver.initializer(*pickle.loads(state_bytes(tasks, query['state'])))
        function, args, kwargs = pickle.loads(query['task'])
        if function is not driver.mp_execute or kwargs or len(args) != 2:
            raise ValueError('unknown worker task')
        observed = []
        def optimize(*a, **kw):
            started = time.process_time()
            before = state_digest((driver._schema, driver._tables))
            result = original(*a, **kw)
            try:
                output = checkpoint_digest((result, a, kw))
            except (ValueError, TypeError, RecursionError) as error:
                print(f'Unsupported checkpoint {index}: {error}', flush=True)
                output = None
            observed.append(dict(kind='expression_graph_v1', output=output,
                                 pure=before == state_digest((driver._schema, driver._tables)),
                                 cpu=time.process_time() - started))
            raise Complete()
        executor.optimize = optimize
        try:
            function(*args, **kwargs)
        except Complete:
            checkpoints[str(index)] = observed
        finally:
            executor.optimize = original
        if index % 10 == 0:
            print(f'Checkpoint task {index}/{len(tasks["queries"])}', flush=True)
    return dict(revision=REVISION, kind='optimizer_checkpoints', version=1, driver=tasks['driver'],
                sources=tasks['sources'], checkpoints=checkpoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--only', nargs='*', type=int)
    args = parser.parse_args()
    tasks = load_profile(args.tasks.resolve())
    if args.only is not None:
        tasks['queries'] = {i: tasks['queries'][i] for i in args.only}
    data = record(args.source.resolve(), tasks)
    with gzip.GzipFile(filename=str(args.out), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(data, sort_keys=True, separators=(',', ':')).encode())
    eligible = sum(len(v) == 1 and v[0]['pure'] and bool(v[0]['output']) for v in data['checkpoints'].values())
    print(f'Complete usable checkpoints {eligible}/{len(data["checkpoints"])}', flush=True)
