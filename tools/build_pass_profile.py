"""Certify late optimizer-pass inputs against complete cold executor evidence."""

import argparse
import hashlib
import lzma
import os
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiledcheck import identity, load_profile as load_compiled
from comparisoncheck import REVISION, source_hash
from fixturecheck import digest, prepare_worker, shared_state
from passcheck import snapshot, run_suffix
from queryreplay import checkpoint_digest, load_profile, state_bytes, state_digest


def record(root, tasks, compiled):
    if (os.environ.get('PYTHONHASHSEED') != '0' or not compiled.get('query_complete')
            or compiled['driver'] != tasks['driver']
            or digest((root / 'tests/test_executor.py').read_bytes()) != tasks['driver']
            or any(source_hash((root / p).read_text()) != h for p, h in compiled['sources'].items())):
        raise ValueError('complete pristine executor inputs required')
    os.chdir(root)
    sys.path.insert(0, str(root))
    from tests import test_executor as driver
    import sqlglot.executor as executor
    from sqlglot.optimizer import optimizer
    prepare_worker()
    original = executor.optimize
    calls, codes, cache = [], {}, {}
    active, recording = None, True
    outer_args, outer_kwargs = None, None
    prefix = str(root) + '/'
    m = sys.monitoring
    tool = next(i for i in range(5, -1, -1) if m.get_tool(i) is None)
    m.use_tool_id(tool, 'optimizer-pass-consumers')
    def enter(code, offset):
        if recording:
            if code not in cache:
                path = code.co_filename.removeprefix(prefix)
                cache[code] = (codes.setdefault(path, {}).setdefault(identity(code), {'outside': False, 'calls': set()})
                               if path.startswith('sqlglot/optimizer/') else None)
            row = cache[code]
            if row is not None:
                if active is None:
                    row['outside'] = True
                else:
                    row['calls'].add(active)
        return m.DISABLE
    m.register_callback(tool, m.events.PY_START, enter)
    m.register_callback(tool, m.events.PY_RESUME, enter)
    m.set_events(tool, m.events.PY_START | m.events.PY_RESUME)
    targets = {'eliminate_joins', 'eliminate_ctes', 'canonicalize'}
    states = lru_cache(maxsize=2)(lambda key: state_bytes(tasks, key))
    class Complete(BaseException):
        pass
    try:
        for index, task in tasks['queries'].items():
            expected = compiled.get('checkpoints', {}).get(str(index))
            if not expected:
                continue
            if len(expected) != 1 or not expected[0]['pure'] or not expected[0].get('output'):
                raise ValueError('unsupported complete optimizer checkpoint')
            recording = False
            driver.initializer(*pickle.loads(states(task['state'])))
            recording = True
            m.restart_events()
            before_task = state_digest((driver._schema, driver._tables))
            def wrap(rule, start):
                def call(expression):
                    nonlocal active, recording
                    context = (sys._getframe(1).f_locals['possible_kwargs'], outer_args, outer_kwargs, expression)
                    state = shared_state()
                    payload = snapshot(context)
                    initial = checkpoint_digest(context)
                    recording = False
                    if checkpoint_digest(pickle.loads(payload)) != initial:
                        raise ValueError('optimizer pass snapshot loses observable state')
                    recording = True
                    active = len(calls)
                    calls.append(None)
                    m.restart_events()
                    result = rule(expression)
                    active = None
                    m.restart_events()
                    if not state or shared_state() != state:
                        raise ValueError('optimizer pass changes shared state')
                    output = expected[0]['output']
                    recording = False
                    m.set_events(tool, 0)
                    started = time.process_time()
                    restored = pickle.loads(payload)
                    if (checkpoint_digest(restored) != initial or shared_state() != state
                            or checkpoint_digest(run_suffix(restored, start)) != output or shared_state() != state):
                        raise ValueError('pristine isolated optimizer pass differs')
                    cpu = time.process_time() - started
                    calls[-1] = dict(query=index, rule=rule.__module__+':'+rule.__qualname__, start=start, input=payload,
                                     bytes=hashlib.sha256(payload).hexdigest(), initial=initial, output=output, state=state, cpu=cpu)
                    recording = True
                    m.set_events(tool, m.events.PY_START | m.events.PY_RESUME)
                    m.restart_events()
                    return result
                return call
            rules = tuple(wrap(rule, start) if rule.__name__ in targets else rule for start, rule in enumerate(optimizer.RULES))
            def optimize(*a, **kw):
                nonlocal outer_args, outer_kwargs
                if 'rules' in kw:
                    raise ValueError('nonstandard optimizer rule sequence')
                outer_args, outer_kwargs = a, kw
                result = original(*a, **kw, rules=rules)
                if (checkpoint_digest((result, a, kw)) != expected[0]['output']
                        or state_digest((driver._schema, driver._tables)) != before_task):
                    raise ValueError('complete pristine optimizer checkpoint differs')
                raise Complete()
            executor.optimize = optimize
            function, args, kwargs = pickle.loads(task['task'])
            if function is not driver.mp_execute or kwargs or len(args) != 2:
                raise ValueError('unknown executor worker task')
            try:
                function(*args, **kwargs)
            except Complete:
                pass
            else:
                raise ValueError('complete optimizer checkpoint not reached')
            finally:
                executor.optimize = original
            if index % 10 == 0:
                print(f'Pass checkpoints {index}/{len(tasks["queries"])}; calls {len(calls)}', flush=True)
    finally:
        m.set_events(tool, 0)
        m.free_tool_id(tool)
        executor.optimize = original
    certified = {}
    for path, entries in codes.items():
        for key, row in entries.items():
            baseline = compiled['codes'].get(path, {}).get(key)
            if (not row['outside'] and row['calls'] and baseline and not baseline['query_outside']
                    and {calls[i]['query'] for i in row['calls']} == set(baseline['queries'])):
                certified.setdefault(path, {})[key] = sorted(row['calls'])
    return dict(revision=REVISION, kind='optimizer_passes', version=1, python=list(sys.version_info[:2]),
                hashseed='0', complete=True, driver=tasks['driver'], sources={p: compiled['sources'][p] for p in certified},
                drivers={p: digest((root / p).read_bytes()) for p in ('tests/test_executor.py', 'sqlglot/optimizer/optimizer.py')},
                codes=certified, calls=calls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--compiled', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = record(args.source.resolve(), load_profile(args.tasks.resolve()), load_compiled(args.compiled.resolve()))
    with lzma.open(args.out, 'wb', filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME, 'pb': 0}]) as stream:
        pickle.dump(data, stream, protocol=5)
    print(f'Certified scopes: {sum(map(len, data["codes"].values()))}; bytes: {args.out.stat().st_size}', flush=True)
