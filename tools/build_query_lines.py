"""Record optimizer statement consumers from pristine serialized executor tasks."""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiledcheck import identity, load_profile as load_compiled
from queryreplay import checkpoint_digest, load_profile, state_bytes, state_digest
from comparisoncheck import source_hash


def record(root, tasks, compiled):
    if (os.environ.get('PYTHONHASHSEED') != '0' or not compiled.get('query_complete')
            or compiled['driver'] != tasks['driver']
            or hashlib.sha256((root / 'tests/test_executor.py').read_bytes()).hexdigest() != tasks['driver']
            or any(source_hash((root / p).read_text()) != h for p, h in compiled['sources'].items())):
        raise ValueError('complete pristine executor observations required')
    os.chdir(root)
    sys.path.insert(0, str(root))
    from tests import test_executor as driver
    import sqlglot.executor as executor
    original = executor.optimize
    prefix = str(root) + '/'
    codes, cache, local = {}, {}, set()
    active = None
    m = sys.monitoring
    tool = next(i for i in range(5, -1, -1) if m.get_tool(i) is None)
    m.use_tool_id(tool, 'executor-query-statements')

    def row(code):
        if code not in cache:
            path = code.co_filename.removeprefix(prefix)
            cache[code] = (codes.setdefault(path, {}).setdefault(identity(code), {})
                           if path.startswith('sqlglot/optimizer/') else None)
        return cache[code]

    def enter(code, offset):
        if active is not None and row(code) is not None:
            if code not in local:
                local.add(code)
                m.set_local_events(tool, code, m.events.LINE)
        return m.DISABLE

    def line(code, number):
        entry = row(code)
        if active is not None and entry is not None:
            entry[number] = entry.get(number, 0) | (1 << active)
        return m.DISABLE

    m.register_callback(tool, m.events.PY_START, enter)
    m.register_callback(tool, m.events.PY_RESUME, enter)
    m.register_callback(tool, m.events.LINE, line)
    m.set_events(tool, m.events.PY_START | m.events.PY_RESUME)
    snapshot = lru_cache(maxsize=2)(lambda key: state_bytes(tasks, key))

    class Complete(BaseException):
        pass

    try:
        for index, task in tasks['queries'].items():
            expected = compiled.get('checkpoints', {}).get(str(index))
            if not expected:
                continue
            if len(expected) != 1 or not expected[0]['pure'] or not expected[0].get('output'):
                raise ValueError('unsupported baseline checkpoint')
            driver.initializer(*pickle.loads(snapshot(task['state'])))
            function, args, kwargs = pickle.loads(task['task'])
            if function is not driver.mp_execute or kwargs or len(args) != 2:
                raise ValueError('unknown executor task')
            before = state_digest((driver._schema, driver._tables))

            def optimize(*a, **kw):
                nonlocal active
                active = index
                m.restart_events()
                try:
                    expression = original(*a, **kw)
                finally:
                    active = None
                    m.restart_events()
                if (checkpoint_digest((expression, a, kw)) != expected[0]['output']
                        or state_digest((driver._schema, driver._tables)) != before):
                    raise ValueError('pristine optimizer checkpoint differs')
                raise Complete()

            executor.optimize = optimize
            try:
                function(*args, **kwargs)
            except Complete:
                pass
            else:
                raise ValueError('recorded optimizer checkpoint was not reached')
            finally:
                executor.optimize = original
            if index % 10 == 0:
                print(f'Query statements {index}/{len(tasks["queries"])}', flush=True)
    finally:
        m.set_events(tool, 0)
        for code in local:
            m.set_local_events(tool, code, 0)
        m.free_tool_id(tool)
        executor.optimize = original
    # Cold worker observations remain the authority for complete reachability.
    # A scope is usable only when this independent pass saw precisely that set.
    matched = {}
    for path, entries in codes.items():
        for key, lines in entries.items():
            mask = 0
            for value in lines.values():
                mask |= value
            baseline = compiled['codes'].get(path, {}).get(key)
            if (baseline and mask == sum(1 << i for i in baseline['queries'])
                    and not baseline['query_outside']):
                matched.setdefault(path, {})[key] = {str(n): format(v, 'x') for n, v in lines.items()}
    return matched


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--compiled', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = load_compiled(args.compiled)
    if data is None:
        raise ValueError('invalid compiled profile')
    data['query_lines'] = record(args.source.resolve(), load_profile(args.tasks.resolve()), data)
    with gzip.GzipFile(filename=str(args.out), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(data, sort_keys=True, separators=(',', ':')).encode())
    print(f'Certified statement scopes: {sum(map(len, data["query_lines"].values()))}; bytes: {args.out.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
