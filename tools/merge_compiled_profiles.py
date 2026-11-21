"""Merge independent pristine passes while retaining every incomplete context."""

import argparse
import copy
import gzip
import json
from pathlib import Path


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def union(left, right):
    return format(int(left, 16) | int(right, 16), 'x')


def merge(compiled, fields, checkpoints=None):
    result, consumers = copy.deepcopy(compiled[0]), copy.deepcopy(fields[0])
    for profile in [*compiled, *fields]:
        if any(profile[k] != result[k] for k in ('revision', 'version', 'tests', 'sources')):
            raise ValueError('incompatible pristine observations')
    complete = set()
    for profile in compiled:
        if profile['python'] != result['python'] or profile['kind'] != 'compiled_consumers':
            raise ValueError('incompatible executable observations')
        if profile.get('universe') != result.get('universe'):
            raise ValueError('compiled scope universes differ')
        complete.update(profile['complete'])
        for path, entries in profile['codes'].items():
            target = result['codes'].setdefault(path, {})
            for key, row in entries.items():
                if key not in target:
                    target[key] = copy.deepcopy(row)
                    continue
                saved = target[key]
                saved['tests'] = union(saved['tests'], row['tests'])
                saved['startup'] |= row['startup']
                saved['query_outside'] |= row['query_outside']
                saved['queries'] = sorted(set(saved['queries']) | set(row['queries']))
                for line, mask in row['lines'].items():
                    saved['lines'][line] = union(saved['lines'].get(line, '0'), mask)
    result['complete'] = sorted(complete)
    result['incomplete'] = format(sum(1 << i for i in range(len(result['tests'])) if i not in complete), 'x')
    result['unseen_scopes'] = all(p.get('unseen_scopes') for p in compiled)
    from queryreplay import TEST
    result['query_complete'] = TEST in [result['tests'][i] for i in complete]
    result['checkpoints'] = {}
    if checkpoints:
        if (checkpoints['revision'] != result['revision'] or checkpoints['kind'] != 'optimizer_checkpoints'
                or any(result['sources'].get(p) != h for p, h in checkpoints['sources'].items())):
            raise ValueError('incompatible optimizer checkpoints')
        result['checkpoints'] = checkpoints['checkpoints']
        result['driver'] = checkpoints['driver']
    for profile in fields:
        if profile['validator_source'] != consumers['validator_source'] or profile['kind'] != 'field_consumers':
            raise ValueError('incompatible validator observations')
        for key, row in profile['fields'].items():
            target = consumers['fields'].setdefault(key, dict(whole='0', keys={}, missing={}))
            target['whole'] = union(target['whole'], row['whole'])
            for kind in ('keys', 'missing'):
                for name, mask in row[kind].items():
                    target[kind][name] = union(target[kind].get(name, '0'), mask)
    # Executable and field passes share the same complete class contexts.
    for profile, companion in zip(compiled, fields, strict=True):
        if profile['incomplete'] != companion['incomplete']:
            raise ValueError('field and code completion differ')
    consumers['incomplete'] = result['incomplete']
    return result, consumers


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiled', type=Path, nargs='+', required=True)
    parser.add_argument('--fields', type=Path, nargs='+', required=True)
    parser.add_argument('--checkpoints', type=Path)
    parser.add_argument('--compiled-out', type=Path, required=True)
    parser.add_argument('--fields-out', type=Path, required=True)
    args = parser.parse_args()
    data = merge([read(p) for p in args.compiled], [read(p) for p in args.fields],
                 read(args.checkpoints) if args.checkpoints else None)
    for path, payload in zip((args.compiled_out, args.fields_out), data):
        with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode())
    print(f'Complete methods {len(data[0]["complete"])}/{len(data[0]["tests"])}', flush=True)
