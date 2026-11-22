"""Concrete generator receiver evidence for materialized dispatch callbacks."""
import ast
import gzip
import hashlib
import json
from pathlib import Path

from comparisoncheck import REVISION


def driver_hash(source):
    tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in {'sql', '_build_dispatch', '__init__'}]
    return hashlib.sha256(repr([ast.dump(n) for n in nodes]).encode()).hexdigest()


def load_profile(path):
    try:
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        if (data['revision'] != REVISION or data['version'] != 1 or not data['collection_complete']
                or len(data['tests']) != len(set(data['tests']))):
            return None
        limit = (1 << len(data['tests'])) - 1
        if any(type(mask) is not str or int(mask, 16) & ~limit for rows in data['receivers'].values() for mask in rows.values()):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def consumers(profile, changes, catalog, root):
    if (not profile or driver_hash((Path(root) / 'sqlglot/generator.py').read_text()) != profile['driver']):
        return None
    mask = 0
    for receiver, key in changes:
        rows = profile['receivers'].get(receiver)
        if rows is None:
            return None
        mask |= int(rows.get(key, '0'), 16) | int(rows.get('*', '0'), 16)
    complete = set(profile['complete'])
    return {t['number'] for t in catalog if t['id'] in {
        profile['tests'][i] for i in range(len(profile['tests'])) if i not in complete or mask & (1 << i)}}
