"""Dependency proofs for unchanged dictionary shape and validator requirements."""

from __future__ import annotations

import ast
import gzip
import json

from comparisoncheck import REVISION, source_hash


CORE = 'sqlglot/expressions/core.py'


def validator_hash(source):
    import hashlib
    tree = ast.parse(source)
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'Expr'
               for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {'__init_subclass__', 'error_messages'}]
    if len(methods) != 2:
        return None
    return hashlib.sha256(repr([ast.dump(n) for n in methods]).encode()).hexdigest()


def dictionary_changes(original, updated, path):
    """Accept primitive value changes only, with construction and key order fixed."""
    from selector import same_ast
    result = []

    def visit(a, b, owner='', body=False):
        if same_ast(a, b):
            return True
        if isinstance(a, ast.ClassDef) and isinstance(b, ast.ClassDef):
            return all(visit(value, getattr(b, field), owner + '.' + a.name if owner else a.name,
                             field == 'body') for field, value in ast.iter_fields(a))
        if body and isinstance(a, (ast.Assign, ast.AnnAssign)) and type(a) is type(b):
            targets = a.targets if isinstance(a, ast.Assign) else [a.target]
            if (len(targets) == 1 and isinstance(targets[0], ast.Name) and isinstance(a.value, ast.Dict)
                    and isinstance(b.value, ast.Dict) and all(same_ast(x, y) for x, y in zip(a.value.keys, b.value.keys))
                    and len(a.value.keys) == len(b.value.keys)
                    and all(same_ast(value, getattr(b, field)) for field, value in ast.iter_fields(a) if field != 'value')):
                try:
                    before, after = ast.literal_eval(a.value), ast.literal_eval(b.value)
                except (ValueError, TypeError):
                    return False
                primitive = (bool, int, str, type(None))
                if (all(type(k) is str for k in before) and len(before) == len(a.value.keys)
                        and list(before) == list(after) and all(type(v) in primitive for v in [*before.values(), *after.values()])):
                    keys = [k for k in before if (type(before[k]), before[k]) != (type(after[k]), after[k])]
                    if keys:
                        result.append((f'{path}:{owner}:{targets[0].id}', keys, before, after))
                    return bool(keys)
        if type(a) is not type(b):
            return False
        if isinstance(a, ast.AST):
            return all(visit(value, getattr(b, field), owner, False) for field, value in ast.iter_fields(a))
        if isinstance(a, list):
            return len(a) == len(b) and all(visit(x, y, owner, body) for x, y in zip(a, b))
        return a == b

    return result if visit(ast.parse(original), ast.parse(updated)) and result else None


def load_profile(path):
    try:
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        if (data['revision'] != REVISION or data['kind'] != 'field_consumers' or data['version'] != 1
                or len(set(data['tests'])) != len(data['tests'])):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def consumers(profile, path, original, updated, *, updates):
    if not profile or profile['sources'].get(path) != source_hash(original):
        return None
    changes = dictionary_changes(original, updated, path)
    if changes is None:
        return None
    selected = int(profile['incomplete'], 16)
    for key, changed, before, after in changes:
        row = profile['fields'].get(key)
        if row is None:
            return None
        selected |= int(row['whole'], 16)
        for name in changed:
            selected |= int(row['keys'].get(name, '0'), 16)
        if key.endswith(':arg_types'):
            # The derived required_args proof depends on the entire initializer
            # and validator file. Simultaneous edits invalidate that evidence.
            if (CORE in updates and validator_hash(updates[CORE]) != profile.get('validator_source')
                    or any(type(v) is not bool for v in [*before.values(), *after.values()])):
                return None
            for name in changed:
                selected |= int(row.get('missing', {}).get(name, '0'), 16)
    return selected
