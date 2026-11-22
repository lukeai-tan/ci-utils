"""Operator-only rewrites certified by exact primitive operand observations."""
from __future__ import annotations

import ast
import operator

OPS = ('Is', 'IsNot', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE')
OPERATORS = (operator.is_, operator.is_not, operator.eq, operator.ne,
             operator.lt, operator.le, operator.gt, operator.ge)
PRIMITIVES = (type(None), bool, int, float, str, bytes)


def observe(left, right):
    """Extra comparisons are permitted only for exact immutable builtins.

    Record each operator independently: NaN and incompatible ordering types do
    not obey the usual equality/order identities. Custom methods are never run.
    """
    if type(left) not in PRIMITIVES or type(right) not in PRIMITIVES:
        return 0
    known = truth = 0
    for i, operation in enumerate(OPERATORS):
        try:
            result = operation(left, right)
        except TypeError:
            continue
        known |= 1 << i
        truth |= int(result) << i
    return (known << 8) | truth


def same(code, before, after):
    a, b = OPS.index(before), OPS.index(after)
    known, truth = code >> 8, code & 255
    return bool(known & (1 << a) and known & (1 << b)) and bool(truth & (1 << a)) == bool(truth & (1 << b))


def edits(old, new, path):
    """Require all differences to be single operator edits in ordinary bodies."""
    from comparisoncheck import location
    from selector import same_ast
    found = {}

    def walk(a, b, body=False):
        if type(a) is not type(b):
            return False
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if same_ast(a, b):
                return True
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                   ast.Lambda, ast.Global, ast.Nonlocal))
                   for stmt in a.body for n in ast.walk(stmt)):
                return False
            return all(walk(value, getattr(b, key), key == 'body') for key, value in ast.iter_fields(a))
        if isinstance(a, ast.Compare) and len(a.ops) == len(b.ops) == 1:
            before, after = type(a.ops[0]).__name__, type(b.ops[0]).__name__
            if before != after:
                if not body or before not in OPS or after not in OPS:
                    return False
                found[location(path, a)] = before, after
                return all(walk(value, getattr(b, key), body)
                           for key, value in ast.iter_fields(a) if key != 'ops')
        if isinstance(a, ast.AST):
            return all(walk(value, getattr(b, key), body and not isinstance(a, ast.ClassDef))
                       for key, value in ast.iter_fields(a))
        if isinstance(a, list):
            return len(a) == len(b) and all(walk(x, y, body) for x, y in zip(a, b))
        return a == b

    return found if walk(old, new) and found else None


def certified(profile, updates, originals, catalog, *, trees=None):
    from comparisoncheck import source_hash
    import warnings
    if not profile or not profile.get('relations'):
        return set()
    found = {}
    for path, text in updates.items():
        if profile['sources'].get(path) != source_hash(originals[path]):
            return set()
        pair = trees[path] if trees is not None else (ast.parse(originals[path]), ast.parse(text))
        changes = edits(*pair, path)
        if not changes:
            return set()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always', SyntaxWarning)
            compile(originals[path], path, 'exec')
            compile(text, path, 'exec')
        if observed:
            return set()
        found.update(changes)
    contexts = set(profile['complete'])
    for key, (before, after) in found.items():
        rows = profile['relations'].get(key)
        if not rows or any(not same(code, before, after) for code in rows.get('startup', [])):
            return set()
        contexts.intersection_update(int(context) for context, codes in rows.items()
                                    if context != 'startup' and codes
                                    and all(same(code, before, after) for code in codes))
    ids = {profile['tests'][i] for i in contexts}
    return {test['number'] for test in catalog if test['id'] in ids}
