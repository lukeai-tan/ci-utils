"""Locate executable changes using baseline code objects and their consumers."""

from __future__ import annotations

import dis
import gzip
import hashlib
import marshal
import json
import sys
from types import CodeType


def load_profile(path):
    from comparisoncheck import REVISION
    try:
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        if (data['revision'] != REVISION or data['kind'] != 'compiled_consumers' or data['version'] != 1
                or data['python'] != list(sys.version_info[:2]) or len(data['tests']) != len(set(data['tests']))):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def signature(code, *, constants=True):
    values = tuple(('code', value.co_name) if isinstance(value, CodeType)
                   else (type(value).__name__, marshal.dumps(value)) for value in code.co_consts)
    return (code.co_code, values if constants else (), code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars, code.co_argcount, code.co_posonlyargcount,
            code.co_kwonlyargcount, code.co_flags, code.co_exceptiontable)


def identity(code):
    digest = hashlib.sha256(marshal.dumps(signature(code))).hexdigest()[:20]
    return f'{code.co_firstlineno}:{code.co_qualname}:{digest}'


def children(code):
    return [value for value in code.co_consts if isinstance(value, CodeType)]


def changes(old, new):
    """Match nested bodies independently; changed closures still affect their owner."""
    before, after = children(old), children(new)
    if [(c.co_name, c.co_flags) for c in before] != [(c.co_name, c.co_flags) for c in after]:
        raise ValueError('changed executable scope structure')
    if signature(old) != signature(new):
        arguments = lambda c: (c.co_argcount, c.co_posonlyargcount, c.co_kwonlyargcount,
                               c.co_varnames[:c.co_argcount + c.co_kwonlyargcount
                                            + bool(c.co_flags & 4) + bool(c.co_flags & 8)])
        if arguments(old) != arguments(new):
            # Argument binding can raise before PY_START. Changed signatures
            # cannot be narrowed from observations of entered frames.
            raise ValueError('changed argument binding')
        # A function docstring is consumed when the object is constructed, even
        # if no test calls that function. Leave it to the existing broad path.
        if old.co_flags & 1 and old.co_consts and new.co_consts:
            a, b = old.co_consts[0], new.co_consts[0]
            if (isinstance(a, str) or isinstance(b, str)) and a != b:
                raise ValueError('changed function metadata')
        yield old, new
    for a, b in zip(before, after):
        yield from changes(a, b)


def constant_locations(old, new):
    """Narrow only identical instructions with a complete set of constant loads."""
    if signature(old, constants=False) != signature(new, constants=False):
        return None
    if any(i.opname in {'STORE_ATTR', 'STORE_SUBSCR', 'STORE_GLOBAL', 'STORE_DEREF',
                        'DELETE_ATTR', 'DELETE_SUBSCR', 'DELETE_GLOBAL', 'DELETE_DEREF'}
           for i in dis.get_instructions(old)):
        return None
    if len(old.co_consts) != len(new.co_consts):
        return None
    changed = {i for i, (a, b) in enumerate(zip(old.co_consts, new.co_consts))
               if not isinstance(a, CodeType) and marshal.dumps(a) != marshal.dumps(b)}
    loaded, lines = set(), set()
    for instruction in dis.get_instructions(old):
        if instruction.opcode in dis.hasconst and instruction.arg in changed:
            loaded.add(instruction.arg)
            position = instruction.positions
            if position.lineno is None or position.end_lineno is None:
                return None
            lines.update(range(position.lineno, position.end_lineno + 1))
    return lines if changed and loaded == changed else None


def query_reachability(profile, base, patch):
    """Slice complete real tasks only when edited scopes ran inside optimize."""
    from pathlib import Path
    from selector import apply_edit, parse_patch, same_ast
    from querycheck import supported_patch
    from comparisoncheck import source_hash
    import ast

    if (not profile or not profile.get('query_complete') or not supported_patch(base, patch)
            or profile.get('python') != list(sys.version_info[:2])):
        return None
    # Queries with no optimize call (e.g. an explicit skip) remain covered by
    # whole-task replay; they cannot be inferred from a partial recording.
    indices = set()
    for edit in parse_patch(patch):
        if not edit.path.startswith('sqlglot/optimizer/'):
            return None
        old = (Path(base) / edit.path).read_text()
        new, _ = apply_edit(old, edit)
        if (profile['sources'].get(edit.path) != source_hash(old)
                or not same_ast(ast.parse(old), ast.parse(new), structure=True)):
            return None
        try:
            pairs = list(changes(compile(old, edit.path, 'exec', dont_inherit=True), compile(new, edit.path, 'exec', dont_inherit=True)))
        except ValueError:
            return None
        for before, after in pairs:
            if not before.co_flags & 1:
                return None
            row = profile['codes'].get(edit.path, {}).get(identity(before))
            if (row is None and profile.get('unseen_scopes')
                    and identity(before) in profile.get('universe', {}).get(edit.path, [])):
                continue
            if row is None or row['query_outside']:
                return None
            indices.update(row['queries'])
    return indices


def query_plan(profile, base, patch, tasks):
    from selector import parse_patch
    indices = query_reachability(profile, base, patch)
    if indices is None or profile.get('driver') != tasks.get('driver'):
        return None
    if any(tasks['sources'].get(e.path) != profile['sources'].get(e.path) for e in parse_patch(patch)):
        return None
    checkpoints = profile.get('checkpoints', {})
    if not indices <= set(tasks['queries']):
        return None
    eligible = {str(i): checkpoints[str(i)] for i in indices
                if str(i) in checkpoints and len(checkpoints[str(i)]) == 1
                and checkpoints[str(i)][0]['pure'] and checkpoints[str(i)][0].get('output')
                and checkpoints[str(i)][0].get('kind') == 'expression_graph_v1'}
    estimate = sum(eligible[str(i)][0].get('cpu', tasks['queries'][i]['cpu'])
                   if str(i) in eligible else tasks['queries'][i]['cpu'] for i in indices)
    return sorted(indices), estimate, eligible


def consumers(profile, path, original, updated, *, trees=None, statements=True):
    """Return a baseline-index mask; unsupported or startup changes stay broad."""
    if not profile or profile.get('python') != list(sys.version_info[:2]):
        return None
    from comparisoncheck import source_hash
    if profile.get('sources', {}).get(path) != source_hash(original):
        return None
    try:
        pairs = list(changes(compile(original, path, 'exec', dont_inherit=True), compile(updated, path, 'exec', dont_inherit=True)))
        if trees is None:
            import ast
            trees = ast.parse(original), ast.parse(updated)
        from statementcheck import locations as statement_locations
        entries = profile['codes'].get(path, {})
        selected = 0
        for before, after in pairs:
            if not before.co_flags & 1:
                return None
            row = entries.get(identity(before))
            if (row is None and profile.get('unseen_scopes')
                    and identity(before) in profile.get('universe', {}).get(path, [])):
                continue
            if row is None or row['startup']:
                return None
            locations = constant_locations(before, after) if before.co_flags & 1 else None
            if locations is None and statements:
                spans = statement_locations(*trees, before, after)
                if spans and all(any(str(n) in row['lines'] for n in range(start, end + 1))
                                 for start, end in spans):
                    locations = {n for start, end in spans for n in range(start, end + 1)
                                 if str(n) in row['lines']}
            # Captured values are consumed after the outer call. Frame entry is
            # the fallback whenever a changed function creates closure cells.
            if locations and not before.co_cellvars and all(str(n) in row['lines'] for n in locations):
                selected |= int(row.get('extra', '0'), 16)
                for line in locations:
                    selected |= int(row['lines'][str(line)], 16)
            else:
                selected |= int(row['tests'], 16)
        return selected | int(profile['incomplete'], 16)
    except (SyntaxError, ValueError, KeyError, TypeError):
        return None
