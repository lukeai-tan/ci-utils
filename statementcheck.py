"""Baseline statements that must execute before a function-body edit matters."""

from __future__ import annotations

import ast
import dis


FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
COMPOUNDS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)


def dump(node):
    return ast.dump(node, include_attributes=False)


def locations(old_tree, new_tree, before, after):
    """Return old statement spans, or None when frame-wide effects are possible.

    Stable compiler bindings prevent an edit in one branch from changing name
    resolution in another. Nested definitions, closures, exception structure,
    insertions and deletions retain the existing whole-frame fallback.
    """
    binding = lambda c: (c.co_varnames, c.co_freevars, c.co_cellvars, c.co_flags,
                         c.co_argcount, c.co_posonlyargcount, c.co_kwonlyargcount)
    if binding(before) != binding(after) or before.co_cellvars or before.co_freevars:
        return None

    def owner(tree, code):
        matches = [n for n in ast.walk(tree) if isinstance(n, FUNCTIONS)
                   and n.name == code.co_name
                   and min([n.lineno, *[d.lineno for d in n.decorator_list]]) == code.co_firstlineno]
        return matches[0] if len(matches) == 1 else None

    old, new = owner(old_tree, before), owner(new_tree, after)
    if old is None or new is None:
        return None
    # Lists of decorators/type parameters need structural comparison without
    # AST object identities or source positions.
    def signature(node):
        return ast.dump(ast.Module(body=[type(node)(**{k: [] if k == 'body' else getattr(node, k)
                                                      for k in node._fields})], type_ignores=[]))
    if signature(old) != signature(new):
        return None
    forbidden = (*FUNCTIONS, ast.Lambda, ast.ClassDef, ast.Global, ast.Nonlocal,
                 ast.Yield, ast.YieldFrom, ast.Await)
    if any(isinstance(n, forbidden) for fn in (old, new) for stmt in fn.body for n in ast.walk(stmt)):
        return None
    if (old.body and new.body and isinstance(old.body[0], ast.Expr)
            and isinstance(old.body[0].value, ast.Constant)
            and isinstance(old.body[0].value.value, str)
            and dump(old.body[0]) != dump(new.body[0])):
        return None

    spans = []
    def block(left, right):
        if len(left) != len(right):
            return False
        for a, b in zip(left, right):
            if dump(a) == dump(b):
                continue
            if isinstance(a, COMPOUNDS) or isinstance(b, COMPOUNDS):
                if type(a) is not type(b):
                    return False
                fields = [k for k in a._fields if k not in {'body', 'orelse', 'type_comment'}]
                def field(n, k):
                    value = getattr(n, k)
                    return [dump(v) for v in value] if isinstance(value, list) else dump(value)
                if any(field(a, k) != field(b, k) for k in fields):
                    spans.append((a.lineno, a.body[0].lineno - 1))
                for name in ('body', 'orelse'):
                    if not block(getattr(a, name, []), getattr(b, name, [])):
                        return False
            elif isinstance(a, (ast.Try, ast.TryStar, ast.Match)) or isinstance(b, (ast.Try, ast.TryStar, ast.Match)):
                return False
            else:
                # A changed simple statement takes effect when the old statement
                # is reached, even if its expression or statement type changes.
                spans.append((a.lineno, a.end_lineno))
        return True

    if not block(old.body, new.body) or not spans:
        return None
    executable = {line for _, line in dis.findlinestarts(before) if line is not None}
    if any(not any(start <= line <= end for line in executable) for start, end in spans):
        return None
    return spans
