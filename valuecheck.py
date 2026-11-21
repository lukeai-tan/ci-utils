"""Certify literal comparisons from complete observations of the pinned source."""

from __future__ import annotations

import ast
import gzip
import json
import operator
import warnings
from pathlib import Path

from comparisoncheck import REVISION, location, source_hash


OPS = {"Eq": operator.eq, "NotEq": operator.ne, "Lt": operator.lt,
       "LtE": operator.le, "Gt": operator.gt, "GtE": operator.ge}
MISSING = object()


class LazyProfile:
    """Decode operand domains only after an eligible comparison is found."""

    def __init__(self, path):
        self.path = Path(path)
        self.loaded = False
        self.data = None

    def load(self):
        if not self.loaded:
            self.data = load_profile(self.path)
            self.loaded = True
        return self.data

    def __getitem__(self, key):
        return self.load()[key]


def literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, str):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int):
        return node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
    return MISSING


def comparison_site(node):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or type(node.ops[0]).__name__ not in OPS:
        return None
    left, right = literal(node.left), literal(node.comparators[0])
    if (left is MISSING) == (right is MISSING):
        return None
    value, side = (right, "left") if left is MISSING else (left, "right")
    return {"op": type(node.ops[0]).__name__, "literal": value,
            "side": side, "type": type(value).__name__}


def comparison_change(old, new, path):
    """Require exactly one comparison edit in an otherwise identical function body."""
    from selector import same_ast

    changes = []

    def visit(a, b, body=False):
        if type(a) is not type(b):
            return False
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and same_ast(a, b):
            return True
        if isinstance(a, ast.Compare) and body:
            before, after = comparison_site(a), comparison_site(b)
            if before and after and before["side"] == after["side"] and before["type"] == after["type"]:
                x = a.left if before["side"] == "left" else a.comparators[0]
                y = b.left if after["side"] == "left" else b.comparators[0]
                if ast.dump(x) == ast.dump(y):
                    if before == after:
                        return True
                    changes.append((location(path, a), before, after))
                    return True
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Closures/shared bindings need a separate state proof.
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                  ast.ClassDef, ast.Global, ast.Nonlocal))
                   for statement in a.body for n in ast.walk(statement)):
                return ast.dump(a) == ast.dump(b)
            return all(visit(value, getattr(b, field), field == "body") for field, value in ast.iter_fields(a))
        if isinstance(a, ast.ClassDef):
            return all(visit(value, getattr(b, field), False) for field, value in ast.iter_fields(a))
        if isinstance(a, ast.AST):
            return all(visit(value, getattr(b, field), body) for field, value in ast.iter_fields(a))
        if isinstance(a, list):
            return len(a) == len(b) and all(visit(x, y, body) for x, y in zip(a, b))
        return a == b

    return changes[0] if visit(old, new) and len(changes) == 1 else None


def unchanged(domain, before, after):
    """Check every recorded primitive, or every integer in a conservative interval."""
    if not domain or domain.get("unknown") or domain.get("type") != before["type"]:
        return False
    values = domain.get("values", [])
    if not values:
        return False
    if domain["kind"] == "range":
        lo, hi = values
        # Both predicates are constant between their literal boundaries. Include
        # the boundaries and neighbors as well as the interval endpoints.
        values = {lo, hi} | {v + delta for v in (before["literal"], after["literal"])
                                 for delta in (-1, 0, 1) if lo <= v + delta <= hi}
    for value in values:
        a, b = (value, before["literal"]) if before["side"] == "left" else (before["literal"], value)
        x, y = (value, after["literal"]) if after["side"] == "left" else (after["literal"], value)
        if OPS[before["op"]](a, b) != OPS[after["op"]](x, y):
            return False
    return True


def load_profile(path):
    try:
        with gzip.open(path, "rt") as stream:
            profile = json.load(stream)
        if (profile["revision"] != REVISION or profile["version"] != 1
                or profile["kind"] != "literal_values" or not profile["collection_complete"]
                or len(profile["tests"]) != len(set(profile["tests"]))
                or not isinstance(profile["sources"], dict)
                or any(type(path) is not str or type(digest) is not str for path, digest in profile["sources"].items())):
            return None
        complete = set(profile["complete"])
        if (any(type(i) is not int or not 0 <= i < len(profile["tests"]) for i in profile["complete"])
                or any(profile["tests"][i] not in profile["cpu"] for i in complete)):
            return None
        for key, site in profile["sites"].items():
            if (not isinstance(key, str) or site["op"] not in OPS or site["side"] not in ("left", "right")
                    or type(site["literal"]).__name__ != site["type"] or site["type"] not in ("int", "str")):
                return None
        for key, contexts in profile["rows"].items():
            if key not in profile["sites"]:
                return None
            for context, row in contexts.items():
                if context != "startup" and (not context.isdecimal() or int(context) >= len(profile["tests"])):
                    return None
                if (row["kind"] not in ("exact", "range") or row["type"] != profile["sites"][key]["type"]
                        or type(row["unknown"]) is not bool
                        or any(type(v).__name__ != row["type"] for v in row["values"])
                        or (row["kind"] == "range" and not (row["unknown"] and row["values"] == [])
                            and (row["type"] != "int" or len(row["values"]) != 2
                                 or row["values"][0] > row["values"][1]))):
                    return None
        return profile
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def validated_change(profile, updates, originals, *, trees=None):
    if not profile or len(updates) != 1:
        return None
    path, updated = next(iter(updates.items()))
    old, new = trees[path] if trees is not None else (ast.parse(originals[path]), ast.parse(updated))
    change = comparison_change(old, new, path)
    if not change:
        return None
    if isinstance(profile, LazyProfile):
        profile = profile.load()
    if (not profile or profile["sources"].get(path) != source_hash(originals[path])
            or profile["sites"].get(change[0]) != change[1]):
        return None
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", SyntaxWarning)
        compile(originals[path], path, "exec")
        compile(updated, path, "exec")
    return None if observed else change


def certified_tests(profile, updates, originals, catalog, *, trees=None):
    change = validated_change(profile, updates, originals, trees=trees)
    if change is None:
        return set()
    key, before, after = change
    contexts = profile["rows"].get(key, {})
    if "startup" in contexts and not unchanged(contexts["startup"], before, after):
        return set()
    safe = {profile["tests"][i] for i in profile["complete"]
            if unchanged(contexts.get(str(i)), before, after)}
    return {test["number"] for test in catalog if test["id"] in safe}
