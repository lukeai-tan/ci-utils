"""Certify local Boolean and dictionary edits from pristine observations."""

from __future__ import annotations

import ast
import gzip
import itertools
import json
import warnings
from pathlib import Path

from comparisoncheck import REVISION, location, source_hash


FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
MAX_ATOMS = 10
NONBOOL = 8


def formula(node, atoms):
    """Keep operand occurrences separate and model Python's returned operand."""
    if isinstance(node, ast.BoolOp):
        return [type(node.op).__name__, [formula(value, atoms) for value in node.values]]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ["Not", formula(node.operand, atoms)]
    if isinstance(node, ast.Constant) and type(node.value) in (bool, type(None), int, str):
        return ["Const", type(node.value).__name__, node.value]
    if any(isinstance(n, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom, ast.Lambda))
           or isinstance(n, ast.Name) and n.id in {"super", "__class__"} for n in ast.walk(node)):
        raise ValueError("unsupported Boolean operand")
    key = ast.dump(node)
    if key in atoms:
        raise ValueError("repeated operand occurrence")
    atoms.append(key)
    if len(atoms) > MAX_ATOMS:
        raise ValueError("Boolean expression too large")
    return ["Atom", len(atoms) - 1]


def boolean_site(node, *, truth=False):
    if not (isinstance(node, ast.BoolOp) or isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
        return None
    atoms = []
    try:
        value = formula(node, atoms)
    except ValueError:
        return None
    return dict(kind="boolean", formula=value, atoms=atoms, truth=truth) if atoms else None


def lookup_site(node):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "args"
            and not node.keywords and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) is str):
        return dict(kind="lookup", key=node.args[0].value)
    return None


def local_change(old, new, path):
    """Accept one modeled expression; everything around it must remain identical."""
    from selector import same_ast

    changes = []

    def visit(a, b, body=False, truth=False):
        if same_ast(a, b):
            return True
        if isinstance(a, ast.AST) and isinstance(b, ast.AST) and body:
            before, after = boolean_site(a, truth=truth), boolean_site(b, truth=truth)
            if before and after is None:
                atoms = []
                try:
                    after = dict(kind="boolean", formula=formula(b, atoms), atoms=atoms, truth=truth)
                except ValueError:
                    pass
            if before and after and set(before["atoms"]) == set(after["atoms"]):
                # Renumber replacement operands into the original occurrence index.
                translate = {i: before["atoms"].index(key) for i, key in enumerate(after["atoms"])}
                def renumber(value):
                    if value[0] == "Atom":
                        return ["Atom", translate[value[1]]]
                    if value[0] in ("And", "Or"):
                        return [value[0], [renumber(child) for child in value[1]]]
                    if value[0] == "Not":
                        return ["Not", renumber(value[1])]
                    return value
                after.update(formula=renumber(after["formula"]), atoms=before["atoms"])
                changes.append((location(path, a), before, after))
                return True
            before, after = lookup_site(a), lookup_site(b)
            if before and after and same_ast(a.func, b.func):
                changes.append((location(path, a), before, after))
                return True
        if type(a) is not type(b):
            return False
        if isinstance(a, FUNCTIONS):
            if any(isinstance(n, (*FUNCTIONS, ast.ClassDef, ast.Global, ast.Nonlocal))
                   for statement in a.body for n in ast.walk(statement)):
                return False
            return all(visit(value, getattr(b, field), field == "body") for field, value in ast.iter_fields(a))
        if isinstance(a, ast.AST):
            return all(visit(value, getattr(b, field), body,
                             isinstance(a, (ast.If, ast.While, ast.IfExp)) and field == "test")
                       for field, value in ast.iter_fields(a))
        if isinstance(a, list):
            return len(a) == len(b) and all(visit(x, y, body, truth) for x, y in zip(a, b))
        return a == b

    return changes[0] if visit(old, new) and len(changes) == 1 else None


def evaluate(expression, values, masks, trace):
    kind = expression[0]
    if kind == "Const":
        value = expression[2]
        return bool(value), ("bool", value) if expression[1] == "bool" else tuple(expression)
    if kind == "Atom":
        index = expression[1]
        trace.append(index)
        value = values[index]
        return value, ("atom", index) if masks[index] & NONBOOL else ("bool", value)
    if kind == "Not":
        value, _ = evaluate(expression[1], values, masks, trace)
        return not value, ("bool", not value)
    result = None
    for child in expression[1]:
        result = evaluate(child, values, masks, trace)
        if (kind == "And" and not result[0]) or (kind == "Or" and result[0]):
            break
    return result


def unchanged(row, before, after):
    """Require the same operand evaluation trace, truth, and returned object."""
    if row is None:
        # A complete context that never reached the sole changed expression.
        return True
    if before["kind"] == "lookup":
        return (row["seen"] and not row["unknown"] and after["key"] not in row["different"]
                and (row["same"] is None or after["key"] in row["same"]))
    masks = row["masks"]
    if row["unknown"] or len(masks) != len(before["atoms"]):
        return False
    choices = [[bool(bit == 2) for bit in (1, 2) if mask & bit] or [False, True] for mask in masks]
    possible = False
    for values in itertools.product(*choices):
        original_trace, replacement_trace = [], []
        original = evaluate(before["formula"], values, masks, original_trace)
        # Missing operands may have raised before their observation callback.
        # Independent domains can also admit extra paths; retain either case.
        if any(masks[i] == 0 for i in original_trace):
            return False
        possible = True
        replacement = evaluate(after["formula"], values, masks, replacement_trace)
        if (original_trace != replacement_trace or original[0] != replacement[0]
                or not before["truth"] and original[1] != replacement[1]):
            return False
    return possible


class LazyProfile:
    def __init__(self, path):
        self.path = Path(path)
        self.loaded = False
        self.data = None

    def load(self):
        if not self.loaded:
            self.data = load_profile(self.path)
            self.loaded = True
        return self.data


def load_profile(path):
    try:
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        if (data["revision"] != REVISION or data["version"] not in (1, 2) or data["kind"] != "local_behavior"
                or not data["collection_complete"] or len(data["tests"]) != len(set(data["tests"]))
                or any(type(i) is not int or not 0 <= i < len(data["tests"]) or data["tests"][i] not in data["cpu"]
                       for i in data["complete"])):
            return None
        if data["version"] == 2:
            templates = data.pop("row_templates")
            for field, limit in (("rows", len(data["tests"])), ("query_rows", len(data["queries"]))):
                if field in data:
                    return None
                expanded = {}
                for key, groups in data.pop(field + "_groups").items():
                    contexts, seen, started = {}, 0, False
                    for index, bits, startup in groups:
                        if (type(index) is not int or not 0 <= index < len(templates)
                                or type(bits) is not str or type(startup) is not bool):
                            return None
                        mask = int(bits, 16)
                        if (mask < 0 or mask.bit_length() > limit or mask & seen
                                or startup and (field != "rows" or started)):
                            return None
                        seen |= mask
                        if startup:
                            contexts["startup"] = templates[index]
                            started = True
                        while mask:
                            bit = mask & -mask
                            contexts[str(bit.bit_length() - 1)] = templates[index]
                            mask ^= bit
                    expanded[key] = contexts
                data[field] = expanded
        checked = set()
        for key, site in data["sites"].items():
            if site["kind"] not in ("boolean", "lookup"):
                return None
            if site["kind"] == "boolean" and (not 0 < len(site["atoms"]) <= MAX_ATOMS or type(site["truth"]) is not bool):
                return None
            initialization = data.get("initialization", {})
            if key in initialization and site["kind"] != "lookup":
                return None
            for contexts in (data["rows"].get(key, {}), data["query_rows"].get(key, {}),
                             {"initialization": initialization[key]} if key in initialization else {}):
                for row in contexts.values():
                    shape = site["kind"], len(site["atoms"]) if site["kind"] == "boolean" else 0, id(row)
                    if shape in checked:
                        continue
                    checked.add(shape)
                    if type(row["unknown"]) is not bool:
                        return None
                    if site["kind"] == "boolean":
                        if len(row["masks"]) != len(site["atoms"]) or any(type(m) is not int or not 0 <= m <= 15 for m in row["masks"]):
                            return None
                    elif (type(row["seen"]) is not bool or any(type(k) is not str for k in row["different"])
                          or row["same"] is not None and any(type(k) is not str for k in row["same"])):
                        return None
        if (type(data.get("initialization_complete", False)) is not bool
                or not set(data.get("initialization", {})) <= set(data["sites"])):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def validated_change(profile, updates, originals, *, trees=None):
    if profile is None or len(updates) != 1:
        return None
    path, updated = next(iter(updates.items()))
    old, new = trees[path] if trees is not None else (ast.parse(originals[path]), ast.parse(updated))
    change = local_change(old, new, path)
    if change is None:
        return None
    profile = profile.load() if isinstance(profile, LazyProfile) else profile
    if (profile is None or profile["sources"].get(path) != source_hash(originals[path])
            or profile["sites"].get(change[0]) != change[1]):
        return None
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", SyntaxWarning)
        compile(originals[path], path, "exec")
        compile(updated, path, "exec")
    return None if observed else change


def certified_tests(profile, updates, originals, catalog, *, trees=None, require_initialization=False):
    change = validated_change(profile, updates, originals, trees=trees)
    if change is None:
        return set()
    profile = profile.load() if isinstance(profile, LazyProfile) else profile
    key, before, after = change
    if require_initialization and (before["kind"] != "lookup"
            or profile.get("initialization_complete") is not True
            or not unchanged(profile.get("initialization", {}).get(key), before, after)):
        return set()
    contexts = profile["rows"].get(key, {})
    if "startup" in contexts and not unchanged(contexts["startup"], before, after):
        return set()
    decisions = {}
    def same(row):
        identity = id(row)
        if identity not in decisions:
            decisions[identity] = unchanged(row, before, after)
        return decisions[identity]
    safe = {profile["tests"][i] for i in profile["complete"] if same(contexts.get(str(i)))}
    return {test["number"] for test in catalog if test["id"] in safe}
