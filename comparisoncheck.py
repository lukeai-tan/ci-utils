"""Prove a narrow comparison rewrite from pristine operand type evidence."""

from __future__ import annotations

import ast
import enum
import gzip
import hashlib
import json
import warnings


REVISION = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
COMPARISONS = (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)
FAMILIES = ({ast.Eq, ast.Is}, {ast.NotEq, ast.IsNot})
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
BUILTIN_TYPES = {id(cls): ("builtin", "builtins", cls.__name__) for cls in
                 (type(None), bool, int, float, complex, str, bytes, list, tuple, dict, set, frozenset, range)}


def location(path, node):
    return f"{path}:{node.lineno}:{node.col_offset}:{node.end_lineno}:{node.end_col_offset}"


def source_hash(source):
    return hashlib.sha256(source.encode()).hexdigest()


def comparison_changes(old, new, path):
    """Accept only operator substitutions with unchanged operands and bindings."""
    from selector import same_ast

    changes = set()

    def compare(a, b, in_function=False):
        if type(a) is not type(b):
            return False
        if isinstance(a, (*FUNCTIONS, ast.ClassDef)) and same_ast(a, b):
            return True
        if isinstance(a, ast.Compare):
            if len(a.ops) == len(b.ops) == 1 and type(a.ops[0]) is not type(b.ops[0]):
                if (not in_function or not any({type(a.ops[0]), type(b.ops[0])} <= family
                                              for family in FAMILIES)
                        or any(isinstance(operand, ast.Constant) and operand.value is not None
                               for operand in (a.left, *a.comparators))):
                    return False
                if not compare(a.left, b.left, in_function) or not compare(a.comparators, b.comparators, in_function):
                    return False
                changes.add(location(path, a))
                return True
        if isinstance(a, ast.AST):
            if isinstance(a, FUNCTIONS):
                # Retain functions with closures or explicit shared bindings.
                if any(isinstance(node, (*FUNCTIONS, ast.Lambda, ast.ClassDef, ast.Global, ast.Nonlocal))
                       for child in a.body for node in ast.walk(child)):
                    return ast.dump(a) == ast.dump(b)
                return all(compare(av, getattr(b, name), name == "body")
                           for name, av in ast.iter_fields(a))
            if isinstance(a, ast.ClassDef):
                return all(compare(av, getattr(b, name), False) for name, av in ast.iter_fields(a))
            return all(compare(av, getattr(b, name), in_function) for name, av in ast.iter_fields(a))
        if isinstance(a, list):
            return len(a) == len(b) and all(compare(x, y, in_function) for x, y in zip(a, b))
        return a == b

    return changes if compare(old, new) and changes else None


def operand_type(value):
    """Describe actual special-method resolution without invoking user equality."""
    cls = type(value)
    builtin = BUILTIN_TYPES.get(id(cls))
    if builtin is not None:
        return builtin
    if type(cls) is not type and type(cls) is not enum.EnumMeta:
        return ("opaque", "unknown", "custom-metaclass")
    namespace = type.__getattribute__(cls, "__dict__")
    module = namespace.get("__module__", "builtins")
    name = type.__getattribute__(cls, "__qualname__")
    module = module if type(module) is str else "unknown"
    name = name if type(name) is str else "unknown"
    bases = type.__getattribute__(cls, "__mro__")
    eq = next((type.__getattribute__(base, "__dict__")["__eq__"] for base in bases
               if "__eq__" in type.__getattribute__(base, "__dict__")), None)
    ne = next((type.__getattribute__(base, "__dict__")["__ne__"] for base in bases
               if "__ne__" in type.__getattribute__(base, "__dict__")), None)
    kind = "identity" if eq is object.__eq__ and ne is object.__ne__ else "opaque"
    # String enum members can use inherited str equality. Require real canonical
    # members and unique immutable string payloads; enum-looking subclasses or
    # objects constructed directly with str.__new__ remain opaque.
    members = namespace.get("_member_map_")
    if (eq is str.__eq__ and ne is str.__ne__ and any(base is enum.Enum for base in bases)
            and any(base is str for base in bases)
            and type(members) is dict):
        unique = {}
        canonical = False
        valid = True
        for member in members.values():
            if type(member) is not cls:
                valid = False
                break
            text = str.__str__(member)
            previous = unique.get(text)
            if previous is not None and previous is not member:
                valid = False
                break
            unique[text] = member
            canonical |= member is value
        if valid and canonical:
            kind = "string-enum"
    return (kind, module, name)


def operand_types(left, right):
    a, b = operand_type(left), operand_type(right)
    if a[0] == b[0] == "string-enum" and type(left) is type(right):
        return (("same-string-enum", *a[1:]), ("same-string-enum", *b[1:]))
    return a, b


def identity_equality_pair(pair):
    """These type pairs agree for every value, without extra user method calls."""
    a, b = pair
    if a[0] == b[0] == "same-string-enum" and tuple(a[1:]) == tuple(b[1:]):
        return True
    if tuple(a) == tuple(b) == ("builtin", "builtins", "bool"):
        return True
    none = ("builtin", "builtins", "NoneType")
    if all(value[0] == "identity" or tuple(value) == none for value in pair):
        return True
    # Exact builtins reject None without comparing container elements.
    builtin_names = {"NoneType", "bool", "int", "float", "complex", "str", "bytes",
                     "list", "tuple", "dict", "set", "frozenset", "range"}
    return ((tuple(a) == none and tuple(b[:2]) == ("builtin", "builtins") and b[2] in builtin_names)
            or (tuple(b) == none and tuple(a[:2]) == ("builtin", "builtins") and a[2] in builtin_names))


def load_comparison_profile(path):
    try:
        with gzip.open(path, "rt") as stream:
            profile = json.load(stream)
        if (profile["revision"] != REVISION or profile["version"] not in (1, 2, 3)
                or profile["kind"] != "comparison_types" or not profile["collection_complete"]
                or "startup" in profile.get("bad", [])
                or not isinstance(profile["sources"], dict)
                or len(profile["tests"]) != len(set(profile["tests"]))):
            return None
        complete = set(profile["complete"])
        if (not complete <= set(range(len(profile["tests"])))
                or not all(profile["tests"][i] in profile["cpu"] for i in complete)):
            return None
        if profile["version"] == 3:
            # The offline builder applies the same type proof once for every
            # baseline source location. No edit or alternative result is used.
            if not isinstance(profile["certified"], dict) or any(
                not isinstance(key, str) or not isinstance(contexts, list)
                or any(type(i) is not int or i not in complete for i in contexts)
                for key, contexts in profile["certified"].items()
            ):
                return None
            return profile
        if not isinstance(profile["operands"], dict):
            return None
        if profile["version"] == 2:
            pairs = profile["pairs"]
            if not isinstance(pairs, list) or any(len(pair) != 2 or any(len(value) != 3 or not all(isinstance(x, str) for x in value)
                                                                      for value in pair) for pair in pairs):
                return None
        for contexts in profile["operands"].values():
            for context, pairs in contexts.items():
                if context != "startup" and int(context) not in range(len(profile["tests"])):
                    return None
                if not isinstance(pairs, list) or not pairs:
                    return None
                if profile["version"] == 2:
                    if any(type(index) is not int or not 0 <= index < len(profile["pairs"]) for index in pairs):
                        return None
                    contexts[context] = [profile["pairs"][index] for index in pairs]
                elif any(len(pair) != 2 or any(len(value) != 3 or not all(isinstance(x, str) for x in value)
                                              for value in pair) for pair in pairs):
                    return None
        return profile
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def certified_tests(profile, updates, originals, catalog, *, trees=None):
    """Every edited comparison must have safe evidence in each removed context."""
    if not profile:
        return set()
    changes = set()
    for path, updated in updates.items():
        if profile["sources"].get(path) != source_hash(originals[path]):
            return set()
        old, new = trees[path] if trees is not None else (ast.parse(originals[path]), ast.parse(updated))
        found = comparison_changes(old, new, path)
        if not found:
            return set()
        # Identity comparisons with folded literals can change import warnings
        # even when the comparison itself has the same result.
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", SyntaxWarning)
            compile(originals[path], path, "exec")
            compile(updated, path, "exec")
        if observed:
            return set()
        changes.update(found)
    if not changes:
        return set()
    safe_contexts = set(profile["complete"])
    for key in changes:
        if profile["version"] == 3:
            contexts = profile["certified"].get(key)
            if not contexts:
                return set()
            safe_contexts.intersection_update(contexts)
            continue
        contexts = profile["operands"].get(key)
        if not contexts:
            return set()
        # Startup is shared by tests; an uncertain startup observation blocks refinement.
        if any(not identity_equality_pair(pair) for pair in contexts.get("startup", [])):
            return set()
        safe_contexts.intersection_update(int(context) for context, pairs in contexts.items()
                                         if context != "startup" and pairs
                                         and all(identity_equality_pair(pair) for pair in pairs))
    ids = {profile["tests"][i] for i in safe_contexts}
    return {test["number"] for test in catalog if test["id"] in ids}
