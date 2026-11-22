"""Select tests from baseline dependencies; no patches or labels are precomputed."""

from __future__ import annotations
import argparse
import os
import subprocess
import tempfile
import ast
import gzip
import json
import math
import re
import resource
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from contextvars import ContextVar
from pathlib import Path, PurePosixPath

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Edit:
    path: str
    hunks: list = field(default_factory=list)


def parse_patch(text):
    lines = text.splitlines(keepends=True)
    edits = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            old = line[4:].rstrip("\n").split("\t")[0]
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise ValueError("missing new path")
            new = lines[i][4:].rstrip("\n").split("\t")[0]
            if (
                not old.startswith("a/")
                or not new.startswith("b/")
                or old[2:] != new[2:]
            ):
                raise ValueError("new, deleted, or renamed file")
            path = new[2:]
            if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                raise ValueError("unsafe path")
            edits.append(Edit(path))
        elif line.startswith("@@"):
            match = HUNK.match(line)
            if not match or not edits:
                raise ValueError("invalid hunk")
            start, count, newstart, newcount = match.groups()
            start, count, newstart, newcount = (
                int(start),
                int(count or 1),
                int(newstart),
                int(newcount or 1),
            )
            body = []
            old_seen = new_seen = 0
            i += 1
            while i < len(lines) and (old_seen < count or new_seen < newcount):
                entry = lines[i]
                if entry.startswith("\\ No newline"):
                    if not body:
                        raise ValueError("unexpected newline marker")
                    body[-1] = body[-1].removesuffix("\n")
                    i += 1
                    continue
                if not entry or entry[0] not in " +-":
                    raise ValueError("invalid hunk body")
                old_seen += entry[0] in " -"
                new_seen += entry[0] in " +"
                body.append(entry)
                i += 1
            if i < len(lines) and lines[i].startswith("\\ No newline"):
                body[-1] = body[-1].removesuffix("\n")
                i += 1
            if (old_seen, new_seen) != (count, newcount):
                raise ValueError("hunk length mismatch")
            edits[-1].hunks.append((start, count, newstart, newcount, body))
            continue
        elif line.startswith(
            ("rename ", "new file ", "deleted file ", "GIT binary", "Binary files")
        ):
            raise ValueError("structural diff")
        i += 1
    if not edits or any(not e.hunks for e in edits):
        raise ValueError("no supported edits")
    return edits


def apply_edit(source, edit):
    original = source.splitlines(keepends=True)
    result = []
    cursor = 0
    touched = set()
    for start, count, newstart, newcount, body in edit.hunks:
        offset = start - 1 if count else start
        if offset < cursor or offset > len(original):
            raise ValueError("overlapping or out of range hunk")
        result.extend(original[cursor:offset])
        cursor = offset
        deleted_block = False
        for entry in body:
            kind, content = entry[0], entry[1:]
            if kind in " -":
                if cursor >= len(original) or original[cursor] != content:
                    raise ValueError("diff does not match baseline")
                if kind == "-":
                    touched.add(cursor + 1)
                    deleted_block = True
                else:
                    deleted_block = False
                cursor += 1
            if kind == "+":
                # Replacement lines use the removed range; a pure insertion
                # belongs to the next old statement, or triggers a boundary fallback.
                if not deleted_block:
                    touched.add(cursor + 1)
            if kind in " +":
                result.append(content)
    result.extend(original[cursor:])
    return "".join(result), touched


FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_ast_comparisons = ContextVar('ast_comparisons', default=None)


def same_ast(old, new, *, structure=False, settings=False):
    """Reuse comparisons only within a selection's immutable parsed trees."""
    cache = _ast_comparisons.get()
    if cache is None or not isinstance(old, ast.AST) or not isinstance(new, ast.AST):
        return _same_ast(old, new, structure=structure, settings=settings)
    key = old, new, structure, settings
    if key not in cache:
        cache[key] = _same_ast(old, new, structure=structure, settings=settings)
    return cache[key]


def _same_ast(old, new, *, structure=False, settings=False):
    """Compare syntax without allocating dumps or modifying cached trees."""
    pending = [(old, new)]
    while pending:
        a, b = pending.pop()
        if type(a) is not type(b):
            return False
        if isinstance(a, ast.AST):
            for name, av in ast.iter_fields(a):
                if structure and isinstance(a, FUNCTIONS) and name == "body":
                    continue
                bv = getattr(b, name)
                # A setting may alter literal arguments, including those inside
                # factory calls and lambdas. Keep names, calls, operators and
                # dictionary keys intact; initialization is checked separately.
                if settings and isinstance(a, ast.Constant) and name == "value":
                    if type(av) is not type(bv):
                        return False
                elif settings and isinstance(a, ast.Dict) and name == "keys":
                    if not same_ast(av, bv):
                        return False
                else:
                    pending.append((av, bv))
        elif isinstance(a, list):
            if len(a) != len(b):
                return False
            pending.extend(zip(a, b))
        elif a != b:
            return False
    return True


def class_setting_changes(old, new):
    """Find class settings that preserve construction or use pure literals.

    Literal containers may change their keys or size without adding calls,
    bindings, or attribute access during class initialization. Those edits
    require the class's full consumer set, including inherited consumers.
    """
    owners = set()

    def compare(a, b, owner=None):
        if type(a) is not type(b):
            return False
        if isinstance(a, ast.ClassDef) and same_ast(a, b):
            return True
        if isinstance(a, ast.ClassDef):
            owner = (owner + "." if owner else "") + a.name
        if isinstance(a, FUNCTIONS):
            return same_ast(a, b)
        if owner and isinstance(a, (ast.Assign, ast.AnnAssign)):
            if same_ast(a, b):
                return True
            targets = a.targets if isinstance(a, ast.Assign) else [a.target]
            if not all(isinstance(target, ast.Name) for target in targets):
                return False
            literal_container = (
                isinstance(a.value, (ast.Dict, ast.List, ast.Tuple, ast.Set))
                and type(a.value) is type(b.value)
                and literal_setting_value(a.value)
                and literal_setting_value(b.value)
            )
            if not literal_container and not same_ast(a.value, b.value, settings=True):
                return False
            if any(
                not same_ast(av, getattr(b, name))
                for name, av in ast.iter_fields(a)
                if name != "value"
            ):
                return False
            owners.add(owner)
            return True
        for field_name, av in ast.iter_fields(a):
            bv = getattr(b, field_name)
            if isinstance(av, ast.AST):
                if not compare(av, bv, owner):
                    return False
            elif isinstance(av, list):
                if len(av) != len(bv):
                    return False
                for x, y in zip(av, bv):
                    if isinstance(x, ast.AST):
                        if not isinstance(y, ast.AST) or not compare(x, y, owner):
                            return False
                    elif x != y:
                        return False
            elif av != bv:
                return False
        return True

    return owners if compare(old, new) else set()


def literal_setting_value(node):
    """Literal construction cannot introduce initializer call side effects."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(literal_setting_value(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and literal_setting_value(key) and literal_setting_value(value)
                   for key, value in zip(node.keys, node.values))
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float, complex))


def value_change_scopes(old, new):
    """Find statements affected by literal/operator edits inside function bodies.

    Keep bindings, expression shape, definitions and control-flow statements
    intact. A test must execute an affected statement to observe these edits.
    More general edits still use the enclosing function's dependencies.
    """
    scopes = set()
    operators = (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)

    def compare(a, b, statement=None, in_function=False):
        if isinstance(a, operators) and isinstance(b, operators):
            if not any(isinstance(a, family) and isinstance(b, family) for family in operators):
                return False
            if type(a) is not type(b):
                if not in_function or statement is None:
                    return False
                scopes.add((statement.lineno, statement.end_lineno))
            return True
        if type(a) is not type(b):
            return False
        if isinstance(a, (*FUNCTIONS, ast.ClassDef)) and same_ast(a, b):
            return True
        if isinstance(a, ast.AST):
            if isinstance(a, FUNCTIONS) and any(
                isinstance(node, (*FUNCTIONS, ast.Lambda, ast.ClassDef, ast.Global, ast.Nonlocal))
                for child in a.body for node in ast.walk(child)
            ):
                # Captured values may be consumed after the outer call, and
                # explicit shared bindings can affect later calls elsewhere.
                return same_ast(a, b)
            if isinstance(a, ast.stmt):
                statement = a
            for name, av in ast.iter_fields(a):
                bv = getattr(b, name)
                if isinstance(a, FUNCTIONS):
                    if name != "body":
                        if not same_ast(av, bv):
                            return False
                        continue
                    if not compare(av, bv, statement, True):
                        return False
                elif isinstance(a, ast.ClassDef):
                    # Class bodies, even inside a function, execute when the
                    # definition is reached and may feed metaclass state.
                    if in_function or name != "body":
                        if not same_ast(av, bv):
                            return False
                    elif not compare(av, bv, statement, False):
                        return False
                elif isinstance(a, ast.Constant) and name == "value" and av != bv:
                    if (type(av) is not type(bv) or not in_function
                            or statement is None or isinstance(statement, FUNCTIONS)):
                        return False
                    # Docstrings are observable without executing their line.
                    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                        return False
                    scopes.add((statement.lineno, statement.end_lineno))
                elif not compare(av, bv, statement, in_function):
                    return False
            return True
        if isinstance(a, list):
            return len(a) == len(b) and all(
                compare(x, y, statement, in_function) for x, y in zip(a, b)
            )
        return a == b

    return scopes if compare(old, new) else None


class ImportCheck:
    """Use a fresh interpreter to check changed class initialization."""

    def __init__(self, sources):
        self.sources = sources
        self.directory = None
        self.remaining_seconds = 12.0

    def check(self, path, source, updates=None):
        if self.remaining_seconds <= 0:
            return False
        started = time.monotonic()
        if self.directory is None:
            self.directory = tempfile.TemporaryDirectory(prefix="ci-import-")
            for name, text in self.sources.items():
                if name.startswith("sqlglot/"):
                    target = Path(self.directory.name) / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text)
        updates = updates or {path: source}
        modules = [name.removesuffix(".py").replace("/", ".") for name in updates]
        try:
            for name, text in updates.items():
                target = Path(self.directory.name) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    "import sys,importlib;sys.path.insert(0,sys.argv[1]);"
                    "[importlib.import_module(name) for name in sys.argv[2:]]",
                    self.directory.name,
                    *modules,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                timeout=min(2.0, self.remaining_seconds),
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
        finally:
            self.remaining_seconds -= time.monotonic() - started
            for name in updates:
                (Path(self.directory.name) / name).write_text(self.sources[name])


class Selector:
    def __init__(self, mapping, tests, import_check=None):
        self.mapping = mapping
        self.import_check = import_check
        self.import_environment = (
            None if import_check is not None else ImportCheck(mapping.get("sources", {}))
        )
        self.tests = tests
        self.all = {t["number"] for t in tests}
        by_id = {t["id"]: t["number"] for t in tests}
        self.numbers = {
            i: by_id[name] for i, name in enumerate(mapping["tests"]) if name in by_id
        }
        self.unknown = (self.all - set(self.numbers.values())) | self.translated(
            mapping.get("incomplete", [])
        )
        self.ms = {t["number"]: math.floor(t["seconds"] * 1000 + 0.5) for t in tests}
        self.ranked = sorted(self.all, key=lambda n: (-self.ms[n], n))

    def setting_flow_consumers(self, old, new, path, fallback):
        """Use complete measured forwarding only for unchanged literal construction."""
        profile = self.mapping.get("setting_flow")
        if not profile:
            return None
        by_id = {test["id"]: test["number"] for test in self.tests}
        numbers = {i: by_id[name] for i, name in enumerate(profile["tests"]) if name in by_id}
        module = path.removesuffix(".py").replace("/", ".")
        changes = []

        def walk(a, b, owner=""):
            if isinstance(a, ast.ClassDef):
                owner = (owner + "." if owner else "") + a.name
            if isinstance(a, FUNCTIONS):
                return
            if owner and isinstance(a, (ast.Assign, ast.AnnAssign)) and not same_ast(a, b):
                targets = a.targets if isinstance(a, ast.Assign) else [a.target]
                if not literal_setting_value(a.value) or not literal_setting_value(b.value):
                    raise ValueError("nonliteral forwarding")
                for target in targets:
                    if not isinstance(target, ast.Name):
                        raise ValueError("unsupported forwarding target")
                    changes.append((module + ":" + owner + ":" + target.id, a.value, b.value))
                return
            for ac, bc in zip(ast.iter_child_nodes(a), ast.iter_child_nodes(b)):
                walk(ac, bc, owner)

        try:
            walk(old, new)
            if not changes:
                return None
            consumers = set()
            for key, before, after in changes:
                if (key not in profile["known"] or key in profile["blocked"]
                        or key.rsplit(":", 1)[0] not in self.mapping.get("setting_precise_classes", [])
                        or key.rsplit(":", 1)[-1].startswith("__")):
                    return None
                consumers.update(profile["consumers"].get(key, []))
                if isinstance(before, ast.Dict):
                    names = [node.value for node in before.keys if isinstance(node, ast.Constant) and type(node.value) is str]
                    updated_names = [node.value for node in after.keys if isinstance(node, ast.Constant) and type(node.value) is str]
                    if len(names) != len(before.keys) or len(set(names)) != len(names) or names != updated_names:
                        return None
                    entries = profile.get("entries", {}).get(key, {})
                    for name, av, bv in zip(names, before.values, after.values):
                        if not same_ast(av, bv):
                            consumers.update(entries.get(name, []))
                else:
                    consumers.update(i for ids in profile.get("entries", {}).get(key, {}).values() for i in ids)
            # Failed/unsupported children and unknown cores retain their contexts.
            consumers.update(set(profile.get("untrusted", [])) - set(profile.get("uninstrumented", [])))
            retained = {numbers[i] for i in consumers if i in numbers}
            retained.update(fallback & {numbers[i] for i in profile.get("uninstrumented", []) if i in numbers})
            retained.update(set(by_id.values()) - set(numbers.values()))
            return retained
        except (ValueError, KeyError, TypeError, AttributeError):
            return None

    def translated(self, values):
        return {self.numbers[i] for i in values if i in self.numbers}

    @lru_cache(maxsize=32)
    def analysis(self, path):
        tree = ast.parse(self.mapping["sources"][path])
        functions, assignments = [], []
        for node in ast.walk(tree):
            if isinstance(node, FUNCTIONS):
                functions.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                assignments.append(node)
        return tree, functions, assignments

    @lru_cache(maxsize=256)
    def scope_dependencies(self, path, start, end):
        startup = self.mapping.get("startup", {}).get(path, [])
        initialization = any(start <= line <= end for line in startup)
        entries = self.mapping["lines"].get(path, {})
        impacted = self.translated(
            {i for line in range(start, end + 1) for i in entries.get(str(line), [])}
        )
        return initialization, impacted

    def select(self, patch, *, use_cold=True, _prepared=None):
        if _ast_comparisons.get() is not None:
            return self._select(patch, use_cold=use_cold, _prepared=_prepared)
        token = _ast_comparisons.set({})
        try:
            return self._select(patch, use_cold=use_cold, _prepared=_prepared)
        finally:
            _ast_comparisons.reset(token)

    def _select(self, patch, *, use_cold=True, _prepared=None):
        selected = set(self.unknown)
        reasons = []
        try:
            edits = parse_patch(patch) if _prepared is None else _prepared[0]
            if len({edit.path for edit in edits}) != len(edits):
                return self.all, ["repeated file in patch"]
            trees = {}
            fallback_chosen = None
            if _prepared is None:
                updates, touches, parsed = {}, {}, {}
                for edit in edits:
                    source = self.mapping["sources"].get(edit.path)
                    if edit.path.endswith(".py") and source is not None:
                        updates[edit.path], touches[edit.path] = apply_edit(source, edit)
                _prepared = edits, updates, touches, parsed
            else:
                _, updates, touches, parsed = _prepared
            import_ok = None
            for edit in edits:
                path = edit.path
                readers = self.translated(self.mapping["reads"].get(path, []))
                if not path.endswith(".py"):
                    if not readers:
                        return self.all, ["unmapped data file: " + path]
                    selected.update(readers)
                    reasons.append("fixture: " + path)
                    continue
                source = self.mapping["sources"].get(path)
                if source is None:
                    return self.all, ["unknown source: " + path]
                updated, touched = updates[path], touches[path]
                if path not in parsed:
                    new_tree = ast.parse(updated)
                    old_tree, functions, assignments = self.analysis(path)
                    unchanged = same_ast(old_tree, new_tree)
                    owners = set() if unchanged else class_setting_changes(old_tree, new_tree)
                    parsed[path] = old_tree, new_tree, functions, assignments, unchanged, owners
                old_tree, new_tree, functions, assignments, unchanged, owners = parsed[path]
                trees[path] = old_tree, new_tree
                if re.search(r'coding\s*[:=]', ''.join(source.splitlines(True)[:2] + updated.splitlines(True)[:2])):
                    return self.all, ['encoding directive: ' + path]
                if unchanged:
                    # Preserve consumers of source text and code metadata. Syntax
                    # equality alone is not evidence that every test is unaffected.
                    selected.update(readers)
                    selected.update(
                        self.translated(self.mapping.get("introspection", []))
                    )
                    reasons.append(
                        "unchanged AST with source-reader retention: " + path
                    )
                    continue
                # Cold observations cover original fixtures and real workers.
                # Keep explicit source readers and unknown contexts protected.
                evidence = None
                field_profile = self.mapping.get('field_consumers') if use_cold else None
                if field_profile and owners:
                    from fieldcheck import consumers
                    mask = consumers(field_profile, path, source, updated, updates=updates)
                    if mask is not None:
                        evidence = field_profile, mask, 'dictionary keys and validator inputs'
                compiled_profile = self.mapping.get('compiled_consumers') if use_cold else None
                if evidence is None and compiled_profile:
                    from compiledcheck import consumers
                    mask = consumers(compiled_profile, path, source, updated, trees=(old_tree, new_tree),
                                     statements=self.mapping.get('statement_selection', True))
                    if mask is not None:
                        evidence = compiled_profile, mask, 'cold executable scopes'
                if evidence is not None:
                    trace, mask, reason = evidence
                    by_id = {test['id']: test['number'] for test in self.tests}
                    # Each complete dependency pass supplies an independent
                    # bound. Intersect them so conservative cold startup or
                    # reflection observations cannot broaden existing evidence.
                    if fallback_chosen is None:
                        fallback_chosen, _ = self.select(patch, use_cold=False, _prepared=_prepared)
                    incomplete = int(trace['incomplete'], 16)
                    if incomplete:
                        uncertain = {by_id[name] for i, name in enumerate(trace['tests'])
                                     if incomplete & (1 << i) and name in by_id}
                        selected.update(fallback_chosen & uncertain)
                        mask &= ~incomplete
                    selected.update(by_id[name] for i, name in enumerate(trace['tests'])
                                    if mask & (1 << i) and name in by_id and by_id[name] in fallback_chosen)
                    selected.update(readers & fallback_chosen)
                    selected.update(self.translated(self.mapping.get('introspection', [])) & fallback_chosen)
                    reasons.append(reason + ': ' + path)
                    continue
                if owners:
                    # Entry/size changes can affect iteration, membership, and
                    # derived initialization. Existing per-value evidence is
                    # only valid when the construction's shape is preserved.
                    setting_shape_changed = not same_ast(old_tree, new_tree, settings=True)
                    module = path.removesuffix(".py").replace("/", ".")
                    class_map = self.mapping.get("classes", {})
                    keys = [module + ":" + owner for owner in owners]
                    if all(key in class_map for key in keys):
                        if import_ok is None:
                            import_ok = (
                                self.import_environment.check(path, updated, updates)
                                if self.import_environment is not None
                                else self.import_check(path, updated)
                            )
                        if not import_ok:
                            return self.all, ["class import check failed: " + path]
                        selected.update(readers)
                        selected.update(
                            self.translated(self.mapping.get("introspection", []))
                        )
                        broad = {number for key in keys for number in self.translated(class_map[key])}
                        flow_consumers = (
                            None if setting_shape_changed else
                            self.setting_flow_consumers(old_tree, new_tree, path, broad)
                        )
                        if flow_consumers is not None:
                            selected.update(flow_consumers)
                            reasons.append("forwarded setting readers: " + path)
                            continue
                        setting_keys = set()
                        literal_settings = True
                        for node in assignments:
                            if any(
                                node.lineno <= line <= node.end_lineno
                                for line in touched
                            ):
                                literal_settings = literal_settings and literal_setting_value(node.value)
                                targets = (
                                    node.targets
                                    if isinstance(node, ast.Assign)
                                    else [node.target]
                                )
                                for target in targets:
                                    if isinstance(target, ast.Name):
                                        setting_keys.update(key + ":" + target.id for key in keys)
                                        qualified = self.mapping.get("qualified_attributes")
                                        if qualified is None:
                                            consumers = self.mapping.get("attributes", {}).get(target.id, [])
                                        else:
                                            consumers = {i for key in keys for i in qualified.get(key + ":" + target.id, [])}
                                        selected.update(self.translated(consumers))
                        precise = (
                            not setting_shape_changed
                            and self.mapping.get("attribute_complete", False)
                            and literal_settings
                            and all(key in self.mapping.get("setting_precise_classes", []) for key in keys)
                            and bool(setting_keys)
                            and all(not key.rsplit(":", 1)[-1].startswith("__") for key in setting_keys)
                            and all(key in self.mapping.get("qualified_attributes", {}) for key in setting_keys)
                            and not setting_keys.intersection(self.mapping.get("setting_collection", []))
                        )
                        if not precise:
                            for key in keys:
                                selected.update(self.translated(class_map[key]))
                        reasons.append(("observed setting reads: " if precise else "observed class consumers: ") + path)
                        continue
                if not same_ast(old_tree, new_tree, structure=True):
                    return self.all, ["module/class structure: " + path]
                value_scopes = value_change_scopes(old_tree, new_tree)
                chosen = set()
                for line in touched:
                    enclosing = [
                        n for n in functions if n.lineno < line <= n.end_lineno
                    ]
                    if not enclosing:
                        return self.all, ["import/definition boundary: " + path]
                    # The outer function includes captured state and nested functions.
                    scope = max(enclosing, key=lambda n: n.end_lineno - n.lineno)
                    chosen.add((scope.body[0].lineno, scope.end_lineno))
                impacted = set()
                narrowed = value_scopes is not None and bool(value_scopes)
                for start, end in chosen:
                    initialization, consumers = self.scope_dependencies(path, start, end)
                    # Module initialization can create shared state for later tests.
                    if initialization:
                        # Complete observations can establish that the changed
                        # expression leaves initialization itself unchanged.
                        # Preserve the broad fallback unless that proof exists.
                        behavior = self.mapping.get("local_behavior")
                        if behavior is not None and len(edits) == 1:
                            from behaviorcheck import certified_tests
                            proven = certified_tests(behavior, updates, self.mapping["sources"], self.tests,
                                                     trees=trees, require_initialization=True)
                            protected = set(self.unknown) | readers | self.translated(self.mapping.get("introspection", []))
                            removable = proven - protected
                            if removable:
                                return self.all - removable, ["complete initialization and test observations: " + path]
                        return self.all, ["initialization dependency: " + path]
                    impacted.update(consumers)
                if narrowed:
                    statement_consumers = set()
                    for start, end in value_scopes:
                        _, consumers = self.scope_dependencies(path, start, end)
                        # Missing statement evidence cannot establish safety.
                        if not consumers:
                            narrowed = False
                            break
                        statement_consumers.update(consumers)
                    if narrowed:
                        impacted = statement_consumers
                        selected.update(self.translated(self.mapping.get("introspection", [])))
                if not impacted:
                    impacted = self.translated(self.mapping["files"].get(path, []))
                if not impacted:
                    return self.all, ["missing coverage: " + path]
                selected.update(impacted | readers)
                reasons.append(("statement values: " if narrowed else "function: ") + path)
            if fallback_chosen is not None:
                # The independent selection already evaluated every comparison,
                # value and behavior proof for this entire patch. Reuse that
                # bound instead of repeating those expensive AST traversals.
                return selected & fallback_chosen, reasons
            if len(updates) == len(edits) and self.mapping.get("comparison_types"):
                from comparisoncheck import certified_tests

                proven = certified_tests(self.mapping["comparison_types"], updates,
                                         self.mapping["sources"], self.tests, trees=trees)
                retained = set(self.unknown) | self.translated(self.mapping.get("introspection", []))
                for edit in edits:
                    retained.update(self.translated(self.mapping["reads"].get(edit.path, [])))
                removable = selected & (proven - retained)
                if removable:
                    selected.difference_update(removable)
                    reasons.append(f"baseline operand types certify {len(removable)} comparison consumers")
            if len(updates) == len(edits) and self.mapping.get("literal_values"):
                from valuecheck import certified_tests

                proven = certified_tests(self.mapping["literal_values"], updates,
                                         self.mapping["sources"], self.tests, trees=trees)
                retained = set(self.unknown) | self.translated(self.mapping.get("introspection", []))
                for edit in edits:
                    retained.update(self.translated(self.mapping["reads"].get(edit.path, [])))
                removable = selected & (proven - retained)
                if removable:
                    selected.difference_update(removable)
                    reasons.append(f"baseline primitive values certify {len(removable)} comparison consumers")
            if len(updates) == len(edits) and self.mapping.get("local_behavior"):
                from behaviorcheck import certified_tests

                proven = certified_tests(self.mapping["local_behavior"], updates,
                                         self.mapping["sources"], self.tests, trees=trees)
                retained = set(self.unknown) | self.translated(self.mapping.get("introspection", []))
                for edit in edits:
                    retained.update(self.translated(self.mapping["reads"].get(edit.path, [])))
                removable = selected & (proven - retained)
                if removable:
                    selected.difference_update(removable)
                    reasons.append(f"baseline Boolean or dict identity observations certify {len(removable)} consumers")
        except (ValueError, SyntaxError, KeyError, TypeError, RecursionError) as exc:
            return self.all, ["fallback: " + str(exc)]
        return selected, reasons

    def schedule(self, selected, fill=True):
        if not selected:
            return []
        ranked = sorted(selected, key=lambda n: (-self.ms[n], n))
        total = sum(self.ms[n] for n in ranked)
        longest = self.ms[ranked[0]]
        best = None
        for count in range(1, min(8, len(ranked)) + 1):
            lower_bound = max(longest, (total + count - 1) // count) + 2000 * count
            if best is not None and lower_bound >= best[0]:
                continue
            workers = [[] for _ in range(count)]
            loads = [0] * count
            for number in ranked:
                index = min(range(count), key=lambda i: (loads[i], len(workers[i]), i))
                workers[index].append(number)
                loads[index] += self.ms[number]
            pairs = [(worker, load) for worker, load in zip(workers, loads) if worker]
            workers, loads = [p[0] for p in pairs], [p[1] for p in pairs]
            cost = max(loads) + 2000 * len(workers)
            if best is None or cost < best[0]:
                best = cost, workers, loads
        _, workers, loads = best
        if fill:
            ceiling = max(loads)
            for number in self.ranked:
                if number in selected:
                    continue
                index = min(range(len(loads)), key=lambda i: loads[i])
                if loads[index] + self.ms[number] <= ceiling:
                    workers[index].append(number)
                    loads[index] += self.ms[number]
        return workers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument(
        "--map", type=Path, default=Path(__file__).parent / "maps/baseline.json.gz"
    )
    parser.add_argument("--no-fill", action="store_true")
    parser.add_argument("--profile", type=Path, default=Path(__file__).parent / "maps/runtime.json.gz")
    parser.add_argument("--setting-flow", type=Path, default=Path(__file__).parent / "maps/settings.json.gz")
    parser.add_argument("--comparisons", type=Path, default=Path(__file__).parent / "maps/comparisons.json.gz")
    parser.add_argument("--source", type=Path, default=Path("/app/source"))
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--queries", type=Path, default=Path(__file__).parent / "maps/queries.pkl.gz")
    parser.add_argument("--query-checks", action="store_true")
    parser.add_argument("--values", type=Path, default=Path(__file__).parent / "maps/values.json.gz")
    parser.add_argument("--task-queries", type=Path, default=Path(__file__).parent / "maps/query-values.pkl.gz")
    parser.add_argument("--no-query-replay", action="store_true")
    parser.add_argument("--behavior", type=Path, default=Path(__file__).parent / "maps/behavior.json.gz")
    parser.add_argument("--timing", type=Path, default=Path(__file__).parent / "maps/timing.json.gz")
    parser.add_argument('--compiled', type=Path, default=Path(__file__).parent / 'maps/compiled.json.gz')
    parser.add_argument('--fields', type=Path, default=Path(__file__).parent / 'maps/fields.json.gz')
    parser.add_argument('--no-statement-selection', action='store_true')
    parser.add_argument('--fixture-cases', type=Path, default=Path(__file__).parent / 'maps/fixtures.json.gz')
    parser.add_argument('--no-fixture-replay', action='store_true')
    parser.add_argument('--no-query-statements', action='store_true')
    parser.add_argument('--no-fixture-delta', action='store_true')
    parser.add_argument('--no-mixed-planning', action='store_true')
    parser.add_argument('--no-progressive-verification', action='store_true')
    parser.add_argument('--pass-checkpoints', type=Path, default=Path(__file__).parent / 'maps/passes.pkl.xz')
    parser.add_argument('--no-pass-replay', action='store_true')
    parser.add_argument('--stage-profile', type=Path, default=Path(__file__).parent / 'maps/stages.pkl.xz')
    parser.add_argument('--import-profile', type=Path, default=Path(__file__).parent / 'maps/import-state.json.gz')
    parser.add_argument('--dispatch-profile', type=Path, default=Path(__file__).parent / 'maps/dispatch.json.gz')
    parser.add_argument('--no-stage-replay', action='store_true')
    parser.add_argument('--stage-replay', action='store_true')
    parser.add_argument('--no-import-verification', action='store_true')
    parser.add_argument('--import-verification', action='store_true')
    parser.add_argument('--no-warm-verification', action='store_true')
    parser.add_argument('--warm-verification', action='store_true')
    parser.add_argument('--aggressive', action='store_true')
    parser.add_argument('--aggressive-min-cpu', type=float, choices=(0.5, 1.0, 2.0, 4.0), default=4.0)
    parser.add_argument('--aggressive-broad', action='store_true')
    args = parser.parse_args()
    start = time.process_time()
    children_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    tests = json.loads((args.data / "tests.json").read_text())["tests"]
    from runtime import Verifier, load_profile, load_setting_flow, load_timing, merge_profile

    profile = load_profile(args.profile)
    with gzip.open(args.map, "rt") as f:
        mapping = json.load(f)
    merge_profile(mapping, profile, load_setting_flow(args.setting_flow))
    mapping['statement_selection'] = not args.no_statement_selection
    from compiledcheck import load_profile as load_compiled
    from fieldcheck import load_profile as load_fields
    compiled = load_compiled(args.compiled)
    if compiled and args.no_query_statements:
        compiled.pop('query_lines', None)
    fields = load_fields(args.fields)
    if compiled:
        mapping['compiled_consumers'] = compiled
    if fields:
        mapping['field_consumers'] = fields
    from comparisoncheck import load_comparison_profile
    comparison_profile = load_comparison_profile(args.comparisons)
    if comparison_profile:
        mapping["comparison_types"] = comparison_profile
    from valuecheck import LazyProfile
    values = LazyProfile(args.values) if args.values.is_file() else None
    if values:
        mapping["literal_values"] = values
    from behaviorcheck import LazyProfile as LazyBehavior
    behavior = LazyBehavior(args.behavior) if args.behavior.is_file() else None
    if behavior:
        mapping["local_behavior"] = behavior
    selector = Selector(mapping, tests)
    from dispatchcheck import load_profile as load_dispatch
    verifier = None if args.static else Verifier(args.source, profile, tests, selector.schedule,
                                               queries=args.queries if args.query_checks else None,
                                               values=values, behavior=behavior,
                                               timing=load_timing(args.timing),
                                               task_queries=None if args.no_query_replay else args.task_queries,
                                               fixtures=None if args.no_fixture_replay else args.fixture_cases,
                                               passes=None if args.no_pass_replay else args.pass_checkpoints,
                                               stages=args.stage_profile if args.stage_replay and not args.no_stage_replay else None,
                                               imported=args.import_profile if args.import_verification and not args.no_import_verification else None,
                                               dispatch=load_dispatch(args.dispatch_profile) if args.import_verification else None,
                                               warm=mapping.get('startup') if args.warm_verification and not args.no_warm_verification else None,
                                               fixture_delta=not args.no_fixture_delta,
                                               progressive=not args.no_progressive_verification,
                                               aggressive=args.aggressive,
                                               aggressive_min_cpu=args.aggressive_min_cpu,
                                               aggressive_broad=args.aggressive_broad,
                                               compiled=compiled)
    try:
        for path in sorted((args.data / "commits").glob("*.patch")):
            patch = path.read_text()
            chosen, reasons = selector.select(patch)
            if verifier is not None:
                protected = set(selector.unknown) | selector.translated(mapping.get("introspection", []))
                data_readers, code_patch, mixed = set(), None, False
                try:
                    edits = parse_patch(patch)
                    mixed = (not args.no_mixed_planning and any(e.path.endswith('.py') for e in edits)
                             and any(not e.path.endswith('.py') for e in edits))
                    code_parts = []
                    for edit in edits:
                        protected.update(selector.translated(mapping["reads"].get(edit.path, [])))
                        if not edit.path.endswith('.py'):
                            data_readers.update(selector.translated(mapping['reads'].get(edit.path, [])) or selector.all)
                        elif mixed:
                            import difflib
                            old = mapping['sources'][edit.path]
                            new, _ = apply_edit(old, edit)
                            code_parts.append(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                              fromfile='a/' + edit.path, tofile='b/' + edit.path)))
                    if mixed:
                        code_patch = ''.join(code_parts)
                    protected.update(data_readers)
                except (ValueError, KeyError):
                    protected.update(selector.all)
                    data_readers.update(selector.all)
                verifier.protected = protected
                chosen, checked = verifier.import_refine(patch, chosen, protected=protected)
                reasons.extend(checked)
                chosen, checked = verifier.stage_refine(patch, chosen, protected=protected)
                reasons.extend(checked)
                chosen, checked = verifier.aggressive_refine(patch, chosen, protected=protected)
                reasons.extend(checked)
                chosen, checked = verifier.task_query_refine(patch, chosen, protected=protected, planning_patch=code_patch)
                reasons.extend(checked)
                chosen, checked = verifier.query_refine(patch, chosen)
                reasons.extend(checked)
                chosen, checked = verifier.refine(patch, chosen, data_readers=data_readers, mixed=mixed)
                reasons.extend(checked)
            workers = selector.schedule(chosen, fill=not args.no_fill)
            print(
                json.dumps({"commit": path.stem, "workers": workers}, separators=(",", ":"))
            )
            print(
                json.dumps(
                    {
                        "commit": path.stem,
                        "selected": len(chosen),
                        "scheduled": sum(map(len, workers)),
                        "reasons": reasons,
                    }
                ),
                file=sys.stderr,
            )
    finally:
        if verifier is not None:
            verifier.close()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (
        time.process_time()
        - start
        + children.ru_utime
        + children.ru_stime
        - children_start.ru_utime
        - children_start.ru_stime
    )
    print(
        f"CPU seconds including import checks (excluding interpreter startup): {cpu:.4f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
