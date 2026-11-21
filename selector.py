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


def same_ast(old, new, *, structure=False, settings=False):
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
    """Find class settings whose expression only changes literal values."""
    owners = set()

    def compare(a, b, owner=None):
        if type(a) is not type(b):
            return False
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
            if not same_ast(a.value, b.value, settings=True):
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

    def select(self, patch):
        selected = set(self.unknown)
        reasons = []
        try:
            edits = parse_patch(patch)
            if len({edit.path for edit in edits}) != len(edits):
                return self.all, ["repeated file in patch"]
            updates, touches = {}, {}
            for edit in edits:
                source = self.mapping["sources"].get(edit.path)
                if edit.path.endswith(".py") and source is not None:
                    updates[edit.path], touches[edit.path] = apply_edit(source, edit)
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
                new_tree = ast.parse(updated)
                old_tree, functions, assignments = self.analysis(path)
                if same_ast(old_tree, new_tree):
                    # Preserve consumers of source text and code metadata. Syntax
                    # equality alone is not evidence that every test is unaffected.
                    if re.search(
                        r"coding\s*[:=]",
                        "".join(
                            source.splitlines(True)[:2] + updated.splitlines(True)[:2]
                        ),
                    ):
                        return self.all, ["encoding directive: " + path]
                    selected.update(readers)
                    selected.update(
                        self.translated(self.mapping.get("introspection", []))
                    )
                    reasons.append(
                        "unchanged AST with source-reader retention: " + path
                    )
                    continue
                owners = class_setting_changes(old_tree, new_tree)
                if owners:
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
                            self.mapping.get("attribute_complete", False)
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
    parser.add_argument("--source", type=Path, default=Path("/app/source"))
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--queries", type=Path, default=Path(__file__).parent / "maps/queries.pkl.gz")
    parser.add_argument("--query-checks", action="store_true")
    args = parser.parse_args()
    start = time.process_time()
    children_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    tests = json.loads((args.data / "tests.json").read_text())["tests"]
    from runtime import Verifier, load_profile, merge_profile

    profile = load_profile(args.profile)
    with gzip.open(args.map, "rt") as f:
        mapping = json.load(f)
    merge_profile(mapping, profile)
    selector = Selector(mapping, tests)
    verifier = None if args.static else Verifier(args.source, profile, tests, selector.schedule,
                                               queries=args.queries if args.query_checks else None)
    try:
        for path in sorted((args.data / "commits").glob("*.patch")):
            patch = path.read_text()
            chosen, reasons = selector.select(patch)
            if verifier is not None:
                chosen, checked = verifier.query_refine(patch, chosen)
                reasons.extend(checked)
                chosen, checked = verifier.refine(patch, chosen)
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
