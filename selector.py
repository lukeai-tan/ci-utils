"""Select tests from baseline dependencies; no patches or labels are precomputed."""

from __future__ import annotations
import argparse
import ast
import gzip
import json
import re
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


class Structure(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.body = [ast.Pass()]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


class Selector:
    def __init__(self, mapping, tests):
        self.mapping = mapping
        self.tests = tests
        self.all = {t["number"] for t in tests}
        by_id = {t["id"]: t["number"] for t in tests}
        self.numbers = {
            i: by_id[name] for i, name in enumerate(mapping["tests"]) if name in by_id
        }
        self.unknown = (self.all - set(self.numbers.values())) | self.translated(
            mapping.get("incomplete", [])
        )
        self.ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}

    def translated(self, values):
        return {self.numbers[i] for i in values if i in self.numbers}

    @lru_cache(maxsize=256)
    def scopes(self, path):
        tree = ast.parse(self.mapping["sources"][path])
        return [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    def select(self, patch):
        selected = set(self.unknown)
        reasons = []
        try:
            for edit in parse_patch(patch):
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
                updated, touched = apply_edit(source, edit)
                new_tree = ast.parse(updated)
                old_structure = ast.dump(Structure().visit(ast.parse(source)))
                new_structure = ast.dump(Structure().visit(new_tree))
                if old_structure != new_structure:
                    return self.all, ["module/class structure: " + path]
                functions = self.scopes(path)
                chosen = set()
                for line in touched:
                    enclosing = [
                        n for n in functions if n.lineno < line <= n.end_lineno
                    ]
                    if not enclosing:
                        return self.all, ["import/definition boundary: " + path]
                    # The outer function includes captured state and nested functions.
                    scope = max(enclosing, key=lambda n: n.end_lineno - n.lineno)
                    chosen.update(range(scope.body[0].lineno, scope.end_lineno + 1))
                # Module initialization can create shared state for later tests.
                if chosen.intersection(self.mapping.get("startup", {}).get(path, [])):
                    return self.all, ["initialization dependency: " + path]
                entries = self.mapping["lines"].get(path, {})
                impacted = self.translated(
                    {i for line in chosen for i in entries.get(str(line), [])}
                )
                if not impacted:
                    impacted = self.translated(self.mapping["files"].get(path, []))
                if not impacted:
                    return self.all, ["missing coverage: " + path]
                selected.update(impacted | readers)
                reasons.append("function: " + path)
        except (ValueError, SyntaxError, KeyError, TypeError, RecursionError) as exc:
            return self.all, ["fallback: " + str(exc)]
        return selected, reasons

    def schedule(self, selected, fill=True):
        if not selected:
            return []
        ranked = sorted(selected, key=lambda n: (-self.ms[n], n))
        best = None
        for count in range(1, min(8, len(ranked)) + 1):
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
            for number in sorted(self.all - selected, key=lambda n: (-self.ms[n], n)):
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
    args = parser.parse_args()
    start = time.process_time()
    tests = json.loads((args.data / "tests.json").read_text())["tests"]
    with gzip.open(args.map, "rt") as f:
        selector = Selector(json.load(f), tests)
    for path in sorted((args.data / "commits").glob("*.patch")):
        chosen, reasons = selector.select(path.read_text())
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
    print(f"CPU seconds: {time.process_time() - start:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
