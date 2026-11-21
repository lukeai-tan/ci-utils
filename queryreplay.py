"""Replay affected baseline executor tasks with their actual serialized input state."""

from __future__ import annotations

import ast
import datetime
import decimal
import gzip
import hashlib
import json
import os
import pickle
import sys
import zlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from comparisoncheck import REVISION, source_hash
from valuecheck import unchanged, validated_change


TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"


def result_digest(table):
    if table is None:
        return "none"

    def scalar(value):
        if value is None or type(value) in (str, int, bool):
            return [type(value).__name__, value]
        if type(value) is float:
            return ["float", value.hex()]
        if type(value) is decimal.Decimal:
            return ["decimal", str(value)]
        if type(value) in (datetime.date, datetime.time, datetime.datetime):
            return [type(value).__name__, value.isoformat()]
        raise ValueError("unsupported executor result")

    if type(table.columns) is not tuple or any(type(c) is not str for c in table.columns):
        raise ValueError("unsupported executor columns")
    payload = [list(table.columns), [[scalar(value) for value in row] for row in table.rows]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def state_digest(state):
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()


def compact_states(data):
    """Share identical pickle frames without changing a byte of worker state."""
    chunks, states = {}, {}
    for key, payload in data["states"].items():
        offsets = []
        cursor = 0
        if payload[:2] == b"\x80\x05":
            offsets.append((0, 2))
            cursor = 2
        while cursor < len(payload):
            if payload[cursor:cursor + 1] == b"\x95" and cursor + 9 <= len(payload):
                end = cursor + 9 + int.from_bytes(payload[cursor + 1:cursor + 9], "little")
                if end > len(payload):
                    end = len(payload)
            else:
                end = min(len(payload), cursor + 65536)
            offsets.append((cursor, end))
            cursor = end
        parts = []
        for start, end in offsets:
            raw = payload[start:end]
            digest = hashlib.sha256(raw).hexdigest()
            if digest not in chunks:
                chunks[digest] = zlib.compress(raw)
            parts.append(digest)
        states[key] = parts
    return dict(data, version=2, states=states, chunks=chunks)


def state_bytes(data, key, *, decode=None):
    if data["version"] == 1:
        return data["states"][key]
    result = b"".join(decode(part) if decode is not None else zlib.decompress(data["chunks"][part])
                      for part in data["states"][key])
    if hashlib.sha256(result).hexdigest() != key:
        raise ValueError("worker state checksum mismatch")
    return result


def load_profile(path):
    try:
        with gzip.open(path, "rb") as stream:
            data = pickle.load(stream)
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError("invalid executor task baseline") from error
    if (data["revision"] != REVISION or data["version"] not in (1, 2) or data["kind"] != "executor_tasks"
            or data["test"] != TEST or not data["complete"] or data["hashseed"] != "0"
            or not data["queries"] or not data["states"]
            or set(data["queries"]) != set(range(len(data["queries"])))):
        raise ValueError("incomplete executor task baseline")
    for index, query in data["queries"].items():
        if (type(index) is not int or query["state"] not in data["states"]
                or type(query["task"]) is not bytes or not query["result"] or not query["after"]
                or query["cpu"] < 0):
            raise ValueError("invalid executor task")
    for rows in data["rows"].values():
        if not set(rows) <= set(data["queries"]):
            raise ValueError("unknown executor query observation")
    if data["version"] == 2 and any(part not in data["chunks"] for parts in data["states"].values() for part in parts):
        raise ValueError("missing worker state chunk")
    return data


def plan(base, patch, values, tasks, behavior=None):
    """Keep unsupported changes and every comparison used outside task execution."""
    from selector import apply_edit, parse_patch
    from querycheck import supported_patch

    edits = parse_patch(patch)
    if len(edits) != 1 or not edits[0].path.startswith("sqlglot/optimizer/"):
        return None
    edit = edits[0]
    old = (Path(base) / edit.path).read_text()
    updated, _ = apply_edit(old, edit)
    change = validated_change(values, {edit.path: updated}, {edit.path: old})
    behavioral = False
    if not change and behavior is not None:
        from behaviorcheck import LazyProfile, validated_change as behavior_change
        change = behavior_change(behavior, {edit.path: updated}, {edit.path: old})
        behavioral = change is not None
        behavior = behavior.load() if isinstance(behavior, LazyProfile) else behavior
    if not change or tasks["sources"].get(edit.path) != source_hash(old):
        return None
    if not supported_patch(base, patch):
        return None
    key, before, after = change
    # Imports, setup, parsing and reference SQL construction must be unchanged.
    if behavioral:
        from behaviorcheck import unchanged as behavior_unchanged
        if (not behavior["query_complete"] or set(behavior["queries"]) != set(tasks["queries"])
                or key in behavior["outside"]):
            return None
        # Function reachability must cover every observed changed expression.
        tree = ast.parse(old)
        line = int(key.rsplit(":", 4)[1])
        owners = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.lineno < line <= node.end_lineno]
        if not owners:
            return None
        owner = min(owners, key=lambda node: node.end_lineno - node.lineno)
        function = f"{edit.path}:{owner.lineno}"
        if owner.decorator_list or function in behavior["outside_functions"]:
            return None
        rows = behavior["query_rows"].get(key, {})
        reachable = set(behavior["functions"].get(function, []))
        if not {int(index) for index in rows} <= reachable:
            return None
        indices = sorted(index for index in reachable
                         if not behavior_unchanged(rows.get(str(index)), before, after))
    else:
        if key in tasks["outside"]:
            return None
        rows = tasks["rows"].get(key)
        if not rows or any(row.get("unknown") for row in rows.values()):
            return None
        indices = sorted(index for index in tasks["queries"]
                         if index in rows and not unchanged(rows[index], before, after))
    if len(indices) > 8 and not tasks.get("uninstrumented_cpu"):
        return None
    return indices, sum(tasks["queries"][i]["cpu"] for i in indices)


def replay(root, data, indices):
    """Execute the actual patched mp_execute function, including its whole backend.

    Each task receives the precise schema/table state that its real baseline
    worker had before invocation. Results and the complete shared input state
    must both match. Unknown state, failures and incomplete work retain the test.
    """
    root = Path(root)
    if hashlib.sha256((root / "tests/test_executor.py").read_bytes()).hexdigest() != data["driver"]:
        raise ValueError("executor driver changed")
    os.chdir(root)
    sys.path.insert(0, str(root))
    from tests import test_executor as driver

    chunks = OrderedDict()
    cache_bytes = 0
    def decode(key):
        nonlocal cache_bytes
        if key in chunks:
            chunks.move_to_end(key)
            return chunks[key]
        raw = zlib.decompress(data["chunks"][key])
        if len(raw) <= 32 * 1024**2:
            while chunks and cache_bytes + len(raw) > 32 * 1024**2:
                _, expired = chunks.popitem(last=False)
                cache_bytes -= len(expired)
            chunks[key] = raw
            cache_bytes += len(raw)
        return raw

    @lru_cache(maxsize=2)
    def snapshot(key):
        return state_bytes(data, key, decode=decode)

    for index in indices:
        query = data["queries"][index]
        schema, tables = pickle.loads(snapshot(query["state"]))
        driver.initializer(schema, tables)
        function, args, kwargs = pickle.loads(query["task"])
        if function is not driver.mp_execute or kwargs or len(args) != 2:
            raise ValueError("unrecognized executor task")
        result = pickle.loads(pickle.dumps(function(*args, **kwargs), protocol=5))
        if (result_digest(result) != query["result"]
                or state_digest((driver._schema, driver._tables)) != query["after"]):
            raise ValueError("executor task output or shared state changed")
    return {"passed": [TEST], "replayed_queries": len(indices)}
