"""Index tuple fields consumed by readers of SQL fixture pairs in class attributes."""
import ast
import gzip
import json
import sys
from pathlib import Path

from fixture_map import class_bindings

ALL = {0, 1, 2}


def fields(node, method, methods, parents, active):
    parent = parents.get(node)
    if isinstance(parent, ast.IfExp) and node in (parent.body, parent.orelse):
        return fields(parent, method, methods, parents, active)
    if isinstance(parent, ast.Subscript) and parent.value is node:
        target = parents.get(parent)
        while isinstance(target, ast.IfExp) and parent in (target.body, target.orelse):
            parent, target = target, parents.get(target)
        if isinstance(target, ast.Assign) and len(target.targets) == 1:
            return unpack(target.targets[0], method)
    if isinstance(parent, (ast.comprehension, ast.For)) and parent.iter is node:
        return unpack(parent.target, method)
    if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute)
            and isinstance(parent.func.value, ast.Name) and parent.func.value.id == "self"
            and node in parent.args):
        callee = methods.get(parent.func.attr)
        if callee is None or callee in active or callee.decorator_list:
            return ALL
        offset = 1 + parent.args.index(node)
        if offset >= len(callee.args.args):
            return ALL
        param = callee.args.args[offset].arg
        nodes = list(ast.walk(callee))
        if any(isinstance(n, ast.Name) and n.id in {"locals", "eval", "vars", "exec"} for n in nodes):
            return ALL
        uses = [n for n in nodes if isinstance(n, ast.Name) and n.id == param]
        if any(not isinstance(n.ctx, ast.Load) for n in uses):
            return ALL
        return set().union(*(fields(n, callee, methods, parents, active | {callee}) for n in uses))
    return ALL


def unpack(target, method):
    if not isinstance(target, ast.Tuple) or len(target.elts) != 3:
        return ALL
    if any(not isinstance(n, ast.Name) for n in target.elts):
        return ALL
    reads = {n.id for n in ast.walk(method) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if reads & {"locals", "exec", "vars", "eval"}:
        return ALL
    return {i for i, n in enumerate(target.elts) if n.id in reads}


def build(src, coverage):
    with gzip.open(coverage, "rt") as f:
        data = json.load(f)
    result = {}
    for path in (src / "tests").rglob("test_*.py"):
        tree = ast.parse(path.read_bytes())
        file = path.relative_to(src).as_posix()
        rows = data["lines"].get(file, {})
        parents = {n: p for p in ast.walk(tree) for n in ast.iter_child_nodes(p)}
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            for attr, fixture in class_bindings(cls):
                group = sum(1 << i for i, name in enumerate(data["names"]) if name.startswith(f"{file}::{cls.name}::"))
                key = "tests/fixtures/" + fixture
                # Keep every reader outside the class, plus any reader whose
                # access cannot be traced to a direct and accounted-for method.
                if key in result:
                    raise ValueError("multiple fixture bindings")
                masks = [int(data["files"].get(key, "0"), 16) & ~group] * 3
                for use in ast.walk(tree):
                    if isinstance(use, ast.Constant) and use.value == attr:
                        break
                    if not (isinstance(use, ast.Attribute) and use.attr == attr and isinstance(use.ctx, ast.Load)):
                        continue
                    method = parents.get(use)
                    while method is not None and not isinstance(method, ast.FunctionDef):
                        method = parents.get(method)
                    if method not in methods.values():
                        break
                    for i in fields(use, method, methods, parents, {method}):
                        masks[i] |= int(rows.get(str(use.lineno), "0"), 16)
                else:
                    result[key] = [hex(m) for m in masks]
    return result


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2]))))
