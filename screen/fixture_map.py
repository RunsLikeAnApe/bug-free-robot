"""Index fixture reads routed through class attributes initialized from SQL pairs."""
import ast
import gzip
import json
from pathlib import Path

HERE = Path(__file__).parent


def class_bindings(cls):
    setup = next((fn for fn in cls.body if isinstance(fn, ast.FunctionDef) and fn.name == "setUpClass"), None)
    if setup is None:
        return
    for stmt in setup.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        lhs, rhs = stmt.targets[0], stmt.value
        if not (isinstance(lhs, ast.Attribute) and isinstance(lhs.value, ast.Name) and lhs.value.id == "cls"
                and isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id == "list"
                and len(rhs.args) == 1 and isinstance(rhs.args[0], ast.Call)):
            continue
        load = rhs.args[0]
        if (isinstance(load.func, ast.Name) and load.func.id == "load_sql_fixture_pairs" and len(load.args) == 1
                and isinstance(load.args[0], ast.Constant) and isinstance(load.args[0].value, str)):
            yield lhs.attr, load.args[0].value


def build():
    with gzip.open(HERE / "coverage.json.gz", "rt") as f:
        data = json.load(f)
    sites = {}
    for path in Path("tests").rglob("test_*.py"):
        file = path.as_posix()
        tree = ast.parse(path.read_bytes())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            for attr, fixture in class_bindings(cls):
                # Allow one direct initialization and no indirect string-based access.
                if path.read_text().count('"' + fixture + '"') != 1:
                    continue
                if any(isinstance(n, ast.Constant) and n.value == attr for n in ast.walk(tree)):
                    continue
                group = sum(1 << i for i, name in enumerate(data["names"])
                            if name.startswith(f"{file}::{cls.name}::"))
                path_key = "tests/fixtures/" + fixture
                raw = int(data["files"].get(path_key, "0"), 16)
                if not group or raw & group != group or path_key in sites:
                    continue
                mask = raw & ~group
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and node.attr == attr and isinstance(node.ctx, ast.Load):
                        mask |= int(data["lines"].get(file, {}).get(str(node.lineno), "0"), 16)
                sites[path_key] = hex(mask)
    with gzip.open(HERE / "fixture-attrs.gz", "wt") as f:
        json.dump({"names": data["names"], "sites": sites}, f, separators=(",", ":"))


if __name__ == "__main__":
    build()
