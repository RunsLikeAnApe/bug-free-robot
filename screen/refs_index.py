"""Index static reads of module and class bindings in the pinned source.

Runs during the image build on the unmodified tree. For every name bound by a
module-level or class-level assignment in sqlglot, it lists each load of the
name as [file, line, kind, context]: kind is "n" for a bare name and "a" for an
attribute; context is null for a load that runs only when a function or lambda
is called, or else the module or class binding that receives the evaluated
value during import ("" when the load reaches anything else, such as
decorators, defaults, bases or bare statements), prefixed "C:" inside a class
body and "M:" at module level. Function loads also list the
first line of the enclosing statement.

"imports" lists "from X import name [as alias]" bindings per file, so a module
global can be followed into modules that import it. "strings" holds names that
also occur inside string constants of the package, which getattr, globals() or
doctests can reach without a static load; "test_strings" maps names that occur
in strings of test modules to those modules. "special" lists classes consumed as a whole
(enums, dataclasses, named tuples, typed dicts, protocols, slotted classes).

usage: refs_index.py SOURCE_ROOT OUTPUT
"""
import ast
import dataclasses
import enum
import gzip
import importlib
import json
import re
import sys
from pathlib import Path

DEFERRED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
DYNAMIC_CALLS = {"getattr", "hasattr", "setattr", "delattr", "attrgetter", "import_module", "__import__", "eval", "exec",
                 "patch", "object", "dict"}
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def bindings(tree):
    names = set()

    def body(stmts):
        for node in stmts:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
                body(node.body)
            elif isinstance(node, (ast.If, ast.Try)):
                body(node.body)
                body(getattr(node, "orelse", []))

    body(tree.body)
    return names


class Loads(ast.NodeVisitor):
    """Collect loads with the context in which they are evaluated."""

    def __init__(self, file, wanted, out, future=False):
        self.file = file
        # Under "from __future__ import annotations" annotations stay strings.
        self.future = future
        self.wanted = wanted
        self.out = out
        self.deferred = 0
        self.classes = 0
        self.target = []
        self.statement = []

    def record(self, name, node, kind):
        if name not in self.wanted:
            return
        loads = self.out.setdefault(name, [])
        if self.deferred:
            loads.append([self.file, node.lineno, kind, None])
            if self.statement and self.statement[-1] != node.lineno:
                loads.append([self.file, self.statement[-1], kind, None])
        else:
            target = self.target[-1] if self.target else ""
            if target:
                target = ("C:" if self.classes else "M:") + target
            loads.append([self.file, node.lineno, kind, target])

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.record(node.id, node, "n")

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            self.record(node.attr, node, "a")
        self.generic_visit(node)

    def generic_visit(self, node):
        if isinstance(node, ast.stmt) and self.deferred:
            self.statement.append(node.lineno)
            super().generic_visit(node)
            self.statement.pop()
        else:
            super().generic_visit(node)

    def visit_function(self, node):
        # Decorators, defaults and (possibly) annotations run with the
        # enclosing scope; the body runs later.
        for item in getattr(node, "decorator_list", []):
            self.visit(item)
        self.visit(node.args)
        if getattr(node, "returns", None) is not None and not self.future:
            self.visit(node.returns)
        self.deferred += 1
        self.statement.append(node.lineno)
        body = node.body if isinstance(node.body, list) else [node.body]
        for item in body:
            self.visit(item)
        self.statement.pop()
        self.deferred -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = visit_function

    def visit_arguments(self, node):
        for item in [*node.defaults, *[d for d in node.kw_defaults if d is not None]]:
            self.visit(item)
        if not self.deferred and not self.future:
            for arg in [*node.posonlyargs, *node.args, *node.kwonlyargs, node.vararg, node.kwarg]:
                if arg is not None and arg.annotation is not None:
                    self.visit(arg.annotation)

    def visit_assignment(self, node, targets, value):
        named = [t.id for t in targets if isinstance(t, ast.Name)]
        simple = not self.deferred and len(named) == len(targets) == 1
        for target in targets:
            self.visit(target)
        if getattr(node, "annotation", None) is not None and not self.deferred and not self.future:
            self.target.append("")
            self.visit(node.annotation)
            self.target.pop()
        if value is not None:
            self.target.append(named[0] if simple else ("" if not self.deferred else ""))
            self.visit(value)
            self.target.pop()

    def visit_Assign(self, node):
        if self.deferred:
            return self.generic_visit(node)
        self.visit_assignment(node, node.targets, node.value)

    def visit_AnnAssign(self, node):
        if self.deferred:
            # Annotations of local variables are never evaluated.
            self.statement.append(node.lineno)
            self.visit(node.target)
            if node.value is not None:
                self.visit(node.value)
            self.statement.pop()
            return
        self.visit_assignment(node, [node.target], node.value)

    def visit_If(self, node):
        test = node.test
        if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"):
            for item in node.orelse:
                self.visit(item)
            return
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if self.deferred:
            return self.generic_visit(node)
        self.target.append("")
        for item in node.decorator_list:
            self.visit(item)
        self.target.pop()
        # Bases and class keywords only shape the class being created.
        self.target.append(node.name)
        for item in [*node.bases, *node.keywords]:
            self.visit(item)
        self.target.pop()
        self.classes += 1
        for item in node.body:
            self.visit(item)
        self.classes -= 1


def cached_lines(file, tree):
    """Lines of functions whose results persist across tests in module-level containers.

    A function that stores into a subscript of a module-level name (a memo
    table), and every same-module function it calls by name, runs only for the
    first test that needs the value, so coverage of their loads does not list
    every test that observes them.
    """
    module_names = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(target, ast.Name):
                    module_names.add(target.id)
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, []).append(node)
    marked = []
    for group in functions.values():
        for fn in group:
            local = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    local.add(node.id)
            for node in ast.walk(fn):
                if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                        and isinstance(node.value, ast.Name) and node.value.id in module_names
                        and node.value.id not in local):
                    marked.append(fn)
                    break
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"setdefault", "update"} and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in module_names and node.func.value.id not in local):
                    marked.append(fn)
                    break
    lines = set()
    seen = set()
    while marked:
        fn = marked.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        lines.update(range(fn.lineno, fn.end_lineno + 1))
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                marked.extend(functions.get(node.func.id, []))
    return sorted(lines)


def defaulted_functions(trees):
    """Names of package functions and methods that declare parameter defaults."""
    names = {}
    for file, tree in trees.items():
        if not file.startswith("sqlglot/"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    node.args.defaults or any(d is not None for d in node.args.kw_defaults)):
                names.setdefault(node.name, []).append(file)
    return names


def receiver_root(node):
    """Root name of an attribute call's receiver chain, or None for a plain name call."""
    if not isinstance(node, ast.Attribute):
        return None
    value = node.value
    while isinstance(value, ast.Attribute):
        value = value.value
    return value.id if isinstance(value, ast.Name) else ""


def call_sites(trees, names):
    """Every call of a defaulted name, and every name used other than by calling it.

    A site is [file, line, positional count, has *args, keyword names, has **kwargs,
    runs only when an enclosing function or lambda is called, receiver root name
    (None for a plain call, "" for a computed receiver)].
    """
    calls = {}
    escapes = set()
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

    def local_names(scope):
        found = {a.arg for a in ast.walk(scope.args) if isinstance(a, ast.arg)}
        pending = list(scope.body) if isinstance(scope.body, list) else [scope.body]
        while pending:
            item = pending.pop()
            if isinstance(item, scopes) or isinstance(item, ast.ClassDef):
                if not isinstance(item, ast.Lambda):
                    found.add(item.name)
                continue
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                found.add(item.id)
            elif isinstance(item, (ast.Import, ast.ImportFrom)):
                found.update((a.asname or a.name).split(".")[0] for a in item.names)
            elif isinstance(item, (ast.Global, ast.Nonlocal)):
                found.difference_update(item.names)
            pending.extend(ast.iter_child_nodes(item))
        return found

    for file, tree in trees.items():
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        locals_cache = {}

        def bound_locally(node, name):
            scope = parents.get(node)
            while scope is not None and not isinstance(scope, scopes):
                scope = parents.get(scope)
            if scope is None:
                return False
            if id(scope) not in locals_cache:
                locals_cache[id(scope)] = local_names(scope)
            return name in locals_cache[id(scope)]

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                name = node.attr
            else:
                continue
            if name not in names:
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                scope = parents.get(parent)
                while scope is not None and not isinstance(scope, scopes):
                    scope = parents.get(scope)
                deferred = scope is not None and (
                    isinstance(scope, ast.Lambda) or not any(
                        parent is d or any(parent is x for x in ast.walk(d))
                        for d in [*scope.decorator_list, *scope.args.defaults,
                                  *[k for k in scope.args.kw_defaults if k is not None]]))
                calls.setdefault(name, []).append([
                    file, parent.lineno,
                    sum(1 for a in parent.args if not isinstance(a, ast.Starred)),
                    any(isinstance(a, ast.Starred) for a in parent.args),
                    [k.arg for k in parent.keywords if k.arg is not None],
                    any(k.arg is None for k in parent.keywords),
                    deferred, receiver_root(node)])
            elif not (isinstance(node, ast.Name) and bound_locally(node, name)):
                escapes.add(name)
    return calls, escapes


def special_classes(root):
    """Name classes whose members are consumed without being named."""
    sys.path.insert(0, str(root))
    found = []
    for path in sorted((root / "sqlglot").rglob("*.py")):
        module_name = ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
        if module_name.endswith("__main__"):
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for value in list(vars(module).values()):
            if not isinstance(value, type) or value.__module__ != module_name:
                continue
            stack = [value]
            while stack:
                cls = stack.pop()
                meta = type(cls).__name__
                if (issubclass(cls, (enum.Enum, tuple)) or dataclasses.is_dataclass(cls)
                        or meta in {"_TypedDictMeta", "_ProtocolMeta"}):
                    found.append(f"{cls.__module__}.{cls.__qualname__}")
                stack.extend(v for v in vars(cls).values() if isinstance(v, type) and v.__module__ == module_name)
    return sorted(set(found))


def main():
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    sources = sorted([*root.joinpath("sqlglot").rglob("*.py"), *root.joinpath("tests").rglob("*.py")])
    trees = {p.relative_to(root).as_posix(): ast.parse(p.read_bytes()) for p in sources}
    wanted = set()
    for file, tree in trees.items():
        if file.startswith("sqlglot/"):
            wanted |= bindings(tree)
    callables = defaulted_functions(trees)
    calls, escapes = call_sites(trees, set(callables))
    wanted_all = wanted | set(callables)
    # A package string can reach a binding only through a dynamic lookup:
    # getattr-style calls, namespace dictionaries, __all__, imports by name or a
    # doctest example. Any string in a test module counts for that module.
    words = set()
    dynamic_words = set()
    doc_words = set()
    patterns = set()
    test_words = {}
    for file, tree in trees.items():
        found = set()
        testing = file.startswith("tests/")
        for node in ast.walk(tree):
            strings = []
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in DYNAMIC_CALLS:
                    strings = node.args[1:2] if name in {"getattr", "hasattr", "setattr", "delattr", "object"} else node.args[:1]
                    for arg in strings:
                        if isinstance(arg, ast.JoinedStr):
                            parts = arg.values
                            head = parts[0].value if parts and isinstance(parts[0], ast.Constant) else ""
                            tail = parts[-1].value if parts and isinstance(parts[-1], ast.Constant) else ""
                            if head or tail:
                                # A pattern without fixed text is as dynamic as a plain
                                # variable name, which the selector cannot bound either.
                                patterns.add((head, tail))
                        elif not isinstance(arg, ast.Constant):
                            pass
                if isinstance(func, ast.Attribute) and func.attr in {"get", "pop", "setdefault", "__getitem__"}:
                    owner = func.value
                    if ((isinstance(owner, ast.Attribute) and owner.attr == "__dict__")
                            or (isinstance(owner, ast.Call) and getattr(owner.func, "id", "") in {"globals", "vars", "locals"})):
                        strings = node.args[:1]
            elif isinstance(node, ast.Subscript):
                owner = node.value
                if ((isinstance(owner, ast.Attribute) and owner.attr == "__dict__")
                        or (isinstance(owner, ast.Call) and getattr(owner.func, "id", "") in {"globals", "vars", "locals"})):
                    strings = [node.slice]
            elif isinstance(node, ast.Compare) and any(
                    (isinstance(c, ast.Attribute) and c.attr == "__dict__")
                    or (isinstance(c, ast.Call) and getattr(c.func, "id", "") in {"globals", "vars", "locals"})
                    for c in node.comparators):
                strings = [node.left]
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                # Star imports bind listed names under the same name.
                for leaf in ast.walk(node.value):
                    if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
                        found.update(WORD.findall(leaf.value))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for line in node.value.value.splitlines():
                    if line.lstrip().startswith((">>>", "...")):
                        found.update(WORD.findall(line))
                        if not testing:
                            doc_words.update(WORD.findall(line))
            for item in strings:
                for leaf in ast.walk(item):
                    if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
                        found.update(WORD.findall(leaf.value))
                        if not testing:
                            dynamic_words.update(WORD.findall(leaf.value))
        if testing:
            for word in found & wanted_all:
                test_words.setdefault(word, []).append(file)
        else:
            words |= found

    out = {}
    imports = {}
    cached = {}
    for file, tree in trees.items():
        future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                     and any(a.name == "annotations" for a in n.names) for n in tree.body)
        Loads(file, wanted, out, future).visit(tree)
        lines = cached_lines(file, tree)
        if lines:
            cached[file] = lines
        package = file[:-3].replace("/", ".").removesuffix(".__init__").split(".")
        if not file.endswith("__init__.py"):
            package = package[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.setdefault(file, []).append([alias.name, "", alias.asname or alias.name])
            if isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join([*base, *([node.module] if node.module else [])])
                for alias in node.names:
                    imports.setdefault(file, []).append([module, alias.name, alias.asname or alias.name])
    with gzip.open(output, "wt", encoding="utf8") as f:
        json.dump({"loads": out, "imports": imports, "cached": cached, "strings": sorted(wanted_all & words), "test_strings": test_words,
                   "dynamic_strings": sorted(wanted_all & dynamic_words), "doc_strings": sorted(wanted_all & doc_words),
                   "calls": calls, "escapes": sorted(escapes),
                   "patterns": sorted(patterns),
                   "special": special_classes(root)}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
