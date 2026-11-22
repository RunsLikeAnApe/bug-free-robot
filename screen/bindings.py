"""Follow an edited module or class binding to the tests that read it.

An edit confined to the value of a module-level or class-level assignment can
only change behavior where that value is read. The build-time index from
refs_index.py lists every static load of each bound name. A load inside a
function or lambda body runs only when that code runs, so baseline line
coverage of the load lists the tests that could observe the new value. A load
evaluated while a module or class body runs feeds another binding, which is
followed in turn. Anything else gives up and leaves the ordinary selection.

The edit itself must not be able to fail during import: it may only change
constants, existing names and attributes inside literal containers.
"""
import ast
from tree_equal import same_tree
import builtins
import gzip
import json

CONTAINERS = (ast.Dict, ast.List, ast.Tuple, ast.Set)
# Typing factories accept any name or bound string at runtime.
TYPING_FACTORIES = {"TypeVar", "NewType", "ParamSpec", "TypeVarTuple"}


def _typing_factory(node):
    func = node.func if isinstance(node, ast.Call) else None
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name in TYPING_FACTORIES


def _typed_dict(cls, tree, seen=()):
    """True for a class deriving only from TypedDict, directly or through same-module TypedDicts."""
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if not cls.bases or cls.decorator_list:
        return False
    for base in cls.bases:
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", None)
        if name == "TypedDict":
            continue
        if name in classes and name not in seen and _typed_dict(classes[name], tree, (*seen, cls.name)):
            continue
        return False
    return True
SAFE_BINOPS = (ast.BitOr, ast.Add, ast.Sub)
SWAPPABLE = {ast.Eq: "eq", ast.NotEq: "eq", ast.Is: "is", ast.IsNot: "is", ast.In: "in", ast.NotIn: "in"}
# Expressions that evaluate a comparison during import without adding failure modes
# beyond those of the unchanged operands.
EVALUATING = (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.IfExp, ast.comprehension, ast.DictComp,
              ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.Dict, ast.List, ast.Tuple, ast.Set)
LIMIT = 40


def _diff(old, new, chain, out):
    """Collect minimal differing subtrees with their ancestor chains."""
    if type(old) is not type(new):
        out.append((old, new, chain))
        return
    if isinstance(old, ast.AST):
        for field in old._fields:
            if field in ("ctx", "type_comment", "kind"):
                continue
            a, b = getattr(old, field, None), getattr(new, field, None)
            if isinstance(a, list) or isinstance(b, list):
                if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
                    out.append((old, new, chain))
                    return
                pairs = list(zip(a, b))
            elif isinstance(a, ast.AST) or isinstance(b, ast.AST):
                pairs = [(a, b)]
            else:
                if a != b or type(a) is not type(b):
                    out.append((old, new, chain))
                    return
                continue
            for x, y in pairs:
                if isinstance(x, ast.AST) or isinstance(y, ast.AST):
                    if x is None or y is None:
                        out.append((old, new, chain))
                        return
                    _diff(x, y, chain + [old], out)
                elif x != y:
                    out.append((old, new, chain))
                    return
    elif old != new:
        out.append((old, new, chain))


def _selected_by(node):
    """Subexpressions whose evaluation depends on a condition inside node."""
    if isinstance(node, ast.IfExp):
        return [node.body, node.orelse]
    if isinstance(node, ast.BoolOp):
        return list(node.values)
    if isinstance(node, ast.DictComp):
        return [node.key, node.value]
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return [node.elt]
    return []


def _plain(node, names, attributes, targets=()):
    """An expression that evaluates without calls, subscripts or unknown attributes.

    Attributes of comprehension variables are unknown: the edit can apply them
    to items they were never evaluated on.
    """
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            if item.id not in names and item.id not in vars(builtins) and item.id not in targets:
                return False
        elif isinstance(item, ast.Attribute):
            root = item.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if (ast.dump(item.value), item.attr) not in attributes or (
                    isinstance(root, ast.Name) and root.id in targets):
                return False
        elif not isinstance(item, (ast.Constant, ast.Tuple, ast.List, ast.Set, ast.Compare, ast.BoolOp,
                                   ast.UnaryOp, ast.expr_context, ast.cmpop, ast.boolop, ast.unaryop)):
            return False
    return True


def _bound_before(body, lineno):
    names = set()
    for node in body:
        if node.lineno >= lineno:
            break
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def _import_attributes(tree):
    pairs = set()
    stack = [(tree, False)]
    while stack:
        node, deferred = stack.pop()
        if isinstance(node, ast.Attribute) and not deferred:
            pairs.add((ast.dump(node.value), node.attr))
        inner = deferred or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        stack.extend((child, inner) for child in ast.iter_child_nodes(node))
    return pairs


def _safe_value(node, names, attributes):
    """True when evaluating node at import cannot raise and returns a hashable or container value."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in names or (node.id in vars(builtins) and node.id[0].isupper())
    if isinstance(node, ast.Attribute):
        return (ast.dump(node.value), node.attr) in attributes and _safe_value(node.value, names, attributes)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_safe_value(e, names, attributes) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _hashable(k) and _safe_value(k, names, attributes) for k in node.keys) and all(
            _safe_value(v, names, attributes) for v in node.values)
    return False


def _hashable(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Name, ast.Attribute)):
        return True
    if isinstance(node, ast.Tuple):
        return all(_hashable(e) for e in node.elts)
    return False


def _only_field_types_changed(old, new):
    if len(old.body) != len(new.body):
        return False
    saved = []
    try:
        for prev, curr in zip(old.body, new.body):
            if not isinstance(prev, ast.AnnAssign) or not isinstance(curr, ast.AnnAssign):
                continue
            if not isinstance(curr.target, ast.Name) or curr.value is not None:
                continue
            allowed = (ast.Name, ast.Load, ast.Attribute, ast.Subscript, ast.Tuple,
                       ast.BinOp, ast.BitOr, ast.Constant)
            if not all(isinstance(n, allowed) and
                       (not isinstance(n, ast.Constant) or n.value is None or type(n.value) is int)
                       for n in ast.walk(curr.annotation)):
                continue
            saved.append((curr, curr.annotation))
            curr.annotation = prev.annotation
        return bool(saved) and same_tree(old, new)
    finally:
        for node, annotation in saved:
            node.annotation = annotation


def _changed_bindings(path, before, after):
    """Return [(kind, owner, name)] for edited assignments, or None for any other edit."""
    module = path[:-3].replace("/", ".").removesuffix(".__init__")
    attributes = _import_attributes(before)
    found = []

    def walk(old_body, new_body, owner, scope_names):
        if len(old_body) != len(new_body):
            return False
        for old, new in zip(old_body, new_body):
            if same_tree(old, new):
                continue
            if isinstance(old, ast.ClassDef) and isinstance(new, ast.ClassDef):
                header = lambda c: ast.dump(ast.ClassDef(c.name, c.bases, c.keywords, [], c.decorator_list))
                if header(old) != header(new):
                    # A typed dict's keywords (total=...) only change its metadata.
                    same = (old.name == new.name and same_tree(ast.Tuple(old.bases), ast.Tuple(new.bases))
                            and [k.arg for k in old.keywords] == [k.arg for k in new.keywords]
                            and not old.decorator_list and not new.decorator_list
                            and all(isinstance(a.value, ast.Constant) and isinstance(b.value, ast.Constant)
                                    and type(a.value.value) is type(b.value.value)
                                    for a, b in zip(old.keywords, new.keywords)))
                    if not same or not _typed_dict(old, before) or not _typed_dict(new, after):
                        return False
                    found.append(("attr" if owner else "global", ".".join([module, *owner]), old.name))
                    if same_tree(ast.Module(old.body, []), ast.Module(new.body, [])):
                        continue
                # Postponed TypedDict field annotations are stored on the class.
                # Attribute consumers, subclasses and reflection can observe them.
                if (not owner and _typed_dict(old, before) and _typed_dict(new, after)
                        and any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                                and any(a.name == "annotations" for a in n.names) for n in before.body)
                        and _only_field_types_changed(old, new)):
                    found.append(("global", module, old.name))
                    continue
                inner = owner + [old.name]
                if not walk(old.body, new.body, inner, None):
                    return False
                continue
            if type(old) is not type(new) or not isinstance(old, (ast.Assign, ast.AnnAssign)):
                return False
            targets = old.targets if isinstance(old, ast.Assign) else [old.target]
            new_targets = new.targets if isinstance(new, ast.Assign) else [new.target]
            if (len(targets) != 1 or not isinstance(targets[0], ast.Name)
                    or (not same_tree(ast.Module(targets, []), ast.Module(new_targets, [])))):
                return False
            if isinstance(old, ast.AnnAssign) and (not same_tree(old.annotation, new.annotation)):
                return False
            if old.value is None or new.value is None:
                return False
            diffs = []
            _diff(old.value, new.value, [], diffs)
            body = old_body
            names = _bound_before(body, old.lineno) | (scope_names or set())
            for a, b, chain in diffs:
                if type(a) in SWAPPABLE and type(b) in SWAPPABLE and SWAPPABLE[type(a)] == SWAPPABLE[type(b)]:
                    # Swapping == / != (or is / is not, in / not in) calls the same
                    # protocol on the same operands, so it cannot newly raise. What
                    # the outcome selects (branches, comprehension elements, later
                    # operands) may now run, so it must be unable to raise as well.
                    known = attributes | _attributes_in(old.value)
                    loop_vars = {t.id for p in chain if isinstance(p, ast.comprehension)
                                 for t in ast.walk(p.target) if isinstance(t, ast.Name)}
                    loop_vars |= {t.id for p in chain for g in getattr(p, "generators", ())
                                  for t in ast.walk(g.target) if isinstance(t, ast.Name)}
                    plain = lambda e: _plain(e, names | _names_in(old.value), known, loop_vars)
                    if all(isinstance(p, EVALUATING) for p in chain) and all(
                            plain(e) for p in chain for e in _selected_by(p)):
                        continue
                    return False
                if not isinstance(a, ast.expr) or not isinstance(b, ast.expr):
                    return False
                for parent in chain:
                    if isinstance(parent, CONTAINERS):
                        continue
                    if (_typing_factory(parent) and isinstance(a, ast.Constant) and isinstance(b, ast.Constant)
                            and isinstance(a.value, str) and isinstance(b.value, str)):
                        continue
                    if isinstance(parent, ast.BinOp) and isinstance(parent.op, SAFE_BINOPS):
                        continue
                    return False
                parent = chain[-1] if chain else None
                if isinstance(parent, ast.Dict) and any(k is a for k in parent.keys) and not _hashable(b):
                    return False
                if isinstance(parent, ast.Set) and not _hashable(b):
                    return False
                if not _safe_value(b, names | _names_in(a), attributes | _attributes_in(a)):
                    return False
            # Renaming the string passed to a typing factory changes only the
            # object's displayed name: type variables are opaque at runtime.
            if (_typing_factory(old.value) and diffs and all(
                    chain == [old.value] and isinstance(a, ast.Constant) and isinstance(a.value, str)
                    and isinstance(b, ast.Constant) and isinstance(b.value, str) for a, b, chain in diffs)):
                found.append(("display", ".".join([module, *owner]), targets[0].id))
                continue
            found.append(("attr" if owner else "global", ".".join([module, *owner]), targets[0].id))
        return True

    if not walk(before.body, after.body, [], _bound_before(before.body, 10**9)):
        return None
    return found


def _names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _attributes_in(node):
    return {(ast.dump(n.value), n.attr) for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def _module_file(module, files):
    for candidate in (module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py"):
        if candidate in files:
            return candidate
    return None


def load_refs(maps):
    if "refs" not in maps:
        try:
            with gzip.open(maps["refs_path"], "rt", encoding="utf8") as f:
                maps["refs"] = json.load(f)
        except (KeyError, OSError, ValueError):
            maps["refs"] = None
    return maps["refs"]


def binding_mask(path, before, after, maps):
    changed = _changed_bindings(path, before, after)
    if not changed:
        return None
    refs = load_refs(maps)
    if not refs:
        return None
    loads, imports, strings, special = refs["loads"], refs["imports"], set(refs["strings"]), refs["special"]
    files = set(imports) | {path}
    for entries in loads.values():
        files.update(e[0] for e in entries)
    rows_by_file = maps["lines"]
    import_rows = maps["imports"]
    test_files = maps["test_files"]
    mask = 0
    seen = set()
    cached = {}
    work = list(changed)
    while work:
        kind, owner, name = work.pop()
        if (kind, owner, name) in seen:
            continue
        seen.add((kind, owner, name))
        if kind == "display":
            # Only the object's repr and __name__ differ. Tests reading the
            # source are kept by the caller.
            continue
        if len(seen) > LIMIT or name in strings or name.startswith("__") or name.endswith("_sql"):
            return None
        if any(name.startswith(head) and name.endswith(tail) for head, tail in refs.get("patterns", ())):
            return None
        if kind == "attr" and any(owner == s or owner.startswith(s + ".") for s in special):
            return None
        for file in refs.get("test_strings", {}).get(name, ()):
            mask |= test_files.get(file, 0)
        if name not in loads:
            # A binding without any static load cannot be observed by name.
            continue
        if kind == "global":
            home = _module_file(owner, files)
            if home is None:
                return None
            scopes = {home: name}
            module = owner
            for file, entries in imports.items():
                for source, imported, alias in entries:
                    if imported == "*" and source == module:
                        scopes.setdefault(file, name)
                    elif imported == name and source == module:
                        if alias != name:
                            return None
                        scopes.setdefault(file, name)
            relevant = [e for e in loads[name] if e[2] == "a" or e[0] in scopes]
        else:
            module_file = None
            parts = owner.split(".")
            for cut in range(len(parts), 0, -1):
                module_file = _module_file(".".join(parts[:cut]), files)
                if module_file:
                    break
            if module_file is None:
                return None
            relevant = [e for e in loads[name] if e[2] == "a" or (e[0] == module_file and (e[3] or "").startswith("C:"))]
        for file, line, _, context in relevant:
            if context is None and line in cached.setdefault(file, set(refs.get("cached", {}).get(file, ()))):
                # Memoized across tests: coverage lists only the first reader.
                return None
            if context is None:
                key = str(line)
                mask |= int(rows_by_file.get(file, {}).get(key, "0"), 16)
                mask |= int(import_rows.get(file, {}).get(key, "0"), 16)
            elif file.startswith("tests/"):
                mask |= test_files.get(file, 0)
            elif context == "":
                return None
            else:
                derived_module = file[:-3].replace("/", ".").removesuffix(".__init__")
                scope, target = context.split(":", 1)
                if scope == "M":
                    work.append(("global", derived_module, target))
                else:
                    work.append(("attr", derived_module + ".?", target))
    return mask


def importer_mask(path, maps):
    """Tests in a test module and in every test module that imports it, directly or not."""
    refs = load_refs(maps)
    if not refs or not path.startswith("tests/") or not path.endswith(".py"):
        return None
    module = path[:-3].replace("/", ".").removesuffix(".__init__")
    package = path.endswith("/__init__.py")
    importers = {}
    for file, entries in refs["imports"].items():
        for source, name, _ in entries:
            for target in (source, f"{source}.{name}" if name and name != "*" else source):
                importers.setdefault(target, set()).add(file)
    done, todo = set(), [(path, module, package)]
    mask = 0
    while todo:
        file, name, is_package = todo.pop()
        if file in done:
            continue
        done.add(file)
        mask |= maps["test_files"].get(file, 0)
        followers = set(importers.get(name, ()))
        if is_package:
            folder = file.rsplit("/", 1)[0] + "/"
            followers |= {f for f in maps["test_files"] if f.startswith(folder)}
            followers |= {f for f in refs["imports"] if f.startswith(folder)}
        for other in followers:
            todo.append((other, other[:-3].replace("/", ".").removesuffix(".__init__"), other.endswith("/__init__.py")))
    return mask


def signature_safe(func, before, after):
    """True when a function's definition-time expressions differ only in ways that cannot raise."""
    twin = next((n for n in ast.walk(after) if type(n) is type(func) and n.name == func.name
                 and n.lineno == func.lineno and n.col_offset == func.col_offset), None)
    if twin is None or (not same_tree(ast.Module(func.decorator_list, []), ast.Module(twin.decorator_list, []))):
        return False
    attributes = _import_attributes(before)
    names = _bound_before(before.body, 10**9) | {n.id for n in ast.walk(before) if isinstance(n, ast.Name)}
    future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                 and any(a.name == "annotations" for a in n.names) for n in before.body)
    olds = [*func.args.defaults, *func.args.kw_defaults]
    news = [*twin.args.defaults, *twin.args.kw_defaults]
    if len(olds) != len(news):
        return False
    pairs = list(zip(olds, news))
    if not future:
        old_args = [a for a in ast.walk(func.args) if isinstance(a, ast.arg)]
        new_args = [a for a in ast.walk(twin.args) if isinstance(a, ast.arg)]
        if len(old_args) != len(new_args):
            return False
        pairs += [(a.annotation, b.annotation) for a, b in zip(old_args, new_args)]
        pairs.append((func.returns, twin.returns))
    for old, new in pairs:
        if old is None or new is None:
            if old is not new:
                return False
            continue
        diffs = []
        _diff(old, new, [], diffs)
        for a, b, chain in diffs:
            if not isinstance(b, ast.expr) or not all(isinstance(p, CONTAINERS) for p in chain):
                return False
            if not _safe_value(b, names | _names_in(a), attributes | _attributes_in(a)):
                return False
    return True


def default_callers(func, before, after, path, maps):
    """Tests reaching a call that relies on a default whose constant value changed.

    A default is used only by calls that omit its argument. When the function's
    name never escapes as a value (callbacks, aliases, partials) and cannot be
    looked up dynamically, every such call is a static call site; coverage of
    the sites that omit the argument lists the tests that observe the change.
    """
    if func.name.startswith("__") or any(
            not (isinstance(d, ast.Name) and d.id in {"staticmethod", "classmethod"}) for d in func.decorator_list):
        return None
    twin = next((n for n in ast.walk(after) if type(n) is type(func) and n.name == func.name
                 and n.lineno == func.lineno and n.col_offset == func.col_offset), None)
    if twin is None:
        return None
    old, new = func.args, twin.args
    shape = lambda a: ast.dump(ast.arguments(a.posonlyargs, a.args, a.vararg, a.kwonlyargs, [None] * len(a.kw_defaults),
                                             a.kwarg, []), include_attributes=False)
    if shape(old) != shape(new) or len(old.defaults) != len(new.defaults):
        return None
    positional = [a.arg for a in [*old.posonlyargs, *old.args]]
    changed = []
    for i, (a, b) in enumerate(zip(old.defaults, new.defaults)):
        if (not same_tree(a, b)):
            if not (isinstance(a, ast.Constant) and isinstance(b, ast.Constant)):
                return None
            index = len(positional) - len(old.defaults) + i
            changed.append((positional[index], index, index >= len(old.posonlyargs)))
    for arg, a, b in zip(old.kwonlyargs, old.kw_defaults, new.kw_defaults):
        if (a is None) != (b is None):
            return None
        if a is not None and (not same_tree(a, b)):
            if not (isinstance(a, ast.Constant) and isinstance(b, ast.Constant)):
                return None
            changed.append((arg.arg, None, True))
    if not changed or (not same_tree(ast.Module(func.body, []), ast.Module(twin.body, []))):
        return None
    refs = load_refs(maps)
    if not refs or "calls" not in refs:
        return None
    name = func.name
    if any(imported == name and alias != name for entries in refs["imports"].values()
           for _, imported, alias in entries):
        # Called under another name: the call index does not list those sites.
        return None
    # Names listed in __all__ are bound under the same name by star imports;
    # only lookups by computed strings can call the function unseen. Doctest
    # examples run in the documentation tests.
    dynamic = refs.get("dynamic_strings", refs["strings"])
    if name in refs["escapes"] or name in set(dynamic) or any(
            name.startswith(h) and name.endswith(t) for h, t in refs.get("patterns", ())):
        return None
    mask = maps.get("documentation_readers", 0) if name in set(refs.get("doc_strings", ())) else 0
    for file in refs.get("test_strings", {}).get(name, ()):
        mask |= maps["test_files"].get(file, 0)
    rows_by_file, import_rows = maps["lines"], maps["imports"]
    module_level = any(node is func for node in before.body)
    imported = {}
    for file, line, npos, starred, keywords, double_star, deferred, root in refs["calls"].get(name, ()):
        if module_level and root is not None:
            # A module function is reachable through an attribute only on an
            # imported module, never on a local object such as a connection.
            if file not in imported:
                imported[file] = {alias.split(".")[0] for _, _, alias in refs["imports"].get(file, ())}
            if root not in imported[file]:
                continue
        if not starred and not double_star and all(
                (index is not None and index < npos) or (by_name and arg in keywords)
                for arg, index, by_name in changed):
            continue
        if not deferred and not file.startswith("tests/"):
            # Evaluated while a module imports: its result reaches tests that
            # line coverage does not attribute.
            return None
        if not deferred:
            mask |= maps["test_files"].get(file, 0)
            continue
        key = str(line)
        mask |= int(rows_by_file.get(file, {}).get(key, "0"), 16) | int(import_rows.get(file, {}).get(key, "0"), 16)
    return mask
