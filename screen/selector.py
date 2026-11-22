"""Select tests from baseline coverage, fixture reads and syntax changes."""
import ast
from tree_equal import same_tree
import bisect
import gzip
import json
import os
import re
import time
import runpy
from io import StringIO
from types import FunctionType
from replay import failing_tests
import tpcds_replay
from bindings import binding_mask, default_callers, importer_mask, load_refs, signature_safe
from profiles import boolop_edit, membership_edit, default_edit, conditional_edit, spanning_nodes, keyword_edit, metadata_callback_edit, baseline_dump, child_mask, operation_edit, lookup_edit, copy_edit, callback_edit, changed_predicate, config_edit, flag_edit, module_edit
from copy import copy
from functools import cache
from pathlib import Path

APP = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("SOURCE", "/src"))
DATA = Path(os.environ.get("DATA", "/data"))
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


# Files outside the package, its tests and the extension sources that pytest,
# Python or the build never load implicitly. Tests that read one are listed in
# the baseline file map.
LOADED = ("sqlglot", "tests", "sqlglotc", "sqlglot-integration-tests")
CONFIG = {"setup.cfg", "pyproject.toml", "tox.ini", "pytest.ini", "setup.py", "conftest.py",
          "sitecustomize.py", "usercustomize.py", "__init__.py"}


def inert(path):
    parts = Path(path).parts
    return (bool(parts) and parts[0] not in LOADED and len(parts) > 0 and parts[-1] not in CONFIG
            and not parts[-1].endswith(".pth") and ".." not in parts and not Path(path).is_absolute())


def header_lines(lines):
    """The lines of one diff block before its first hunk.

    Hunk lines can also start with "--- " or "+++ ", for example a removed SQL
    comment line "-- note" becomes "--- note", so only the header names files.
    """
    for i, line in enumerate(lines):
        if line.startswith("@@"):
            return lines[:i]
    return lines


def side_paths(block):
    """Old and new paths named by one git diff block, or None when a side is absent."""
    header = block.split(chr(10), 1)[0].split()
    heading = header_lines(block.splitlines())
    olds = [l[4:].strip() for l in heading if l.startswith("--- ")]
    news = [l[4:].strip() for l in heading if l.startswith("+++ ")]
    names = set()
    for value in olds + news + header:
        if value.startswith(("a/", "b/")):
            names.add(value[2:])
    for line in heading:
        for prefix in ("rename from ", "rename to ", "copy from ", "copy to "):
            if line.startswith(prefix):
                names.add(line[len(prefix):].strip())
    return names


def fixture_structure(diff, maps, all_mask):
    """Select observers of added, deleted, or renamed SQL data in an existing folder."""
    if diff.count("diff --git ") != 1:
        return None
    block = diff.partition("diff --git ")[2]
    header = block.splitlines()[0] if block else ""
    if not re.fullmatch(r"a/[\w./-]+ b/[\w./-]+", header):
        return None
    if not any(line.startswith(("new file mode 100644", "deleted file mode 100644", "rename from "))
               for line in block.splitlines()):
        return None
    paths = side_paths(block)
    if not paths or any(not p.startswith("tests/fixtures/") or not p.endswith(".sql")
                        or ".." in Path(p).parts or not (SOURCE/p).parent.is_dir() for p in paths):
        return None
    # Native database readers of the execution benchmark assets are outside
    # the Python filesystem hooks.
    if any(p.startswith(("tests/fixtures/optimizer/tpc-ds/", "tests/fixtures/optimizer/tpc-h/")) for p in paths):
        return None
    profile = maps["fixture_paths"]
    mask = int(profile["external"], 16)
    for path in paths:
        folder = str(Path(path).parent)
        for key in ("read:" + path, "file:" + path, "dir:" + folder, "file:" + folder):
            mask |= int(profile["events"].get(key, "0"), 16)
        mask |= int(maps["files"].get(path, "0"), 16)
    return mask & all_mask


def changes(diff, skipped=None):
    """Read old-side coordinates and rebuild the changed text for each file."""
    for block in re.split(r"(?m)^diff --git ", diff)[1:]:
        lines = block.splitlines(keepends=True)
        heading = header_lines(lines)
        olds = [line[4:].strip() for line in heading if line.startswith("--- ")]
        news = [line[4:].strip() for line in heading if line.startswith("+++ ")]
        regular = (len(olds) == len(news) == 1 and olds[0].startswith("a/") and news[0] == "b/" + olds[0][2:]
                   and not any(l.startswith(("rename ", "copy ", "new file", "deleted file", "Binary files", "GIT binary"))
                               for l in heading))
        if not regular:
            names = side_paths(block)
            if skipped is not None and names and all(inert(n) for n in names) and '"' not in block.split(chr(10), 1)[0]:
                skipped.update(names)
                continue
            raise ValueError("new, removed, renamed or binary file")
        old, new = olds[0], news[0]
        path = old[2:]
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("invalid source path")
        base = (SOURCE / path).read_text(encoding="utf8").splitlines(keepends=True)
        result, removed, inserted = [], set(), set()
        cursor = 0
        active = False
        replacing = False
        for line in lines:
            match = HUNK.match(line)
            if match:
                pos = int(match[1]) - (match[2] != "0")
                if pos < cursor or pos > len(base):
                    raise ValueError("hunk range")
                result.extend(base[cursor:pos])
                cursor = pos
                active = True
                replacing = False
            elif active and line.startswith((" ", "-")):
                if line[1:].rstrip("\n") != base[cursor].rstrip("\n"):
                    raise ValueError("unmatched context")
                if line.startswith("-"):
                    removed.add(cursor + 1)
                    replacing = True
                else:
                    result.append(base[cursor])
                    replacing = False
                cursor += 1
            elif active and line.startswith("+"):
                if not replacing:
                    inserted.add(cursor + 1)
                result.append(line[1:])
        if not active:
            raise ValueError("no text hunks")
        result.extend(base[cursor:])
        yield path, "".join(base), "".join(result), removed, inserted


@cache
def syntax(source):
    return ast.parse(source)


def identity_equivalent(left, right):
    if left == right == "identity":
        return True
    if left == right and left.startswith("enum:"):
        return True
    if "none" in (left,right):
        return all(t in {"none","identity","builtin"} or t.startswith("enum:") for t in (left,right))
    return False


def normalize_comparisons(before, after, path, operands):
    groups = {ast.Is:1, ast.Eq:1, ast.IsNot:2, ast.NotEq:2}
    for old,new in zip(ast.walk(before),ast.walk(after)):
        if not isinstance(old,ast.Compare) or not isinstance(new,ast.Compare):
            continue
        if len(old.ops) != 1 or len(new.ops) != 1:
            continue
        a,b = type(old.ops[0]),type(new.ops[0])
        if a is b or a not in groups or groups[a] != groups.get(b):
            continue
        key = f"{path}:{old.lineno}:{old.col_offset}:"
        left,right = operands.get(key+"0"),operands.get(key+"1")
        if left and right and all(identity_equivalent(x,y) for x in left for y in right):
            new.ops[0] = old.ops[0]


def added_method_receivers(path, before, after, maps):
    """Identify insertions of plain instance methods into existing profiled classes."""
    baseline = {cls.name: cls for cls in before.body if isinstance(cls, ast.ClassDef)}
    selected = 0
    remainder = copy(after)
    remainder.body = list(after.body)
    for index, cls in enumerate(after.body):
        if not isinstance(cls, ast.ClassDef) or cls.name not in baseline:
            continue
        owner = path[:-3].replace("/", ".") + "." + cls.name
        receivers = maps["types"].get(owner)
        if receivers is None:
            continue
        existing = {getattr(n,"name",None) for n in baseline[cls.name].body}
        retained = []
        for member in cls.body:
            if (isinstance(member, ast.FunctionDef) and member.name not in existing
                    and not member.name.startswith("__") and not member.decorator_list
                    and not member.type_params and not member.returns and not member.args.defaults
                    and not any(member.args.kw_defaults) and member.args.args
                    and member.args.args[0].arg == "self"
                    and all(not arg.annotation for arg in ast.walk(member.args) if isinstance(arg,ast.arg))):
                selected |= int(receivers,16)
            else:
                retained.append(member)
        remainder.body[index] = copy(cls)
        remainder.body[index].body = retained
    if selected and same_tree(before, remainder):
        return selected
    return None


def scalar_setting(path, before, after, removed, maps):
    """Look up reads only when one existing class scalar changed value."""
    if not path.startswith("sqlglot/dialects/") or len(removed) != 1:
        return None
    number = next(iter(removed))
    for old_cls in before.body:
        if not isinstance(old_cls,ast.ClassDef) or not old_cls.lineno < number <= old_cls.end_lineno:
            continue
        for old in old_cls.body:
            if (not isinstance(old,ast.Assign) or old.lineno != number or old.end_lineno != number
                    or len(old.targets) != 1 or not isinstance(old.targets[0],ast.Name)):
                continue
            name = old.targets[0].id
            key = ".".join((path[:-3].replace("/","."),old_cls.name,name))
            field = maps["fields"].get(key)
            if field is None:
                continue
            for cls in after.body:
                if not isinstance(cls,ast.ClassDef) or cls.name != old_cls.name:
                    continue
                for new in cls.body:
                    if not isinstance(new,ast.Assign) or new.lineno != number:
                        continue
                    try:
                        lhs, rhs = ast.literal_eval(old.value), ast.literal_eval(new.value)
                    except (ValueError,TypeError):
                        continue
                    if type(lhs) is not type(rhs) or type(lhs) not in (int,bool,str,float):
                        continue
                    revised = new.value
                    new.value = old.value
                    try:
                        if same_tree(before, after):
                            return int(field,16) | child_mask(path,maps)
                    finally:
                        new.value = revised
    return None


@cache
def fixture_helpers():
    return runpy.run_path(str(SOURCE / "tests/helpers.py"))


def fixture_values(mode, content):
    loader = fixture_helpers()[mode]
    namespace = dict(loader.__globals__, open=lambda *args, **kwargs: StringIO(content))
    read = FunctionType(loader.__code__, namespace)
    if mode == "load_sql_fixture_pairs":
        return [(tuple(meta.items()), text, expected) for meta, text, expected in read("fixture.sql")]
    return list(read("fixture.sql"))


def fixture_selection(path, base, revised, readers, maps):
    modes = maps["fixture_modes"].get(path)
    if not modes:
        return readers
    observed = 0
    selected = 0
    for mode, hits in modes.items():
        mask = int(hits, 16)
        observed |= mask
        if mode in {"load_sql_fixtures", "load_sql_fixture_pairs"}:
            try:
                old = fixture_values(mode, base)
                new = fixture_values(mode, revised)
                if old == new:
                    continue
            except (IndexError, ValueError):
                selected |= mask
                continue
            if mode == "load_sql_fixture_pairs" and path in maps["fixture_attributes"]:
                mask &= int(maps["fixture_attributes"][path], 16)
                fields = maps["fixture_fields"].get(path)
                if fields and len(old) == len(new):
                    changed = 0
                    for i, bits in enumerate(fields):
                        if any(a[i] != b[i] for a, b in zip(old, new)):
                            changed |= int(bits, 16)
                    mask &= changed
        selected |= mask
    return selected | (readers & ~observed)


@cache
def single_definition(node):
    return ast.Module(body=[node],type_ignores=[])


def edited_function(before, after, base, revised, removed):
    """Scope a whole-tree comparison when every changed byte belongs to one function."""
    if not removed:
        return before, after
    first, last = min(removed), max(removed)
    def definitions(tree):
        for item in tree.body:
            if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):
                yield item
            elif isinstance(item,ast.ClassDef):
                for member in item.body:
                    if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        yield member
    old = next((f for f in definitions(before) if f.lineno < first <= last <= f.end_lineno), None)
    if old is None:
        return before, after
    old_lines, new_lines = base.splitlines(True), revised.splitlines(True)
    delta = len(new_lines) - len(old_lines)
    new = next((f for f in definitions(after) if type(f) is type(old) and f.name == old.name and
                f.lineno == old.lineno and f.end_lineno == old.end_lineno + delta), None)
    if new is None or old_lines[:old.lineno-1] != new_lines[:new.lineno-1] or old_lines[old.end_lineno:] != new_lines[new.end_lineno:]:
        return before, after
    return single_definition(old), ast.Module(body=[new],type_ignores=[])


def outside_test_methods(tree, lines):
    """True when one of the edited lines lies outside the body of every test function."""
    tests = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith("test")]
    return any(not any(f.lineno < line <= f.end_lineno for f in tests) for line in lines)


def shallow_functions(tree):
    """Functions and methods of a module, without looking inside function bodies."""
    found, stack = [], list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(node)
        elif isinstance(node, ast.ClassDef):
            stack.extend(node.body)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for block in ("body", "orelse", "finalbody"):
                stack.extend(getattr(node, block, []))
            for handler in getattr(node, "handlers", []):
                stack.extend(handler.body)
    return found


def cache_guard(test, func):
    """True for the test of a miss on a cache shared across tests (a module global or a class
    attribute, not self): `if found is None:` after `found = CACHE.get(key)`, `if key not in
    cls._classes:`, `if len(NAMES) != len(cls._classes):`."""
    local = {a.arg for a in ast.walk(func.args) if isinstance(a, ast.arg)}
    local |= {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}

    def shared(node):
        if isinstance(node, ast.Name):
            return node.id not in local
        if isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            return isinstance(root, ast.Name) and root.id != "self"
        return False

    for c in ast.walk(test):
        if not isinstance(c, ast.Compare) or len(c.ops) != 1:
            continue
        op, left, right = c.ops[0], c.left, c.comparators[0]
        if isinstance(op, (ast.In, ast.NotIn)) and shared(right):
            return True
        if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) and right.value is None:
            if isinstance(left, ast.Attribute) and shared(left):
                return True
            if isinstance(left, ast.Name) and any(
                    isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id == left.id for t in a.targets)
                    and isinstance(a.value, ast.Call) and isinstance(a.value.func, ast.Attribute)
                    and a.value.func.attr == "get" and shared(a.value.func.value) for a in ast.walk(func)):
                return True
        if isinstance(op, (ast.Eq, ast.NotEq)) and all(
                isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == "len"
                for side in (left, right)):
            return True
    return False


def once_per_process_mask(path, tree, base, lines, maps):
    """Tests reaching the callers of an edited function that runs only on a cache miss.

    A function that is called only under a cache-miss guard (cache_guard) runs once per
    process or class, in whichever test gets there first, so line coverage credits it to
    that test alone, while every test that reaches its callers uses what it built."""
    functions = shallow_functions(tree)
    rows = maps["lines"].get(path, {})
    imports = maps["imports"].get(path, {})
    mask = 0
    for func in functions:
        if not any(func.lineno <= line <= func.end_lineno for line in lines):
            continue
        callers, line, pos = [], 1, 0
        # call sites in file order, stopping at the first one outside a cache-miss guard
        for site in re.finditer(r"(?<!def )\b%s\(" % re.escape(func.name), base):
            line += base.count("\n", pos, site.start())
            pos = site.start()
            if func.lineno <= line <= func.end_lineno:
                continue
            caller = min((f for f in functions if f.lineno < line <= f.end_lineno),
                         key=lambda f: f.end_lineno - f.lineno, default=None)
            if caller is None:
                callers = None
                break
            parents = {child: node for node in ast.walk(caller) for child in ast.iter_child_nodes(node)}
            calls = [n for n in parents if isinstance(n, ast.Call) and n.lineno == line and (
                (isinstance(n.func, ast.Name) and n.func.id == func.name) or
                (isinstance(n.func, ast.Attribute) and n.func.attr == func.name))]
            for call in calls:
                node, guard = call, None
                while node in parents and node is not caller:
                    node = parents[node]
                    if isinstance(node, ast.If):
                        guard = node
                        break
                if guard is None or not cache_guard(guard.test, caller):
                    callers = None
                    break
            if callers is None:
                break
            if calls:
                callers.append(caller)
        for caller in callers or ():
            for i in range(caller.lineno, caller.end_lineno + 1):
                mask |= int(rows.get(str(i), "0"), 16) | int(imports.get(str(i), "0"), 16)
    return mask


DIALECT_COPIED = re.compile(
    r"^[ \t]+(?:QUOTES|IDENTIFIERS|BIT_STRINGS|HEX_STRINGS|BYTE_STRINGS|UNICODE_STRINGS|STRING_ESCAPES"
    r"|BYTE_STRING_ESCAPES|IDENTIFIER_ESCAPES|NUMERIC_ESCAPES|UNESCAPED_SEQUENCES)[ \t]*(?::[^=\n]*)?=(?!=)", re.M)


def dialect_copied_edit(base, lines):
    """True when an edited line lies in a class-level statement that assigns a tokenizer table
    which the dialect metaclass copies onto the dialect (QUOTES -> QUOTE_START, IDENTIFIERS ->
    IDENTIFIER_START, BIT/HEX/BYTE/UNICODE_STRINGS -> BIT_START..., the escape tables ->
    ESCAPED_SEQUENCES). Every dialect's generator reads those copies when it writes this
    dialect, without running its tokenizer, so the receivers of the tokenizer class do not
    bound the tests that can fail."""
    spans = set()
    for m in DIALECT_COPIED.finditer(base):
        line = base.count("\n", 0, m.start()) + 1
        first, depth, i = line, 0, m.end()
        while i < len(base):
            ch = base[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "\n":
                if depth <= 0 and base[i - 1] != "\\":
                    break
                line += 1
            i += 1
        spans.update(range(first, line + 1))
    return any(line in spans for line in lines)


def lambda_edit(path, before, after, maps):
    """Confine an edit to lambdas defined outside functions.

    A lambda body runs only when the lambda is called, so line coverage of its
    lines lists every test that executed it. Import-time calls are covered by
    the import trace of each test module and lazily loaded dialect.
    """
    olds = {(n.lineno, n.col_offset): n for n in ast.walk(before) if isinstance(n, ast.Lambda)}
    changed = []
    for new in ast.walk(after):
        if isinstance(new, ast.Lambda):
            old = olds.get((new.lineno, new.col_offset))
            if old is not None and (not same_tree(old, new)):
                changed.append((old, new))
    if not changed:
        return None
    saved = [(new, new.args, new.body) for _, new in changed]
    try:
        for old, new in changed:
            new.args, new.body = old.args, old.body
        if (not same_tree(before, after)):
            return None
    finally:
        for new, args, body in saved:
            new.args, new.body = args, body
    functions = [n for n in ast.walk(before) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    rows = maps["lines"].get(path, {})
    imports = maps["imports"].get(path, {})
    mask = child_mask(path, maps)
    for old, _ in changed:
        if any(f.lineno <= old.lineno <= f.end_lineno for f in functions):
            return None
        covered = 0
        for line in range(old.lineno, old.end_lineno + 1):
            key = str(line)
            covered |= int(rows.get(key, "0"), 16)
            mask |= int(rows.get(key, "0"), 16) | int(imports.get(key, "0"), 16)
        if not covered:
            # no test is recorded running this lambda's lines (the coverage map can miss a
            # one-line lambda that a table holds): the coverage cannot confine the edit
            return None
    return mask


def future_annotations(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names) for n in tree.body)


def equal_without_annotations(before, after):
    """Compare trees ignoring annotations that the interpreter never evaluates.

    Under "from __future__ import annotations" function signatures keep their
    annotations as unevaluated strings, and annotations of local variables are
    never evaluated. Class and module level annotations still define fields of
    dataclasses, named tuples and typed dicts, so they are compared.
    """
    if not future_annotations(before) or not future_annotations(after):
        return False
    saved = []
    for tree in (before, after):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                saved.append((node, "returns", node.returns))
                node.returns = None
                for arg in ast.walk(node.args):
                    if isinstance(arg, ast.arg):
                        saved.append((arg, "annotation", arg.annotation))
                        arg.annotation = None
                pending = list(node.body)
                while pending:
                    inner = pending.pop()
                    if isinstance(inner, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if isinstance(inner, ast.AnnAssign):
                        saved.append((inner, "annotation", inner.annotation))
                        inner.annotation = ast.Constant(None)
                    pending.extend(ast.iter_child_nodes(inner))
    try:
        return same_tree(before, after)
    finally:
        for node, field, value in reversed(saved):
            setattr(node, field, value)


def annotation_on_line(tree, line):
    for node in spanning_nodes(tree, line, line):
        if isinstance(node, (ast.arg, ast.AnnAssign)):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        else:
            continue
        if annotation is not None and annotation.lineno <= line <= annotation.end_lineno:
            return True
    return False


def frame_readers_for(path, maps):
    """Frame-inspecting tests that can reach code objects defined in path."""
    mask = 0
    for file, bits in maps.get("frame_reader_files", {}).items():
        scope = maps.get("frame_scopes", {}).get(file)
        if scope is None or path in scope:
            mask |= bits
    return mask if maps.get("frame_reader_files") else maps["frame_readers"]


def inert_statement(path, base, before, after, inserted, maps):
    """Recognize an isolated store or unreachable statement with no ordinary effects."""
    if len(inserted) != 1 or not path.startswith("sqlglot/"):
        return None
    line = next(iter(inserted))
    candidates = [n for n in spanning_nodes(after, line, line)
                  if isinstance(n, ast.FunctionDef) and n.lineno < line]
    if len(candidates) != 1:
        return None
    func = candidates[0]
    for pos, stmt in enumerate(func.body):
        if stmt.lineno != line or func.decorator_list:
            continue
        if isinstance(stmt, ast.Assign):
            if not (len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and type(stmt.value.value) in (type(None), bool, int, str)):
                continue
            name = stmt.targets[0].id
            if name in base or name.startswith("__"):
                return None
        elif isinstance(stmt, ast.If):
            if not (isinstance(stmt.test, ast.Constant) and stmt.test.value is False
                    and not stmt.orelse and all(isinstance(n, ast.Pass) or
                        isinstance(n, ast.Return) and (n.value is None or isinstance(n.value, ast.Constant))
                        for n in stmt.body)):
                continue
        else:
            continue
        if (pos == 0 and len(func.body) > 1 and isinstance(func.body[1], ast.Expr)
                and isinstance(func.body[1].value, ast.Constant) and isinstance(func.body[1].value.value, str)):
            return None
        reflective = {"locals", "vars", "dir", "eval", "exec", "inspect", "sys", "super",
                      "f_locals", "gi_frame", "cr_frame", "tb_frame", "currentframe", "_getframe"}
        if any(isinstance(n, ast.Name) and n.id in reflective or
               isinstance(n, ast.Attribute) and n.attr in reflective for n in ast.walk(func)):
            return None
        func.body.pop(pos)
        try:
            if same_tree(before, after):
                # Memory accounting can observe the larger frame only when
                # the function runs. Code-object inspection needs no invocation.
                reached = 0
                for table in (maps["lines"], maps["imports"]):
                    rows = table.get(path, {})
                    for line_number in range(func.lineno, func.end_lineno + 1):
                        reached |= int(rows.get(str(line_number), "0"), 16)
                return (frame_readers_for(path, maps) | (maps["memory_readers"] & reached)
                        | child_mask(path, maps))
        finally:
            func.body.insert(pos, stmt)
    return None


def documentation_edit(path, before, after, removed, inserted, maps):
    """An isolated prose change in a module or undecorated top-level function."""
    if not path.startswith("sqlglot/") or not (removed or inserted):
        return None
    first = min(removed | inserted)
    # Module, class and function docstrings at any depth. Only doctests and
    # code-object comparisons read them in the pinned suite.
    candidates = [before] + [n for n in ast.walk(before)
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                             and not n.decorator_list and n.lineno <= first <= n.end_lineno]
    for owner in candidates:
        if not owner.body or not isinstance(owner.body[0], ast.Expr):
            continue
        old = owner.body[0].value
        if not (isinstance(old, ast.Constant) and type(old.value) is str and old.lineno <= first <= old.end_lineno):
            continue
        twin = after if owner is before else next((n for n in ast.walk(after) if type(n) is type(owner)
                   and n.name == owner.name and n.lineno == owner.lineno
                   and n.col_offset == owner.col_offset), None)
        if twin is None or not twin.body or not isinstance(twin.body[0], ast.Expr):
            continue
        new = twin.body[0].value
        if not isinstance(new, ast.Constant) or type(new.value) is not str or new.value == old.value:
            continue
        import doctest
        def examples(doc):
            return [(e.source, e.want, e.exc_msg, e.options) for e in doctest.DocTestParser().get_examples(doc)]
        try:
            if examples(new.value) != examples(old.value):
                continue
        except ValueError:
            continue
        twin.body[0].value = old
        try:
            if same_tree(before, after):
                mask = maps["documentation_readers"] | child_mask(path, maps)
                if owner is not before:
                    mask |= frame_readers_for(path, maps)
                return mask
        finally:
            twin.body[0].value = new
    return None


def removed_parser_members(path, before, after, removed, maps):
    """Recognize removed definitions and deferred dispatch entries on a parser subclass."""
    if not path.startswith("sqlglot/parsers/") or path not in maps["cold_imports"] or not future_annotations(before):
        return None
    remainder = copy(before)
    remainder.body = list(before.body)
    observers = maps["cold_imports"][path]
    changed = False
    for index, cls in enumerate(before.body):
        if not (isinstance(cls, ast.ClassDef) and not cls.decorator_list and not cls.keywords
                and len(cls.bases) == 1 and ast.unparse(cls.bases[0]) == "parser.Parser"
                and all(cls.lineno < line <= cls.end_lineno for line in removed)):
            continue
        owner = path[:-3].replace("/", ".") + "." + cls.name
        if owner not in maps["types"] or maps.get("import_types") is None:
            return None
        observers |= int(maps["types"][owner], 16) | int(maps["import_types"].get(owner, "0"), 16)
        retained = []
        for member in cls.body:
            if (isinstance(member, ast.FunctionDef) and not member.name.startswith("__")
                    and not member.decorator_list and not member.args.defaults and not any(member.args.kw_defaults)
                    and set(range(member.lineno, member.end_lineno + 1)) <= removed):
                changed = True
                continue
            if (isinstance(member, ast.Assign) and isinstance(member.value, ast.Dict)
                    and len(member.targets) == 1 and isinstance(member.targets[0], ast.Name)):
                table = member.value
                keep = []
                for j, (k, v) in enumerate(zip(table.keys, table.values)):
                    if (isinstance(k, ast.Constant) and type(k.value) is str and isinstance(v, ast.Lambda)
                            and not v.args.defaults and not any(v.args.kw_defaults)
                            and set(range(k.lineno, v.end_lineno + 1)) <= removed):
                        continue
                    keep.append(j)
                if len(keep) != len(table.keys):
                    changed = True
                    member = copy(member)
                    member.value = copy(table)
                    member.value.keys = [table.keys[j] for j in keep]
                    member.value.values = [table.values[j] for j in keep]
            retained.append(member)
        replacement = copy(cls)
        replacement.body = retained
        remainder.body[index] = replacement
    if changed and same_tree(remainder, after):
        return observers
    return None


def mask_for_file(path, base, revised, removed, inserted, maps, all_mask, neighbors, isolated=False, widen=True):
    readers = int(maps["files"].get(path, "0"), 16)
    if inert(path):
        return readers | int(maps["source_readers"].get(path, "0"), 16)
    module = path[:-3].replace("/", ".").removesuffix(".__init__") if path.endswith(".py") else None
    if (module and path.startswith("sqlglot/") and maps.get("modules") and module not in maps["modules"]
            and path not in maps.get("child_paths", {})):
        # Never imported by the pinned suite: only source readers and opaque children can notice.
        return readers | int(maps["source_readers"].get(path, "0"), 16) | maps.get("unknown_children", 0)
    if not path.endswith(".py"):
        neighbors.append(readers)
        return fixture_selection(path, base, revised, readers, maps) if readers else all_mask
    try:
        before, after = syntax(base), ast.parse(revised)
        compile(after, path, "exec", dont_inherit=True)
    except SyntaxError:
        # A test module that no longer compiles fails to import, which fails its
        # own tests and those of every test module importing it.
        importers = importer_mask(path, maps) if path.startswith("tests/") else None
        if importers is None:
            raise
        return readers | importers
    scoped_before, scoped_after = edited_function(before, after, base, revised, removed) if not inserted else (before, after)
    readers = int(maps["source_readers"].get(path, "0"), 16)
    if removed:
        original_lines = base.splitlines()
        if any(re.search(r"\bis\b|!=|==", original_lines[n-1]) for n in removed):
            normalize_comparisons(scoped_before, scoped_after, path, maps.get("operands",{}))
    if same_tree(scoped_before, scoped_after):
        return readers
    doc = documentation_edit(path, before, after, removed, inserted, maps)
    if doc is not None:
        return readers | doc
    if widen:
        extra = 0
        if path.startswith("tests/") and outside_test_methods(before, removed | inserted):
            # Module, class and helper code of a test file (constants, setUpClass, helpers that
            # only class setup calls) runs outside the tests that line coverage credits, so the
            # coverage cannot confine the edit: every test of the module and of its importers.
            importers = importer_mask(path, maps)
            extra = importers if importers is not None else all_mask
        elif path.startswith("sqlglot/"):
            # Code that runs once per process (on a cache miss) is credited to one test only.
            extra = once_per_process_mask(path, before, base, removed | inserted, maps)
            if dialect_copied_edit(base, removed | inserted):
                extra = all_mask
        if extra:
            # added to what the other rules select (they still run, so nothing is dropped)
            rest = mask_for_file(path, base, revised, removed, inserted, maps, all_mask, neighbors, isolated, widen=False)
            return readers | rest | extra
    # Module attribute hooks can populate globals once at import. Later callers
    # see the cached value without executing the profiled hook.
    if any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" and
           any(n.lineno <= line <= n.end_lineno for line in removed | inserted)
           for n in before.body):
        return all_mask
    if inserted and not removed:
        local = inert_statement(path, base, before, after, inserted, maps)
        if local is not None:
            return readers | (local & maps["cold_imports"].get(path, all_mask))
    # A one-line replacement outside an annotation cannot be erased by
    # ignoring annotations. Avoid copying and dumping the full module twice.
    check_annotations = True
    if len(removed) == 1 and not inserted and base.count("\n") == revised.count("\n"):
        line = next(iter(removed))
        check_annotations = annotation_on_line(scoped_before, line) or annotation_on_line(scoped_after, line)
    if check_annotations and equal_without_annotations(before, after):
        return int(maps["source_readers"].get(path, "0"),16)
    neighbors.append(0)
    if not inserted:
        default = default_edit(path,scoped_before,scoped_after,removed,maps)
        if default is not None:
            return readers | default
        keywords = keyword_edit(path, before, after, removed, maps)
        if keywords is not None:
            return readers | keywords
        metadata = metadata_callback_edit(path, before, after, removed, maps)
        if metadata is not None:
            return readers | metadata
        field_scope = before is scoped_before or any(isinstance(n, ast.ClassDef) for n in ast.walk(scoped_before))
        refs = load_refs(maps) if field_scope and maps.get("flips") else None
        refined = flag_edit(path,before,after,removed,maps,set(refs["strings"]) if refs else None) if refs else None
        if refined is None:
            refined = membership_edit(path,scoped_before,scoped_after,removed,maps)
        if refined is None:
            refined = copy_edit(path,scoped_before,scoped_after,removed,maps)
        if refined is None:
            refined = lookup_edit(path,scoped_before,scoped_after,removed,maps)
        if refined is None:
            refined = config_edit(path,scoped_before,scoped_after,removed,maps)
        if refined is None:
            refined = module_edit(path,scoped_before,scoped_after,removed,maps)
        if refined is None:
            refined = changed_predicate(path,scoped_before,scoped_after,removed,maps)
        if refined is None:
            refined = callback_edit(path,scoped_before,scoped_after,removed,maps)
        if refined is None:
            refined = conditional_edit(path,scoped_before,scoped_after,removed,maps)
        if refined is None:
            refined = operation_edit(path,scoped_before,scoped_after,removed,maps)
        if refined is None and before is scoped_before:
            refined = lambda_edit(path,before,after,maps)
        if refined is not None:
            return readers | refined
        field = scalar_setting(path, before, after, removed, maps)
        if field is not None:
            return readers | field
    if inserted and not removed:
        added = added_method_receivers(path, before, after, maps)
        if added is not None:
            return readers | added
    selected = coverage_mask(path, base, before, after, removed, inserted, maps, all_mask, neighbors, readers, scoped_before)
    if isolated and selected == all_mask and removed and not inserted and scoped_before is before:
        narrower = removed_parser_members(path, before, after, removed, maps)
        if narrower is not None:
            return readers | narrower
    if not inserted and before is scoped_before:
        bound = binding_mask(path, before, after, maps)
        if bound is not None:
            selected &= bound | readers
    if not inserted:
        operand = boolop_edit(path, scoped_before, scoped_after, removed, inserted, maps)
        if operand is not None:
            selected &= operand | readers
    return selected


def confined(func, before, after):
    """True when the only differences between the trees lie inside func's body."""
    twin = next((n for n in ast.walk(after) if type(n) is type(func) and n.name == func.name
                 and n.lineno == func.lineno and n.col_offset == func.col_offset), None)
    if twin is None:
        return False
    body = twin.body
    twin.body = func.body
    try:
        return same_tree(before, after)
    finally:
        twin.body = body


def insertion_reach(func, number, before, after):
    """Lines whose execution every run reaching an insertion before line number must pass.

    Reaching the inserted code requires starting the statement that ends just
    before it (at any nesting level) or a block header that encloses it.
    """
    if number <= func.body[0].lineno or not confined(func, before, after):
        return None
    statements = [n for n in ast.walk(func) if isinstance(n, ast.stmt) and n is not func]
    ends = [n.end_lineno for n in statements if n.end_lineno < number]
    if not ends:
        return None
    last = max(ends)
    lines = {n.lineno for n in statements if n.end_lineno == last}
    for n in statements:
        if n.lineno < number <= n.end_lineno:
            lines.add(n.lineno)
            if not isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
                                  ast.Try, ast.Match, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                lines.update(range(n.lineno, n.end_lineno + 1))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return None
    return lines


def coverage_mask(path, base, before, after, removed, inserted, maps, all_mask, neighbors, readers, scoped_before):
    rows = maps["lines"].get(path, {})
    imports = maps["imports"].get(path, {})
    selected = readers
    functions = [n for n in ast.walk(scoped_before) if isinstance(n,(ast.FunctionDef, ast.AsyncFunctionDef))]
    source_lines = base.splitlines()
    for number in removed | inserted:
        enclosing = [f for f in functions if f.lineno < number <= f.end_lineno]
        if not enclosing and path.startswith("tests/"):
            # Test module and class bodies run at collection. Their effects reach
            # the module's own tests and those of the modules importing it.
            importers = importer_mask(path, maps)
            if importers is None:
                return all_mask
            selected |= importers
            continue
        if not enclosing:
            classes = sorted((n for n in ast.walk(before) if isinstance(n, ast.ClassDef)
                              and n.lineno < number <= n.end_lineno), key=lambda n:n.lineno)
            # Limit a class policy edit to receivers observed in the baseline.
            # A changed class declaration or unknown receiver gets the full suite.
            if classes and number in removed and path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/expressions/")):
                assignments = (n for n in classes[-1].body if isinstance(n,(ast.Assign,ast.AnnAssign)))
                if any(n.lineno <= number <= n.end_lineno for n in assignments):
                    # New global names or attribute lookups can fail at import, before
                    # any receiver exists. Include every test in that case.
                    def references(tree):
                        return {(type(n), n.id if isinstance(n,ast.Name) else n.attr)
                                for n in ast.walk(tree) if isinstance(n,(ast.Name,ast.Attribute))}
                    if references(after) - references(before):
                        return all_mask
                    owner = ".".join([path[:-3].replace("/", ".")] + [n.name for n in classes])
                    mask = maps.get("types", {}).get(owner)
                    if mask and maps.get("import_types") is not None:
                        # Instances built while modules import (class-level
                        # templates) reach every test importing those modules.
                        selected |= int(mask,16) | int(maps["import_types"].get(owner, "0"), 16)
                        continue
            return all_mask
        func = min(enclosing, key=lambda f:f.end_lineno-f.lineno)
        nearby = 0
        for i in range(func.lineno,func.end_lineno+1):
            nearby |= int(rows.get(str(i), "0"),16)
        neighbors.append(nearby)
        reach = insertion_reach(func, number, before, after) if number in inserted else None
        if reach is not None:
            for line in reach:
                key = str(line)
                selected |= int(rows.get(key, "0"),16) | int(imports.get(key,"0"),16)
        elif number in inserted or number < func.body[0].lineno:
            # Defaults, decorators and evaluated annotations run when the module
            # imports. Only edits that cannot raise there stay within the function.
            if number not in inserted and not signature_safe(func, before, after):
                return all_mask
            callers = default_callers(func, before, after, path, maps) if number not in inserted else None
            if callers is not None:
                selected |= callers | child_mask(path, maps)
                continue
            selected |= nearby
            for i in range(func.lineno,func.end_lineno+1):
                selected |= int(imports.get(str(i),"0"),16)
        else:
            # Include the opening line of multiline expressions. A changed boolean
            # connector can newly reach an operand that the original skipped.
            hits = {number}
            nodes = [n for n in ast.walk(func) if isinstance(n, ast.expr)
                     and n.lineno <= number <= n.end_lineno]
            hits.update(n.lineno for n in nodes)
            if source_lines[number-1].lstrip().startswith(("else:", "finally:", "except ", "except:", "case ")):
                selected |= nearby
            for line in hits:
                key = str(line)
                selected |= int(rows.get(key, "0"),16) | int(imports.get(key,"0"),16)
    return selected


def rebalance(workers, loads, floor, rounds=100):
    """Shift load off the busiest worker by moving one test or swapping a pair.

    Each step lowers the larger load of the two workers involved, so the
    busiest load never rises. The set of scheduled tests is unchanged.
    """
    for _ in range(rounds):
        b = loads.index(max(loads))
        if loads[b] <= floor:
            return
        best = None  # (new pair maximum, j, index in b or None, index in j or None)
        for j in range(len(loads)):
            gap = loads[b] - loads[j]
            if j == b or gap <= 1:
                continue
            # Trade a test of duration t on b for one of duration u on j (u = 0
            # is a plain move). The pair improves when 0 < t - u < gap and is
            # most even when t - u is near gap / 2.
            options = sorted((ms, k) for k, (ms, _) in enumerate(workers[j]))
            options.insert(0, (0, None))
            sizes = [ms for ms, _ in options]
            for i, (t, _) in enumerate(workers[b]):
                at = bisect.bisect_left(sizes, t - gap / 2)
                for pos in (at - 1, at):
                    if 0 <= pos < len(options):
                        delta = t - sizes[pos]
                        if 0 < delta < gap:
                            peak = max(loads[b] - delta, loads[j] + delta)
                            if best is None or peak < best[0]:
                                best = (peak, j, i, options[pos][1])
        if best is None:
            return
        _, j, i, k = best
        item = workers[b].pop(i)
        loads[b] -= item[0]
        if k is not None:
            other = workers[j].pop(k)
            workers[b].append(other)
            loads[b] += other[0]
            loads[j] -= other[0]
        workers[j].append(item)
        loads[j] += item[0]


def schedule(tests):
    if not tests:
        return []
    best_cost = float("inf")
    ordered = sorted(tests, reverse=True)
    best = []
    total = sum(ms for ms, _ in ordered)
    largest = ordered[0][0]
    for count in range(1,min(8,len(tests))+1):
        floor = max(largest, (total + count - 1) // count)
        if floor + 2000 * count >= best_cost:
            continue
        workers = [[] for _ in range(count)]
        loads = [0]*count
        for item in ordered:
            i = loads.index(min(loads))
            workers[i].append(item)
            loads[i] += item[0]
        rebalance(workers, loads, floor)
        cost = max(loads) + 2000*sum(bool(w) for w in workers)
        if cost < best_cost:
            best_cost, best = cost, workers
    return [[number for _, number in w] for w in best if w]


def schedule_with_margin(mask, nearby, numbered, duration, widen):
    required = [t for bit,t in numbered if bit is None or bit & mask]
    workers = schedule(required)
    if not widen:
        return workers
    if not workers:
        return []
    loads = [sum(duration[n] for n in w) for w in workers]
    ceiling = max(loads)
    optional = sorted((not bool(bit & nearby), ms, n)
                      for bit,(ms,n) in numbered if bit is not None and not bit & mask)
    for _,ms,n in optional:
        i = loads.index(min(loads))
        if loads[i] + ms <= ceiling:
            workers[i].append(n)
            loads[i] += ms
    return workers


# Wall seconds the TPC-DS replays may take across the batch, the fixed CPU
# cost of one replay (imports and table loading), and a margin for a slower
# scoring machine. A replay runs only when its estimate is at most half the
# simulated seconds that dropping the test saves.
TPCDS_BUDGET = 45.0
TPCDS_OVERHEAD = 3.0
TPCDS_SLOWDOWN = 1.3


def plan_cost(mask, numbered):
    plan = schedule([t for bit, t in numbered if bit is None or bit & mask])
    lookup = {n: ms for _, (ms, n) in numbered}
    return 2000 * len(plan) + max((sum(lookup[n] for n in w) for w in plan), default=0)


def drop_tpcds(mask, files, skipped, file_masks, maps, numbered, bit, left):
    """Clear the TPC-DS bit when a replay of the queries the patch reaches passes."""
    profile = maps["tpcds"]
    if any(file_masks.get(path, bit) & bit for path in skipped):
        return mask, left
    found = tpcds_replay.affected_queries(profile, files, file_masks, bit)
    if found is not None and not found[0]:
        found = tpcds_replay.affected_queries(profile, files, file_masks, bit, granular=False)
    if found is None or not found[0]:
        return mask, left
    queries, reached = found
    order = [i for i in range(profile["count"]) if queries >> i & 1]
    estimate = TPCDS_OVERHEAD + TPCDS_SLOWDOWN * sum(profile["cpu"][i] for i in order)
    gain = (plan_cost(mask, numbered) - plan_cost(mask & ~bit, numbered)) / 1000
    if estimate > gain / 2 or estimate > left:
        return mask, left
    start = time.monotonic()
    passed = tpcds_replay.passes(SOURCE, files, order, reached, profile, min(left, 2 * estimate + 5))
    left -= time.monotonic() - start
    import sys
    print(f"tpcds replay: {len(order)} queries, {'passed' if passed else 'kept'}", file=sys.stderr)
    return (mask & ~bit if passed else mask), left


def main(printed):
    with gzip.open(APP/"coverage.json.gz", "rt") as f:
        maps = json.load(f)
    with gzip.open(APP/"operands.gz", "rt") as f:
        maps["operands"] = json.load(f)
    with gzip.open(APP/"fields.gz", "rt") as f:
        profile = json.load(f)
    maps["fields"] = profile["fields"] if profile["names"] == maps["names"] else {}
    with gzip.open(APP/"imports.gz", "rt") as f:
        imports = json.load(f)
    if imports["names"] != maps["names"]:
        raise ValueError("inconsistent build-time test inventory")
    maps["imports"] = imports["lines"]
    profiles = [
        ("arguments.gz", "sites", "arguments"), ("config.gz", "sites", "config"), ("predicates.gz", "values", "predicates"),
        ("cold-orders.gz", "orders", "cold_orders"),
        ("globals.gz", "sites", "globals"), ("fixture-attrs.gz", "sites", "fixture_attributes"),
        ("lookups.gz", "sites", "lookups"), ("fixture-modes.gz", "sites", "fixture_modes"),
        ("copies.gz", "sites", "copies"), ("order.gz", "sites", "orders"),
    ]
    for file, source, target in profiles:
        with gzip.open(APP/file,"rt") as f:
            profile = json.load(f)
        if profile["names"] != maps["names"]:
            raise ValueError("inconsistent build-time test inventory")
        maps[target] = profile[source]
        if target == "orders":
            maps["unmeasured_orders"] = int(profile["unmeasured"],16)
        if "external" in profile:
            maps["unknown_children"] = int(profile.get("unknown",profile["external"]),16)
            maps["child_paths"] = profile.get("child_paths",{})
    with gzip.open(APP/"branches.gz","rt") as f:
        branch = json.load(f)
    if branch.pop("names") != maps["names"]:
        raise ValueError("branch observations use a different test inventory")
    maps["unmeasured_branch"] = int(branch["unmeasured"], 16)
    maps["branch_orders"] = branch["orders"]
    maps["choices"] = branch["choices"]
    with gzip.open(APP/"instructions.gz","rt") as f:
        maps["instructions"] = {path:set(keys) for path,keys in json.load(f).items()}
    index = {name:i for i,name in enumerate(maps["names"])}
    maps["test_files"] = {}
    for name, i in index.items():
        file = name.split("::")[0]
        maps["test_files"][file] = maps["test_files"].get(file, 0) | (1 << i)
    with gzip.open(APP/"typing-calls.gz", "rt") as f:
        callbacks = json.load(f)
    if callbacks["names"] != maps["names"]:
        raise ValueError("typing callback inventory mismatch")
    maps["typing_calls"] = callbacks["sites"]
    maps["metadata_readers"] = 0
    for file in json.loads((APP/"metadata-readers.json").read_text()):
        maps["metadata_readers"] |= maps["test_files"].get(file, 0)
    with gzip.open(APP/"words.gz", "rt") as f:
        word_profile = json.load(f)
    if word_profile.pop("names") != maps["names"]:
        raise ValueError("tokenizer lookup inventory mismatch")
    maps["word_profile"] = word_profile
    maps["fixture_fields"] = json.loads((APP/"fixture-fields.json").read_text())
    maps["frame_readers"] = 0
    maps["memory_readers"] = 0
    maps["documentation_readers"] = 0
    maps["frame_reader_files"] = {}
    for file, kinds in json.loads((APP/"frame-readers.json").read_text()).items():
        for kind in kinds:
            maps[kind] |= maps["test_files"].get(file, 0)
            if kind == "frame_readers":
                maps["frame_reader_files"][file] = maps["test_files"].get(file, 0)
    try:
        maps["frame_scopes"] = json.loads((APP/"frame-scope.json").read_text())
    except (OSError, ValueError):
        maps["frame_scopes"] = {}
    defaults = json.loads((APP/"default-flow.json").read_text())
    maps["default_sites"] = defaults["sites"]
    maps["default_observers"] = maps["frame_readers"]
    for file, lines in defaults["observers"].items():
        for line in lines:
            for table in (maps["lines"], maps["imports"]):
                maps["default_observers"] |= int(table.get(file, {}).get(str(line),"0"),16)
    maps["refs_path"] = APP/"refs.json.gz"
    try:
        with gzip.open(APP/"boolops.gz", "rt") as f:
            profile = json.load(f)
        maps["boolops"] = profile if profile.pop("names") == maps["names"] else None
    except (OSError, ValueError, KeyError):
        maps["boolops"] = None
    try:
        with gzip.open(APP/"members.gz", "rt") as f:
            profile = json.load(f)
        maps["members"] = profile if profile.pop("names") == maps["names"] else None
    except (OSError, ValueError, KeyError):
        maps["members"] = None
    try:
        with gzip.open(APP/"flips.gz", "rt") as f:
            profile = json.load(f)
        maps["flips"] = profile["sites"] if profile["names"] == maps["names"] else None
    except (OSError, ValueError, KeyError):
        maps["flips"] = None
    if maps["flips"] is not None:
        with gzip.open(APP/"forward.gz", "rt") as f:
            forwarded = json.load(f)
        if forwarded["names"] != maps["names"]:
            raise ValueError("forwarded settings inventory mismatch")
        for site, mask in forwarded["sites"].items():
            if site in maps["flips"]:
                maps["flips"][site] = hex(int(mask, 16) & int(maps["flips"][site], 16))
    try:
        with gzip.open(APP/"import_types.gz", "rt") as f:
            profile = json.load(f)
        maps["import_types"] = profile["types"] if profile["names"] == maps["names"] else None
    except (OSError, ValueError, KeyError):
        maps["import_types"] = None
    try:
        maps["modules"] = set(json.loads((APP/"modules.json").read_text()))
    except (OSError, ValueError):
        maps["modules"] = None
    with gzip.open(APP/"cold-imports.gz", "rt") as f:
        cold = json.load(f)
    if cold["names"] != maps["names"]:
        raise ValueError("cold import inventory mismatch")
    maps["cold_imports"] = {}
    for path, bits in cold["sites"].items():
        mask = int(bits, 16) | child_mask(path, maps)
        mask |= int(maps["source_readers"].get(path, "0"), 16)
        # Augment isolation with evidence from the ordered full suite, in
        # case an earlier test initialized a cached object for a later one.
        for value in maps["lines"].get(path, {}).values():
            mask |= int(value, 16)
        maps["cold_imports"][path] = mask
    with gzip.open(APP/"fixture-paths.gz", "rt") as f:
        maps["fixture_paths"] = json.load(f)
    if maps["fixture_paths"]["names"] != maps["names"]:
        raise ValueError("fixture path inventory mismatch")
    tests = json.loads((DATA/"tests.json").read_text())["tests"]
    all_mask = (1 << len(index)) - 1
    numbered = [(1 << index[t["id"]] if t["id"] in index else None,
                 (round(1000*t["seconds"]), t["number"])) for t in tests]
    duration = {n:ms for _,(ms,n) in numbered}
    by_number = {test["number"]:test["id"] for test in tests}
    maps["tpcds"] = tpcds_replay.load(APP/"tpcds.gz")
    tpcds_id = maps["tpcds"]["test"] if maps["tpcds"] else None
    tpcds_bit = 1 << index[tpcds_id] if tpcds_id in index and any(t["id"] == tpcds_id for t in tests) else 0
    if maps["tpcds"] and maps["tpcds"]["count"] != len(maps["tpcds"]["cpu"]):
        tpcds_bit = 0
    tpcds_left = float(os.environ.get("SCREEN_TPCDS_BUDGET", TPCDS_BUDGET))
    schedules = {}
    remaining = float(os.environ.get("SCREEN_REPLAY_BUDGET", "22.0"))
    for patch in sorted((DATA/"commits").glob("*.patch")):
        mask = 0
        neighbors = []
        files = []
        skipped = set()
        file_masks = {}
        analysed = False
        try:
            diff = patch.read_text(encoding="utf8")
            structural = fixture_structure(diff, maps, all_mask)
            if structural is not None:
                required = [pair for bit, pair in numbered if bit is None or bit & structural]
                emit(printed, patch.stem, schedule(required))
                continue
            files = list(changes(diff, skipped))
            if not files and not skipped:
                raise ValueError("missing diff")
            for path in skipped:
                file_masks[path] = int(maps["files"].get(path, "0"), 16) | int(maps["source_readers"].get(path, "0"), 16)
                mask |= file_masks[path]
            for args in files:
                try:
                    file_masks[args[0]] = mask_for_file(*args,maps,all_mask,neighbors,isolated=len(files) == 1 and not skipped)
                except SyntaxError:
                    if len(files) != 1 or skipped or args[0] not in maps["cold_imports"]:
                        raise
                    file_masks[args[0]] = maps["cold_imports"][args[0]]
                mask |= file_masks[args[0]]
            if neighbors:
                mask |= int(maps.get("opaque","0"),16)
            analysed = True
        except Exception:
            mask = all_mask
        if analysed and files and mask != all_mask and tpcds_bit and mask & tpcds_bit:
            try:
                mask, tpcds_left = drop_tpcds(mask, files, skipped, file_masks, maps, numbered, tpcds_bit, tpcds_left)
            except Exception:
                import sys, traceback
                traceback.print_exc(file=sys.stderr)
        nearby = 0
        for value in neighbors:
            nearby |= value
        key = mask, nearby, bool(neighbors)
        if key not in schedules:
            schedules[key] = schedule_with_margin(mask, nearby, numbered, duration, bool(neighbors))
        workers = schedules[key]
        if (remaining > 1 and workers and files and all(p.startswith("sqlglot/") and p.endswith(".py") for p,*_ in files)):
            ids = [n for bit,(_,n) in numbered if bit is None or bit & mask]
            replayable = ("tests/dialects/", "tests/test_parser.py::", "tests/test_transpile.py::",
                          "tests/test_expressions.py::", "tests/test_generator.py::", "tests/test_typing.py::",
                          "tests/test_serde.py::", "tests/test_tokens.py::", "tests/test_build.py::")
            eligible = [n for n in ids if by_number[n].startswith(replayable)]
            retained = [n for n in ids if not by_number[n].startswith(replayable)]
            if eligible and len(eligible) <= 120 and sum(duration[n] for n in eligible) <= 2200:
                if retained:
                    def cost(plan):
                        return 2000 * len(plan) + max((sum(duration[n] for n in w) for w in plan), default=0)
                    if cost(workers) - cost(schedule([(duration[n],n) for n in retained])) < 650:
                        eligible = []
            else:
                eligible = []
            if eligible:
                ids = eligible
                extras = sorted((ms,n) for bit,(ms,n) in numbered if bit is not None
                                and bit & nearby and not bit & mask and by_number[n].startswith("tests/dialects/"))
                spare = 250
                for ms,n in extras:
                    if ms > spare or len(ids) >= 48:
                        break
                    ids.append(n)
                    spare -= ms
                start = time.monotonic()
                failed = failing_tests(SOURCE,files,[by_number[n] for n in ids],min(2.4,remaining))
                if failed is not None:
                    required = set(retained) | {n for n in ids if by_number[n] in failed}
                    if retained:
                        reduced = sum(bit for bit, (_, n) in numbered if bit is not None and n in required)
                        workers = schedule_with_margin(reduced, nearby, numbered, duration, True)
                    else:
                        workers = schedule([(duration[n], n) for n in required])
                remaining -= time.monotonic()-start
        emit(printed, patch.stem, workers)


def emit(printed, commit, workers):
    if commit not in printed:
        printed.add(commit)
        print(json.dumps(dict(commit=commit, workers=workers),separators=(",",":")), flush=True)


def fallback(printed):
    """Run every test for any commit left unanswered by an unexpected failure."""
    tests = json.loads((DATA/"tests.json").read_text())["tests"]
    workers = schedule([(round(1000*t["seconds"]), t["number"]) for t in tests])
    for patch in sorted((DATA/"commits").glob("*.patch")):
        emit(printed, patch.stem, workers)


if __name__ == "__main__":
    answered = set()
    try:
        main(answered)
    except BaseException as error:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        fallback(answered)

