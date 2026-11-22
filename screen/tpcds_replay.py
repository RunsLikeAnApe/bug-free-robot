"""Decide whether test_execute_tpcds can fail, by replaying only the queries an edit reaches.

The build-time profile (tpcds_map.py) lists, per TPC-DS query, every sqlglot
line its pipeline executes, and separately every line executed while the test
module imports or setUpClass runs.

A patch is eligible only when every Python edit lies inside function bodies
and none of those functions runs during import or setup. Queries that execute
no line of an edited function behave as on the baseline, which passes. The
others run in a forked child against a private tree holding the patched
modules, exactly as the test runs them. Any error, mismatch, timeout or
incomplete run keeps the test.
"""
import ast
from tree_equal import same_tree
import gzip
import json
import os
import pickle
import signal
import sys
import tempfile
import multiprocessing
from pathlib import Path

from replay import overlay


def load(path):
    try:
        with gzip.open(path, "rt") as f:
            profile = json.load(f)
    except (OSError, ValueError):
        return None
    profile["setup"] = {p: set(ns) for p, ns in profile["setup"].items()}
    profile["lines"] = {p: {int(n): int(m, 16) for n, m in row.items()} for p, row in profile["lines"].items()}
    return profile


def outer_definitions(body):
    """Functions not nested in other functions, in source order."""
    found = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(node)
        elif isinstance(node, ast.ClassDef):
            found.extend(outer_definitions(node.body))
        else:
            for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    found.extend(outer_definitions(inner))
    return found


def edited_bodies(base, revised):
    """Baseline functions whose bodies hold every difference, or None.

    None means some difference lies outside function bodies (module or class
    level code, signatures, decorators) or the files do not parse.
    """
    try:
        before, after = ast.parse(base), ast.parse(revised)
    except SyntaxError:
        return None
    if same_tree(before, after):
        return []
    olds, news = outer_definitions(before.body), outer_definitions(after.body)
    if len(olds) != len(news):
        return None
    edited, saved = [], []
    for old, new in zip(olds, news):
        if type(old) is not type(new) or old.name != new.name:
            return None
        if (not same_tree(old, new)):
            edited.append(old)
            saved.append((new, new.body))
            new.body = old.body
    try:
        confined = same_tree(before, after)
    finally:
        for new, body in saved:
            new.body = body
    return edited if confined else None


def statement_lines(func, number):
    """Lines of the innermost statement of func holding line number, or None.

    A query that executed none of those lines never ran the statement, so an
    edit inside it cannot change what that query does. For a compound
    statement edited in its header (an if or while test, a for iterable, a
    with item) only the header lines count, since the header runs whenever
    the statement does. An edit to a nested definition's signature or
    decorators returns None.
    """
    best = None
    for node in ast.walk(func):
        if node is not func and isinstance(node, ast.stmt) and node.lineno <= number <= node.end_lineno:
            if best is None or node.end_lineno - node.lineno <= best.end_lineno - best.lineno:
                best = node
    if best is None or isinstance(best, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    body = getattr(best, "body", None)
    if isinstance(body, list) and body and number < body[0].lineno:
        return range(best.lineno, body[0].lineno)
    return range(best.lineno, best.end_lineno + 1)


def reach(func, first, removed, inserted, granular=True):
    """Baseline lines whose execution any query affected by edits to func must show."""
    whole = range(first, func.end_lineno + 1)
    if not granular or any(first <= n <= func.end_lineno + 1 for n in inserted):
        return whole
    lines = set()
    for number in removed:
        if first <= number <= func.end_lineno:
            found = statement_lines(func, number)
            if found is None:
                return whole
            lines.update(found)
    return sorted(lines) if lines else whole


def affected_queries(profile, files, other_masks, bit, granular=True):
    """(bitmask of TPC-DS queries the patch can reach, paths they reach), or None when ineligible.

    files are (path, base, revised, removed, inserted) tuples. other_masks maps
    each non-Python path to the selector's test mask for it.
    """
    queries = 0
    reached = set()
    for path, base, revised, removed, inserted in files:
        if not path.endswith(".py"):
            if other_masks.get(path, bit) & bit:
                return None
            continue
        if not path.startswith("sqlglot/"):
            return None
        edited = edited_bodies(base, revised)
        if edited is None:
            return None
        setup = profile["setup"].get(path, set())
        rows = profile["lines"].get(path, {})
        for func in edited:
            body_start = func.body[0].lineno
            if any(n in setup for n in range(body_start, func.end_lineno + 1)):
                return None
            first = min([func.lineno] + [d.lineno for d in func.decorator_list])
            for n in reach(func, first, removed, inserted, granular):
                if rows.get(n, 0):
                    queries |= rows[n]
                    reached.add(path)
    return queries, reached


def passes(source, files, queries, reached, profile, seconds):
    """True only when every listed query completes and matches DuckDB on the patched tree."""
    try:
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)
            overlay(source, dest, {path: new for path, old, new, *_ in files})
            order = sorted(queries, key=lambda i: profile["cpu"][i])
            limits = [3 * profile["cpu"][i] + 1.0 for i in order]
            proc = multiprocessing.get_context("fork").Process(
                target=run, args=(dest, [p for p, *_ in files if p.endswith(".py")], sorted(reached), order, limits))
            proc.start()
            proc.join(seconds)
            if proc.is_alive():
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    proc.kill()
                proc.join()
            if proc.exitcode != 0:
                return False
            result = json.loads((dest / "tpcds-result.json").read_text())
            return result == {"done": sorted(order), "failed": False}
    except (OSError, ValueError, KeyError, TypeError):
        return False


class Timeout(Exception):
    pass


def loaded_from(root, path):
    """True, False when imported from elsewhere, or None when not imported."""
    name = path[:-3].replace("/", ".").removesuffix(".__init__")
    module = sys.modules.get(name)
    if module is None:
        return None
    loaded = getattr(module, "__file__", None)
    return bool(loaded) and Path(loaded).resolve() == (root / path).resolve()


def run(root, paths, reached, order, limits):
    os.setsid()
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    os.chdir(root)
    for name in [m for m in sys.modules if m.split(".")[0] in ("sqlglot", "tests")]:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONPATH=str(root), SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1")

    def expire(*_):
        raise Timeout()
    signal.signal(signal.SIGALRM, expire)

    import tests.test_executor as te
    cls = te.TestExecutor
    if getattr(cls, "__unittest_skip__", False):
        sys.exit(1)
    cls.setUpClass()
    te.initializer(te.TPCDS_SCHEMA, pickle.loads(pickle.dumps(cls.tpcds_tables)))
    case = cls("test_execute_tpcds")
    # Lazily loaded modules import during the queries, so they are checked after.
    if any(loaded_from(root, path) is False for path in paths):
        sys.exit(1)
    done, failed = [], False
    for i, limit in zip(order, limits):
        args, sql, _ = cls.tpcds_sqls[i]
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            expression = te.parse_one(sql)
            expression, args = pickle.loads(pickle.dumps((expression, args)))
            table = te.mp_execute(expression, args)
            table = pickle.loads(pickle.dumps(table))
            if table is not None:
                case.subtestHelper(i, table, tpch=False)
        except BaseException:
            failed = True
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        done.append(i)
        if failed:
            break
    if any(loaded_from(root, path) is False for path in paths):
        sys.exit(1)
    if not failed and any(loaded_from(root, path) is not True for path in reached):
        sys.exit(1)
    (root / "tpcds-result.json").write_text(json.dumps({"done": sorted(done), "failed": failed}))
