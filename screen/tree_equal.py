"""Exact comparison of two syntax trees, equal to ast.dump(a) == ast.dump(b) without the text.

ast.dump writes the class name and, for every field, name=value; it leaves out a field that
is missing, None or an empty list (on Constant and MatchSingleton only a field whose class
default is None, such as kind). Positions are not written. The comparison walks both trees
by the same rules and stops at the first difference."""
import ast
import os

_AST = ast.AST
_SHOW = (ast.Constant, ast.MatchSingleton)
_PLAIN = (str, int, bytes, bool, type(None))
_CHECK = os.environ.get("SCREEN_CHECK_EQUAL") == "1"


def _same(a, b):
    stack = [(a, b)]
    while stack:
        a, b = stack.pop()
        if isinstance(a, _AST):
            cls = type(a)
            if cls is not type(b):
                if not isinstance(b, _AST) or cls.__name__ != type(b).__name__ or a._fields != b._fields:
                    return False
            shown = isinstance(a, _SHOW)
            for name in a._fields:
                x = getattr(a, name, None)
                y = getattr(b, name, None)
                if x is y:
                    if x is None and shown and getattr(cls, name, ...) is not None and (
                            hasattr(a, name) != hasattr(b, name)):
                        return False
                    continue
                if shown:
                    # None is written unless the class default of the field is None
                    if x is None or y is None:
                        if getattr(cls, name, ...) is None or hasattr(a, name) != hasattr(b, name):
                            return False
                else:
                    if x is None or (type(x) is list and not x):
                        if y is None or (type(y) is list and not y):
                            continue
                        return False
                    if y is None or (type(y) is list and not y):
                        return False
                stack.append((x, y))
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                return False
            stack.extend(zip(a, b))
        elif isinstance(b, (_AST, list)):
            return False
        elif type(a) is type(b) and type(a) in _PLAIN:
            if a != b:
                return False
        elif repr(a) != repr(b):
            return False
    return True


def same_tree(a, b):
    """ast.dump(a) == ast.dump(b)"""
    if not isinstance(a, _AST) or not isinstance(b, _AST):
        return ast.dump(a) == ast.dump(b)  # as before (ast.dump raises for anything else)
    result = _same(a, b)
    if _CHECK and result != (ast.dump(a) == ast.dump(b)):
        raise AssertionError("same_tree differs from ast.dump")
    return result
