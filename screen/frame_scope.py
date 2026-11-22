"""List the source files whose code objects each frame-inspecting test module can reach.

tests/test_typing.py compares the code of "annotator" callables between the
EXPRESSION_METADATA tables of sqlglot.typing modules. It reads those modules'
sources and inspects only callables reachable from those tables, so edits
elsewhere cannot change what it observes. Test modules without an entry here
are treated as able to inspect any code.

usage: frame_scope.py SOURCE_ROOT > frame-scope.json
"""
import importlib
import json
import pkgutil
import sys
from pathlib import Path


def callables(value, seen):
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        for item in value.values():
            yield from callables(item, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from callables(item, seen)
    elif hasattr(value, "__code__"):
        yield value
        for cell in getattr(value, "__closure__", None) or ():
            try:
                yield from callables(cell.cell_contents, seen)
            except ValueError:
                pass


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    import sqlglot.typing
    package = Path(sqlglot.typing.__file__).parent
    files = {p.relative_to(root).as_posix() for p in package.glob("*.py")}
    seen = set()
    for info in pkgutil.iter_modules([str(package)]):
        module = importlib.import_module(f"sqlglot.typing.{info.name}")
        for function in callables(getattr(module, "EXPRESSION_METADATA", {}), seen):
            source = Path(function.__code__.co_filename).resolve()
            try:
                files.add(source.relative_to(root).as_posix())
            except ValueError:
                pass
    json.dump({"tests/test_typing.py": sorted(files)}, sys.stdout)


if __name__ == "__main__":
    main()
