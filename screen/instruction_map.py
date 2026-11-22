"""Locate operations for which the pinned CPython emits an instruction on the edited line."""
import ast
import dis
import gzip
import json
from pathlib import Path
from types import CodeType

OPCODES = {
    "COMPARE_OP": "Compare", "IS_OP": "Compare", "CONTAINS_OP": "Compare",
    "BINARY_OP": "BinOp", "UNARY_NEGATIVE": "UnaryOp", "UNARY_INVERT": "UnaryOp",
    "UNARY_NOT": "UnaryOp", "CALL": "Call", "CALL_KW": "Call",
}


def instruction_sites(code):
    for inst in dis.get_instructions(code):
        kind = OPCODES.get(inst.opname)
        pos = inst.positions
        if kind and pos.lineno and pos.lineno == pos.end_lineno:
            yield (kind, pos.lineno, pos.col_offset, pos.end_col_offset)
    for item in code.co_consts:
        if isinstance(item, CodeType):
            yield from instruction_sites(item)


def build(root, output):
    data = {}
    for path in sorted((root/"sqlglot").rglob("*.py")):
        filename = path.relative_to(root).as_posix()
        source = path.read_bytes()
        tree = ast.parse(source,filename)
        bytecode = set(instruction_sites(compile(tree,filename,"exec",dont_inherit=True)))
        sites = sorted({f"{type(node).__name__}:{node.lineno}:{node.col_offset}:{node.end_col_offset}"
                        for node in ast.walk(tree) if isinstance(node,(ast.Compare,ast.BinOp,ast.UnaryOp,ast.Call))
                        and node.lineno == node.end_lineno and
                        (type(node).__name__,node.lineno,node.col_offset,node.end_col_offset) in bytecode})
        if sites:
            data[filename] = sites
    with gzip.open(output,"wt") as f:
        json.dump(data,f,separators=(",",":"))


if __name__ == "__main__":
    import sys
    build(Path(sys.argv[1]), Path(sys.argv[2]))
