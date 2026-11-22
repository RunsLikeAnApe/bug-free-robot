"""Certify immutable falsy defaults whose value is discarded at every use."""
import ast
import importlib
import inspect
import json
import sys
from collections import Counter
from functools import cache
from pathlib import Path


def build(root):
    sys.path.insert(0, str(root))
    funcs = {}
    module_names = {}
    overwrites = set()
    definitions = Counter()
    observers = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if not relative.startswith(("tests/", "sqlglot/")):
            continue
        tree = ast.parse(path.read_bytes())
        parents = {child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        module = relative[:-3].replace("/", ".").removesuffix(".__init__")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] += 1
            elif isinstance(node, ast.Attribute):
                if not isinstance(node.ctx, ast.Load):
                    overwrites.add(node.attr)
                if node.attr in {"signature","getfullargspec","__defaults__","__kwdefaults__","__func__"}:
                    parent = parents.get(node)
                    access = parents.get(parent)
                    if (isinstance(parent, ast.Call) and isinstance(access, ast.Attribute) and
                            ((node.attr == "getfullargspec" and access.attr == "args") or
                             (node.attr == "signature" and access.attr == "parameters" and
                              isinstance(parents.get(access),ast.Call) and
                              getattr(parents[access].func,"id",None) == "enumerate"))):
                        continue
                    observers.setdefault(relative,[]).append(node.lineno)
            elif isinstance(node, ast.Constant) and isinstance(node.value,str):
                # A reference by string can be used in patch.object, getattr or a registry.
                tail = node.value.rsplit(".",1)[-1]
                if tail.isidentifier():
                    overwrites.add(tail)
        if not relative.startswith("sqlglot/"):
            continue
        module_names[module] = relative
        for top in tree.body:
            if isinstance(top, ast.ClassDef) and (top.decorator_list or top.keywords):
                continue
            for owner,fn in ([(top.name, n) for n in top.body] if isinstance(top,ast.ClassDef) else [(None,top)]):
                if isinstance(fn,ast.FunctionDef):
                    funcs[(module, owner, fn.name)] = (fn, parents)

    @cache
    def fixed_classmethod(module, owner, name):
        try:
            cls = getattr(importlib.import_module(module), owner)
            if type(cls) is not type or not isinstance(inspect.getattr_static(cls,name), classmethod):
                return False
            method = inspect.getattr_static(cls,name)
            pending = list(cls.__subclasses__())
            seen = set()
            while pending:
                child = pending.pop()
                if child in seen:
                    continue
                if type(child) is not type or inspect.getattr_static(child,name) is not method:
                    return False
                seen.add(child)
                pending.extend(child.__subclasses__())
            return True
        except (AttributeError, ImportError, TypeError):
            return False

    @cache
    def fixed_function(module, name):
        try:
            fn = getattr(importlib.import_module(module), name)
            source, _ = funcs[(module,None,name)]
            return (inspect.isfunction(fn) and fn.__module__ == module and
                    fn.__code__.co_firstlineno == source.lineno)
        except (AttributeError, ImportError, KeyError):
            return False

    @cache
    def safe(key, name, depth):
        if depth > 3:
            return False
        module, owner, _ = key
        fn,parents = funcs[key]
        if fn.decorator_list and not (len(fn.decorator_list)==1 and
            getattr(fn.decorator_list[0],"id",None) in {"classmethod","staticmethod"}):
            return False
        args = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
        imports = {}
        nodes = [n for stmt in fn.body for n in ast.walk(stmt)]
        for n in nodes:
            if isinstance(n, (ast.FunctionDef,ast.ClassDef,ast.Lambda)) or (
                isinstance(n,ast.Name) and n.id in {"locals","exec","eval","vars","dir","super","inspect"}):
                return False
            if isinstance(n,ast.ImportFrom) and not n.level and n.module:
                for a in n.names:
                    imports[a.asname or a.name] = (n.module,None,a.name)
        for n in nodes:
            if not isinstance(n,ast.Name) or n.id!=name or not isinstance(n.ctx,ast.Load):
                continue
            parent = parents[n]
            if ((isinstance(parent,(ast.If,ast.While,ast.IfExp)) and parent.test is n) or
                (isinstance(parent,ast.UnaryOp) and isinstance(parent.op,ast.Not)) or
                (isinstance(parent,ast.BoolOp) and isinstance(parent.op,ast.Or) and parent.values[-1] is not n)):
                continue
            call = parents[parent] if isinstance(parent,ast.keyword) else parent
            if not isinstance(call,ast.Call):
                return False
            target = call.func
            offset = 0
            if isinstance(target,ast.Name) and target.id in imports and target.id not in args and not any(
                isinstance(x,ast.Name) and x.id==target.id and not isinstance(x.ctx,ast.Load) for x in nodes):
                dest = imports[target.id]
                if not fixed_function(dest[0],dest[2]):
                    return False
            elif (isinstance(target,ast.Attribute) and owner and isinstance(target.value,ast.Name) and
                  args and target.value.id==args[0] and fn.decorator_list and
                  getattr(fn.decorator_list[0],"id",None)=="classmethod" and
                  definitions[target.attr]==1 and fixed_classmethod(module,owner,target.attr)):
                dest = (module,owner,target.attr)
                offset = 1
            else:
                return False
            if dest not in funcs or dest[-1] in overwrites:
                return False
            target_fn = funcs[dest][0]
            slots = target_fn.args.posonlyargs + target_fn.args.args
            if isinstance(parent,ast.keyword):
                arg = parent.arg
            elif n in call.args and call.args.index(n)+offset < len(slots) and all(not isinstance(v,ast.Starred) for v in call.args):
                arg = slots[call.args.index(n)+offset].arg
            else:
                return False
            if not arg or arg not in [a.arg for a in slots+target_fn.args.kwonlyargs] or not safe(dest,arg,depth+1):
                return False
        return True

    sites = {}
    for key,(fn,_) in funcs.items():
        args = fn.args.posonlyargs+fn.args.args
        for a,v in [*zip(args[len(args)-len(fn.args.defaults):],fn.args.defaults),
                    *zip(fn.args.kwonlyargs,fn.args.kw_defaults)]:
            if isinstance(v,ast.Constant) and not v.value and safe(key,a.arg,0):
                sites[f"{module_names[key[0]]}:{fn.lineno}:{a.arg}"] = True
    return dict(sites=sites,observers=observers)


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1]))))
