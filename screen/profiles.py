"""Narrow a selection when a complete syntax-tree comparison isolates one edit."""
import ast
from tree_equal import same_tree
import operator
import json
import math
from functools import cache

OPS = {ast.Eq:operator.eq, ast.NotEq:operator.ne, ast.Lt:operator.lt, ast.LtE:operator.le,
       ast.Gt:operator.gt, ast.GtE:operator.ge}


@cache
def baseline_dump(tree):
    return ast.dump(tree)


def spanning_nodes(root, first, last):
    pending = [root]
    while pending:
        node = pending.pop()
        if hasattr(node, "lineno") and (node.lineno > first or node.end_lineno < last):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def child_mask(path, maps):
    return maps["unknown_children"] | int(maps["child_paths"].get(path,"0"),16)


def unchanged_with_original(root, expected, node, replacements):
    saved = {attr:getattr(node,attr) for attr in replacements}
    try:
        for attr,value in replacements.items():
            setattr(node,attr,value)
        return same_tree(expected, root)
    finally:
        for attr,value in saved.items():
            setattr(node,attr,value)


def scalar_key(value):
    if type(value) is int and value.bit_length() <= 64:
        return "i" + str(value)
    if type(value) is str and len(value) <= 48:
        return "s" + value
    if type(value) is float and math.isfinite(value):
        return "f" + value.hex()
    if type(value) is tuple and len(value) <= 6 and all(type(n) is int and n.bit_length() <= 64 for n in value):
        return "t" + json.dumps(value, separators=(",", ":"))
    return "?"


def literal_operand(node):
    if not isinstance(node, (ast.Constant, ast.UnaryOp, ast.Tuple)):
        return None
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return None
    return value if scalar_key(value) != "?" else None


def operand_value(key):
    if key.startswith("i"):
        return int(key[1:])
    if key.startswith("f"):
        return float.fromhex(key[1:])
    if key.startswith("t"):
        return tuple(json.loads(key[1:]))
    return key[1:]


def changed_predicate(path, before, after, removed, maps):
    if len(removed) != 1:
        return None
    number = next(iter(removed))
    old_nodes = {(n.lineno,n.col_offset):n for n in spanning_nodes(before,number,number) if isinstance(n,ast.Compare)
                 and n.lineno <= number <= n.end_lineno and len(n.ops)==1 and type(n.ops[0]) in OPS}
    for new in spanning_nodes(after,number,number):
        if not isinstance(new,ast.Compare) or len(new.ops)!=1 or type(new.ops[0]) not in OPS:
            continue
        old = old_nodes.get((new.lineno,new.col_offset))
        if old is None or same_tree(old, new):
            continue
        for side in (0,1):
            prev = (old.left,old.comparators[0])[side]
            revised = (new.left,new.comparators[0])[side]
            args = (old.left,old.comparators[0])[1-side], (new.left,new.comparators[0])[1-side]
            old_literal, new_literal = literal_operand(prev), literal_operand(revised)
            if old_literal is None or type(old_literal) is not type(new_literal) or (not same_tree(args[0], args[1])):
                continue
            obs = maps["predicates"].get(f"{path}:{old.lineno}:{old.col_offset}:{side}")
            if not obs or not unchanged_with_original(after,before,new,
                                    dict(left=old.left,ops=old.ops,comparators=old.comparators)):
                continue
            mask = child_mask(path,maps)
            imports = 0
            for line in range(old.lineno,old.end_lineno+1):
                imports |= int(maps["imports"].get(path,{}).get(str(line),"0"),16)
            if old_literal == new_literal:
                imports = cold_order_mask(path, old, new, imports, maps)
            mask |= imports
            for key,bits in obs.items():
                keep = True
                if key != "?":
                    v = operand_value(key)
                    def outcome(node, literal):
                        op = OPS[type(node.ops[0])]
                        return op(v,literal) if side else op(literal,v)
                    try:
                        keep = outcome(old,old_literal) != outcome(new,new_literal)
                    except TypeError:
                        pass
                if keep:
                    mask |= int(bits,16)
            return mask
    return None


def module_edit(path, before, after, removed, maps):
    if not removed or not path.startswith(("sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/", "sqlglot/dialects/")):
        return None
    for old, new in zip(before.body, after.body):
        if type(old) is not type(new) or not isinstance(old, (ast.Assign, ast.AnnAssign)):
            continue
        if not old.lineno <= min(removed) <= max(removed) <= old.end_lineno:
            continue
        targets = old.targets if isinstance(old, ast.Assign) else [old.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        mask = maps["globals"].get(f"{path}:{old.lineno}:{targets[0].id}")
        if mask is None or not isinstance(old.value, ast.Constant) or not isinstance(new.value, ast.Constant):
            continue
        if type(old.value.value) not in (str, int, bool, float) or type(old.value.value) is not type(new.value.value):
            continue
        if unchanged_with_original(after, before, new, {"value": old.value}):
            return int(mask, 16) | child_mask(path, maps)
    return None


def config_edit(path, before, after, removed, maps):
    if not removed:
        return None
    old_nodes = {n.lineno:n for n in spanning_nodes(before,min(removed),max(removed)) if isinstance(n,ast.Assign)
                 and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                 and n.lineno <= min(removed) <= max(removed) <= n.end_lineno}
    for new in spanning_nodes(after,min(removed),max(removed)):
        if not isinstance(new,ast.Assign) or new.lineno not in old_nodes:
            continue
        old = old_nodes[new.lineno]
        site = maps["config"].get(f"{path}:{old.lineno}:{old.targets[0].id}")
        arguments = maps["arguments"].get(f"{path}:{old.lineno}") if old.targets[0].id == "arg_types" else None
        if (site is None and arguments is None) or not unchanged_with_original(after,before,new,{"value":old.value}):
            continue
        try:
            a,b = ast.literal_eval(old.value),ast.literal_eval(new.value)
        except (ValueError,TypeError):
            continue
        if (arguments is not None and isinstance(a, dict) and isinstance(b, dict)
                and list(a) == list(b) and all(type(v) is bool for v in (*a.values(), *b.values()))):
            changed = [key for key in a if a[key] != b[key]]
            masks = [arguments.get(key, "0") for key in ["!errors", *changed]]
            if "unknown" in masks:
                return None
            mask = child_mask(path, maps)
            for bits in masks:
                mask |= int(bits, 16)
            return mask
        if site is None:
            continue
        if type(a) in (int,bool,str,float) and type(a) is type(b) and "scalar" in site:
            return int(site["scalar"],16) | child_mask(path,maps)
        if (isinstance(a,dict) and isinstance(b,dict) and list(a)==list(b)
                and old.targets[0].id=="NUMERIC_ESCAPES"
                and all(isinstance(v,tuple) and len(v)==4 and all(type(i) is int for i in v)
                        for v in [*a.values(),*b.values()])):
            mask = child_mask(path,maps)
            for key in a:
                if a[key]!=b[key]:
                    mask |= int(site.get(key,"0"),16)
            return mask
    return None


def same_factory(old,new):
    if not isinstance(old,ast.Call) or not isinstance(new,ast.Call) or (not same_tree(old.func, new.func)):
        return False
    if (len(old.args)!=len(new.args) or [k.arg for k in old.keywords]!=[k.arg for k in new.keywords]):
        return False
    for a,b in zip(old.args+[k.value for k in old.keywords],new.args+[k.value for k in new.keywords]):
        if (not same_tree(a, b)) and not (isinstance(a,ast.Constant) and isinstance(b,ast.Constant)
                                              and type(a.value) is type(b.value)):
            return False
    return True


def callback_edit(path,before,after,removed,maps):
    if not removed or not (path.startswith(("sqlglot/parsers/","sqlglot/generators/")) or path in {"sqlglot/parser.py","sqlglot/generator.py"}):
        return None
    old_nodes = {(n.lineno,n.col_offset):n for n in spanning_nodes(before,min(removed),max(removed)) if isinstance(n,(ast.Lambda,ast.Call))
                 and n.lineno <= min(removed) <= max(removed) <= n.end_lineno}
    for parent in spanning_nodes(after,min(removed),max(removed)):
        if not isinstance(parent,ast.Dict):
            continue
        for i,node in enumerate(parent.values):
            old = old_nodes.get((node.lineno,node.col_offset))
            if old is None:
                continue
            site = maps["config"].get(f"{path}:{old.lineno}:{old.col_offset}:callback")
            if site is None:
                continue
            same_lambda = (isinstance(old,ast.Lambda) and isinstance(node,ast.Lambda)
                           and same_tree(old.args, node.args))
            if same_lambda or same_factory(old,node):
                values = parent.values.copy()
                values[i] = old
                if unchanged_with_original(after,before,parent,{"values":values}):
                    return int(site.get("call","0"),16) | child_mask(path,maps)
    return None

def lookup_edit(path, before, after, removed, maps):
    if not removed:
        return None
    def sites(tree):
        for call in spanning_nodes(tree,min(removed),max(removed)):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "get"
                    and 1 <= len(call.args) <= 2 and not call.keywords and isinstance(call.args[0], ast.Constant)
                    and type(call.args[0].value) is str and call.lineno <= min(removed) <= max(removed) <= call.end_lineno):
                yield (call.lineno, call.col_offset), call
    originals = dict(sites(before))
    if not originals:
        return None
    for loc, new in sites(after):
        old = originals.get(loc)
        profile = maps["lookups"].get(f"{path}:{loc[0]}:{loc[1]}")
        if old is None or profile is None or same_tree(old, new):
            continue
        if not unchanged_with_original(after, before, new, {"args": old.args}):
            continue
        keys = [old.args[0].value, new.args[0].value]
        defaults = [c.args[1] if len(c.args) == 2 else ast.Constant(None) for c in (old, new)]
        # Both defaults are evaluated before get(), even for a present key.
        # A new nonliteral expression may therefore have an independent effect.
        if (not same_tree(defaults[0], defaults[1])) and not all(isinstance(d, ast.Constant) for d in defaults):
            continue
        mask = child_mask(path, maps) | int(profile.get("unknown", "0"), 16)
        if keys[0] != keys[1]:
            mask |= int(profile.get("key:" + keys[0], "0"), 16) | int(profile.get("key:" + keys[1], "0"), 16)
        if (not same_tree(defaults[0], defaults[1])):
            mask |= int(profile.get("missing", "0"), 16)
        for line in range(old.lineno, old.end_lineno + 1):
            mask |= int(maps["imports"].get(path, {}).get(str(line), "0"), 16)
        return mask
    return None

def copy_edit(path, before, after, removed, maps):
    if not removed:
        return None

    def calls(tree):
        for call in spanning_nodes(tree,min(removed),max(removed)):
            if not isinstance(call, ast.Call) or not call.lineno <= min(removed) <= max(removed) <= call.end_lineno:
                continue
            name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
            if (name in {"alias_", "to_identifier", "maybe_parse"} and 1 <= len(call.args) <= 2
                    and not any(isinstance(a, ast.Starred) for a in call.args)
                    and all(k.arg is not None for k in call.keywords)):
                yield (call.lineno, call.col_offset), call

    original = dict(calls(before))
    if not original:
        return None
    for loc, new in calls(after):
        old = original.get(loc)
        profile = maps["copies"].get(f"{path}:{loc[0]}:{loc[1]}")
        if old is None or profile is None:
            continue
        values = [[k.value for k in call.keywords if k.arg == "copy"] for call in (old, new)]
        if (not any(values) or any(len(v) > 1 for v in values)
                or not all(isinstance(v, ast.Constant) and type(v.value) is bool for vs in values for v in vs)):
            continue
        if [v.value for v in values[0]] == [v.value for v in values[1]]:
            continue
        keywords = [[ast.dump(k) for k in call.keywords if k.arg != "copy"] for call in (old, new)]
        if keywords[0] != keywords[1] or not unchanged_with_original(after, before, new, {"keywords":old.keywords}):
            continue
        mask = child_mask(path,maps) | int(profile.get("other", "0"),16) | int(profile.get("unverified_factory", "0"),16)
        for line in range(old.lineno, old.end_lineno + 1):
            mask |= int(maps["imports"].get(path,{}).get(str(line),"0"),16)
        return mask
    return None

def comparison_outcome(op, identity, relation):
    if op is ast.Is:
        return identity
    if op is ast.IsNot:
        return not identity
    if relation == "?":
        return None
    if op is ast.Eq:
        return relation in {"=", "equal_only"}
    if op is ast.NotEq:
        return relation not in {"=", "equal_only"}
    if relation in {"equal_only", "unequal_only"}:
        return None
    orders = {ast.Lt: {"<"}, ast.LtE: {"<","="}, ast.Gt: {">"}, ast.GtE: {">","="}}
    return relation in orders[op] if op in orders else None


def cold_order_mask(path, old, new, imports, maps):
    """Narrow import-time hits using operands from independently imported roots."""
    site = f"{path}:{old.lineno}:{old.col_offset}"
    rows = maps["cold_orders"].get(site)
    if not imports or not rows:
        return imports
    seen = changed = 0
    for key, bits in rows.items():
        mask = int(bits, 16)
        seen |= mask
        identity, relation = key.split("/")
        a = comparison_outcome(type(old.ops[0]), identity == "same", relation)
        b = comparison_outcome(type(new.ops[0]), identity == "same", relation)
        if a is None or b is None or a != b:
            changed |= mask
    return imports & (~seen | changed)


def deferred_nodes(tree):
    """Nodes that run only when a function or lambda is called.

    Module and class bodies, decorators, defaults and evaluated annotations run
    during import, before or outside any test, so their line coverage does not
    list the tests that observe them.
    """
    found = set()
    def visit(node, inside):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for item in [*node.decorator_list, node.args, *([node.returns] if node.returns else [])]:
                visit(item, inside)
            for item in node.body:
                visit(item, True)
            return
        if isinstance(node, ast.Lambda):
            visit(node.args, inside)
            visit(node.body, True)
            return
        if inside:
            found.add(id(node))
        for child in ast.iter_child_nodes(node):
            visit(child, inside)
    visit(tree, False)
    return found


def default_edit(path, before, after, removed, maps):
    """A certified parameter treats every falsy immutable default alike."""
    if len(removed) != 1:
        return None
    line = next(iter(removed))
    def values(tree):
        for fn in spanning_nodes(tree,line,line):
            if not isinstance(fn,ast.FunctionDef) or line >= fn.body[0].lineno:
                continue
            args = fn.args.posonlyargs + fn.args.args
            for arg, default in [*zip(args[len(args)-len(fn.args.defaults):],fn.args.defaults),
                                 *zip(fn.args.kwonlyargs,fn.args.kw_defaults)]:
                if (isinstance(default, ast.Constant) and default.lineno == line and
                        type(default.value) in (type(None),int,bool,str,float) and not default.value):
                    yield (fn.lineno,arg.arg), default
    old = dict(values(before))
    for key, node in values(after):
        prev = old.get(key)
        if prev is not None and f"{path}:{key[0]}:{key[1]}" in maps["default_sites"] and (
                type(node.value) is not type(prev.value) or node.value != prev.value):
            if unchanged_with_original(after,before,node,{"value":prev.value}):
                return maps["default_observers"] | child_mask(path,maps)
    return None


def conditional_edit(path, before, after, removed, maps):
    """Inverting a conditional preserves its value when both local arms are identical."""
    if len(removed) != 1:
        return None
    line = next(iter(removed))
    def choices(tree):
        return {n.col_offset:n for n in spanning_nodes(tree,line,line)
                if isinstance(n,ast.IfExp) and n.lineno == n.end_lineno == line
                and isinstance(n.body,ast.Name) and isinstance(n.orelse,ast.Name)}
    for col, old in choices(before).items():
        key = f"{path}:{line}:{col}"
        if key not in maps["choices"]:
            continue
        new = choices(after).get(col)
        if new is None or new.body.id != old.body.id or new.orelse.id != old.orelse.id:
            continue
        enclosing = [n for n in spanning_nodes(before,line,line) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        if not enclosing:
            continue
        func = min(enclosing,key=lambda n:n.end_lineno-n.lineno)
        arguments = {a.arg for a in [*func.args.args,*func.args.posonlyargs,*func.args.kwonlyargs]}
        if old.body.id not in arguments or old.orelse.id not in arguments:
            continue
        a,b = old.test,new.test
        if not any(isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.Not) and same_tree(x.operand, y)
                   for x,y in ((a,b),(b,a))):
            continue
        condition = a.operand if isinstance(a,ast.UnaryOp) and isinstance(a.op,ast.Not) else a
        # The observer snapshots local bindings before the truth test. An
        # identity comparison returns an exact bool with no user truth hook.
        if not (isinstance(condition,ast.Compare) and len(condition.ops)==1
                and isinstance(condition.ops[0],(ast.Is,ast.IsNot))):
            continue
        if unchanged_with_original(after,before,new,{"test":old.test}):
            risk = int(maps["choices"][key],16) | maps["unmeasured_branch"] | child_mask(path,maps)
            reached = int(maps["lines"].get(path,{}).get(str(line),"0"),16)
            return reached & risk | int(maps["imports"].get(path,{}).get(str(line),"0"),16)
    return None


def changed_outcomes(profile, old, new):
    risk = 0
    for key, bits in profile.items():
        identity, relation = key.split("/")
        previous = comparison_outcome(old,identity == "same",relation)
        revised = comparison_outcome(new,identity == "same",relation)
        if previous is None or revised is None or previous != revised:
            risk |= int(bits,16)
    return risk


def operation_edit(path, before, after, removed, maps):
    """An isolated operator or literal-argument edit first takes effect at its instruction."""
    if len(removed) != 1 or path not in maps["instructions"]:
        return None
    deferred = deferred_nodes(before)
    line = next(iter(removed))
    old_nodes = {(type(n), n.col_offset): n for n in spanning_nodes(before,line,line)
                 if isinstance(n,(ast.Compare,ast.BinOp,ast.UnaryOp,ast.Call)) and n.lineno == n.end_lineno == line}
    for new in spanning_nodes(after,line,line):
        if not isinstance(new,(ast.Compare,ast.BinOp,ast.UnaryOp,ast.Call)) or new.lineno != new.end_lineno:
            continue
        old = old_nodes.get((type(new),new.col_offset))
        if old is None or same_tree(new, old):
            continue
        if id(old) not in deferred:
            return None
        key = f"{type(old).__name__}:{line}:{old.col_offset}:{old.end_col_offset}"
        if key not in maps["instructions"][path]:
            continue
        if isinstance(old,ast.Compare):
            if len(old.ops) != 1 or len(new.ops) != 1:
                continue
            replacements = {"ops":old.ops}
        elif isinstance(old,ast.Call):
            if (len(old.args) != len(new.args) or [k.arg for k in old.keywords] != [k.arg for k in new.keywords]
                    or any((not same_tree(a, b)) and not (isinstance(a,ast.Constant) and isinstance(b,ast.Constant))
                           for a,b in zip(old.args + [k.value for k in old.keywords],new.args + [k.value for k in new.keywords]))):
                continue
            replacements = {"args":old.args,"keywords":old.keywords}
        else:
            replacements = {"op":old.op}
        if unchanged_with_original(after,before,new,replacements):
            reached = int(maps["lines"].get(path,{}).get(str(line),"0"),16)
            imports = int(maps["imports"].get(path,{}).get(str(line),"0"),16)
            if isinstance(old, ast.Compare):
                imports = cold_order_mask(path, old, new, imports, maps)
                site = f"{path}:{line}:{old.col_offset}"
                for source, unmeasured in (("orders","unmeasured_orders"), ("branch_orders","unmeasured_branch")):
                    profile = maps[source].get(site)
                    if profile is not None:
                        risk = changed_outcomes(profile,type(old.ops[0]),type(new.ops[0]))
                        reached &= risk | child_mask(path,maps) | maps[unmeasured]
            return reached | imports


def _setting_name(node):
    if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool)):
        return node.targets[0].id
    if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.simple
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool)
            and "ClassVar" in ast.dump(node.annotation)):
        return node.target.id
    return None


def flag_edit(path, before, after, removed, maps, dynamic_names):
    """Select tests whose reads a flipped boolean class setting could change.

    The flip profile records, per setting, the tests where flipping the value
    changes control flow: every plain read, and an "and"/"or" operand read only
    when the observed truthiness of the other operands makes the outcome (or the
    operands evaluated) depend on it.
    """
    flips = maps.get("flips")
    if not flips or not removed:
        return None
    for cls in ast.walk(before):
        if not isinstance(cls, ast.ClassDef):
            continue
        for old in cls.body:
            name = _setting_name(old)
            if name is None or not old.lineno <= min(removed) <= max(removed) <= old.end_lineno:
                continue
            site = f"{path}:{old.lineno}:{name}"
            if site not in flips or name in dynamic_names:
                return None
            for new in ast.walk(after):
                if (type(new) is type(old) and new.lineno == old.lineno and _setting_name(new) == name
                        and new.value.value != old.value.value
                        and unchanged_with_original(after, before, new, {"value": old.value})):
                    return int(flips[site], 16) | child_mask(path, maps)
            return None
    return None

def metadata_callback_edit(path, before, after, removed, maps):
    """Confine one metadata key or callback body edit to readers of that entry."""
    if not path.startswith("sqlglot/typing/") or not removed:
        return None
    def entries(tree):
        for stmt in tree.body:
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if not any(isinstance(t, ast.Name) and t.id == "EXPRESSION_METADATA" for t in targets):
                continue
            if not isinstance(stmt.value, ast.Dict):
                continue
            for spec in stmt.value.values:
                if (isinstance(spec, ast.Dict) and len(spec.keys) == 1
                        and isinstance(spec.keys[0], ast.Constant) and type(spec.keys[0].value) is str
                        and isinstance(spec.values[0], ast.Lambda)
                        and spec.lineno <= min(removed) <= max(removed) <= spec.end_lineno):
                    yield spec
    for old, new in zip(entries(before), entries(after)):
        if old.keys[0].value != "annotator":
            continue
        previous, revised = old.values[0], new.values[0]
        if (not same_tree(previous.args, revised.args)):
            continue
        if new.keys[0].value != "annotator" and (not same_tree(previous, revised)):
            continue
        if not unchanged_with_original(after, before, new, {"keys": old.keys, "values": old.values}):
            continue
        site = f"{path}:{previous.lineno}"
        hits = int(maps["typing_calls"].get(site, "0"), 16)
        if hits < 0:
            return None
        return hits | maps["metadata_readers"] | child_mask(path, maps) | child_mask("sqlglot/optimizer/annotate_types.py", maps)
    return None

def keyword_edit(path, before, after, removed, maps):
    """Use measured tokenizer lookups for a plain word renamed in a keyword table."""
    if len(removed) != 1 or not (path == "sqlglot/tokens.py" or path.startswith("sqlglot/dialects/")):
        return None
    line = next(iter(removed))
    def entry(tree):
        for stmt in spanning_nodes(tree, line, line):
            if isinstance(stmt, ast.AnnAssign):
                target = stmt.target
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
            else:
                continue
            if not (isinstance(target, ast.Name) and target.id == "KEYWORDS" and isinstance(stmt.value, ast.Dict)):
                continue
            for key, value in zip(stmt.value.keys, stmt.value.values):
                if (isinstance(key, ast.Constant) and key.lineno == line and isinstance(key.value, str)
                        and key.value.isascii() and key.value.isidentifier()):
                    yield key, value, stmt.value
    profile = maps["word_profile"]
    for (old, old_value, table), (new, new_value, _) in zip(entry(before), entry(after)):
        words = {old.value, new.value}
        if words.intersection(profile["reserved"]):
            continue
        if any(symbol in word for symbol in profile["single"] for word in words):
            continue
        if len(words) == 2:
            unchanged = unchanged_with_original(after, before, new, {"value": old.value})
        else:
            # Only accept enum members already resolved by this pinned table.
            if not (isinstance(old_value, ast.Attribute) and isinstance(new_value, ast.Attribute)
                    and isinstance(new_value.value, ast.Name) and new_value.value.id == "TokenType"
                    and same_tree(old_value.value, new_value.value)
                    and any(isinstance(v, ast.Attribute) and same_tree(v, new_value)
                            for v in table.values)):
                continue
            unchanged = unchanged_with_original(after, before, new_value, {"attr": old_value.attr})
        if not unchanged:
            continue
        hits = child_mask(path, maps) | child_mask("sqlglot/tokenizer_core.py", maps)
        for word in words:
            observed = int(profile["words"].get(word, "0"), 16)
            if observed < 0:
                return None
            hits |= observed
        return hits


def membership_edit(path, before, after, removed, maps):
    """Changing elements of a literal collection in `x in (...)` matters only for those values.

    The membership profile records, per test, the values tested at each such
    site. Only tests that observed a value equal to an added or removed element
    can see a different outcome; non-literal observations count as unknown.
    """
    members = maps.get("members")
    if not members or len(removed) != 1:
        return None
    line = next(iter(removed))
    def sites(tree):
        return {n.col_offset: n for n in spanning_nodes(tree, line, line)
                if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn))
                and n.lineno == line and isinstance(n.comparators[0], (ast.Tuple, ast.List, ast.Set))
                and n.comparators[0].elts
                and all(isinstance(e, ast.Constant) and type(e.value) in (str, int) for e in n.comparators[0].elts)}
    olds, news = sites(before), sites(after)
    for col, old in olds.items():
        new = news.get(col)
        if new is None or same_tree(new, old) or type(new.ops[0]) is not type(old.ops[0]):
            continue
        if type(new.comparators[0]) is not type(old.comparators[0]) or (not same_tree(new.left, old.left)):
            continue
        if not unchanged_with_original(after, before, new, {"comparators": old.comparators}):
            continue
        site = f"{path}:{old.lineno}:{old.col_offset}"
        observed = members["sites"].get(site)
        if site in members["overflow"]:
            return None
        from bindings import load_refs
        refs = load_refs(maps)
        if not refs or old.lineno in set(refs.get("cached", {}).get(path, ())):
            # Inside a function that memoizes across tests, only the first
            # caller records the value that later tests reuse.
            return None
        oldset = {("s" if isinstance(e.value, str) else "i") + str(e.value) for e in old.comparators[0].elts}
        newset = {("s" if isinstance(e.value, str) else "i") + str(e.value) for e in new.comparators[0].elts}
        flipped = oldset ^ newset
        if flipped & set(members["imported"].get(site, ())) or "?" in members["imported"].get(site, ()):
            return None
        mask = child_mask(path, maps)
        for key, bits in (observed or {}).items():
            if key == "?" or key in flipped:
                mask |= int(bits, 16)
        return mask
    return None


TRUTH_SWAPS = ({ast.Eq, ast.NotEq}, {ast.Is, ast.IsNot}, {ast.In, ast.NotIn})


def truth_only(old, new):
    """True when new can differ from old only in truthiness, never in what it evaluates."""
    for a, b in ((old, new), (new, old)):
        if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.Not) and same_tree(a.operand, b):
            return True
    if (isinstance(old, ast.Constant) and isinstance(new, ast.Constant)
            and isinstance(old.value, bool) and isinstance(new.value, bool)):
        return True
    if (isinstance(old, ast.Compare) and isinstance(new, ast.Compare) and len(old.ops) == len(new.ops) == 1
            and same_tree(old.left, new.left)
            and same_tree(old.comparators[0], new.comparators[0])):
        return any({type(old.ops[0]), type(new.ops[0])} <= pair for pair in TRUTH_SWAPS)
    return False


def boolop_edit(path, before, after, removed, inserted, maps):
    """Tests that evaluated the one and/or operand an edit is confined to.

    The operand profile lists, per operand, the tests that evaluated it and the
    tests where flipping its truthiness could change the operator's outcome,
    the side-effecting operands it runs, or its value. An edit inside an operand
    a test never evaluated cannot change that test.
    """
    profile = maps.get("boolops")
    if not profile or inserted or not removed:
        return None
    first, last = min(removed), max(removed)
    olds = {(n.lineno, n.col_offset): n for n in ast.walk(before) if isinstance(n, ast.BoolOp)
            and n.lineno <= first and last <= n.end_lineno}
    best = None
    for new in ast.walk(after):
        if not isinstance(new, ast.BoolOp):
            continue
        old = olds.get((new.lineno, new.col_offset))
        if (old is None or type(old.op) is not type(new.op) or len(old.values) != len(new.values)
                or same_tree(old, new)):
            continue
        changed = [i for i, (a, b) in enumerate(zip(old.values, new.values)) if (not same_tree(a, b))]
        if len(changed) != 1:
            continue
        i = changed[0]
        if not (old.values[i].lineno <= first and last <= old.values[i].end_lineno):
            continue
        values = list(new.values)
        values[i] = old.values[i]
        if not unchanged_with_original(after, before, new, {"values": values}):
            continue
        if best is None or old.end_lineno - old.lineno < best[0].end_lineno - best[0].lineno:
            best = (old, new, i)
    if best is None:
        return swapped_boolop(path, before, after, removed, maps, olds)
    old, new, i = best
    site = f"{path}:{old.lineno}:{old.col_offset}"
    if i in set(profile["imported"].get(site, ())):
        return None
    from bindings import load_refs
    refs = load_refs(maps)
    if not refs or any(n in set(refs.get("cached", {}).get(path, ())) for n in range(old.lineno, old.end_lineno + 1)):
        return None
    table = profile["unsafe"] if truth_only(old.values[i], new.values[i]) else profile["ev"]
    return (int(table.get(site, {}).get(str(i), "0"), 16) | int(profile["unprofiled"], 16)
            | child_mask(path, maps))



def swapped_boolop(path, before, after, removed, maps, olds):
    """Tests with an evaluation that swapping an operator between and and or could change."""
    profile = maps["boolops"]
    for new in ast.walk(after):
        if not isinstance(new, ast.BoolOp):
            continue
        old = olds.get((new.lineno, new.col_offset))
        if (old is None or type(old.op) is type(new.op) or len(old.values) != len(new.values)
                or any((not same_tree(a, b)) for a, b in zip(old.values, new.values))):
            continue
        if not unchanged_with_original(after, before, new, {"op": old.op}):
            continue
        site = f"{path}:{old.lineno}:{old.col_offset}"
        if profile["imported"].get(site) or "swap" not in profile:
            return None
        from bindings import load_refs
        refs = load_refs(maps)
        if not refs or any(n in set(refs.get("cached", {}).get(path, ()))
                           for n in range(old.lineno, old.end_lineno + 1)):
            return None
        return int(profile["swap"].get(site, "0"), 16) | int(profile["unprofiled"], 16) | child_mask(path, maps)
    return None
