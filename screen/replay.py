"""Run inexpensive candidate groups in disposable copies of the patched source tree."""
import atexit
import os
import json
from pathlib import Path
import signal
import multiprocessing
import sys
import tempfile
import importlib
import unittest


def overlay(source, dest, changes):
    """Copy modified text and link untouched siblings into a private working tree."""
    branches = {}
    for path, text in changes.items():
        first, _, rest = path.partition("/")
        branches.setdefault(first,{})[rest] = text
    for child in source.iterdir():
        if child.name == "__pycache__":
            # Import unchanged siblings from the pinned precompiled tree. A patched
            # sibling always compiles from its new source, even for same-size edits.
            excluded = {Path(name).stem for name in branches if name.endswith(".py")}
            cache = dest / child.name
            if excluded:
                cache.mkdir()
                for pyc in child.iterdir():
                    if pyc.name.split(".", 1)[0] not in excluded:
                        (cache / pyc.name).symlink_to(pyc)
            else:
                cache.symlink_to(child, target_is_directory=True)
            continue
        target = dest / child.name
        if child.name not in branches:
            target.symlink_to(child, target_is_directory=child.is_dir())
        elif child.is_dir():
            target.mkdir()
            overlay(child,target,branches[child.name])
        else:
            target.write_text(branches[child.name][""],encoding="utf-8")


_server = None


def run_child(root, paths, ids, seconds, source=None):
    proc = multiprocessing.get_context("fork").Process(target=run_tests, args=(root, paths, ids, source))
    proc.start()
    proc.join(seconds)
    if proc.is_alive():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
        proc.join()
    success = proc.exitcode == 0
    proc.close()
    return success


def warm_manager(source, connection):
    # This process imports only the pinned package. Every test and every
    # patch, including sys.path and package-path edits, lives in a fork.
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    accessed = set()
    collecting = True
    def audit(event, args):
        if collecting and event == "open" and isinstance(args[0], (str, bytes)):
            accessed.add(os.path.abspath(os.fsdecode(args[0])))
    sys.addaudithook(audit)
    if any(name == "sqlglot" or name.startswith(("sqlglot.", "tests.")) for name in sys.modules):
        connection.send(None)
        return
    sys.path.insert(0, str(source))
    os.chdir(source)
    os.environ["SKIP_INTEGRATION"] = "0"
    importlib.import_module("sqlglot")
    collecting = False
    loaded = {name for name in sys.modules if name == "sqlglot" or name.startswith("sqlglot.")}
    connection.send((loaded, accessed))
    while True:
        msg = connection.recv()
        if msg is None:
            return
        root, paths, ids, seconds = msg
        connection.send(run_child(root, paths, ids, seconds, source))


def warm_server(source):
    global _server
    if _server is None:
        ctx = multiprocessing.get_context("fork")
        client, worker = ctx.Pipe()
        proc = ctx.Process(target=warm_manager, args=(source, worker))
        proc.start()
        worker.close()
        def stop():
            try:
                client.send(None)
            except (BrokenPipeError, OSError):
                pass
            proc.join(0.3)
            if proc.is_alive():
                proc.kill()
                proc.join()
            client.close()
        atexit.register(stop)
        state = client.recv() if client.poll(2) else None
        _server = (client, *state) if state else False
    return _server


def warm_run(source, dest, paths, ids, seconds):
    global _server
    server = warm_server(source)
    if server:
        client, loaded, accessed = server
        eligible = all(path[:-3].replace("/", ".").removesuffix(".__init__") not in loaded
                       and str(source / path) not in accessed for path in paths)
        if eligible:
            client.send((dest, paths, ids, seconds))
            if client.poll(seconds + 1.5):
                return client.recv()
            _server = False
            return False
    return run_child(dest, paths, ids, seconds)


def failing_tests(source, files, ids, seconds):
    try:
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)
            overlay(source,dest,{path:new for path,old,new,*_ in files})
            if not warm_run(source, dest, [path for path,*_ in files], ids, seconds):
                return None
            failed = set(json.loads((dest/"replay-result.json").read_text()))
            return failed if failed <= set(ids) else None
    except (EOFError, OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None


def run_tests(root, paths, ids, source=None):
    os.setsid()
    with open(os.devnull,"wb") as sink:
        os.dup2(sink.fileno(),1)
        os.dup2(sink.fileno(),2)
    os.chdir(root)
    sys.path.insert(0,str(root))
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONPATH=str(root), SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1")

    if source is not None:
        for name, module in list(sys.modules.items()):
            if name == "sqlglot" or name.startswith("sqlglot."):
                directory = root.joinpath(*name.split("."))
                if hasattr(module, "__path__"):
                    module.__path__ = [str(directory)]
                    module.__spec__.submodule_search_locations = module.__path__
                filename = getattr(module, "__file__", None)
                if filename and Path(filename).is_relative_to(source):
                    module.__file__ = str(root / Path(filename).relative_to(source))
                    module.__spec__.origin = module.__file__
        changed = {str(source / p) for p in paths}
        basenames = {Path(p).name for p in paths}
        bypassed = []
        def guard(event, args):
            if event == "open" and isinstance(args[0], str) and os.path.basename(args[0]) in basenames:
                if os.path.realpath(args[0]) in changed:
                    bypassed.append(args[0])
        sys.addaudithook(guard)

    suite = unittest.TestSuite()
    labels = {}
    for label in sorted(ids):
        filename,classname,method = label.split("::")
        module = importlib.import_module(filename[:-3].replace("/","."))
        cls = getattr(module,classname)
        if (not issubclass(cls,unittest.TestCase) or not method.startswith("test")
                or hasattr(module,"load_tests") or hasattr(cls,"pytestmark")
                or hasattr(getattr(cls,method),"pytestmark")):
            sys.exit(1)
        test = cls(method)
        suite.addTest(test)
        labels[test] = label

    failed = set()
    completed = set()
    class Recorder(unittest.TestResult):
        def stopTest(self,test):
            completed.add(labels[test])
            super().stopTest(test)

        def addSubTest(self,test,subtest,error):
            if error is not None:
                failed.add(labels[test])

        def addError(self,test,error):
            failed.add(labels[test])

        def addFailure(self,test,error):
            failed.add(labels[test])

    recorder = Recorder()
    suite.run(recorder)
    verified = source is None or not bypassed
    for path in paths:
        name = path[:-3].replace("/",".").removesuffix(".__init__")
        loaded = getattr(sys.modules.get(name),"__file__",None)
        if not loaded or Path(loaded).resolve() != root/path:
            verified = False
            break
    if (not verified or recorder.skipped or recorder.expectedFailures or recorder.unexpectedSuccesses
            or set(ids) != completed):
        sys.exit(1)
    (root/"replay-result.json").write_text(json.dumps(sorted(failed)))
